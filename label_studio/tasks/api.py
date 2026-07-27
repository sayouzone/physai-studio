"""This file and its contents are licensed under the Apache License 2.0. Please see the included NOTICE for copyright information and LICENSE for a copy of the license."""

from importlib import metadata
import logging
import os

from core.feature_flags import flag_set
from core.mixins import GetParentObjectMixin
from core.permissions import ViewClassPermission, all_permissions
from core.utils.common import is_community
from core.utils.params import bool_from_request
from data_manager.api import TaskListAPI as DMTaskListAPI
from data_manager.functions import evaluate_predictions
from data_manager.models import PrepareParams
from data_manager.serializers import DataManagerTaskSerializer
from django.conf import settings
from django.db import transaction
from django.db.models import Q
from django.utils import timezone
from django.utils.decorators import method_decorator
from django_filters.rest_framework import DjangoFilterBackend
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiExample, OpenApiParameter, OpenApiResponse, extend_schema
from projects.functions.stream_history import fill_history_annotation
from projects.models import Project
from rest_framework import generics, viewsets
from rest_framework.exceptions import PermissionDenied, ValidationError
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from tasks.models import Annotation, AnnotationDraft, Prediction, Task
from tasks.openapi_schema import (
    annotation_request_schema,
    annotation_response_example,
    dm_task_response_example,
    prediction_request_schema,
    prediction_response_example,
    task_request_schema,
    task_response_example,
)
from tasks.serializers import (
    AnnotationDraftSerializer,
    AnnotationSerializer,
    PredictionSerializer,
    TaskSerializer,
    TaskSimpleSerializer,
    TaskGeoreferencingSerializer,
)
from webhooks.models import WebhookAction
from webhooks.utils import (
    api_webhook,
    api_webhook_for_delete,
    emit_webhooks_for_instance,
)

from typing import Optional, Sequence, Dict, Any

from sayou.annotation_georeferencer import AnnotationGeoreferencer
from sayou.georeferencing import coordinates
from sayou.georeferencing.footprint import compare_footprints

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 사이트 투영 평면 설정
# ─────────────────────────────────────────────────────────────────────────────
# 검출 대상은 지면이 아니라 '패널 상면'이므로 광선-평면 교차의 평면을
# 패널 상면에 두어야 한다.
#
# 절대고도(panel_plane_altitude)는 지형 표고에 종속되므로 전역 하드코딩 금지.
# 이식 가능한 값은 '이륙지점 지면 대비 패널 상면 높이'다.
# 값이 None 이면 georeferencer.SITE_PANEL_HEIGHT_ABOVE_GROUND 가 쓰인다.
#
# 사이트마다 다르면 Project 에 필드를 두고 _plane_kwargs() 에서 읽어오면 된다.
DEFAULT_PANEL_HEIGHT_ABOVE_GROUND: Optional[float] = None  # None = 모듈 기본값(1.48)
 
 
def _plane_kwargs(project=None) -> Dict[str, Any]:
    """
    georeferencing 투영 평면 인자. **모든 호출부가 이 함수 하나를 써야 한다.**
    검출 폴리곤 / footprint / 오버레이가 서로 다른 평면을 쓰면 조용히 어긋난다.
    """
    # 프로젝트별 설정이 생기면 여기서 우선 적용한다.
    #   height = getattr(project, 'panel_height_above_ground_m', None)
    #   if height is not None:
    #       return {'panel_height_above_ground': float(height)}
    if DEFAULT_PANEL_HEIGHT_ABOVE_GROUND is not None:
        return {'panel_height_above_ground': DEFAULT_PANEL_HEIGHT_ABOVE_GROUND}
    return {}
 
 
def resolve_task_image_path(task) -> str:
    """Task 의 이미지 로컬 경로. (기존 로직 유지, 방어 코드만 추가)"""
    base_data_dir = os.environ.get('LABEL_STUDIO_BASE_DATA_DIR')
    filename = task.data.get('image') or task.data.get('$undefined$')
    if not filename:
        raise ValueError(f'task={task.id}: task.data 에 image 경로가 없습니다')
    filename = filename.split('/data/upload/')[-1]
    return os.path.expanduser(f'{base_data_dir}/media/upload/{filename}')


# TODO: fix after switch to api/tasks from api/dm/tasks
@method_decorator(
    name='post',
    decorator=extend_schema(
        tags=['Tasks'],
        summary='Create task',
        description='Create a new labeling task in Label Studio.',
        request={
            'application/json': task_request_schema,
        },
        responses={
            '201': OpenApiResponse(
                description='Created task',
                response=TaskSerializer,
                examples=[OpenApiExample(name='response', value=task_response_example, media_type='application/json')],
            )
        },
        extensions={
            'x-fern-sdk-group-name': 'tasks',
            'x-fern-sdk-method-name': 'create',
            'x-fern-audiences': ['public'],
        },
    )
    if is_community()
    else lambda f: f,
)
@method_decorator(
    name='get',
    decorator=extend_schema(
        tags=['Tasks'],
        summary='Get tasks list',
        description="""
    Retrieve a list of tasks with pagination for a specific view or project, by using filters and ordering.
    """,
        parameters=[
            OpenApiParameter(name='view', type=OpenApiTypes.INT, location='query', description='View ID'),
            OpenApiParameter(name='project', type=OpenApiTypes.INT, location='query', description='Project ID'),
            OpenApiParameter(
                name='resolve_uri',
                type=OpenApiTypes.BOOL,
                location='query',
                description='Resolve task data URIs using Cloud Storage',
            ),
            OpenApiParameter(
                name='fields',
                type=OpenApiTypes.STR,
                enum=['all', 'task_only'],
                default='task_only',
                location='query',
                description='Set to "all" if you want to include annotations and predictions in the response',
            ),
            OpenApiParameter(
                name='review',
                type=OpenApiTypes.BOOL,
                location='query',
                description='Get tasks for review',
            ),
            OpenApiParameter(
                name='include',
                type=OpenApiTypes.STR,
                location='query',
                description='Specify which fields to include in the response',
            ),
            OpenApiParameter(
                name='query',
                type=OpenApiTypes.STR,
                location='query',
                description='Additional query to filter tasks. It must be JSON encoded string of dict containing '
                'one of the following parameters: `{"filters": ..., "selectedItems": ..., "ordering": ...}`. Check '
                '[Data Manager > Create View > see `data` field](#tag/Data-Manager/operation/api_dm_views_create) '
                'for more details about filters, selectedItems and ordering.\n\n'
                '* **filters**: dict with `"conjunction"` string (`"or"` or `"and"`) and list of filters in `"items"` array. '
                'Each filter is a dictionary with keys: `"filter"`, `"operator"`, `"type"`, `"value"`. '
                '[Read more about available filters](https://labelstud.io/sdk/data_manager.html)<br/>'
                '                   Example: `{"conjunction": "or", "items": [{"filter": "filter:tasks:completed_at", "operator": "greater", "type": "Datetime", "value": "2021-01-01T00:00:00.000Z"}]}`\n'
                '* **selectedItems**: dictionary with keys: `"all"`, `"included"`, `"excluded"`. If "all" is `false`, `"included"` must be used. If "all" is `true`, `"excluded"` must be used.<br/>'
                '                   Examples: `{"all": false, "included": [1, 2, 3]}` or `{"all": true, "excluded": [4, 5]}`\n'
                '* **ordering**: list of fields to order by. Currently, ordering is supported by only one parameter. <br/>\n'
                '                   Example: `["completed_at"]`',
            ),
        ],
        responses={
            '200': OpenApiResponse(
                description='Tasks list',
                response={
                    'type': 'object',
                    'properties': {
                        'tasks': {
                            'description': 'List of tasks',
                            'type': 'array',
                            'items': {
                                'description': 'Task object',
                                'type': 'object',
                            },
                        },
                        'total': {
                            'description': 'Total number of tasks',
                            'type': 'integer',
                        },
                        'total_annotations': {
                            'description': 'Total number of annotations',
                            'type': 'integer',
                        },
                        'total_predictions': {
                            'description': 'Total number of predictions',
                            'type': 'integer',
                        },
                    },
                },
            )
        },
        extensions={
            'x-fern-sdk-group-name': 'tasks',
            'x-fern-sdk-method-name': 'list',
            'x-fern-pagination': {
                'offset': '$request.page',
                'results': '$response.tasks',
            },
            'x-fern-audiences': ['public'],
        },
    )
    if is_community()
    else lambda f: f,
)
class TaskListAPI(DMTaskListAPI):
    serializer_class = TaskSerializer
    permission_required = ViewClassPermission(
        GET=all_permissions.tasks_view,
        POST=all_permissions.tasks_create,
    )
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['project']

    def filter_queryset(self, queryset):
        queryset = super().filter_queryset(queryset)
        return queryset.filter(project__organization=self.request.user.active_organization)

    def get_serializer_context(self):
        context = super().get_serializer_context()
        project_id = self.request.data.get('project')
        if project_id:
            context['project'] = generics.get_object_or_404(Project, pk=project_id)
        return context

    def perform_create(self, serializer):
        project_id = self.request.data.get('project')
        project = generics.get_object_or_404(Project, pk=project_id)
        instance = serializer.save(project=project)
        emit_webhooks_for_instance(
            self.request.user.active_organization, project, WebhookAction.TASKS_CREATED, [instance]
        )


@method_decorator(
    name='get',
    decorator=extend_schema(
        tags=['Tasks'],
        summary='Get task',
        description="""
        Get task data, metadata, annotations and other attributes for a specific labeling task by task ID.
        """,
        parameters=[
            OpenApiParameter(name='id', type=OpenApiTypes.STR, location='path', description='Task ID'),
        ],
        request=None,
        responses={
            '200': OpenApiResponse(
                description='Task',
                response=DataManagerTaskSerializer,
                examples=[
                    OpenApiExample(name='response', value=dm_task_response_example, media_type='application/json')
                ],
            )
        },
        extensions={
            'x-fern-sdk-group-name': 'tasks',
            'x-fern-sdk-method-name': 'get',
            'x-fern-audiences': ['public'],
        },
    ),
)
@method_decorator(
    name='patch',
    decorator=extend_schema(
        tags=['Tasks'],
        summary='Update task',
        description='Update the attributes of an existing labeling task.',
        parameters=[
            OpenApiParameter(name='id', type=OpenApiTypes.STR, location='path', description='Task ID'),
        ],
        request={
            'application/json': task_request_schema,
        },
        responses={
            '200': OpenApiResponse(
                description='Updated task',
                response=TaskSerializer,
                examples=[OpenApiExample(name='response', value=task_response_example, media_type='application/json')],
            )
        },
        extensions={
            'x-fern-sdk-group-name': 'tasks',
            'x-fern-sdk-method-name': 'update',
            'x-fern-audiences': ['public'],
        },
    ),
)
@method_decorator(
    name='delete',
    decorator=extend_schema(
        tags=['Tasks'],
        summary='Delete task',
        description='Delete a task in Label Studio. This action cannot be undone!',
        parameters=[
            OpenApiParameter(name='id', type=OpenApiTypes.STR, location='path', description='Task ID'),
        ],
        request=None,
        extensions={
            'x-fern-sdk-group-name': 'tasks',
            'x-fern-sdk-method-name': 'delete',
            'x-fern-audiences': ['public'],
        },
    ),
)
class TaskAPI(generics.RetrieveUpdateDestroyAPIView):
    parser_classes = (JSONParser, FormParser, MultiPartParser)
    permission_required = ViewClassPermission(
        GET=all_permissions.tasks_view,
        PUT=all_permissions.tasks_change,
        PATCH=all_permissions.tasks_change,
        DELETE=all_permissions.tasks_delete,
    )

    def initial(self, request, *args, **kwargs):
        super().initial(request, *args, **kwargs)
        self.task = self.get_object()

    def prefetch(self, queryset):
        return queryset.prefetch_related(
            'annotations',
            'predictions',
            'annotations__completed_by',
            'project',
            'io_storages_azureblobimportstoragelink',
            'io_storages_gcsimportstoragelink',
            'io_storages_localfilesimportstoragelink',
            'io_storages_redisimportstoragelink',
            'io_storages_s3importstoragelink',
            'file_upload',
            'project__ml_backends',
        )

    def get_retrieve_serializer_context(self, request):
        """Build serializer context for task retrieval.

        The resolve_uri parameter controls whether storage URLs (e.g., s3://bucket/file.jpg)
        are converted to proxy URLs (/tasks/<id>/resolve/?fileuri=...). This is useful for:
        - resolve_uri=True (default): URLs are proxied through Label Studio for security
        - resolve_uri=False: Original storage URLs are preserved, useful for debugging
          or when users need to see the actual source paths in task preview
        """
        fields = ['drafts', 'predictions', 'annotations']

        # Lazy load annotations behind feature flag (FIT-720)
        annotations_stub = False
        if flag_set('fflag_fix_all_fit_720_lazy_load_annotations', user=request.user):
            annotations_stub = bool_from_request(request.GET, 'annotations_stub', False)

        return {
            'resolve_uri': bool_from_request(request.GET, 'resolve_uri', True),
            'predictions': 'predictions' in fields,
            'annotations': 'annotations' in fields,
            'drafts': 'drafts' in fields,
            'annotations_stub': annotations_stub,
            'request': request,
        }

    def get(self, request, pk):
        context = self.get_retrieve_serializer_context(request)
        context['project'] = project = self.task.project

        # get prediction
        if (
            project.evaluate_predictions_automatically or project.show_collab_predictions
        ) and not self.task.predictions.exists():
            evaluate_predictions([self.task])
            # refresh task from db with prefetches
            self.task = self.get_object()

        # Don't use expand for annotations when using stub mode (FIT-720)
        # The expand mechanism would override get_annotations and use AnnotationSerializer
        # instead of AnnotationStubSerializer
        expand = [] if context.get('annotations_stub') else ['annotations.completed_by']
        serializer = self.get_serializer_class()(self.task, many=False, context=context, expand=expand)
        data = serializer.data
        return Response(data)

    def get_excluded_fields_for_evaluation(self):
        return ['annotations_results', 'predictions_results']

    def get_queryset(self):
        task_id = self.request.parser_context['kwargs'].get('pk')
        task = generics.get_object_or_404(Task, pk=task_id)
        review = bool_from_request(self.request.GET, 'review', False)
        selected = {'all': False, 'included': [self.kwargs.get('pk')]}
        if review:
            kwargs = {'fields_for_evaluation': ['annotators', 'reviewed']}
        else:
            kwargs = {
                'all_fields': True,
                'excluded_fields_for_evaluation': self.get_excluded_fields_for_evaluation(),
            }
        project = self.request.query_params.get('project') or self.request.data.get('project')
        if not project:
            project = task.project.id
        return self.prefetch(
            Task.prepared.get_queryset(
                prepare_params=PrepareParams(project=project, selectedItems=selected, request=self.request), **kwargs
            )
        )

    def get_object(self):
        """
        Override to check permissions on a lightweight task first.

        This avoids executing the expensive PreparedTaskManager query
        when the user doesn't have permission to access the task.
        """
        task_id = self.kwargs.get('pk')

        # First check permissions using a lightweight query
        # select_related('project') avoids extra query when permission check accesses task.project
        lean_task = generics.get_object_or_404(
            Task.objects.filter(project__organization=self.request.user.active_organization).select_related('project'),
            pk=task_id,
        )
        self.check_object_permissions(self.request, lean_task)

        # Now fetch full task with heavy queryset (prefetches, annotations, etc.)
        queryset = self.filter_queryset(self.get_queryset())
        return generics.get_object_or_404(queryset, pk=task_id)

    def get_serializer_class(self):
        # GET => task + annotations + predictions + drafts
        if self.request.method == 'GET':
            return DataManagerTaskSerializer

        # POST, PATCH, PUT
        else:
            return TaskSimpleSerializer

    def patch(self, request, *args, **kwargs):
        return super(TaskAPI, self).patch(request, *args, **kwargs)

    @api_webhook_for_delete(WebhookAction.TASKS_DELETED)
    def delete(self, request, *args, **kwargs):
        return super(TaskAPI, self).delete(request, *args, **kwargs)

    @extend_schema(exclude=True)
    def put(self, request, *args, **kwargs):
        return super(TaskAPI, self).put(request, *args, **kwargs)


@method_decorator(
    name='get',
    decorator=extend_schema(
        tags=['Tasks'],
        summary='Get task label distribution',
        description='Get aggregated label distribution across all annotations for a task. '
        'Returns counts of each label value grouped by control tag. '
        'This is an efficient endpoint that avoids N+1 queries.',
        responses={
            '200': OpenApiResponse(
                description='Label distribution data',
                examples=[
                    OpenApiExample(
                        name='response',
                        value={
                            'total_annotations': 100,
                            'distributions': {
                                'label': {
                                    'type': 'rectanglelabels',
                                    'labels': {'Car': 45, 'Person': 30, 'Dog': 25},
                                },
                            },
                        },
                        media_type='application/json',
                    )
                ],
            )
        },
        extensions={
            'x-fern-audiences': ['internal'],
        },
    ),
)
class TaskAgreementAPI(generics.RetrieveAPIView):
    """
    Efficient endpoint for getting label distribution without fetching all annotations.

    This endpoint aggregates annotation results at the database level to avoid N+1 queries.
    It returns pre-computed label counts for the Distribution row in the Summary view.
    """

    permission_required = ViewClassPermission(GET=all_permissions.tasks_view)
    queryset = Task.objects.all()

    def get(self, request, pk):
        # This endpoint is gated by feature flag
        if not flag_set('fflag_fix_all_fit_720_lazy_load_annotations', user=request.user):
            raise PermissionDenied('Feature not enabled')

        try:
            task = Task.objects.get(pk=pk)
        except Task.DoesNotExist:
            return Response({'error': 'Task not found'}, status=404)

        # Check project access using LSO's native permission check
        if not task.project.has_permission(request.user):
            raise PermissionDenied('You do not have permission to view this task')

        # Get all annotations for this task with their results in a single query
        annotations = Annotation.objects.filter(
            task=task,
            was_cancelled=False,
        ).values_list('result', flat=True)

        total_annotations = len(annotations)
        distributions = {}

        def merge_result_into_distributions(result):
            """Merge a single result (list of labeling items) into distributions in place."""
            if not result or not isinstance(result, list):
                return
            for item in result:
                if not isinstance(item, dict):
                    continue
                from_name = item.get('from_name', '')
                result_type = item.get('type', '')
                value = item.get('value', {})

                if from_name not in distributions:
                    distributions[from_name] = {
                        'type': result_type,
                        'labels': {},
                        'values': [],
                    }

                if result_type.endswith('labels'):
                    labels = value.get(result_type, [])
                    if isinstance(labels, list):
                        for label in labels:
                            if label not in distributions[from_name]['labels']:
                                distributions[from_name]['labels'][label] = 0
                            distributions[from_name]['labels'][label] += 1

                elif result_type == 'choices':
                    choices = value.get('choices', [])
                    if isinstance(choices, list):
                        for choice in choices:
                            if choice not in distributions[from_name]['labels']:
                                distributions[from_name]['labels'][choice] = 0
                            distributions[from_name]['labels'][choice] += 1

                elif result_type == 'rating':
                    rating = value.get('rating')
                    if rating is not None:
                        distributions[from_name]['values'].append(rating)

                elif result_type == 'number':
                    number = value.get('number')
                    if number is not None:
                        distributions[from_name]['values'].append(number)

                elif result_type == 'taxonomy':
                    taxonomy = value.get('taxonomy', [])
                    if isinstance(taxonomy, list):
                        for path in taxonomy:
                            if isinstance(path, list) and path:
                                leaf = path[-1]
                                if leaf not in distributions[from_name]['labels']:
                                    distributions[from_name]['labels'][leaf] = 0
                                distributions[from_name]['labels'][leaf] += 1

                elif result_type == 'pairwise':
                    selected = value.get('selected')
                    if selected:
                        if selected not in distributions[from_name]['labels']:
                            distributions[from_name]['labels'][selected] = 0
                        distributions[from_name]['labels'][selected] += 1

        # Process annotation results
        for result in annotations:
            merge_result_into_distributions(result)

        # Include prediction results in distribution counts so aggregate matches
        # client-side (develop / FF off). total_annotations stays annotation count only.
        predictions = Prediction.objects.filter(task=task).values_list('result', flat=True)
        for result in predictions:
            # Prediction.result can be list (same as annotation) or dict
            if isinstance(result, list):
                merge_result_into_distributions(result)

        # Post-process: calculate averages for numeric types
        for from_name, dist in distributions.items():
            if dist['values']:
                dist['average'] = sum(dist['values']) / len(dist['values'])
                dist['count'] = len(dist['values'])
            # Remove raw values from response to keep it lightweight
            del dist['values']

        return Response(
            {
                'total_annotations': total_annotations,
                'distributions': distributions,
            }
        )

@method_decorator(
    name='get',
    decorator=extend_schema(
        tags=['Georeferencing'],
        summary='Get task georeferencing data',
        description='Retrieve georeferencing data (RTK/PPK position, orientation, GSD, '
        'RGB-IR co-registration offsets) associated with a task. '
        'If no data has been computed yet, an empty record is returned.',
        parameters=[
            OpenApiParameter(name='pk', type=OpenApiTypes.INT, location='path', description='Task ID'),
        ],
        request=None,
        responses={
            '200': OpenApiResponse(
                description='Georeferencing data',
                response={
                    "status": "success",
                    "metadata": 'object',
                    "geojson": 'object',
                },
            )
        },
        extensions={
            'x-fern-sdk-group-name': 'georeferencing',
            'x-fern-sdk-method-name': 'get',
            'x-fern-audiences': ['public'],
        },
    ),
)
@method_decorator(
    name='put',
    decorator=extend_schema(exclude=True),
)
@method_decorator(
    name='patch',
    decorator=extend_schema(
        tags=['Georeferencing'],
        summary='Update task georeferencing data',
        description='Create or update georeferencing data for a task. '
        'Derived fields (footprints, pipeline status) are read-only; '
        'they are recomputed by the extraction pipeline.',
        request={'application/json': TaskGeoreferencingSerializer},
        responses={
            '200': OpenApiResponse(
                description='Updated georeferencing data',
                response=TaskGeoreferencingSerializer,
            )
        },
        extensions={
            'x-fern-sdk-group-name': 'georeferencing',
            'x-fern-sdk-method-name': 'update',
            'x-fern-audiences': ['public'],
        },
    ),
)
@method_decorator(
    name='delete',
    decorator=extend_schema(
        tags=['Georeferencing'],
        summary='Delete task georeferencing data',
        description="Remove georeferencing data from a task. "
        "This action can't be undone!",
        request=None,
        extensions={
            'x-fern-sdk-group-name': 'georeferencing',
            'x-fern-sdk-method-name': 'delete',
            'x-fern-audiences': ['public'],
        },
    ),
)
class TaskGeoreferencingAPI(generics.RetrieveUpdateDestroyAPIView):
    """
    Task 단위 georeferencing 데이터
    (RTK/PPK 위치, 자세, GSD, RGB-IR co-registration 보정값 등)를
    조회/생성/수정/삭제하는 엔드포인트.
 
    주의: URL의 <pk>는 Annotation ID가 아니라 Task ID이다.
    Task와 1:1 관계이며, 아직 데이터가 없는 task에 대해
    GET 요청이 오면 빈 레코드를 자동 생성해서 반환한다 (파이프라인이 아직
    georeferencing을 끝내지 않은 task도 200으로 조회 가능하게 하기 위함).
    """
 
    parser_classes = (JSONParser, FormParser, MultiPartParser)
    permission_required = ViewClassPermission(
        GET=all_permissions.tasks_view,
        PUT=all_permissions.tasks_change,
        PATCH=all_permissions.tasks_change,
        DELETE=all_permissions.tasks_change,
    )
 
    #serializer_class = GeoreferencingSerializer
 
    def _get_task(self):
        """pk는 task id이므로 Task를 먼저 조회하고, project 기준으로 권한을 체크한다."""
        task = generics.get_object_or_404(Task.objects.select_related('project'), pk=self.kwargs['pk'])
        if not task.project.has_permission(self.request.user):
            raise PermissionDenied('You do not have permission to access this task')
        return task

    def _get_annotation(self, task, include_predictions: bool = True) -> Optional['Annotation']:
        """
        annotation, draft, prediction 중 가장 최근에 작성/수정된 것을 반환한다.

        우선순위 정책:
        - 세 소스 모두 시간(updated_at 또는 created_at) 기준으로 최신 것 선택
        - 단, prediction은 사람이 만든 annotation/draft가 하나도 없을 때의
          fallback으로만 쓰는 것이 안전하므로, prefer_human=True(기본) 동작을 유지

        Returns:
            Annotation | AnnotationDraft | Prediction | None
            (세 모델 모두 .result 필드를 가지므로 _annotation_to_yolo_labels에서 동일하게 처리 가능)
        """
        annotation = (
            task.annotations
            .filter(was_cancelled=False)
            .order_by('-updated_at')
            .first()
        )

        draft = task.drafts.order_by('-updated_at').first()

        # ── annotation vs draft: 시간 비교로 최신 선택 ──────────────
        latest_human = None
        if annotation and draft:
            annotation_time = annotation.updated_at or annotation.created_at
            draft_time = draft.updated_at or draft.created_at
            latest_human = draft if draft_time > annotation_time else annotation
        else:
            latest_human = annotation or draft

        # ── 사람이 만든 결과가 있으면 그것을 우선 반환 ────────────────
        if latest_human is not None:
            return latest_human

        # ── 둘 다 없으면 prediction fallback ─────────────────────────
        if include_predictions:
            prediction = (
                task.predictions
                .order_by('-updated_at')
                .first()
            )
            if prediction is not None:
                logger.info(
                    f'[TaskGeoreferencingAPI] task={task.id}: annotation/draft 없음 → '
                    f'prediction(id={prediction.id}, model={prediction.model_version}) 사용'
                )
            return prediction

        return None

    def get(self, request, *args, **kwargs):
        task = self._get_task()
        print('task.data', task.data, type(task))
        
        base_data_dir = os.environ.get('LABEL_STUDIO_BASE_DATA_DIR')
        filename = task.data.get('image') or task.data.get('$undefined$')
        filename = filename.split('/data/upload/')[-1]
        image_path = os.path.expanduser(
            f'{base_data_dir}/media/upload/{filename}'
        )

        annotation = self._get_annotation(task)

        georeferencer = AnnotationGeoreferencer(image_path, annotation)
        metadata, geojson = georeferencer.get_geojson()
        geojson['metadata']['task_id'] = task.id
        print('GeoJSON:', geojson)

        return Response({
            "status": "success",
            "metadata": metadata.to_dict(),
            "geojson": geojson
        })
 
    @extend_schema(exclude=True)
    def perform_destroy(self, instance):
        instance.delete()


@method_decorator(
    name='get',
    decorator=extend_schema(
        tags=['Annotations'],
        summary='Get annotation by its ID',
        description='Retrieve a specific annotation for a task using the annotation result ID.',
        request=None,
        responses={
            '200': OpenApiResponse(
                description='Retrieved annotation',
                response=AnnotationSerializer,
                examples=[
                    OpenApiExample(name='response', value=annotation_response_example, media_type='application/json')
                ],
            )
        },
        extensions={
            'x-fern-sdk-group-name': 'annotations',
            'x-fern-sdk-method-name': 'get',
            'x-fern-audiences': ['public'],
        },
    ),
)
@method_decorator(
    name='patch',
    decorator=extend_schema(
        tags=['Annotations'],
        summary='Update annotation',
        description='Update existing attributes on an annotation.',
        request={
            'application/json': annotation_request_schema,
        },
        responses={
            '200': OpenApiResponse(
                description='Updated annotation',
                response=AnnotationSerializer,
                examples=[
                    OpenApiExample(name='response', value=annotation_response_example, media_type='application/json')
                ],
            )
        },
        extensions={
            'x-fern-sdk-group-name': 'annotations',
            'x-fern-sdk-method-name': 'update',
            'x-fern-audiences': ['public'],
        },
    ),
)
@method_decorator(
    name='delete',
    decorator=extend_schema(
        tags=['Annotations'],
        summary='Delete annotation',
        description="Delete an annotation. This action can't be undone!",
        request=None,
        extensions={
            'x-fern-sdk-group-name': 'annotations',
            'x-fern-sdk-method-name': 'delete',
            'x-fern-audiences': ['public'],
        },
    ),
)
class AnnotationAPI(generics.RetrieveUpdateDestroyAPIView):
    parser_classes = (JSONParser, FormParser, MultiPartParser)
    permission_required = ViewClassPermission(
        GET=all_permissions.annotations_view,
        PUT=all_permissions.annotations_change,
        PATCH=all_permissions.annotations_change,
        DELETE=all_permissions.annotations_delete,
    )

    serializer_class = AnnotationSerializer
    queryset = Annotation.objects.all()

    def perform_destroy(self, annotation):
        annotation.delete()

    def update(self, request, *args, **kwargs):
        # save user history with annotator_id, time & annotation result
        annotation = self.get_object()
        # use updated instead of save to avoid duplicated signals
        Annotation.objects.filter(id=annotation.id).update(updated_by=request.user)

        task = annotation.task
        if self.request.data.get('ground_truth'):
            task.ensure_unique_groundtruth(annotation_id=annotation.id)
        task.update_is_labeled()
        task.save()  # refresh task metrics

        result = super(AnnotationAPI, self).update(request, *args, **kwargs)

        task.update_is_labeled()
        task.save(update_fields=['updated_at'])  # refresh task metrics
        return result

    def get(self, request, *args, **kwargs):
        return super(AnnotationAPI, self).get(request, *args, **kwargs)

    @api_webhook(WebhookAction.ANNOTATION_UPDATED)
    @extend_schema(exclude=True)
    def put(self, request, *args, **kwargs):
        return super(AnnotationAPI, self).put(request, *args, **kwargs)

    @api_webhook(WebhookAction.ANNOTATION_UPDATED)
    def patch(self, request, *args, **kwargs):
        return super(AnnotationAPI, self).patch(request, *args, **kwargs)

    @api_webhook_for_delete(WebhookAction.ANNOTATIONS_DELETED)
    def delete(self, request, *args, **kwargs):
        return super(AnnotationAPI, self).delete(request, *args, **kwargs)


@method_decorator(
    name='get',
    decorator=extend_schema(
        tags=['Annotations'],
        summary='Get all task annotations',
        description='List all annotations for a task.',
        parameters=[
            OpenApiParameter(name='id', type=OpenApiTypes.INT, location='path', description='Task ID'),
        ],
        request=None,
        responses={
            '200': OpenApiResponse(
                description='Annotation',
                response=AnnotationSerializer(many=True),
                examples=[
                    OpenApiExample(name='response', value=[annotation_response_example], media_type='application/json')
                ],
            )
        },
        extensions={
            'x-fern-sdk-group-name': 'annotations',
            'x-fern-sdk-method-name': 'list',
            'x-fern-audiences': ['public'],
        },
    ),
)
@method_decorator(
    name='post',
    decorator=extend_schema(
        tags=['Annotations'],
        summary='Create annotation',
        description="""
        Add annotations to a task like an annotator does. The content of the result field depends on your
        labeling configuration. For example, send the following data as part of your POST
        request to send an empty annotation with the ID of the user who completed the task:

        ```json
        {
        "result": {},
        "was_cancelled": true,
        "ground_truth": true,
        "lead_time": 0,
        "task": 0
        "completed_by": 123
        }
        ```
        """,
        parameters=[
            OpenApiParameter(name='id', type=OpenApiTypes.INT, location='path', description='Task ID'),
        ],
        request={
            'application/json': annotation_request_schema,
        },
        responses={
            '201': OpenApiResponse(
                description='Created annotation',
                response=AnnotationSerializer,
                examples=[
                    OpenApiExample(name='response', value=annotation_response_example, media_type='application/json')
                ],
            )
        },
        extensions={
            'x-fern-sdk-group-name': 'annotations',
            'x-fern-sdk-method-name': 'create',
            'x-fern-audiences': ['public'],
        },
    ),
)
class AnnotationsListAPI(GetParentObjectMixin, generics.ListCreateAPIView):
    parser_classes = (JSONParser, FormParser, MultiPartParser)
    permission_required = ViewClassPermission(
        GET=all_permissions.annotations_view,
        POST=all_permissions.annotations_create,
    )
    parent_queryset = Task.objects.all()

    serializer_class = AnnotationSerializer

    def get(self, request, *args, **kwargs):
        return super(AnnotationsListAPI, self).get(request, *args, **kwargs)

    @api_webhook(WebhookAction.ANNOTATION_CREATED)
    def post(self, request, *args, **kwargs):
        return super(AnnotationsListAPI, self).post(request, *args, **kwargs)

    def get_queryset(self):
        task = generics.get_object_or_404(Task.objects.for_user(self.request.user), pk=self.kwargs.get('pk', 0))
        return Annotation.objects.filter(Q(task=task) & Q(was_cancelled=False)).order_by('pk')

    def delete_draft(self, draft_id, annotation_id):
        try:
            draft = AnnotationDraft.objects.get(id=draft_id)
            # We call delete on the individual draft object because
            # AnnotationDraft#delete has special behavior (updating created_labels_drafts).
            # This special behavior won't be triggered if we call delete on the queryset.
            # Only for drafts with empty annotation_id, other ones deleted by signal
            draft.delete()
        except AnnotationDraft.DoesNotExist:
            pass

    def perform_create(self, ser):
        task = self.parent_object
        # annotator has write access only to annotations and it can't be checked it after serializer.save()
        user = self.request.user

        # Check if task is being skipped and if it's allowed
        was_cancelled_get = bool_from_request(self.request.GET, 'was_cancelled', False)
        was_cancelled_data = self.request.data.get('was_cancelled', False)
        is_skipping = was_cancelled_get or was_cancelled_data

        if is_skipping and not task.can_be_skipped():
            raise ValidationError({'detail': 'This task cannot be skipped.'})

        # updates history
        result = ser.validated_data.get('result')
        extra_args = {'task_id': self.kwargs['pk'], 'project_id': task.project_id}

        # save stats about how well annotator annotations coincide with current prediction
        # only for finished task annotations
        if result is not None:
            prediction = Prediction.objects.filter(task=task, model_version=task.project.model_version)
            if prediction.exists():
                prediction = prediction.first()
                prediction_ser = PredictionSerializer(prediction).data
            else:
                logger.debug(f'User={self.request.user}: there are no predictions for task={task}')
                prediction_ser = {}
            # serialize annotation
            extra_args.update({'prediction': prediction_ser, 'updated_by': user})

        if 'was_cancelled' in self.request.GET:
            extra_args['was_cancelled'] = bool_from_request(self.request.GET, 'was_cancelled', False)

        if 'completed_by' not in ser.validated_data:
            extra_args['completed_by'] = self.request.user

        draft_id = self.request.data.get('draft_id')
        draft = AnnotationDraft.objects.filter(id=draft_id).first()
        if draft:
            # draft permission check
            if draft.task_id != task.id or not draft.has_permission(user) or draft.user_id != user.id:
                raise PermissionDenied(f'You have no permission to draft id:{draft_id}')

        if draft is not None:
            # if the annotation will be created from draft - get created_at from draft to keep continuity of history
            extra_args['draft_created_at'] = draft.created_at

        # create annotation
        logger.debug(f'User={self.request.user}: save annotation')
        annotation = ser.save(**extra_args)

        logger.debug(f'Save activity for user={self.request.user}')
        self.request.user.activity_at = timezone.now()
        self.request.user.save()

        # Release task if it has been taken at work (it should be taken by the same user, or it makes sentry error
        logger.debug(f'User={user} releases task={task}')
        task.release_lock(user)

        # if annotation created from draft - remove this draft
        if draft_id is not None:
            logger.debug(f'Remove draft {draft_id} after creating annotation {annotation.id}')
            self.delete_draft(draft_id, annotation.id)

        if self.request.data.get('ground_truth'):
            annotation.task.ensure_unique_groundtruth(annotation_id=annotation.id)

        fill_history_annotation(user, task, annotation)

        return annotation


@extend_schema(exclude=True)
class AnnotationDraftListAPI(generics.ListCreateAPIView):
    parser_classes = (JSONParser, MultiPartParser, FormParser)
    serializer_class = AnnotationDraftSerializer
    permission_required = ViewClassPermission(
        GET=all_permissions.annotations_view,
        POST=all_permissions.annotations_create,
    )
    queryset = AnnotationDraft.objects.all()

    def filter_queryset(self, queryset):
        task_id = self.kwargs['pk']
        return queryset.filter(task_id=task_id)

    def perform_create(self, serializer):
        task_id = self.kwargs['pk']
        annotation_id = self.kwargs.get('annotation_id')
        user = self.request.user
        logger.debug(f'User {user} is going to create draft for task={task_id}, annotation={annotation_id}')
        serializer.save(task_id=self.kwargs['pk'], annotation_id=annotation_id, user=self.request.user)


@extend_schema(exclude=True)
class AnnotationDraftAPI(generics.RetrieveUpdateDestroyAPIView):
    parser_classes = (JSONParser, MultiPartParser, FormParser)
    serializer_class = AnnotationDraftSerializer
    queryset = AnnotationDraft.objects.all()
    permission_required = ViewClassPermission(
        GET=all_permissions.annotations_view,
        PUT=all_permissions.annotations_change,
        PATCH=all_permissions.annotations_change,
        DELETE=all_permissions.annotations_delete,
    )


@method_decorator(
    name='list',
    decorator=extend_schema(
        tags=['Predictions'],
        summary='List predictions',
        description='List all predictions and their IDs.',
        parameters=[
            OpenApiParameter(
                name='task',
                type=OpenApiTypes.INT,
                location='query',
                description='Filter predictions by task ID',
            ),
            OpenApiParameter(
                name='project',
                type=OpenApiTypes.INT,
                location='query',
                description='Filter predictions by project ID',
            ),
        ],
        request=None,
        responses={
            '200': OpenApiResponse(
                description='Predictions list',
                response=PredictionSerializer(many=True),
                examples=[
                    OpenApiExample(name='response', value=[prediction_response_example], media_type='application/json')
                ],
            )
        },
        extensions={
            'x-fern-sdk-group-name': 'predictions',
            'x-fern-sdk-method-name': 'list',
            'x-fern-audiences': ['public'],
        },
    ),
)
@method_decorator(
    name='create',
    decorator=extend_schema(
        tags=['Predictions'],
        summary='Create prediction',
        description='Create a prediction for a specific task.',
        request={
            'application/json': prediction_request_schema,
        },
        responses={
            '201': OpenApiResponse(
                description='Created prediction',
                response=PredictionSerializer,
                examples=[
                    OpenApiExample(name='response', value=prediction_response_example, media_type='application/json')
                ],
            )
        },
        extensions={
            'x-fern-sdk-group-name': 'predictions',
            'x-fern-sdk-method-name': 'create',
            'x-fern-audiences': ['public'],
        },
    ),
)
@method_decorator(
    name='retrieve',
    decorator=extend_schema(
        tags=['Predictions'],
        summary='Get prediction details',
        description='Get details about a specific prediction by its ID.',
        parameters=[
            OpenApiParameter(name='id', type=OpenApiTypes.INT, location='path', description='Prediction ID'),
        ],
        request=None,
        responses={
            '200': OpenApiResponse(
                description='Prediction details',
                response=PredictionSerializer,
                examples=[
                    OpenApiExample(name='response', value=prediction_response_example, media_type='application/json')
                ],
            )
        },
        extensions={
            'x-fern-sdk-group-name': 'predictions',
            'x-fern-sdk-method-name': 'get',
            'x-fern-audiences': ['public'],
        },
    ),
)
@method_decorator(
    name='update',
    decorator=extend_schema(
        tags=['Predictions'],
        summary='Put prediction',
        description='Overwrite prediction data by prediction ID.',
        parameters=[
            OpenApiParameter(name='id', type=OpenApiTypes.INT, location='path', description='Prediction ID'),
        ],
        request={
            'application/json': prediction_request_schema,
        },
        responses={
            '200': OpenApiResponse(
                description='Updated prediction',
                response=PredictionSerializer,
                examples=[
                    OpenApiExample(name='response', value=prediction_response_example, media_type='application/json')
                ],
            )
        },
        extensions={
            'x-fern-audiences': ['internal'],
        },
    ),
)
@method_decorator(
    name='partial_update',
    decorator=extend_schema(
        tags=['Predictions'],
        summary='Update prediction',
        description='Update prediction data by prediction ID.',
        parameters=[
            OpenApiParameter(name='id', type=OpenApiTypes.INT, location='path', description='Prediction ID'),
        ],
        request={
            'application/json': prediction_request_schema,
        },
        responses={
            '200': OpenApiResponse(
                description='Updated prediction',
                response=PredictionSerializer,
                examples=[
                    OpenApiExample(name='response', value=prediction_response_example, media_type='application/json')
                ],
            )
        },
        extensions={
            'x-fern-sdk-group-name': 'predictions',
            'x-fern-sdk-method-name': 'update',
            'x-fern-audiences': ['public'],
        },
    ),
)
@method_decorator(
    name='destroy',
    decorator=extend_schema(
        tags=['Predictions'],
        summary='Delete prediction',
        description='Delete a prediction by prediction ID.',
        parameters=[
            OpenApiParameter(name='id', type=OpenApiTypes.INT, location='path', description='Prediction ID'),
        ],
        request=None,
        extensions={
            'x-fern-sdk-group-name': 'predictions',
            'x-fern-sdk-method-name': 'delete',
            'x-fern-audiences': ['public'],
        },
    ),
)
class PredictionAPI(viewsets.ModelViewSet):
    serializer_class = PredictionSerializer
    permission_required = all_permissions.predictions_any
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['task', 'task__project', 'project']

    def get_queryset(self):
        return Prediction.objects.filter(project__organization=self.request.user.active_organization)


@method_decorator(name='get', decorator=extend_schema(exclude=True))
@method_decorator(
    name='post',
    decorator=extend_schema(
        tags=['Annotations'],
        summary='Convert annotation to draft',
        description='Convert annotation to draft',
        extensions={
            'x-fern-audiences': ['internal'],
        },
    ),
)
class AnnotationConvertAPI(generics.RetrieveAPIView):
    permission_required = ViewClassPermission(POST=all_permissions.annotations_change)
    queryset = Annotation.objects.all()

    def process_intermediate_state(self, annotation, draft):
        pass

    def post(self, request, *args, **kwargs):
        annotation = self.get_object()
        organization = annotation.project.organization
        project = annotation.project

        pk = annotation.pk

        with transaction.atomic():
            draft = AnnotationDraft.objects.create(
                result=annotation.result,
                lead_time=annotation.lead_time,
                task=annotation.task,
                annotation=None,
                user=request.user,
            )

            self.process_intermediate_state(annotation, draft)

            annotation.delete()

        emit_webhooks_for_instance(organization, project, WebhookAction.ANNOTATIONS_DELETED, [pk])
        data = AnnotationDraftSerializer(instance=draft).data
        return Response(status=201, data=data)

@method_decorator(
    name='get',
    decorator=extend_schema(
        tags=['Annotations'],
        summary='Search images from annotations',
        description='Search images from annotations',
        parameters=[
            OpenApiParameter(name='id', type=OpenApiTypes.INT, location='path', description='Task ID'),
        ],
        request=None,
        responses={
            '200': OpenApiResponse(
                description='Annotation',
                response=AnnotationSerializer(many=True),
                examples=[
                    OpenApiExample(name='response', value=[annotation_response_example], media_type='application/json')
                ],
            )
        },
        extensions={
            'x-fern-sdk-group-name': 'annotations',
            'x-fern-sdk-method-name': 'list',
            'x-fern-audiences': ['public'],
        },
    ),
)
class AnnotationGeoreferencingAPI(generics.RetrieveAPIView):
    """
    region 의 지리 폴리곤을 계산하고, 그 영역을 커버하는 다른 이미지를 찾는다.
 
    URL: /api/annotations/<region_id>/georeferencing?task_id=<id>
         pk 는 Annotation ID 가 아니라 **LSF region.cleanId** 다.
    """
 
    queryset = Annotation.objects.all()
 
    # 저장 footprint 가 현재 파이프라인과 몇 % 이상 다르면 경고할지
    FOOTPRINT_TOLERANCE_PCT = 1.5
 
    def _get_task(self, task_id):
        task = generics.get_object_or_404(
            Task.objects.select_related('project'), pk=task_id
        )
        if not task.project.has_permission(self.request.user):
            raise PermissionDenied('You do not have permission to access this task')
        return task
 
    def _check_stored_footprint(self, task, metadata) -> Optional[dict]:
        """
        이 task 의 **저장된** footprint 가 현재 파이프라인 산출물인지 검사한다.
 
        metadata 는 이미 로드돼 있으므로 비용이 거의 없다.
        stale 이면 검색 결과와 프론트 오버레이 위치가 둘 다 어긋나므로
        응답에 실어 보내 호출부가 인지하게 한다.
        """
        try:
            from tasks.models import TaskImageGeoreferencing
        except ImportError:
            return None
 
        row = (
            TaskImageGeoreferencing.objects
            .filter(task_id=task.id)
            .order_by('id')
            .first()
        )
        if row is None or row.footprint is None:
            return {'stale': None, 'reason': '저장된 footprint 없음 — 인제스트 필요'}
 
        try:
            ring = list(row.footprint.coords[0])[:4]
            result = compare_footprints(
                ring, metadata,
                tolerance_pct=self.FOOTPRINT_TOLERANCE_PCT,
                **_plane_kwargs(task.project),
            )
        except Exception as e:  # 검사 실패가 본 기능을 막으면 안 된다
            logger.warning('[georeferencing] footprint 검사 실패: %s', e)
            return None
 
        if result.get('stale'):
            logger.error(
                '[georeferencing] task=%s 저장 footprint 가 현재 파이프라인과 불일치 — '
                '저장 %s m, 기대 %s m (비율 %s). 검색 커버리지와 gps_ref 오버레이 '
                '위치가 어긋납니다. rebuild_footprints 실행 필요.',
                task.id, result['stored_m'], result['expected_m'], result['ratio'],
            )
        return result
 
    def get(self, request, *args, **kwargs):
        from tasks.search_image_georeferencing import find_images_covering_polygon
 
        region_id = self.kwargs['pk']          # LSF region.cleanId (str)
        task_id = request.query_params.get('task_id')
        if not task_id:
            return Response({'detail': 'task_id query param is required'}, status=400)
 
        task = self._get_task(task_id)
 
        # ── annotation 선택은 AnnotationGeoreferencer 에 위임 ──────────
        # (api.py / AnnotationGeoreferencer 에 같은 로직이 세 벌 있었다)
        try:
            image_path = resolve_task_image_path(task)
        except ValueError as e:
            return Response({'detail': str(e)}, status=409)
 
        georeferencer = AnnotationGeoreferencer(
            image_path, annotation=None, **_plane_kwargs(task.project),
        )
        annotation = georeferencer._get_annotation(task)
        if annotation is None:
            return Response(
                {'detail': f'annotation/draft/prediction 없음 (task={task.id})'},
                status=409,
            )
 
        metadata, geojson = georeferencer.get_geojson(annotation=annotation)
        geojson['metadata']['task_id'] = task.id
        ground_plane = geojson.get('metadata', {}).get('ground_plane')
 
        # ── region_id 에 해당하는 검출 폴리곤 ──────────────────────────
        raw_coords = next(
            (
                f['geometry']['coordinates']
                for f in geojson.get('features', [])
                if f.get('properties', {}).get('region_id') == region_id
            ),
            None,
        )
        if not raw_coords:
            # 예전에는 coordinates[0] 에서 IndexError → 500 이 났다.
            logger.info(
                '[georeferencing] region_id=%s 에 해당하는 검출 폴리곤 없음 (task=%s)',
                region_id, task.id,
            )
            return Response(
                {
                    'detail': f'region_id={region_id} 에 해당하는 지리 폴리곤이 없습니다.',
                    'task_id': task.id,
                    'ground_plane': ground_plane,
                },
                status=404,
            )
 
        ring = [[float(x), float(y)] for x, y in raw_coords[0]]
 
        # ── 저장 footprint 신선도 검사 (검색 신뢰도에 직결) ───────────
        footprint_check = self._check_stored_footprint(task, metadata)
 
        # ── 커버 이미지 검색 ──────────────────────────────────────────
        hits = find_images_covering_polygon(
            ring, min_coverage=0.3, limit=20, as_dict=True,
        )
 
        selected_hits = [
            {
                'task_id': h['task_id'],
                'project_id': h['project_id'],
                'filename': h['filename'],
                'modality': h['modality'],
                'rtk_flag': h['rtk_flag'],
                'coverage_ratio': h['coverage_ratio'],
                'fully_covered': h['fully_covered'],
                'low_confidence': h['low_confidence'],
            }
            # 주의: 여기서 루프 변수 이름에 task 를 쓰면 위의 Task 객체를 덮어쓴다.
            for h in hits
        ]
 
        logger.info(
            '[georeferencing] task=%s region=%s → %d hits (plane=%s)',
            task.id, region_id, len(hits),
            (ground_plane or {}).get('source'),
        )
 
        return Response({
            'task_id': task.id,
            'region_id': region_id,
            'count': len(hits),
            'hits': selected_hits,
            'region': {
                'coordinates': ring,
            },
            # 프론트가 gps_ref 오버레이 정합을 교차검증할 수 있도록 함께 내려준다.
            'ground_plane': ground_plane,
            'footprint_check': footprint_check,
        })
 
    def post(self, request, *args, **kwargs):
        """annotation → draft 로 되돌리기 (기존 동작 유지)."""
        from django.db import transaction
 
        from core.utils.common import temporary_disconnect_all_signals  # noqa: F401
        from webhooks.models import WebhookAction
        from webhooks.utils import emit_webhooks_for_instance
        from tasks.serializers import AnnotationDraftSerializer
 
        annotation = self.get_object()
        organization = annotation.project.organization
        project = annotation.project
        pk = annotation.pk
 
        with transaction.atomic():
            draft = AnnotationDraft.objects.create(
                result=annotation.result,
                lead_time=annotation.lead_time,
                task=annotation.task,
                annotation=None,
                user=request.user,
            )
            annotation.delete()
 
        emit_webhooks_for_instance(
            organization, project, WebhookAction.ANNOTATIONS_DELETED, [pk]
        )
        return Response(status=201, data=AnnotationDraftSerializer(instance=draft).data)

class FrontendConfigAPI(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]  # 로그인 사용자만

    google_maps_api_key = settings.GOOGLE_MAPS_API_KEY
    print("GOOGLE_MAPS_API_KEY", google_maps_api_key)

    def get(self, request, *args, **kwargs):
        return Response({
            'google_maps_api_key': self.google_maps_api_key,
        })
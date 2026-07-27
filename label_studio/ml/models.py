"""This file and its contents are licensed under the Apache License 2.0. Please see the included NOTICE for copyright information and LICENSE for a copy of the license."""

from aiohttp._websocket import models
import logging
import os
from typing import Dict, List

from core.utils.common import conditional_atomic, db_is_not_sqlite, load_func
from django.conf import settings
from django.db import models, transaction
from django.db.models import Count, JSONField, Q
from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver
from django.utils.translation import gettext_lazy as _
from ml.api_connector import PREDICT_URL, TIMEOUT_PREDICT, MLApi
from projects.models import Project
from tasks.serializers import PredictionSerializer, TaskSimpleSerializer
from webhooks.serializers import Webhook, WebhookSerializer

logger = logging.getLogger(__name__)

MAX_JOBS_PER_PROJECT = 1

InteractiveAnnotatingDataSerializer = load_func(settings.INTERACTIVE_DATA_SERIALIZER)


class MLBackendState(models.TextChoices):
    CONNECTED = 'CO', _('Connected')
    DISCONNECTED = 'DI', _('Disconnected')
    ERROR = 'ER', _('Error')
    TRAINING = 'TR', _('Training')
    PREDICTING = 'PR', _('Predicting')


class MLBackendAuth(models.TextChoices):
    NONE = 'NONE', _('None')
    BASIC_AUTH = 'BASIC_AUTH', _('Basic Auth')


class MLBackend(models.Model):
    """ """

    state = models.CharField(
        max_length=2,
        choices=MLBackendState.choices,
        default=MLBackendState.DISCONNECTED,
    )
    is_local = models.BooleanField(
        _('is_local'),
        default=False,
        help_text=('Used to run the model locally. If true, model runs locally.'),
    )
    is_interactive = models.BooleanField(
        _('is_interactive'),
        default=False,
        help_text=('Used to interactively annotate tasks. If true, model returns one list with results'),
    )
    url = models.TextField(
        _('url'),
        help_text='URL for the machine learning model server',
    )
    error_message = models.TextField(
        _('error_message'),
        blank=True,
        null=True,
        help_text='Error message in error state',
    )
    title = models.TextField(
        _('title'),
        blank=True,
        null=True,
        default='default',
        help_text='Name of the machine learning backend',
    )

    auth_method = models.CharField(
        max_length=255,
        choices=MLBackendAuth.choices,
        default=MLBackendAuth.NONE,
    )

    basic_auth_user = models.TextField(
        _('basic auth user'),
        blank=True,
        null=True,
        default='',
        help_text='HTTP Basic Auth user',
    )

    basic_auth_pass = models.TextField(
        _('basic auth password'),
        blank=True,
        null=True,
        default='',
        help_text='HTTP Basic Auth password',
    )

    description = models.TextField(
        _('description'),
        blank=True,
        null=True,
        default='',
        help_text='Description for the machine learning backend',
    )

    extra_params = JSONField(
        _('extra params'),
        null=True,
        help_text='Any extra parameters passed to the ML Backend during the setup',
    )

    model_version = models.TextField(
        _('model version'),
        blank=True,
        null=True,
        default='',
        help_text='Current model version associated with this machine learning backend',
    )
    timeout = models.FloatField(
        _('timeout'),
        blank=True,
        default=100.0,
        help_text='Response model timeout',
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name='ml_backends',
    )
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)
    auto_update = models.BooleanField(
        _('auto_update'),
        default=True,
        help_text='If false, model version is set by the user, if true - getting latest version from backend.',
    )

    def __str__(self):
        return f'{self.title} (id={self.id}, url={self.url})'

    def __init__(self, *args, **kwargs):
        super(MLBackend, self).__init__(*args, **kwargs)
        self.__original_title = self.title

    def save(self, *args, **kwargs):
        """
        Overrides the save() method to update the associated project's model_version field.
        If the title of the model instance is changed and the model_version
        of the related project is currently the same as the original title,
        the project's model_version is updated to the new title.
        """
        print("MLBackend > save", args, kwargs)
        p = self.project

        if self.title != self.__original_title and p.model_version == self.__original_title:
            with transaction.atomic():
                p.model_version = self.title
                p.save(update_fields=['model_version'])
                super().save(*args, **kwargs)
                # reset original field to current field after save
                self.__original_title = self.title
        else:
            super().save(*args, **kwargs)

    @staticmethod
    def healthcheck_(url, auth_method=None, **kwargs):
        print("MLBackend > healthcheck_", url, auth_method, kwargs)
        is_local = kwargs.get("is_local", False)
        if is_local:
            from ml.api_connector import MLApiResult

            return MLApiResult(
                url=url,
                request={},
                response={"model_version": "local"},
                headers={},
                type='ok',
                status_code=200,
            )
        
        return MLApi(url=url, auth_method=auth_method, **kwargs).health()

    def has_permission(self, user):
        user.project = self.project  # link for activity log
        return self.project.has_permission(user)

    @staticmethod
    def setup_(url, project, auth_method=None, **kwargs):
        print("MLBackend > setup_", url, auth_method, kwargs)
        is_local = kwargs.get("is_local", False)
        if is_local:
            from ml.api_connector import MLApiResult

            return MLApiResult(
                url=url,
                request={},
                response={"model_version": "local"},
                headers={},
                type='ok',
                status_code=200,
            )
        
        api = MLApi(url=url, auth_method=auth_method, **kwargs)

        if not isinstance(project, Project):
            project = Project.objects.get(pk=project)
        response = api.setup(project, **kwargs)

        print("MLBackend > setup_", response)
        return response

    def healthcheck(self):
        print("MLBackend > healthcheck")
        print("MLBackend > healthcheck", self.url,
            self.project,
            self.is_local,
            self.auth_method,
            self.extra_params,
            self.basic_auth_user,
            self.basic_auth_pass)

        return self.healthcheck_(
            self.url, self.auth_method, basic_auth_user=self.basic_auth_user, basic_auth_pass=self.basic_auth_pass
        )

    def setup(self):
        print("MLBackend > setup")
        print("MLBackend > setup", self.url,
            self.project,
            self.is_local,
            self.auth_method,
            self.extra_params,
            self.basic_auth_user,
            self.basic_auth_pass)
        
        return self.setup_(
            self.url,
            self.project,
            self.auth_method,
            extra_params=self.extra_params,
            basic_auth_user=self.basic_auth_user,
            basic_auth_pass=self.basic_auth_pass,
        )

    @property
    def api(self):
        return MLApi(
            url=self.url,
            timeout=self.timeout,
            auth_method=self.auth_method,
            basic_auth_user=self.basic_auth_user,
            basic_auth_pass=self.basic_auth_pass,
        )

    @property
    def not_ready(self):
        return self.state in (MLBackendState.DISCONNECTED, MLBackendState.ERROR)

    def update_state(self, request=None):
        print("MLBackend > update_state")
        print("MLBackend > update_state", request)

        model_version = None

        # request body 접근
        is_local = self.is_local
        if request is not None:
            import json
            try:
                body = json.loads(request.body)
                print("Request body:", body)
                is_local = body.get('is_local', False)
                model_version = body.get('url', None)
            except Exception:
                body = {}
        
        print("MLBackend > update_state", is_local, self.url)
        if is_local:
            model_path = self.url
            if not os.path.exists(self.url):
                logger.warning(f'Not exist local ML backend')
                self.state = MLBackendState.ERROR
                self.error_message = 'Not exist local ML backend'
            else:
                self.state = MLBackendState.CONNECTED
                self.model_version = model_path
                self.error_message = None
        elif self.healthcheck().is_error:
            self.state = MLBackendState.DISCONNECTED
        else:
            setup_response = self.setup()
            if setup_response.is_error:
                logger.info(f'ML backend responds with error: {setup_response.error_message}')
                self.state = MLBackendState.ERROR
                self.error_message = setup_response.error_message
            else:
                self.state = MLBackendState.CONNECTED
                model_version = setup_response.response.get('model_version')
                logger.info(f'ML backend responds with success: {setup_response.response}')
                if self.auto_update:
                    logger.debug(f'Changing model version: {self.model_version} -> {model_version}')
                    self.model_version = model_version
                self.error_message = None
        self.save()
        return model_version

    def train(self):
        train_response = self.api.train(self.project)
        if train_response.is_error:
            self.state = MLBackendState.ERROR
            self.error_message = train_response.error_message
        else:
            self.state = MLBackendState.TRAINING
            current_train_job = train_response.response.get('job')
            if current_train_job:
                MLBackendTrainJob.objects.create(job_id=current_train_job, ml_backend=self)
        self.save()

    def _predict(self, task):
        """This is low level prediction method that is used for debugging"""
        ml_api = self.api
        task_ser = TaskSimpleSerializer(task).data

        request_params = ml_api._prep_prediction_req([task_ser], self.project)
        ml_api_result = ml_api._request(PREDICT_URL, request_params, verbose=False, timeout=TIMEOUT_PREDICT)

        if ml_api_result.is_error:
            logger.info(f'Prediction not created for project {self}: {ml_api_result.error_message}')
            return

        results = ml_api_result.response.get('results', None)

        return {
            'status': 200,
            'data': {
                'status': ml_api_result.status_code,
                'error_message': ml_api_result.error_message,
                'url': ml_api._get_url(PREDICT_URL),
                'task': task_ser,
                'request': request_params,
                'response': results,
            },
        }

    def _get_predictions_from_ml_backend_one_by_one(
        self, serialized_tasks: List[Dict], current_responses: List[Dict]
    ) -> List[Dict]:
        """
        This is helper method to get predictions from ML backend one by one
        in case when tasks length doesn't match responses length
        Note: don't use this function outside of this class
        """

        if len(current_responses) == 1:
            # In case ML backend doesn't support batch of tasks, do it one by one
            # TODO: remove this block after all ML backends will support batch processing
            logger.warning(
                f"'ML backend '{self.title}' doesn't support batch processing of tasks, "
                f'switched to one-by-one task retrieval'
            )
            predictions = []
            for serialized_task in serialized_tasks:
                # get predictions per task
                predictions.extend(self._get_predictions_from_ml_backend([serialized_task]))

            return predictions
        else:
            # complete failure - likely ML backend skipped some tasks, we can't match them
            logger.error(
                f'Number of tasks and responses are not equal: '
                f'{len(serialized_tasks)} tasks != {len(current_responses)} responses. '
                f'Returning empty predictions.'
            )
            return []

    def _get_predictions_from_local(self, serialized_tasks: List[Dict]) -> List[Dict]:
        from ultralytics import YOLO

        print("MLBackend > _get_predictions_from_local", serialized_tasks, self.url)

        base_data_dir = os.environ.get('LABEL_STUDIO_BASE_DATA_DIR')
        print("MLBackend > _get_predictions_from_local", base_data_dir)

        model_path = self.url
        model = YOLO(model_path)

        task_id = serialized_tasks[0]['id']
        project_id = serialized_tasks[0]['project']
        filename = serialized_tasks[0]['data']['image']
        filename = filename.split('/data/upload/')[-1]
        image_path = os.path.expanduser(
            f'{base_data_dir}/media/upload/{filename}'
        )
        results = model.predict(image_path, conf=0.3, verbose=False)


        boxes = results[0].boxes
        img_h, img_w = results[0].orig_shape

        """ """
        regions = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            regions.append({
                'from_name': 'label',
                'to_name':   'image',
                'type':      'rectanglelabels',
                'score':     float(box.conf[0]),
                'value': {
                    'x':               x1 / img_w * 100,
                    'y':               y1 / img_h * 100,
                    'width':           (x2-x1) / img_w * 100,
                    'height':          (y2-y1) / img_h * 100,
                    'rectanglelabels': [model.names[int(box.cls[0])]],
                }
            })
        """ """

        """
        # ✅ 중복 박스 제거
        deduped_boxes = self._dedup_boxes(boxes, iou_threshold=0.5, strategy="confidence")

        regions = []
        for x1, y1, x2, y2, conf, cls_id in deduped_boxes:
            regions.append({
                'from_name': 'label',
                'to_name':   'image',
                'type':      'rectanglelabels',
                'score':     conf,
                'value': {
                    'x':               x1 / img_w * 100,
                    'y':               y1 / img_h * 100,
                    'width':           (x2 - x1) / img_w * 100,
                    'height':          (y2 - y1) / img_h * 100,
                    'rectanglelabels': [model.names[cls_id]],
                }
            })
        """
        
        print("MLBackend > _get_predictions_from_local", serialized_tasks)
        predictions = []
        predictions.append({
            'task': task_id, 
            'result':  regions, 
            'score': None, 
            'model_version': model_path, 
            'project': project_id
        })
        return predictions

    def _iou(self, box_a, box_b) -> float:
        """xyxy 좌표 두 박스의 IoU 계산"""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        xi1, yi1 = max(ax1, bx1), max(ay1, by1)
        xi2, yi2 = min(ax2, bx2), min(ay2, by2)

        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if inter == 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0


    def _dedup_boxes(
        self,
        boxes,
        iou_threshold: float = 0.5,
        strategy: str = "confidence",
    ) -> list:
        """
        YOLO 추론 결과(ultralytics Boxes)에서 중복(겹치는) 박스 제거.

        Args:
            boxes: results[0].boxes (ultralytics Boxes 객체)
            iou_threshold: 이 값 이상 겹치면 중복으로 간주
            strategy:
                "confidence" - 겹치는 그룹 중 confidence 최고 1개만 유지 (기본, class 무시 NMS)
                "per_class"  - 같은 class끼리만 겹침 비교 (다른 class면 유지)

        Returns:
            중복 제거된 box 정보 리스트: [(x1, y1, x2, y2, conf, cls_id), ...]
        """
        n = len(boxes)
        if n == 0:
            return []

        items = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            items.append((x1, y1, x2, y2, conf, cls_id))

        # confidence 내림차순 정렬 (NMS 표준 방식)
        order = sorted(range(n), key=lambda i: items[i][4], reverse=True)

        keep = []
        suppressed = set()

        for idx in order:
            if idx in suppressed:
                continue
            keep.append(idx)

            for other_idx in order:
                if other_idx == idx or other_idx in suppressed:
                    continue

                if strategy == "per_class" and items[idx][5] != items[other_idx][5]:
                    continue  # 다른 클래스는 비교하지 않음

                iou = self._iou(items[idx][:4], items[other_idx][:4])
                if iou >= iou_threshold:
                    suppressed.add(other_idx)

        return [items[i] for i in keep]

    def _get_predictions_from_ml_backend(self, serialized_tasks: List[Dict]) -> List[Dict]:
        result = self.api.make_predictions(serialized_tasks, self.project)

        # response validation
        if result.is_error:
            logger.error(f'Error occurred: {result.error_message}')
            return []
        elif not isinstance(result.response, dict) or 'results' not in result.response:
            logger.error(f'ML backend returns an incorrect response, it must be a dict: {result.response}')
            return []
        elif not isinstance(result.response['results'], list) or len(result.response['results']) == 0:
            logger.error(
                'ML backend returns an incorrect response, results field must be a list with at least one item'
            )
            return []

        responses = result.response['results']

        predictions = []
        if len(serialized_tasks) != len(responses):
            # Number of tasks and responses are not equal
            # It can happen if ML backend doesn't support batch processing but only process one task at a time
            # In the future versions, we may better consider this as an error and deprecate this code branch
            return self._get_predictions_from_ml_backend_one_by_one(serialized_tasks, responses)

        # ML backend supports batch processing
        for task, response in zip(serialized_tasks, responses):
            if isinstance(response, dict):
                # ML backend can return single prediction per task or multiple predictions
                response = [response]

            # get all predictions per task
            for r in response:
                if 'result' not in r:
                    logger.error(
                        f"ML backend returns an incorrect prediction, it should be a dict with the 'result' field: {r}"
                    )
                    continue
                predictions.append(
                    {
                        'task': task['id'],
                        'result': r['result'],
                        'score': r.get('score'),
                        'model_version': r.get('model_version', self.model_version),
                        'project': task['project'],
                    }
                )
        return predictions

    def predict_tasks(self, tasks):
        print("MLBackend > predict_tasks", tasks, self.is_local, self.url)
        model_version = self.update_state()
        print("MLBackend > predict_tasks", self.not_ready)
        if self.not_ready:
            logger.debug(f'ML backend {self} is not ready')
            return

        if isinstance(tasks, list):
            from tasks.models import Task

            tasks = Task.objects.filter(id__in=[task.id for task in tasks])

        # Filter tasks that already contain the current model version in predictions
        tasks = tasks.annotate(predictions_count=Count('predictions')).exclude(
            Q(predictions_count__gt=0) & Q(predictions__model_version=model_version)
        )

        print("MLBackend > predict_tasks", tasks.exists())
        if not tasks.exists():
            logger.debug(f'All tasks already have prediction from model version={self.model_version}')
            return model_version
        tasks_ser = TaskSimpleSerializer(tasks, many=True).data

        print("MLBackend > predict_tasks", self.is_local, self.url)
        if self.is_local:
            predictions = self._get_predictions_from_local(tasks_ser)
        else:
            predictions = self._get_predictions_from_ml_backend(tasks_ser)
        with conditional_atomic(predicate=db_is_not_sqlite):
            prediction_ser = PredictionSerializer(data=predictions, many=True)
            prediction_ser.is_valid(raise_exception=True)
            instances = prediction_ser.save()
        return instances

    def interactive_annotating(self, task, context=None, user=None):
        result = {}
        options = {}
        if user:
            options = {'user': user}
        if not self.is_interactive:
            result['errors'] = ['Model is not set to be used for interactive preannotations']
            return result

        tasks_ser = InteractiveAnnotatingDataSerializer(
            [task], many=True, expand=['drafts', 'predictions', 'annotations'], context=options
        ).data
        ml_api_result = self.api.make_predictions(
            tasks=tasks_ser,
            project=self.project,
            context=context,
        )
        if ml_api_result.is_error:
            logger.info(f'Prediction not created for project {self}: {ml_api_result.error_message}')
            result['errors'] = [ml_api_result.error_message]
            return result

        if not (isinstance(ml_api_result.response, dict) and 'results' in ml_api_result.response):
            logger.info(f'ML backend returns an incorrect response, it must be a dict: {ml_api_result.response}')
            result['errors'] = [
                'Incorrect response from ML service: ML backend returns an incorrect response, it must be a dict.'
            ]
            return result

        ml_results = ml_api_result.response.get(
            'results',
            [
                None,
            ],
        )
        if not isinstance(ml_results, list) or len(ml_results) < 1:
            logger.warning(f'ML backend has to return list with 1 annotation but it returned: {type(ml_results)}')
            result['errors'] = [
                'Incorrect response from ML service: ML backend has to return list with more than 1 result.'
            ]
            return result
        result['data'] = ml_results[0]
        return result

    @staticmethod
    def get_versions_(url, project, auth_method, **kwargs):
        api = MLApi(url=url, auth_method=auth_method, **kwargs)
        if not isinstance(project, Project):
            project = Project.objects.get(pk=project)
        return api.get_versions(project)

    def get_versions(self):
        return self.get_versions_(
            self.url,
            self.project,
            self.auth_method,
            basic_auth_user=self.basic_auth_user,
            basic_auth_pass=self.basic_auth_pass,
        )


class MLBackendPredictionJob(models.Model):
    job_id = models.CharField(max_length=128)
    ml_backend = models.ForeignKey(MLBackend, related_name='prediction_jobs', on_delete=models.CASCADE)
    model_version = models.TextField(
        _('model version'), blank=True, null=True, help_text='Model version this job is associated with'
    )
    batch_size = models.PositiveSmallIntegerField(
        _('batch size'), default=100, help_text='Number of tasks processed per batch'
    )

    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)


class MLBackendTrainJob(models.Model):
    job_id = models.CharField(max_length=128)
    ml_backend = models.ForeignKey(MLBackend, related_name='train_jobs', on_delete=models.CASCADE)
    model_version = models.TextField(
        _('model version'),
        blank=True,
        null=True,
        help_text='Model version this job is associated with',
    )
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)

    def get_status(self):
        project = self.ml_backend.project
        ml_api = project.get_ml_api()
        if not ml_api:
            logger.error(
                f"Training job {self.id}: Can't collect training jobs for project {project.id}: ML API is null"
            )
            return None
        ml_api_result = ml_api.get_train_job_status(self)
        if ml_api_result.is_error:
            if ml_api_result.status_code == 410:
                return {'job_status': 'removed'}
            logger.info(
                f"Training job {self.id}: Can't collect training jobs for project {project}: "
                f'ML API returns error {ml_api_result.error_message}'
            )
            return None
        return ml_api_result.response

    @property
    def is_running(self):
        status = self.get_status()
        return status['job_status'] in ('queued', 'started')


def _validate_ml_api_result(ml_api_result, tasks, curr_logger):
    if ml_api_result.is_error:
        curr_logger.info(ml_api_result.error_message)
        return False

    results = ml_api_result.response['results']
    if not isinstance(results, list) or len(results) != len(tasks):
        curr_logger.warning('Num input tasks is %d but ML API returns %d results', len(tasks), len(results))
        return False

    return True


@receiver(pre_delete, sender=MLBackend)
def modify_project_model_version(sender, instance, **kwargs):
    project = instance.project

    if project.model_version == instance.title:
        project.model_version = ''
        project.save(update_fields=['model_version'])


@receiver(post_save, sender=MLBackend)
def create_ml_webhook(sender, instance, created, **kwargs):
    if not created:
        return
    ml_backend = instance
    webhook_url = ml_backend.url.rstrip('/') + '/webhook'
    project = ml_backend.project
    if Webhook.objects.filter(project=project, url=webhook_url).exists():
        logger.info(f'Webhook {webhook_url} already exists for project {project}: skip creating new one.')
        return
    logger.info(f'Create ML backend webhook {webhook_url}')
    ser = WebhookSerializer(
        data=dict(project=project.id, url=webhook_url, send_payload=True, send_for_all_actions=True)
    )
    if ser.is_valid():
        ser.save(organization=project.organization)

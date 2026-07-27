"""
tasks/api_task_footprint.py

GPS 오버레이용 footprint 조회 엔드포인트.

배경:
    실제 XMP+EXIF 인덱스 데이터는 TaskImageGeoreferencing(사이드카,
    task 1:N, 컬럼명 footprint 단수)에 쌓인다. TaskGeoreferencing
    (task 1:1, footprint_rgb/footprint_ir)은 비어 있을 수 있으므로,
    오버레이는 사이드카를 우선 조회한다.

응답 형식은 프론트(GpsOverlayLayer)가 기대하는 GeoJSON 형태로 맞춘다:

    {
      "task": 278,
      "footprint_rgb": {"type": "Polygon", "coordinates": [[[lng,lat], ...]]},
      "footprint_ir":  {...} | null,
      "source": "task_image_georeferencing"
    }

URL 등록 (tasks/urls.py):

    from tasks.api_task_footprint import TaskFootprintAPI

    urlpatterns += [
        path('api/tasks/<int:pk>/footprint/',
             TaskFootprintAPI.as_view(), name='task-footprint'),
    ]
"""

import json
import logging

from rest_framework import generics
from rest_framework.exceptions import PermissionDenied
from rest_framework.response import Response
from rest_framework.views import APIView

from core.permissions import ViewClassPermission, all_permissions
from tasks.models import Task, TaskImageGeoreferencing

logger = logging.getLogger(__name__)


def _geojson(geom):
    """GEOSGeometry -> GeoJSON dict (None 안전)."""
    if geom is None:
        return None
    return json.loads(geom.geojson)


class TaskFootprintAPI(APIView):
    """GET api/tasks/<pk>/footprint/ — 오버레이용 footprint 조회.

    TaskImageGeoreferencing(사이드카)에서 modality별 footprint를 찾아
    GeoJSON으로 반환한다. 없으면 TaskGeoreferencing으로 폴백.
    """

    permission_required = ViewClassPermission(
        GET=all_permissions.tasks_view,
    )

    def get(self, request, pk):
        task = generics.get_object_or_404(
            Task.objects.select_related('project'), pk=pk)
        if not task.project.has_permission(request.user):
            raise PermissionDenied(
                'You do not have permission to access this task')

        payload = {
            'task': task.id,
            'footprint_rgb': None,
            'footprint_ir': None,
            'source': None,
        }

        # 1) 사이드카 우선 — task 1:N (RGB/IR 별도 row)
        records = TaskImageGeoreferencing.objects.filter(task_id=task.id)
        for rec in records:
            if rec.footprint is None:
                continue
            gj = _geojson(rec.footprint)
            if rec.modality == 'ir':
                payload['footprint_ir'] = gj
            else:
                # wide/zoom/빈값 모두 rgb 취급 (첫 항목 우선)
                if payload['footprint_rgb'] is None:
                    payload['footprint_rgb'] = gj
            payload['source'] = 'task_image_georeferencing'

        # 2) 폴백 — TaskGeoreferencing (1:1, footprint_rgb/ir 보유)
        if payload['footprint_rgb'] is None and payload['footprint_ir'] is None:
            geo = getattr(task, 'georeferencing', None)
            if geo is not None:
                payload['footprint_rgb'] = _geojson(
                    getattr(geo, 'footprint_rgb_corrected', None)
                    or getattr(geo, 'footprint_rgb', None))
                payload['footprint_ir'] = _geojson(
                    getattr(geo, 'footprint_ir_corrected', None)
                    or getattr(geo, 'footprint_ir', None))
                if payload['footprint_rgb'] or payload['footprint_ir']:
                    payload['source'] = 'task_georeferencing'

        if payload['source'] is None:
            logger.info('[footprint] task=%s georeferencing 데이터 없음', task.id)

        return Response(payload)
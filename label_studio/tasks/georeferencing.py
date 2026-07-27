from django.db.models.expressions import result
import logging
import os
from urllib.parse import parse_qs, unquote, urlparse

from django.conf import settings

from tasks.models import Task, TaskGeoreferencing, TaskImageGeoreferencing
from sayou.gps_image_index_pg import GpsImageIndexPG

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION = '1'


def _index_db_path() -> str:
    return getattr(
        settings, 'GPS_IMAGE_INDEX_DB',
        os.path.join(getattr(settings, 'BASE_DATA_DIR', '.'),
                     'gps_image_index.db'))

# ---------------------------------------------------------------------------
# task.data 이미지 참조 -> 로컬 파일 경로 해석
# ---------------------------------------------------------------------------

def resolve_local_path(value: str) -> str | None:
    """task.data의 이미지 참조를 디스크 경로로 변환한다.
 
    지원:
      - 절대경로: /data/solar/DJI_0001_W.JPG
      - local-files 서빙 URL: /data/local-files/?d=solar/DJI_0001_W.JPG
      - 업로드 파일: /data/upload/<project>/<file>.jpg
    미지원 (None 반환):
      - s3:// gs:// http(s):// — 원격 스토리지는 파일 실체가 로컬에 없음.
        필요 시 presign 다운로드 후 인덱싱하는 별도 배치로 처리할 것.
    """
    if not isinstance(value, str):
        return None
 
    if value.startswith(('s3://', 'gs://', 'azure-blob://',
                         'http://', 'https://')):
        return None
 
    if '/data/local-files/' in value:
        qs = parse_qs(urlparse(value).query)
        rel = unquote(qs.get('d', [''])[0])
        root = getattr(settings, 'LOCAL_FILES_DOCUMENT_ROOT', '')
        path = os.path.join(root, rel)
        return path if os.path.exists(path) else None
 
    if value.startswith('/data/upload/'):
        media_root = getattr(settings, 'MEDIA_ROOT', '')
        rel = value[len('/data/'):]        # upload/<project>/<file>
        path = os.path.join(media_root, rel)
        return path if os.path.exists(path) else None
 
    if os.path.isabs(value) and os.path.exists(value):
        return value
    return None

def extract_georeferencing_job(task_id: int) -> dict:
    """Task 이미지에서 XMP+EXIF를 추출해 TaskGeoreferencing을 채운다.
 
    - task.data의 로컬 이미지 참조를 해석 (원격 스토리지는 미지원)
    - 추출: sayou.dji_metadata_extractor (metadata.py 확장 레이어)
    - footprint 계산·저장까지 수행 (RGB/IR modality별)
    """
    from sayou.dji_metadata_extractor import extract, to_georeferencing_dict
    from sayou.gps_image_index import compute_footprint
    from django.contrib.gis.geos import Polygon
 
    task = Task.objects.select_related('project').get(pk=task_id)
    geo, _ = TaskGeoreferencing.objects.get_or_create(task=task)
 
    # ---- 이미지 경로 수집 (data key -> 로컬 경로) ----
    paths = {}
    if isinstance(task.data, dict):
        for key, value in task.data.items():
            p = resolve_local_path(value)
            if p:
                paths[key] = p
    if not paths:
        geo.mark_failed('로컬 이미지 경로 없음 (원격 스토리지는 미지원)')
        geo.save()
        return {'status': 'failed', 'task': task_id,
                'detail': geo.error_message}
 
    try:
        # 대표(primary) 이미지: rgb 키 우선, 없으면 첫 항목
        primary_key = next(
            (k for k in ('rgb', 'image', 'img') if k in paths),
            next(iter(paths)))
        meta = extract(paths[primary_key])
 
        geo.populate_from_metadata(
            to_georeferencing_dict(meta),
            extractor_version=EXTRACTOR_VERSION)
 
        meta_dict = meta.to_dict()
        meta_dict.pop('R_cam_to_enu', None)
        geo.raw_metadata = meta_dict
 
        # ---- footprint (RGB primary + IR 별도 파일이 있으면 각각) ----
        low_any = False
        if geo.status == TaskGeoreferencing.Status.COMPLETED:
            corners, low = compute_footprint(meta)
            geo.footprint_rgb = Polygon(
                [(c[0], c[1]) for c in corners] + [tuple(corners[0])],
                srid=4326)
            low_any |= low
 
            ir_path = paths.get('ir')
            if ir_path:
                ir_meta = extract(ir_path)
                if ir_meta.gps.lat and ir_meta.gps.lng:
                    ir_corners, ir_low = compute_footprint(ir_meta)
                    geo.footprint_ir = Polygon(
                        [(c[0], c[1]) for c in ir_corners]
                        + [tuple(ir_corners[0])], srid=4326)
                    low_any |= ir_low
        geo.footprint_low_confidence = low_any
 
        geo.save()
        logger.info('[georef-extract] task=%s status=%s modality=%s',
                    task_id, geo.status, geo.modality)
        return {'status': 'done', 'task': task_id,
                'detail': geo.status}
    except Exception as e:
        logger.exception('[georef-extract] task=%s 실패', task_id)
        geo.mark_failed(str(e))
        geo.save()
        return {'status': 'failed', 'task': task_id, 'detail': str(e)}
    
def index_georeferencing(task_id, project_id):
    from sayou.dji_metadata_extractor import extract, modality_of
    from sayou.gps_image_index import compute_footprint
    from django.contrib.gis.geos import Polygon

    task = Task.objects.get(pk=task_id)
    paths = {k: resolve_local_path(v) for k, v in (task.data or {}).items()
             if resolve_local_path(v)}
    if not paths:
        return {'status': 'failed', 'task': task_id, 'detail': '로컬 경로 없음'}

    results = []
    for path in paths.values():
        meta = extract(path)
        if not (meta.gps.lat and meta.gps.lng):
            continue

        corners, low = compute_footprint(meta)
        ring = [(c[0], c[1]) for c in corners] + [tuple(corners[0])]
        meta_dict = meta.to_dict(); meta_dict.pop('R_cam_to_enu', None)

        TaskImageGeoreferencing.objects.update_or_create(
            id=meta.id,                         # ← PK = SHA-1
            defaults={
                'task_id': task_id,             # ← FK는 _id 접미사
                'project_id': project_id,
                'path': path,
                'filename': os.path.basename(path),
                'modality': modality_of(meta),
                'capture_time': meta.capture_time or None,
                'lat': meta.gps.lat, 'lng': meta.gps.lng,
                'altitude': meta.gps.altitude,
                'relative_height': meta.relative_height,
                'ground_alt': (meta.lrf_target_abs_alt if meta.has_valid_lrf
                               else meta.gps.altitude - meta.relative_height),
                'ground_alt_src': 'lrf' if meta.has_valid_lrf else 'relative',
                'gimbal_yaw': meta.gimbal_yaw_deg,
                'gimbal_pitch': meta.gimbal_pitch_deg,
                'gimbal_roll': meta.gimbal_roll_deg,
                'rtk_flag': meta.rtk_flag,
                'width': meta.width, 'height': meta.height,
                'focal_35mm': meta.focal_length_in_35mm,
                'low_confidence': low,
                'footprint': Polygon(ring, srid=4326),
                'meta_json': meta_dict,
            },
        )
        results.append(meta.id)

    return {'status': 'done', 'task': task_id, 'indexed': len(results)}

def run_georeferencing_for_tasks(task_ids, project):

    print("project", project, type(project))
    print(f'Georeferencing 시작: {project} {len(task_ids)}개 태스크 {task_ids}')
    
    for task_id in task_ids:
        print(f'Georeferencing for task {task_id}')
        #result = extract_georeferencing_job(task_id)
        result = index_georeferencing(task_id, project.id)
        print("result", result)

    return {}
# label_studio/data_manager/actions/exclude_from_export.py (신규 파일)
"""Export 제외/포함 토글 액션"""

import logging

from core.permissions import AllPermissions

logger = logging.getLogger(__name__)
all_permissions = AllPermissions()


def retrieve_georeferencing(project, queryset, **kwargs):
    """
    선택된 태스크(들)에 대해 georeferencing 파이프라인 실행.
    - RTK/PPK 좌표 triangulation
    - RGB-IR 호모그래피 정합
    - DJI H20T EXIF/XMP 메타데이터 기반 카메라 pose 계산
    """
    from tasks.georeferencing import run_georeferencing_for_tasks  # 기존 파이프라인 모듈 연결

    task_ids = list(queryset.values_list('id', flat=True))
    logger.info(f'Georeferencing 시작: {len(task_ids)}개 태스크 {task_ids}')

    try:
        result = run_georeferencing_for_tasks(task_ids, project)
        return {
            'processed_items': len(task_ids),
            'detail': f'Georeferencing 완료: {len(task_ids)}개 태스크',
            'result': result,
        }
    except Exception as e:
        logger.exception('Georeferencing 실패')
        return {
            'processed_items': 0,
            'detail': f'Georeferencing 실패: {str(e)}',
        }

actions = [
    {
        'entry_point': retrieve_georeferencing,
        'permission': all_permissions.tasks_change,
        'title': 'Retrieve Georeferencing',
        'order': 97,
        'dialog': {
            'text': '선택한 태스크에 대해 Georeferencing을 계산합니다. 시간이 걸릴 수 있습니다. 계속하시겠습니까?',
            'type': 'confirm',
        },
    },
]
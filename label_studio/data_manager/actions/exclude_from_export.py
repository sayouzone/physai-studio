# label_studio/data_manager/actions/exclude_from_export.py (신규 파일)
"""Export 제외/포함 토글 액션"""

from core.permissions import AllPermissions

all_permissions = AllPermissions()


def exclude_from_export(project, queryset, **kwargs):
    """선택 태스크를 export에서 제외"""
    count = queryset.update(exclude_from_export=True)
    return {'processed_items': count, 'detail': f'Excluded {count} tasks from export'}


def include_in_export(project, queryset, **kwargs):
    """선택 태스크를 export에 다시 포함"""
    count = queryset.update(exclude_from_export=False)
    return {'processed_items': count, 'detail': f'Included {count} tasks in export'}


actions = [
    {
        'entry_point': exclude_from_export,
        'permission': all_permissions.tasks_change,
        'title': 'Exclude from Export',
        'order': 95,
        'dialog': {
            'text': '선택한 태스크를 export에서 제외합니다. 계속하시겠습니까?',
            'type': 'confirm',
        },
    },
    {
        'entry_point': include_in_export,
        'permission': all_permissions.tasks_change,
        'title': 'Include in Export',
        'order': 96,
        'dialog': {
            'text': '선택한 태스크를 export에 다시 포함합니다. 계속하시겠습니까?',
            'type': 'confirm',
        },
    },
]
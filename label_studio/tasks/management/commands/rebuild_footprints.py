"""
tasks/management/commands/rebuild_footprints.py

TaskImageGeoreferencing.footprint 를 **현재 파이프라인**으로 재생성한다.

2026-07 (4차) — 정합 보정 배선
──────────────────────────────
row.task_id 를 build_footprint()/compare_footprints() 에 전달한다.
SAYOU_FRAME_OFFSETS 가 설정돼 있으면 재생성되는 footprint 가 프레임
정합 보정(alignment.py)까지 반영한 좌표가 된다. 검출 폴리곤 쪽
(annotation_georeferencer)에도 동일 보정이 배선되어 있어야 두 좌표계가
계속 일치한다 — 한쪽만 재생성하면 다시 어긋난다.

DB 주의
───────
TaskImageGeoreferencing 은 PostGIS 모델이다. Label Studio 는 DJANGO_DB
환경변수로 백엔드를 고르고 **기본값이 sqlite** 이므로, 셸에서 그냥
`python manage.py` 를 실행하면 서버와 다른 DB(sqlite)에 붙어
'no such table: task_image_georeferencing' 로 죽는다.
서버와 같은 환경변수를 export 한 뒤 실행할 것. 이 명령은 시작 시
연결 대상을 검사해 잘못된 DB 면 즉시 중단한다.

사용
────
    # 정합 보정 없이 물리 모델만 재생성/검사
    python manage.py rebuild_footprints --project 3 --check

    # 정합 보정 포함 (SAYOU_FRAME_OFFSETS 설정 후)
    export SAYOU_FRAME_OFFSETS=/path/to/frame_offsets.json
    python manage.py rebuild_footprints --project 3 --apply
"""

import logging
import os
from typing import Optional

from django.contrib.gis.geos import Polygon
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction

from sayou.image.metadata import extract_metadata
from sayou.georeferencing.footprint import build_footprint, compare_footprints
from sayou.georeferencing.alignment import get_alignment, OFFSETS_PATH_ENV

from tasks.models import TaskImageGeoreferencing

logger = logging.getLogger(__name__)


def plane_kwargs(project_id: Optional[int] = None) -> dict:
    """api.py 의 _plane_kwargs() 와 **같은 값**이어야 한다 (단일 출처)."""
    try:
        from tasks.api import _plane_kwargs
        return _plane_kwargs(None)
    except Exception:
        return {}


def assert_spatial_db():
    d = connection.settings_dict
    engine = d.get('ENGINE', '')
    where = f"{d.get('NAME')} @ {d.get('HOST') or 'local'}:{d.get('PORT') or '-'}"

    if 'postgis' not in engine and 'postgresql' not in engine:
        raise CommandError(
            f"PostGIS 가 아닌 DB 에 연결되었습니다.\n"
            f"  ENGINE : {engine}\n"
            f"  NAME   : {d.get('NAME')}\n\n"
            f"Label Studio 는 DJANGO_DB 기본값이 sqlite 라서, 서버와 다른 DB 에\n"
            f"붙기 쉽습니다. 서버와 동일한 환경변수를 export 한 뒤 다시 실행하세요:\n"
            f"  export DJANGO_DB=default\n"
            f"  export POSTGRE_NAME=... POSTGRE_USER=... POSTGRE_PASSWORD=...\n"
            f"  export POSTGRE_HOST=... POSTGRE_PORT=...\n"
        )

    table = TaskImageGeoreferencing._meta.db_table
    if table not in connection.introspection.table_names():
        with connection.cursor() as cur:
            cur.execute('SHOW search_path')
            search_path = cur.fetchone()[0]
            cur.execute(
                """
                SELECT n.nspname
                FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = %s
                """,
                [table],
            )
            schemas = [r[0] for r in cur.fetchall()]
        raise CommandError(
            f"테이블 '{table}' 을 찾을 수 없습니다 ({where}).\n"
            f"  search_path        : {search_path}\n"
            f"  테이블이 있는 스키마 : {schemas or '없음'}\n"
        )

    return where


class Command(BaseCommand):
    help = 'TaskImageGeoreferencing.footprint 를 현재 georeferencing 파이프라인(+정합 보정)으로 재생성'

    def add_arguments(self, parser):
        parser.add_argument('--project', type=int, default=None, help='프로젝트 ID 필터')
        parser.add_argument('--task', type=int, default=None, help='Task ID 필터')
        parser.add_argument('--modality', type=str, default=None, help="'wide'|'zoom'|'ir'")
        parser.add_argument('--limit', type=int, default=0, help='처리 최대 건수 (0=제한없음)')
        parser.add_argument('--apply', action='store_true', help='실제로 DB 를 갱신')
        parser.add_argument('--check', action='store_true', help='검사만 (기본 동작)')
        parser.add_argument('--tolerance', type=float, default=1.5, help='stale 판정 임계 (%%)')

    def handle(self, *args, **opts):
        where = assert_spatial_db()
        self.stdout.write(f'DB: {where}  (table={TaskImageGeoreferencing._meta.db_table})')

        alignment = get_alignment()
        if alignment.enabled:
            self.stdout.write(
                f'정합 보정: 활성 ({os.environ.get(OFFSETS_PATH_ENV)}) — '
                f'재생성되는 footprint 에 프레임 오프셋이 반영됩니다.'
            )
        else:
            self.stdout.write(self.style.WARNING(
                f'정합 보정: 비활성 ({OFFSETS_PATH_ENV} 미설정 또는 파일 없음). '
                f'물리 모델만 재생성됩니다.'
            ))

        qs = TaskImageGeoreferencing.objects.all().order_by('id')
        if opts['project'] is not None:
            qs = qs.filter(project_id=opts['project'])
        if opts['task'] is not None:
            qs = qs.filter(task_id=opts['task'])
        if opts['modality']:
            qs = qs.filter(modality=opts['modality'])

        total = qs.count()
        if opts['limit']:
            qs = qs[: opts['limit']]
            total = min(total, opts['limit'])

        apply_changes = opts['apply']
        tol = opts['tolerance']

        self.stdout.write(
            f'대상 {total} 건 / 모드: {"APPLY (DB 갱신)" if apply_changes else "CHECK (읽기 전용)"}'
        )
        if total == 0:
            self.stdout.write(self.style.WARNING(
                '대상 레코드가 없습니다. --project 필터와 인제스트 여부를 확인하세요.'
            ))
            return

        stats = {'ok': 0, 'stale': 0, 'updated': 0, 'no_image': 0, 'failed': 0,
                 'aligned_measured': 0, 'aligned_interp': 0, 'aligned_none': 0}
        kwargs = plane_kwargs(opts['project'])

        rows = qs.iterator(chunk_size=200) if not opts['limit'] else list(qs)

        for row in rows:
            path = row.path
            if not path:
                stats['no_image'] += 1
                continue

            try:
                metadata = extract_metadata(path)
            except FileNotFoundError:
                self.stderr.write(f'  [{row.id}] 이미지 없음: {path}')
                stats['no_image'] += 1
                continue
            except Exception as e:
                self.stderr.write(f'  [{row.id}] 메타데이터 추출 실패: {e}')
                stats['failed'] += 1
                continue

            try:
                stored_ring = list(row.footprint.coords[0])[:4] if row.footprint else None
            except Exception:
                stored_ring = None

            # task_id 를 넘겨야 정합 보정이 비교/재생성에 반영된다.
            if stored_ring:
                cmp_result = compare_footprints(
                    stored_ring, metadata, tolerance_pct=tol,
                    task_id=row.task_id, **kwargs,
                )
                al_info = cmp_result.get('alignment') or {}
                if cmp_result.get('stale') is False:
                    stats['ok'] += 1
                    self._tally_alignment(stats, al_info)
                    continue
                if cmp_result.get('stale'):
                    stats['stale'] += 1
                    self.stdout.write(
                        f"  [{row.id}] task={row.task_id} STALE  "
                        f"저장 {cmp_result['stored_m']} m  기대 {cmp_result['expected_m']} m  "
                        f"비율 {cmp_result['ratio']}  정합={al_info.get('source', 'none')}"
                    )
            else:
                stats['stale'] += 1
                self.stdout.write(f'  [{row.id}] task={row.task_id} footprint 없음')

            if not apply_changes:
                continue

            fresh = build_footprint(metadata, task_id=row.task_id, **kwargs)
            if fresh is None:
                self.stderr.write(f'  [{row.id}] footprint 재생성 실패 — 건너뜀')
                stats['failed'] += 1
                continue

            self._tally_alignment(stats, fresh.get('alignment') or {})

            ring = fresh['footprint']['coordinates'][0]
            try:
                with transaction.atomic():
                    row.footprint = Polygon([tuple(p) for p in ring], srid=4326)
                    row.save(update_fields=['footprint'])
                stats['updated'] += 1
            except Exception as e:
                self.stderr.write(f'  [{row.id}] 저장 실패: {e}')
                stats['failed'] += 1

        self.stdout.write('')
        self.stdout.write(
            f"완료 — 정상 {stats['ok']} / 불일치 {stats['stale']} / "
            f"갱신 {stats['updated']} / 이미지없음 {stats['no_image']} / 실패 {stats['failed']}"
        )
        if alignment.enabled:
            self.stdout.write(
                f"정합 보정 — 실측 {stats['aligned_measured']} / "
                f"보간 {stats['aligned_interp']} / 미적용 {stats['aligned_none']}"
            )
        if not apply_changes and stats['stale']:
            self.stdout.write(self.style.WARNING(
                '불일치 레코드가 있습니다. --apply 로 재생성하세요. '
                '재생성 후 프론트 캐시(브라우저 새로고침)도 함께 확인할 것.'
            ))

    @staticmethod
    def _tally_alignment(stats: dict, al_info: dict) -> None:
        if not al_info.get('applied'):
            stats['aligned_none'] += 1
        elif al_info.get('source') == 'measured':
            stats['aligned_measured'] += 1
        else:
            stats['aligned_interp'] += 1
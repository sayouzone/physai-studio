"""
sayou/gps_image_index_pg.py

GpsImageIndex의 PostgreSQL/PostGIS 구현.
SQLite 버전(sayou.gps_image_index.GpsImageIndex)과 동일한 인터페이스를
제공하므로 설정에 따라 교체 사용 가능하다.

SQLite 버전과의 차이:
  - bbox 컬럼/2단계 파이썬 판정 불필요 — PostGIS GiST 인덱스가
    Stage 1(bbox 스캔) + Stage 2(정밀 recheck)를 네이티브로 수행
  - 거리/반경은 geography 캐스트로 미터 단위 정확 계산
    (함수 인덱스 `USING GIST ((footprint::geography))` 로 인덱스 히트)
  - 동시 쓰기 안전 (INSERT ... ON CONFLICT upsert)

전제:
    CREATE EXTENSION postgis;   -- DB에 1회

사용:
    from sayou.gps_image_index_pg import GpsImageIndexPG

    with GpsImageIndexPG(dsn="dbname=public user=... host=...") as idx:
        idx.index_directory("/data/flight_0620", project_id=3)
        hits = idx.find_images_containing(37.3009, 127.0512, radius_m=10)

    # Django 안에서는 settings DB 재사용:
    with GpsImageIndexPG.from_django_default() as idx:
        ...

CLI:
    python -m sayou.gps_image_index_pg index --dsn "..." <image_dir>
    python -m sayou.gps_image_index_pg find  --dsn "..." <lat> <lng> [--radius 10]

의존성: psycopg2, PostGIS 확장 + metadata.py / dji_metadata_extractor.py
footprint 기하는 SQLite 버전의 검증된 compute_footprint를 재사용한다.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras

# 추출기/기하 재사용 — 배치 위치에 따른 폴백 체인
try:
    from sayou.dji_metadata_extractor import extract, modality_of
    from sayou.gps_image_index import compute_footprint
    from sayou.image.metadata import M_PER_DEG_LAT
except ImportError:                                     # 단독 실행/테스트
    from dji_metadata_extractor import extract, modality_of
    from gps_image_index import compute_footprint
    from .image.metadata import M_PER_DEG_LAT

logger = logging.getLogger(__name__)

TABLE_NAME = 'task_image_georeferencing'

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id              TEXT PRIMARY KEY,       -- 파일 SHA-1
    task_id         INTEGER,
    project_id      INTEGER,
    path            TEXT NOT NULL,
    filename        TEXT NOT NULL,
    modality        TEXT,
    capture_time    BIGINT,                 -- epoch seconds
    lat DOUBLE PRECISION, lng DOUBLE PRECISION,
    altitude DOUBLE PRECISION,
    relative_height DOUBLE PRECISION,
    ground_alt      DOUBLE PRECISION,
    ground_alt_src  TEXT,
    gimbal_yaw DOUBLE PRECISION,
    gimbal_pitch DOUBLE PRECISION,
    gimbal_roll DOUBLE PRECISION,
    rtk_flag        INTEGER,
    width INTEGER, height INTEGER,
    focal_35mm      INTEGER,
    low_confidence  BOOLEAN DEFAULT FALSE,
    footprint       geometry(Polygon, 4326) NOT NULL,
    meta_json       JSONB,
    indexed_at      TIMESTAMPTZ DEFAULT now()
);

-- Stage 1+2 통합: GiST가 bbox 스캔 후 정밀 recheck까지 수행
CREATE INDEX IF NOT EXISTS idx_gpsimg_footprint
    ON {TABLE_NAME} USING GIST (footprint);
-- 미터 단위 ST_DWithin/ST_Distance(geography) 가속용 함수 인덱스
CREATE INDEX IF NOT EXISTS idx_gpsimg_footprint_geog
    ON {TABLE_NAME} USING GIST ((footprint::geography));

CREATE INDEX IF NOT EXISTS idx_gpsimg_task ON {TABLE_NAME} (task_id);
CREATE INDEX IF NOT EXISTS idx_gpsimg_project
    ON {TABLE_NAME} (project_id);
CREATE INDEX IF NOT EXISTS idx_gpsimg_capture
    ON {TABLE_NAME} (capture_time);
CREATE INDEX IF NOT EXISTS idx_gpsimg_modality
    ON {TABLE_NAME} (modality);
"""

_UPSERT = f"""
INSERT INTO {TABLE_NAME}
    (id, task_id, project_id, path, filename, modality, capture_time,
     lat, lng, altitude, relative_height, ground_alt, ground_alt_src,
     gimbal_yaw, gimbal_pitch, gimbal_roll,
     rtk_flag, width, height, focal_35mm,
     low_confidence, footprint, meta_json)
VALUES
    (%(id)s, %(task_id)s, %(project_id)s, %(path)s, %(filename)s,
     %(modality)s, %(capture_time)s,
     %(lat)s, %(lng)s, %(altitude)s, %(relative_height)s,
     %(ground_alt)s, %(ground_alt_src)s,
     %(gimbal_yaw)s, %(gimbal_pitch)s, %(gimbal_roll)s,
     %(rtk_flag)s, %(width)s, %(height)s, %(focal_35mm)s,
     %(low_confidence)s,
     ST_SetSRID(ST_GeomFromGeoJSON(%(footprint_geojson)s), 4326),
     %(meta_json)s)
ON CONFLICT (id) DO UPDATE SET
    task_id = EXCLUDED.task_id,
    project_id = EXCLUDED.project_id,
    path = EXCLUDED.path,
    filename = EXCLUDED.filename,
    modality = EXCLUDED.modality,
    capture_time = EXCLUDED.capture_time,
    lat = EXCLUDED.lat, lng = EXCLUDED.lng,
    altitude = EXCLUDED.altitude,
    relative_height = EXCLUDED.relative_height,
    ground_alt = EXCLUDED.ground_alt,
    ground_alt_src = EXCLUDED.ground_alt_src,
    gimbal_yaw = EXCLUDED.gimbal_yaw,
    gimbal_pitch = EXCLUDED.gimbal_pitch,
    gimbal_roll = EXCLUDED.gimbal_roll,
    rtk_flag = EXCLUDED.rtk_flag,
    width = EXCLUDED.width, height = EXCLUDED.height,
    focal_35mm = EXCLUDED.focal_35mm,
    low_confidence = EXCLUDED.low_confidence,
    footprint = EXCLUDED.footprint,
    meta_json = EXCLUDED.meta_json,
    indexed_at = now();
"""


@dataclass
class ImageHit:
    id: str
    task_id: Optional[int]
    project_id: Optional[int]
    path: str
    filename: str
    modality: str
    capture_time: Optional[int]
    lat: float
    lng: float
    rtk_flag: Optional[int]
    distance_m: float           # 0 = footprint가 점을 포함
    low_confidence: bool
    footprint: list             # [[lng, lat], ...] (닫힌 링)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class GpsImageIndexPG:
    """PostgreSQL/PostGIS 기반 GPS 이미지 인덱스 (SQLite 버전과 동일 인터페이스)."""

    def __init__(self, dsn: str | None = None,
                 connection=None, autocreate_schema: bool = True):
        """dsn 또는 기존 psycopg2 connection 중 하나로 초기화.

        connection을 넘기면 소유권은 호출자에 있음 (close 시 닫지 않음).
        """
        if connection is not None:
            self.conn = connection
            self._owns_conn = False
        elif dsn:
            self.conn = psycopg2.connect(dsn)
            self._owns_conn = True
        else:
            raise ValueError('dsn 또는 connection이 필요합니다.')

        if autocreate_schema:
            with self.conn.cursor() as cur:
                cur.execute(_SCHEMA)
            self.conn.commit()

    @classmethod
    def from_django_default(cls, autocreate_schema: bool = True
                            ) -> 'GpsImageIndexPG':
        """Django default DB의 접속 파라미터를 그대로 재사용."""
        from django.db import connection as dj_conn

        engine = dj_conn.settings_dict['ENGINE']
        if 'postgresql' not in engine and 'postgis' not in engine:
            raise RuntimeError(
                f'default DB가 PostgreSQL이 아닙니다 (ENGINE={engine}). '
                f'GpsImageIndex(SQLite 버전)를 사용하세요.')

        params = dj_conn.get_connection_params()
        # Django 5.x/psycopg3 환경에서 섞여 들어오는 비-libpq 키 제거
        params.pop('cursor_factory', None)
        params.pop('context', None)

        conn = psycopg2.connect(**params)
        return cls(connection=conn, autocreate_schema=autocreate_schema)

    def close(self):
        if self._owns_conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # 인덱싱
    # ------------------------------------------------------------------

    @staticmethod
    def row_from_image(image_path, task_id, project_id) -> Optional[dict]:
        """추출+footprint 계산 -> upsert 파라미터 dict. GPS 없으면 None."""
        meta = extract(image_path)
        if meta.gps.lat == 0.0 or meta.gps.lng == 0.0:
            logger.warning('GPS 없음, 건너뜀: %s', image_path)
            return None

        try:
            corners, low_conf = compute_footprint(meta)
        except ValueError as e:
            logger.warning('footprint 계산 불가(%s): %s — GPS 점 주변 최소 '
                           '사각형으로 폴백', e, image_path)
            eps_m = 5.0
            m_lon = meta.m_per_deg_lon
            corners = [
                [meta.gps.lng + sx * eps_m / m_lon,
                 meta.gps.lat + sy * eps_m / M_PER_DEG_LAT]
                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            low_conf = True

        ring = corners + [corners[0]]           # GeoJSON 닫힌 링
        footprint_geojson = json.dumps(
            {'type': 'Polygon', 'coordinates': [ring]})

        meta_dict = meta.to_dict()
        meta_dict.pop('R_cam_to_enu', None)

        return {
            'id': meta.id,
            'task_id': task_id,
            'project_id': project_id,
            'path': str(image_path),
            'filename': Path(str(image_path)).name,
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
            'low_confidence': low_conf,
            'footprint_geojson': footprint_geojson,
            'meta_json': json.dumps(meta_dict, default=str),
        }

    def index_image(self, image_path: str | Path,
                    task_id: Optional[int] = None,
                    project_id: Optional[int] = None,
                    commit: bool = True) -> Optional[str]:
        """이미지 1장 XMP+EXIF 추출 후 upsert. 성공 시 SHA-1 id 반환."""
        row = self._row_from_image(image_path, task_id, project_id)
        print("_UPSERT", _UPSERT)
        print("row", row)
        if row is None:
            return None
        with self.conn.cursor() as cur:
            cur.execute(_UPSERT, row)
            print("cursor", cur)
        if commit:
            self.conn.commit()
        return row['id']

    def index_directory(self, directory: str | Path,
                        project_id: Optional[int] = None,
                        extensions=('.jpg', '.jpeg')) -> dict:
        """폴더 일괄 인덱싱 (단일 트랜잭션)."""
        directory = Path(directory)
        stats = {'indexed': 0, 'skipped': 0, 'errors': []}
        try:
            for p in sorted(directory.iterdir()):
                if p.suffix.lower() not in extensions:
                    continue
                try:
                    if self.index_image(p, project_id=project_id,
                                        commit=False):
                        stats['indexed'] += 1
                    else:
                        stats['skipped'] += 1
                except Exception as e:
                    logger.exception('인덱싱 실패: %s', p)
                    stats['errors'].append(f'{p.name}: {e}')
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return stats

    def index_task_images(self, items: list[tuple[str, int, int]]) -> dict:
        """Label Studio task 연동 일괄 인덱싱.

        Args:
            items: [(image_path, task_id, project_id), ...]
        """
        stats = {'indexed': 0, 'skipped': 0, 'errors': []}
        try:
            for path, task_id, project_id in items:
                try:
                    if self.index_image(path, task_id=task_id,
                                        project_id=project_id, commit=False):
                        stats['indexed'] += 1
                    else:
                        stats['skipped'] += 1
                except Exception as e:
                    logger.exception('인덱싱 실패: task=%s %s', task_id, path)
                    stats['errors'].append(f'task {task_id} ({path}): {e}')
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return stats

    # ------------------------------------------------------------------
    # 검색
    # ------------------------------------------------------------------

    def find_images_containing(
        self,
        lat: float,
        lng: float,
        radius_m: float = 0.0,
        modality: Optional[str] = None,
        project_id: Optional[int] = None,
        rtk_fixed_only: bool = False,
        include_low_confidence: bool = True,
        limit: int = 100,
    ) -> list[ImageHit]:
        """(lat, lng)를 footprint가 포함하거나 radius_m 내로 근접한 이미지.

        radius_m=0  : ST_Covers  (포함만, geometry GiST 히트)
        radius_m>0  : ST_DWithin (geography 미터, 함수 인덱스 히트)
        결과는 footprint까지의 거리[m] 오름차순.
        """
        point = ('ST_SetSRID(ST_MakePoint(%(lng)s, %(lat)s), 4326)')
        params: dict = {'lat': lat, 'lng': lng, 'limit': limit}

        if radius_m > 0:
            where = (f'ST_DWithin(footprint::geography, '
                     f'{point}::geography, %(radius)s)')
            params['radius'] = radius_m
        else:
            where = f'ST_Covers(footprint, {point})'

        filters = [where]
        if modality:
            filters.append('modality = %(modality)s')
            params['modality'] = modality
        if project_id is not None:
            filters.append('project_id = %(project_id)s')
            params['project_id'] = project_id
        if rtk_fixed_only:
            filters.append('rtk_flag = 50')
        if not include_low_confidence:
            filters.append('low_confidence = FALSE')

        sql = f"""
            SELECT id, task_id, project_id, path, filename, modality,
                   capture_time, lat, lng, rtk_flag, low_confidence,
                   ST_AsGeoJSON(footprint) AS footprint_geojson,
                   ST_Distance(footprint::geography,
                               {point}::geography) AS distance_m
            FROM {TABLE_NAME}
            WHERE {' AND '.join(filters)}
            ORDER BY distance_m
            LIMIT %(limit)s
        """
        with self.conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [self._hit_from_row(r) for r in rows]

    def find_images_in_bbox(self, min_lat, min_lng, max_lat, max_lng,
                            modality: Optional[str] = None,
                            project_id: Optional[int] = None,
                            limit: int = 500) -> list[ImageHit]:
        """bbox와 footprint가 교차하는 이미지 (지도 뷰용)."""
        params: dict = {'min_lng': min_lng, 'min_lat': min_lat,
                        'max_lng': max_lng, 'max_lat': max_lat,
                        'limit': limit}
        filters = ['ST_Intersects(footprint, ST_MakeEnvelope('
                   '%(min_lng)s, %(min_lat)s, %(max_lng)s, %(max_lat)s, '
                   '4326))']
        if modality:
            filters.append('modality = %(modality)s')
            params['modality'] = modality
        if project_id is not None:
            filters.append('project_id = %(project_id)s')
            params['project_id'] = project_id

        sql = f"""
            SELECT id, task_id, project_id, path, filename, modality,
                   capture_time, lat, lng, rtk_flag, low_confidence,
                   ST_AsGeoJSON(footprint) AS footprint_geojson,
                   0.0 AS distance_m
            FROM {TABLE_NAME}
            WHERE {' AND '.join(filters)}
            LIMIT %(limit)s
        """
        with self.conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [self._hit_from_row(r) for r in rows]

    @staticmethod
    def _hit_from_row(r) -> ImageHit:
        coords = json.loads(r['footprint_geojson'])['coordinates'][0]
        return ImageHit(
            id=r['id'], task_id=r['task_id'], project_id=r['project_id'],
            path=r['path'], filename=r['filename'],
            modality=r['modality'] or '',
            capture_time=r['capture_time'],
            lat=r['lat'], lng=r['lng'], rtk_flag=r['rtk_flag'],
            distance_m=round(float(r['distance_m']), 2),
            low_confidence=bool(r['low_confidence']),
            footprint=coords)

    def get_metadata(self, image_id: str) -> Optional[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                f'SELECT meta_json FROM {TABLE_NAME} WHERE id = %s',
                (image_id,))
            row = cur.fetchone()
        if row is None:
            return None
        return row[0] if isinstance(row[0], dict) else json.loads(row[0])

    def count(self, project_id: Optional[int] = None) -> int:
        with self.conn.cursor() as cur:
            if project_id is None:
                cur.execute(f'SELECT COUNT(*) FROM {TABLE_NAME}')
            else:
                cur.execute(
                    f'SELECT COUNT(*) FROM {TABLE_NAME} '
                    'WHERE project_id = %s', (project_id,))
            return cur.fetchone()[0]

    def delete_project(self, project_id: int) -> int:
        """프로젝트 재인덱싱 전 정리용."""
        with self.conn.cursor() as cur:
            cur.execute(
                f'DELETE FROM {TABLE_NAME} WHERE project_id = %s',
                (project_id,))
            deleted = cur.rowcount
        self.conn.commit()
        return deleted


# ===========================================================================
# CLI
# ===========================================================================

def _main() -> int:
    ap = argparse.ArgumentParser(
        description='PostGIS GPS 이미지 인덱스 (XMP+EXIF)')
    ap.add_argument('--dsn', required=True,
                    help='예: "dbname=gpsidx user=sayou host=localhost"')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_idx = sub.add_parser('index')
    p_idx.add_argument('image_dir')
    p_idx.add_argument('--project-id', type=int, default=None)

    p_find = sub.add_parser('find')
    p_find.add_argument('lat', type=float)
    p_find.add_argument('lng', type=float)
    p_find.add_argument('--radius', type=float, default=0.0)
    p_find.add_argument('--modality', default=None)
    p_find.add_argument('--json', action='store_true')

    args = ap.parse_args()
    with GpsImageIndexPG(dsn=args.dsn) as idx:
        if args.cmd == 'index':
            stats = idx.index_directory(args.image_dir,
                                        project_id=args.project_id)
            print(f"indexed={stats['indexed']} skipped={stats['skipped']} "
                  f"errors={len(stats['errors'])} (total: {idx.count()})")
            for e in stats['errors']:
                print('  ERR:', e)
        else:
            hits = idx.find_images_containing(
                args.lat, args.lng, radius_m=args.radius,
                modality=args.modality)
            if args.json:
                print(json.dumps([h.to_dict() for h in hits],
                                 ensure_ascii=False, indent=2))
            else:
                for h in hits:
                    task = f' task={h.task_id}' if h.task_id else ''
                    print(f'{h.filename} [{h.modality}]{task} '
                          f'dist={h.distance_m}m rtk={h.rtk_flag}')
                print(f'-- {len(hits)} hits')
    return 0


if __name__ == '__main__':
    sys.exit(_main())

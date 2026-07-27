"""
sayou/gps_image_index.py

이미지 파일의 XMP + EXIF 메타데이터를 추출해 SQLite에 저장하고,
특정 GPS 좌표를 포함(커버)하는 이미지를 찾는 사이드카 인덱스.

Label Studio 연동 포인트:
  - 각 레코드에 task_id / project_id를 함께 저장 → 검색 결과에서
    Label Studio task로 즉시 역참조 가능
  - GeoreferencingData(PostGIS)와 별개로 동작하는 경량 경로:
    PostGIS 미가용 환경, 배치 스크립트, 폐쇄망 도구에서 사용
  - 메타데이터 추출은 sayou.metadata / sayou.dji_metadata_extractor 재사용

검색 방식 (2단계):
  Stage 1 (대략) : footprint bbox 컬럼 + B-tree 인덱스로 후보 축소
  Stage 2 (정밀) : 회전 footprint 폴리곤 point-in-polygon / 최단거리
                   (순수 파이썬 — 의존성 없음)

footprint: 핀홀 코너 광선을 짐벌 yaw/pitch/roll로 ENU 회전 후 평면 지면 교차.
LRF 실측 지면고도가 있으면 우선 사용. near-horizon 광선은 500m clamp 후
low_confidence 플래그. (커버리지 검색용 근사 — defect 정밀 좌표는
기존 georeferencer 파이프라인 사용)

단독 CLI:
    python -m sayou.gps_image_index index <db> <image_dir>
    python -m sayou.gps_image_index find  <db> <lat> <lng> [--radius 10]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# 추출기 임포트 — 배치 위치에 따른 폴백 체인
try:
    from sayou.image.metadata import ImageMetadata, M_PER_DEG_LAT
    from sayou.dji_metadata_extractor import extract, modality_of
except ImportError:                                     # 단독 실행/테스트
    from image.metadata import ImageMetadata, M_PER_DEG_LAT
    from dji_metadata_extractor import extract, modality_of

logger = logging.getLogger(__name__)

MAX_RAY_DISTANCE_M = 500.0     # near-horizon 광선 clamp

_SCHEMA = """
CREATE TABLE IF NOT EXISTS task_georeferencing (
    id              TEXT PRIMARY KEY,       -- 파일 SHA-1 (metadata.py 컨벤션)
    task_id         INTEGER,                -- Label Studio task (없으면 NULL)
    project_id      INTEGER,
    path            TEXT NOT NULL,
    filename        TEXT NOT NULL,
    modality        TEXT,
    capture_time    INTEGER,                -- epoch seconds
    lat REAL, lng REAL, altitude REAL,
    relative_height REAL,
    ground_alt      REAL,
    ground_alt_src  TEXT,                   -- lrf | relative
    gimbal_yaw REAL, gimbal_pitch REAL, gimbal_roll REAL,
    rtk_flag        INTEGER,
    width INTEGER, height INTEGER,
    focal_35mm      INTEGER,
    low_confidence  INTEGER DEFAULT 0,
    footprint_json  TEXT,                   -- [[lng,lat] x 4] (GeoJSON 순서)
    min_lat REAL, max_lat REAL, min_lng REAL, max_lng REAL,
    meta_json       TEXT
);
CREATE INDEX IF NOT EXISTS idx_georeferencing_bbox
    ON task_georeferencing (min_lat, max_lat, min_lng, max_lng);
CREATE INDEX IF NOT EXISTS idx_georeferencing_task ON task_georeferencing (task_id);
CREATE INDEX IF NOT EXISTS idx_georeferencing_project ON task_georeferencing (project_id);
CREATE INDEX IF NOT EXISTS idx_georeferencing_capture ON task_georeferencing (capture_time);
CREATE INDEX IF NOT EXISTS idx_georeferencing_modality ON task_georeferencing (modality);
"""


# ===========================================================================
# footprint 기하 (검증 완료: nadir에서 2h·tan(FOV/2) 정확 일치)
# ===========================================================================

def compute_footprint(meta: ImageMetadata) -> tuple[list, bool]:
    """지면 footprint 4코너 [(lng, lat) x 4]와 low_confidence 플래그.

    카메라 프레임: z=전방(boresight), x=우, y=하
    ENU 프레임:    X=동, Y=북, Z=상
    DJI 짐벌:      yaw=북 기준 시계+, pitch=수평 0/연직 -90
    """
    hfov, vfov = meta.hfov_deg, meta.vfov_deg
    if hfov <= 0 or vfov <= 0:
        raise ValueError(f"FOV 계산 불가 (f35={meta.focal_length_in_35mm})")

    if meta.has_valid_lrf and meta.gps.altitude:
        h = meta.gps.altitude - meta.lrf_target_abs_alt
    else:
        h = meta.relative_height
    if h <= 0:
        raise ValueError(f"카메라 높이 계산 불가 (h={h})")

    yaw = math.radians(meta.gimbal_yaw_deg)
    pitch = math.radians(meta.gimbal_pitch_deg)
    roll = math.radians(meta.gimbal_roll_deg)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy_, cy_ = math.sin(yaw), math.cos(yaw)

    z_ax = (cp * sy_, cp * cy_, sp)          # boresight
    x_ax = (cy_, -sy_, 0.0)                  # 이미지 우
    y_ax = (sp * sy_, sp * cy_, -cp)         # 이미지 하 (= z × x)

    if abs(roll) > 1e-9:
        cr, sr = math.cos(roll), math.sin(roll)
        x_ax, y_ax = (
            tuple(cr * x + sr * y for x, y in zip(x_ax, y_ax)),
            tuple(-sr * x + cr * y for x, y in zip(x_ax, y_ax)))

    th = math.tan(math.radians(hfov / 2.0))
    tv = math.tan(math.radians(vfov / 2.0))

    low_confidence = False
    corners_xy: list[tuple[float, float]] = []

    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):   # 좌상,우상,우하,좌하
        r = tuple(z + th * sx * x + tv * sy * y
                  for z, x, y in zip(z_ax, x_ax, y_ax))
        r_e, r_n, r_z = r

        if r_z >= -1e-6:                     # 지평선 위/근처
            horiz = math.hypot(r_e, r_n) or 1e-9
            scale = MAX_RAY_DISTANCE_M / horiz
            corners_xy.append((r_e * scale, r_n * scale))
            low_confidence = True
            continue

        t = -h / r_z
        gx, gy = t * r_e, t * r_n
        if math.hypot(gx, gy) > MAX_RAY_DISTANCE_M:
            scale = MAX_RAY_DISTANCE_M / math.hypot(gx, gy)
            gx, gy = gx * scale, gy * scale
            low_confidence = True
        corners_xy.append((gx, gy))

    m_lon = M_PER_DEG_LAT * math.cos(math.radians(meta.gps.lat))
    corners = [[meta.gps.lng + x / m_lon, meta.gps.lat + y / M_PER_DEG_LAT]
               for x, y in corners_xy]
    return corners, low_confidence


# ===========================================================================
# 폴리곤 판정 (순수 파이썬)
# ===========================================================================

def _point_in_polygon(px, py, poly) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > py) != (yj > py):
            if px < (xj - xi) * (py - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def _dist_point_segment(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _dist_point_polygon_m(lat, lng, corners_lnglat) -> float:
    """점-폴리곤 거리[m]. 내부면 0."""
    m_lon = M_PER_DEG_LAT * math.cos(math.radians(lat))
    poly = [((c[0] - lng) * m_lon, (c[1] - lat) * M_PER_DEG_LAT)
            for c in corners_lnglat]
    if _point_in_polygon(0.0, 0.0, poly):
        return 0.0
    n = len(poly)
    return min(_dist_point_segment(0.0, 0.0, *poly[i], *poly[(i + 1) % n])
               for i in range(n))


# ===========================================================================
# 검색 결과
# ===========================================================================

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
    rtk_flag: int
    distance_m: float           # 0 = footprint가 점을 포함
    low_confidence: bool
    footprint: list

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ===========================================================================
# 인덱스
# ===========================================================================

class GpsImageIndex:
    """SQLite 기반 GPS 이미지 인덱스.

    스레드 주의: sqlite3 연결은 생성 스레드에서만 사용.
    Django에서 요청마다 새로 열거나(파일 DB는 저렴) with 문 사용 권장.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # 인덱싱
    # ------------------------------------------------------------------

    def index_image(self, image_path: str | Path,
                    task_id: Optional[int] = None,
                    project_id: Optional[int] = None,
                    commit: bool = True) -> Optional[str]:
        """이미지 1장 XMP+EXIF 추출 후 저장. 성공 시 SHA-1 id 반환."""
        meta = extract(image_path)

        if meta.gps.lat == 0.0 or meta.gps.lng == 0.0:
            logger.warning("GPS 없음, 건너뜀: %s", image_path)
            return None

        try:
            corners, low_conf = compute_footprint(meta)
        except ValueError as e:
            logger.warning("footprint 계산 불가(%s): %s — GPS 점만 인덱싱",
                           e, image_path)
            eps_m = 5.0
            m_lon = meta.m_per_deg_lon
            corners = [
                [meta.gps.lng + sx * eps_m / m_lon,
                 meta.gps.lat + sy * eps_m / M_PER_DEG_LAT]
                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            low_conf = True

        lngs = [c[0] for c in corners]
        lats = [c[1] for c in corners]

        ground_alt_src = "lrf" if meta.has_valid_lrf else "relative"
        ground_alt = (meta.lrf_target_abs_alt if meta.has_valid_lrf
                      else meta.gps.altitude - meta.relative_height)

        meta_dict = meta.to_dict()
        meta_dict.pop("R_cam_to_enu", None)

        self.conn.execute(
            """INSERT OR REPLACE INTO images
               (id, task_id, project_id, path, filename, modality,
                capture_time, lat, lng, altitude, relative_height,
                ground_alt, ground_alt_src,
                gimbal_yaw, gimbal_pitch, gimbal_roll,
                rtk_flag, width, height, focal_35mm,
                low_confidence, footprint_json,
                min_lat, max_lat, min_lng, max_lng, meta_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (meta.id, task_id, project_id,
             str(image_path), Path(str(image_path)).name,
             modality_of(meta), meta.capture_time or None,
             meta.gps.lat, meta.gps.lng, meta.gps.altitude,
             meta.relative_height, ground_alt, ground_alt_src,
             meta.gimbal_yaw_deg, meta.gimbal_pitch_deg, meta.gimbal_roll_deg,
             meta.rtk_flag, meta.width, meta.height,
             meta.focal_length_in_35mm,
             int(low_conf), json.dumps(corners),
             min(lats), max(lats), min(lngs), max(lngs),
             json.dumps(meta_dict, default=str)))
        if commit:
            self.conn.commit()
        return meta.id

    def index_directory(self, directory: str | Path,
                        project_id: Optional[int] = None,
                        extensions=(".jpg", ".jpeg")) -> dict:
        """폴더 일괄 인덱싱 (단일 트랜잭션)."""
        directory = Path(directory)
        stats = {"indexed": 0, "skipped": 0, "errors": []}
        for p in sorted(directory.iterdir()):
            if p.suffix.lower() not in extensions:
                continue
            try:
                if self.index_image(p, project_id=project_id, commit=False):
                    stats["indexed"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                logger.exception("인덱싱 실패: %s", p)
                stats["errors"].append(f"{p.name}: {e}")
        self.conn.commit()
        return stats

    def index_task_images(self, items: list[tuple[str, int, int]]) -> dict:
        """Label Studio task 연동 일괄 인덱싱.

        Args:
            items: [(image_path, task_id, project_id), ...]
        """
        stats = {"indexed": 0, "skipped": 0, "errors": []}
        for path, task_id, project_id in items:
            try:
                if self.index_image(path, task_id=task_id,
                                    project_id=project_id, commit=False):
                    stats["indexed"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                logger.exception("인덱싱 실패: task=%s %s", task_id, path)
                stats["errors"].append(f"task {task_id} ({path}): {e}")
        self.conn.commit()
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

        Stage 1: bbox 인덱스 쿼리 / Stage 2: 폴리곤 정밀 판정. 거리순 정렬.
        """
        m_lon = M_PER_DEG_LAT * math.cos(math.radians(lat))
        pad_lat = radius_m / M_PER_DEG_LAT
        pad_lng = radius_m / m_lon

        sql = """SELECT * FROM images
                 WHERE min_lat <= ? AND max_lat >= ?
                   AND min_lng <= ? AND max_lng >= ?"""
        params: list = [lat + pad_lat, lat - pad_lat,
                        lng + pad_lng, lng - pad_lng]
        if modality:
            sql += " AND modality = ?"
            params.append(modality)
        if project_id is not None:
            sql += " AND project_id = ?"
            params.append(project_id)
        if rtk_fixed_only:
            sql += " AND rtk_flag = 50"
        if not include_low_confidence:
            sql += " AND low_confidence = 0"

        hits: list[ImageHit] = []
        for row in self.conn.execute(sql, params):
            corners = json.loads(row["footprint_json"])
            dist = _dist_point_polygon_m(lat, lng, corners)
            if dist <= radius_m:
                hits.append(ImageHit(
                    id=row["id"], task_id=row["task_id"],
                    project_id=row["project_id"],
                    path=row["path"], filename=row["filename"],
                    modality=row["modality"],
                    capture_time=row["capture_time"],
                    lat=row["lat"], lng=row["lng"],
                    rtk_flag=row["rtk_flag"],
                    distance_m=round(dist, 2),
                    low_confidence=bool(row["low_confidence"]),
                    footprint=corners))

        hits.sort(key=lambda h: h.distance_m)
        return hits[:limit]

    def find_images_in_bbox(self, min_lat, min_lng, max_lat, max_lng,
                            modality: Optional[str] = None,
                            project_id: Optional[int] = None,
                            limit: int = 500) -> list[ImageHit]:
        """bbox 교차 이미지 (Stage 1만 — 지도 뷰용)."""
        sql = """SELECT * FROM images
                 WHERE min_lat <= ? AND max_lat >= ?
                   AND min_lng <= ? AND max_lng >= ?"""
        params: list = [max_lat, min_lat, max_lng, min_lng]
        if modality:
            sql += " AND modality = ?"
            params.append(modality)
        if project_id is not None:
            sql += " AND project_id = ?"
            params.append(project_id)
        sql += " LIMIT ?"
        params.append(limit)

        return [ImageHit(
            id=r["id"], task_id=r["task_id"], project_id=r["project_id"],
            path=r["path"], filename=r["filename"], modality=r["modality"],
            capture_time=r["capture_time"], lat=r["lat"], lng=r["lng"],
            rtk_flag=r["rtk_flag"], distance_m=0.0,
            low_confidence=bool(r["low_confidence"]),
            footprint=json.loads(r["footprint_json"]))
            for r in self.conn.execute(sql, params)]

    def get_metadata(self, image_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT meta_json FROM images WHERE id = ?", (image_id,)).fetchone()
        return json.loads(row["meta_json"]) if row else None

    def count(self, project_id: Optional[int] = None) -> int:
        if project_id is None:
            return self.conn.execute(
                "SELECT COUNT(*) FROM images").fetchone()[0]
        return self.conn.execute(
            "SELECT COUNT(*) FROM images WHERE project_id = ?",
            (project_id,)).fetchone()[0]


# ===========================================================================
# CLI
# ===========================================================================

def _main() -> int:
    ap = argparse.ArgumentParser(description="XMP+EXIF SQLite GPS 이미지 인덱스")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index")
    p_idx.add_argument("db")
    p_idx.add_argument("image_dir")

    p_find = sub.add_parser("find")
    p_find.add_argument("db")
    p_find.add_argument("lat", type=float)
    p_find.add_argument("lng", type=float)
    p_find.add_argument("--radius", type=float, default=0.0)
    p_find.add_argument("--modality", default=None)
    p_find.add_argument("--json", action="store_true")

    args = ap.parse_args()
    with GpsImageIndex(args.db) as idx:
        if args.cmd == "index":
            stats = idx.index_directory(args.image_dir)
            print(f"indexed={stats['indexed']} skipped={stats['skipped']} "
                  f"errors={len(stats['errors'])} (total: {idx.count()})")
            for e in stats["errors"]:
                print("  ERR:", e)
        else:
            hits = idx.find_images_containing(
                args.lat, args.lng, radius_m=args.radius,
                modality=args.modality)
            if args.json:
                print(json.dumps([h.to_dict() for h in hits],
                                 ensure_ascii=False, indent=2))
            else:
                for h in hits:
                    task = f" task={h.task_id}" if h.task_id else ""
                    print(f"{h.filename} [{h.modality}]{task} "
                          f"dist={h.distance_m}m rtk={h.rtk_flag}")
                print(f"-- {len(hits)} hits")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
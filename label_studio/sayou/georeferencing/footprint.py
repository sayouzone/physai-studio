"""
sayou/georeferencing/footprint.py

이미지 footprint(지상 커버리지 4코너)를 **수정된 파이프라인과 동일한
카메라 모델·투영 평면**으로 생성한다.

2026-07 (4차) — 프레임 정합 보정 배선
────────────────────────────────────
cross_frame_residuals 로 측정한 프레임별 위치 오프셋(alignment.py)을
footprint 생성에도 적용한다. 검출 폴리곤 쪽(annotation_georeferencer)에만
넣으면 footprint 와 검출이 다시 다른 좌표계가 되어, 지난번 레거시
FOV 불일치 때와 같은 실패가 재발한다 — 반드시 둘 다 같이 보정한다.

task_id 는 정합 조회의 키다. 없으면(신규/미인제스트 이미지) 보정이
비활성(no-op)으로 자동 전환되므로 안전하다.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .dji.camera_pose import compute_camera_axes_from_gimbal
from .dji.georeferencer import DJIImageGeoreferencer, resolve_ground_plane
from .alignment import get_alignment

logger = logging.getLogger(__name__)


def image_pixel_corners(width: int, height: int) -> List[List[float]]:
    """좌상 → 우상 → 우하 → 좌하. 프론트 dst 순서와 1:1 대응."""
    return [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]


def build_footprint(
    metadata,
    task_id: Optional[int] = None,
    **plane_kwargs,
) -> Optional[Dict[str, Any]]:
    """
    이미지 한 장의 footprint 페이로드를 만든다.

    Parameters
    ----------
    task_id : 이 이미지가 속한 Task ID. 프레임 정합 보정(alignment.py)의
              조회 키다. None 이면 보정을 적용하지 않는다(신규 이미지 등).
    plane_kwargs : 검출 폴리곤 생성 때와 **동일한 값**을 넘겨야 한다
              (panel_height_above_ground 등). 다르면 두 좌표계가 어긋난다.

    Returns
    -------
    {
      "footprint"     : GeoJSON Polygon (닫힌 링, 정합 보정 적용됨),
      "pixel_corners" : [[0,0],[W,0],[W,H],[0,H]],
      "image_size"    : [W, H],
      "ground_plane"  : {"altitude_m","agl_m","gsd_m_per_px","source"},
      "alignment"     : {"applied","source","east_m","north_m"} — 진단용.
                         프론트/운영이 이 이미지에 정합 보정이 실제로
                         적용됐는지 여기서 바로 확인한다.
      "extent_m"      : {"width_m","height_m"},
    }
    실패 시 None.
    """
    if getattr(metadata, "R_cam_to_enu", None) is None:
        axes = compute_camera_axes_from_gimbal(
            gimbal_yaw_compass_deg=metadata.gimbal_yaw_deg,
            gimbal_pitch_deg=metadata.gimbal_pitch_deg,
            gimbal_roll_deg=metadata.gimbal_roll_deg,
        )
        metadata.R_cam_to_enu = axes["R_camera_to_enu"]

    plane_alt, plane_source = resolve_ground_plane(metadata, **plane_kwargs)

    try:
        gr = DJIImageGeoreferencer(metadata, ground_altitude=plane_alt)
    except ValueError as e:
        logger.error("[footprint] georeferencer 생성 실패: %s (image=%s)", e, metadata.origin_path)
        return None

    W, H = metadata.width, metadata.height
    corners_px = image_pixel_corners(W, H)

    ring: List[List[float]] = []
    for px, py in corners_px:
        geo = gr.pixel_to_geodetic((px, py))
        if geo is None:
            logger.error(
                "[footprint] 코너 (%.1f, %.1f) 역투영 실패 — footprint 생성 중단 (image=%s)",
                px, py, metadata.origin_path,
            )
            # 여기서 (0,0) 같은 더미를 채우면 프론트 호모그래피가 조용히 망가진다.
            return None
        ring.append([geo.longitude, geo.latitude])
    ring.append(ring[0])  # 닫기

    # ── 프레임 정합 보정 ────────────────────────────────────────────
    # 검출 폴리곤(annotation_georeferencer._enu_to_lonlat)과 반드시 같은
    # 보정을 받아야 한다. 여기서 빠지면 gps_ref 오버레이가 다시 어긋난다.
    alignment = get_alignment()
    ring = alignment.shift_ring(ring, task_id, metadata.m_per_deg_lon, metadata.m_per_deg_lat)
    alignment_info = alignment.describe(task_id)

    agl = gr.agl_m
    gsd = gr.nominal_gsd_m

    return {
        "footprint": {"type": "Polygon", "coordinates": [ring]},
        "pixel_corners": corners_px,
        "image_size": [W, H],
        "ground_plane": {
            "altitude_m": round(plane_alt, 3),
            "agl_m": round(agl, 3),
            "gsd_m_per_px": round(gsd, 6),
            "source": plane_source,
        },
        "alignment": alignment_info,
        "extent_m": {
            "width_m": round(gsd * W, 3),
            "height_m": round(gsd * H, 3),
        },
    }


def build_footprint_payload(
    rgb_metadata=None,
    ir_metadata=None,
    task_id: Optional[int] = None,
    **plane_kwargs,
) -> Dict[str, Any]:
    """
    엔드포인트 응답 페이로드. RGB / IR 을 한 번에 만든다.

    RGB / IR 이 같은 task_id 를 공유하므로 정합 보정도 동일하게 적용된다.
    (오프셋은 카메라 모델과 무관한 '위치' 보정이라 센서별로 다를 이유가 없다.)
    """
    out: Dict[str, Any] = {
        "footprint_rgb": None, "pixel_corners_rgb": None, "image_size_rgb": None,
        "footprint_ir": None, "pixel_corners_ir": None, "image_size_ir": None,
        "ground_plane": None,
        "alignment": None,
        "extent_m_rgb": None, "extent_m_ir": None,
    }

    planes = []

    if rgb_metadata is not None:
        r = build_footprint(rgb_metadata, task_id=task_id, **plane_kwargs)
        if r:
            out["footprint_rgb"] = r["footprint"]
            out["pixel_corners_rgb"] = r["pixel_corners"]
            out["image_size_rgb"] = r["image_size"]
            out["extent_m_rgb"] = r["extent_m"]
            out["ground_plane"] = r["ground_plane"]
            out["alignment"] = r["alignment"]
            planes.append(("rgb", r["ground_plane"]["altitude_m"]))

    if ir_metadata is not None:
        r = build_footprint(ir_metadata, task_id=task_id, **plane_kwargs)
        if r:
            out["footprint_ir"] = r["footprint"]
            out["pixel_corners_ir"] = r["pixel_corners"]
            out["image_size_ir"] = r["image_size"]
            out["extent_m_ir"] = r["extent_m"]
            if out["ground_plane"] is None:
                out["ground_plane"] = r["ground_plane"]
            if out["alignment"] is None:
                out["alignment"] = r["alignment"]
            planes.append(("ir", r["ground_plane"]["altitude_m"]))

    if len(planes) == 2 and abs(planes[0][1] - planes[1][1]) > 1e-6:
        logger.warning(
            "[footprint] RGB/IR 투영 평면 불일치: %s. 두 센서 오버레이가 어긋납니다.",
            planes,
        )

    return out


def compare_footprints(
    stored_ring: Sequence[Sequence[float]],
    metadata,
    tolerance_pct: float = 1.0,
    task_id: Optional[int] = None,
    **plane_kwargs,
) -> Dict[str, Any]:
    """
    저장된 footprint 가 현재(수정된) 파이프라인 산출물인지 판정한다.

    주의: task_id 를 넘기면 비교 기준(fresh)에 정합 보정이 들어간다.
    저장값이 보정 이전 것이면 물리 치수 비교(스케일)에는 영향이 없지만
    (보정은 평행이동이라 치수를 바꾸지 않는다) 위치 자체는 어긋난 채로
    비교된다. 순수 스케일 판정 목적이면 task_id 를 생략해도 무방하다.
    """
    import math

    fresh = build_footprint(metadata, task_id=task_id, **plane_kwargs)
    if fresh is None:
        return {"stale": None, "reason": "현재 파이프라인 footprint 생성 실패"}

    ring = list(stored_ring)[:4]
    lat0 = sum(c[1] for c in ring) / 4
    m_lat = metadata.m_per_deg_lat
    m_lon = metadata.m_per_deg_lon

    def dist(a, b):
        return math.hypot((b[0] - a[0]) * m_lon, (b[1] - a[1]) * m_lat)

    stored_w = (dist(ring[0], ring[1]) + dist(ring[3], ring[2])) / 2
    stored_h = (dist(ring[0], ring[3]) + dist(ring[1], ring[2])) / 2

    exp_w = fresh["extent_m"]["width_m"]
    exp_h = fresh["extent_m"]["height_m"]

    rx = stored_w / exp_w if exp_w else float("nan")
    ry = stored_h / exp_h if exp_h else float("nan")
    stale = (
        abs(rx - 1.0) * 100 > tolerance_pct or abs(ry - 1.0) * 100 > tolerance_pct
    )

    return {
        "stale": bool(stale),
        "stored_m": [round(stored_w, 3), round(stored_h, 3)],
        "expected_m": [round(exp_w, 3), round(exp_h, 3)],
        "ratio": [round(rx, 4), round(ry, 4)],
        "ground_plane": fresh["ground_plane"],
        "alignment": fresh["alignment"],
    }
"""
src/georeferencing/dji/georeferencer.py
DJI 이미지 georeferencing 메인 클래스

수정 이력
─────────
2026-07 (1차)
- LRF 지면고도 판정 버그 수정 (_to_float 기본값 0.0 → `is not None` 이 항상 True).
- pixel_to_camera_ray 왜곡보정 경로의 cv2 미import.
- yolo_bbox_to_geodetic 의 image_width/height 오타.
- K, D 를 등방 초점거리/DewarpData 기반으로 개선.
- 레거시 DJIGeoreferencer 클래스 제거.

2026-07 (2차) — 투영 평면 정책
- 검출 대상은 '지면'이 아니라 '패널 상면'이므로 광선-평면 교차의 평면을
  패널 상면에 두어야 (a) 절대 스케일이 맞고 (b) 기복변위(d = h·r/H)가 사라진다.
- LRF 는 기본 비사용. nadir 촬영에서 조준점이 프레임마다 행간 지면 /
  패널 상면을 번갈아 맞아 GSD 가 수 % 흔들린다.

2026-07 (3차) — **사이트 이식성 수정**
- 2차에서 도입한 SITE_PANEL_PLANE_ALT(절대고도)는 지형 표고에 종속되어
  다른 사이트/비행에서 반드시 깨진다.
  (실제 발생: plane=114.87 인데 drone=73.892 → ValueError)
- 이식 가능한 값은 절대고도가 아니라 **지면 대비 패널 상면 높이**다.
  SITE_PANEL_HEIGHT_ABOVE_GROUND 를 1순위 정책으로 삼고,
  평면 = (abs_alt - relative_height) + 패널높이 로 매 이미지마다 계산한다.
- 절대고도 지정(SITE_PANEL_PLANE_ALT)은 실측 표고가 있을 때의 override 로만
  남기되, 현재 이미지의 지면 추정과 크게 어긋나면(다른 사이트로 판단)
  **크래시 대신 경고 후 무시**한다. 다중 사이트 플랫폼에서 한 장의 잘못된
  설정이 엔드포인트 전체를 죽이면 안 된다.
- 평면 결정 로직을 resolve_ground_plane() 하나로 통합.
  annotation_georeferencer 도 이 함수를 호출해야 한다 — 두 곳에 같은 정책을
  복제하면 반드시 갈라진다(2차에서 실제로 그랬다).
"""

import logging

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import sys

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sayou.image.metadata import ImageMetadata, extract_metadata, estimate_intrinsics_from_metadata

from .metadata import (
    #DJIMetadata,
    CameraIntrinsics,
    parse_dji_metadata,
    estimate_zh20t_zoom_intrinsics,
    #estimate_intrinsics_from_metadata,
    dji_gimbal_to_camera_rotation, estimate_zh20t_thermal_intrinsics
)
from .coordinates import (
    ENUPoint,
    GeodeticPoint,
    enu_to_geodetic,
    geodetic_to_enu,
)
from .camera_pose import compute_camera_axes_from_gimbal, verify_nadir_orientation

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 투영 평면 설정
# ─────────────────────────────────────────────────────────────────────────────
# [권장] 이륙지점 지면 대비 패널 상면 높이 (m).
#   구조물 기하이므로 지형 표고와 무관하다 → 사이트가 바뀌어도 그대로 쓸 수 있다.
#   평면 = (abs_alt - relative_height) + 이 값
#
#   캘리브레이션:
#       AGL_true    = 모듈_실제폭_m * metadata.focal_px / 모듈_평균픽셀폭
#       패널높이     = metadata.relative_height - AGL_true
#   (calibrate_panel_height() 헬퍼 참고)
#
#   주의: relative_height 는 '이륙지점' 기준이다. 패널 부지와 표고가 다른 곳
#   (도로, 둑 등)에서 이륙했다면 그 차이가 이 값에 섞여 들어간다.
SITE_PANEL_HEIGHT_ABOVE_GROUND: Optional[float] = 1.48

# [선택] 특정 사이트의 실측 패널면 절대고도 (m).
#   측량값이 있을 때만 쓴다. 지형 표고에 종속되므로 전역 상수로 두면
#   다른 사이트에서 깨진다. 프로젝트/Task 단위로 주입하는 것을 권장.
SITE_PANEL_PLANE_ALT: Optional[float] = None

# 절대 평면고도가 현재 이미지와 같은 사이트인지 판정하는 허용 범위 (m).
#   |SITE_PANEL_PLANE_ALT - (abs_alt - relative_height)| 가 이 값을 넘으면
#   다른 사이트로 보고 무시한다.
SITE_PLANE_SANITY_RANGE_M: float = 30.0


class GroundPlaneSource(str):
    """지면 평면 출처 태그 (디버깅/GeoJSON 기록용)"""
    EXPLICIT = "explicit_ground_altitude"
    SITE_PLANE = "site_panel_plane_abs"
    PANEL_OFFSET = "takeoff_ground_plus_panel_height"
    LRF = "lrf_target_abs_alt"
    REL_HEIGHT = "abs_alt_minus_relative_height"


def resolve_ground_plane(
    metadata: ImageMetadata,
    ground_altitude: Optional[float] = None,
    panel_plane_altitude: Optional[float] = None,
    panel_height_above_ground: Optional[float] = None,
    prefer_lrf: bool = False,
) -> Tuple[float, str]:
    """
    투영 평면의 절대고도와 그 출처를 결정한다. **평면 정책의 단일 진입점.**

    RGB 경로(annotation_georeferencer)와 IR 경로(DJIImageGeoreferencer)가
    반드시 이 함수 하나만 호출해야 두 센서가 같은 평면을 공유한다.

    우선순위
    --------
    1. ground_altitude          : 호출자가 직접 지정 (최우선)
    2. panel_plane_altitude / SITE_PANEL_PLANE_ALT
                                : 절대고도. 단, 현재 이미지의 지면 추정과
                                  SITE_PLANE_SANITY_RANGE_M 이내일 때만.
                                  어긋나면 경고 후 다음 단계로 넘어간다.
    3. panel_height_above_ground / SITE_PANEL_HEIGHT_ABOVE_GROUND
                                : (abs_alt - relative_height) + 패널높이.
                                  ← 권장 경로. 사이트 표고와 무관.
    4. prefer_lrf 이고 LRF 유효  : LRF 조준점 절대고도
    5. 폴백                      : abs_alt - relative_height (지면)

    Returns
    -------
    (plane_altitude_m, source_tag)
    """
    abs_alt = metadata.gps.altitude
    rel_h = metadata.relative_height
    takeoff_ground = abs_alt - rel_h

    # 1) 명시값
    if ground_altitude is not None:
        return float(ground_altitude), GroundPlaneSource.EXPLICIT

    # 2) 절대고도 지정 (같은 사이트일 때만)
    site_abs = (
        panel_plane_altitude
        if panel_plane_altitude is not None
        else SITE_PANEL_PLANE_ALT
    )
    if site_abs is not None:
        gap = abs(float(site_abs) - takeoff_ground)
        if gap <= SITE_PLANE_SANITY_RANGE_M:
            return float(site_abs), GroundPlaneSource.SITE_PLANE
        logger.error(
            "[resolve_ground_plane] 지정된 패널면 절대고도 %.2f m 가 이 이미지의 "
            "지면 추정 %.2f m 와 %.1f m 차이납니다 (허용 %.1f m). 다른 사이트/비행의 "
            "설정으로 판단해 무시하고 패널높이 오프셋으로 대체합니다. "
            "사이트별 설정이면 프로젝트/Task 단위로 주입하세요. "
            "(image=%s, abs_alt=%.3f, rel_height=%.3f)",
            float(site_abs), takeoff_ground, gap, SITE_PLANE_SANITY_RANGE_M,
            metadata.origin_path, abs_alt, rel_h,
        )

    # 3) 패널높이 오프셋 (권장 경로, 사이트 이식 가능)
    panel_h = (
        panel_height_above_ground
        if panel_height_above_ground is not None
        else SITE_PANEL_HEIGHT_ABOVE_GROUND
    )
    if panel_h is not None:
        if rel_h <= 0:
            logger.warning(
                "[resolve_ground_plane] relative_height=%.3f 가 유효하지 않아 "
                "패널높이 오프셋을 적용할 수 없습니다. LRF/폴백으로 진행합니다. "
                "(image=%s)", rel_h, metadata.origin_path,
            )
        elif panel_h >= rel_h:
            logger.error(
                "[resolve_ground_plane] 패널높이 %.2f m 가 대지고도 %.2f m 이상입니다. "
                "설정값을 확인하세요. 폴백으로 진행합니다. (image=%s)",
                panel_h, rel_h, metadata.origin_path,
            )
        else:
            return float(takeoff_ground + panel_h), GroundPlaneSource.PANEL_OFFSET

    # 4) LRF (옵션)
    if prefer_lrf and metadata.has_valid_lrf:
        logger.warning(
            "[resolve_ground_plane] LRF 기반 평면 사용 (abs_alt=%.3f). 조준점이 "
            "행간 지면/패널 상면 중 무엇을 맞았는지에 따라 프레임 간 스케일이 "
            "흔들립니다.", metadata.lrf_target_abs_alt,
        )
        return float(metadata.lrf_target_abs_alt), GroundPlaneSource.LRF

    # 5) 폴백
    logger.info(
        "[resolve_ground_plane] 패널면 평면 미설정 → relative_height 폴백 "
        "(AGL=%.3f). 검출 대상이 패널 상면이면 스케일이 과대평가됩니다.",
        rel_h,
    )
    return float(takeoff_ground), GroundPlaneSource.REL_HEIGHT


def calibrate_panel_height(
    metadata: ImageMetadata,
    known_size_m: float,
    measured_size_px: float,
) -> dict:
    """
    알려진 물리 치수(예: PV 모듈 폭)로부터 **패널 상면 높이**를 역산한다.
    사이트당 1회. 결과의 panel_height_m 을 SITE_PANEL_HEIGHT_ABOVE_GROUND 에 넣는다.

    절대고도(plane_altitude_m)도 함께 반환하지만, 그 값은 이 사이트에서만
    유효하다는 점에 주의. 사이트 간 이식에는 panel_height_m 을 쓸 것.

    Parameters
    ----------
    known_size_m     : 객체의 실제 치수 (m). 예: 모듈 폭 0.992
    measured_size_px : 같은 축 방향의 평균 픽셀 치수

    사용 예
    -------
        m = extract_metadata(path)
        calibrate_panel_height(m, 0.992, 160.4)
        # -> {'agl_m': 43.53, 'panel_height_m': 1.48, 'plane_altitude_m': 114.87, ...}
    """
    if measured_size_px <= 0:
        raise ValueError("measured_size_px 는 0보다 커야 합니다")

    agl = known_size_m * metadata.focal_px / measured_size_px
    plane_alt = metadata.gps.altitude - agl

    out = {
        "agl_m": float(agl),
        "panel_height_m": float(metadata.relative_height - agl),
        "plane_altitude_m": float(plane_alt),
        "gsd_m_per_px": float(agl / metadata.focal_px),
        "takeoff_ground_alt_m": float(metadata.gps.altitude - metadata.relative_height),
        "panel_height_over_lrf_m": None,
    }
    if metadata.has_valid_lrf:
        out["panel_height_over_lrf_m"] = float(plane_alt - metadata.lrf_target_abs_alt)
    return out


# 하위 호환 별칭 (2차 수정에서 쓰던 이름)
calibrate_plane_altitude = calibrate_panel_height


@dataclass
class GeoreferencingResult:
    """결과 묶음"""
    image_corners_geo: List[GeodeticPoint]
    ground_sample_distance_m: float
    coverage_area_m2: float
    nadir_check: Optional[dict] = None
    ground_height_used: float = 0.
    coverage_width_m: float = 0.
    coverage_height_m: float = 0.
    ground_plane_source: str = ""


class DJIImageGeoreferencer:
    """
    DJI 드론 사진 한 장에 대한 georeferencing

    핵심 알고리즘:
    1. 픽셀 좌표 → 카메라 좌표계 광선 (K^-1)
    2. 카메라 광선 → ENU(월드) 광선 (R_camera_to_enu)
    3. 광선 - 지면 평면 교차 → ENU 좌표
    4. ENU → WGS84 (위도, 경도, 고도)
    """

    def __init__(
        self,
        metadata: ImageMetadata,
        K: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
        ground_altitude: Optional[float] = None,
        panel_plane_altitude: Optional[float] = None,
        panel_height_above_ground: Optional[float] = None,
        prefer_lrf: bool = False,
    ):
        """
        평면 관련 인자는 모두 resolve_ground_plane() 으로 위임된다.
        상세 우선순위는 그 함수의 docstring 참고.
        """
        self.metadata = metadata

        # 내부 파라미터: DewarpData(공장 캘리브레이션)가 있으면 그 값을,
        # 없으면 등방 초점거리(대각선 기준) 기반으로 추정.
        if K is None or D is None:
            self.K, self.D = estimate_intrinsics_from_metadata(metadata)
        else:
            self.K, self.D = K, D

        self.origin = GeodeticPoint(
            latitude=metadata.gps.lat,
            longitude=metadata.gps.lng,
            altitude=metadata.gps.altitude
        )

        self.camera_position_enu = np.array([0.0, 0.0, 0.0])

        # ── 투영 평면 (단일 진입점) ────────────────────────────────
        self.ground_altitude, self.ground_plane_source = resolve_ground_plane(
            metadata,
            ground_altitude=ground_altitude,
            panel_plane_altitude=panel_plane_altitude,
            panel_height_above_ground=panel_height_above_ground,
            prefer_lrf=prefer_lrf,
        )

        # ENU 원점(=드론)에서 본 지면 평면의 up 좌표 (보통 음수: 아래)
        self.ground_up_in_enu = self.ground_altitude - metadata.gps.altitude

        if self.ground_up_in_enu >= 0:
            # 여기까지 왔다면 명시 지정(ground_altitude)이 잘못된 경우가 대부분이다.
            raise ValueError(
                f"투영 평면이 드론보다 높습니다 "
                f"(plane={self.ground_altitude}, drone={metadata.gps.altitude}, "
                f"rel_height={metadata.relative_height}, "
                f"source={self.ground_plane_source}, image={metadata.origin_path}). "
                f"평면 고도를 절대값으로 지정했다면 해당 사이트의 값인지 확인하세요. "
                f"사이트 간 이식에는 SITE_PANEL_HEIGHT_ABOVE_GROUND(지면 대비 "
                f"패널 높이)를 사용하세요."
            )

        axes = compute_camera_axes_from_gimbal(
            gimbal_yaw_compass_deg=metadata.orientation[0],
            gimbal_pitch_deg=metadata.orientation[1],
            gimbal_roll_deg=metadata.orientation[2]
        )
        self.R_camera_to_enu = axes["R_camera_to_enu"]
        self.optical_axis_enu = axes["axis_z_enu"]

        self._cache = {}

    # ----- 편의 프로퍼티 -----
    @property
    def agl_m(self) -> float:
        """투영 평면까지의 대지고도 (m)"""
        return -self.ground_up_in_enu

    @property
    def nominal_gsd_m(self) -> float:
        """공칭 GSD (m/px). 등방 초점거리 기준, nadir 가정."""
        return self.agl_m / self.metadata.focal_px

    def pixel_to_camera_ray(self, pixel: Tuple[float, float]) -> np.ndarray:
        """픽셀 → 카메라 좌표계 광선 (정규화된 단위 벡터)"""
        u, v = pixel

        if np.any(self.D != 0):
            # DewarpData 등 렌즈 왜곡계수가 있을 때만 실행되는 경로.
            # cv2 를 상단에서 항상 import 하면 폐쇄망/최소 환경에서 모듈 로딩이
            # 깨질 수 있으므로 필요한 시점에만 지연 import.
            import cv2
            pts = np.array([[[u, v]]], dtype=np.float32)
            undistorted = cv2.undistortPoints(pts, self.K, self.D)
            x, y = undistorted[0, 0]
        else:
            K_inv = np.linalg.inv(self.K)
            homogeneous = np.array([u, v, 1.0])
            normalized = K_inv @ homogeneous
            x, y = normalized[0], normalized[1]

        ray = np.array([x, y, 1.0])
        return ray / np.linalg.norm(ray)

    def pixel_to_enu(self, pixel: Tuple[float, float]) -> Optional[np.ndarray]:
        """픽셀 → ENU 좌표 (광선-평면 교차)"""
        ray_camera = self.pixel_to_camera_ray(pixel)

        ray_enu = self.R_camera_to_enu @ ray_camera
        ray_enu = ray_enu / np.linalg.norm(ray_enu)

        plane_d = self.ground_up_in_enu

        denom = ray_enu[2]
        if abs(denom) < 1e-9:
            return None

        t = (plane_d - self.camera_position_enu[2]) / denom

        if t < 0:
            return None

        intersection = self.camera_position_enu + t * ray_enu
        return intersection

    def pixel_to_geodetic(self, pixel: Tuple[float, float]) -> Optional[GeodeticPoint]:
        """픽셀 → 위도, 경도, 고도"""
        enu_array = self.pixel_to_enu(pixel)
        if enu_array is None:
            return None

        enu = ENUPoint(east=enu_array[0], north=enu_array[1], up=enu_array[2])
        return enu_to_geodetic(enu, self.origin)

    def bbox_to_geodetic(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[List[GeodeticPoint]]:
        """박스 (x1, y1, x2, y2) → 4개 모서리 지리 좌표"""
        x1, y1, x2, y2 = bbox
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        result = []
        for px in corners:
            geo = self.pixel_to_geodetic(px)
            if geo is None:
                return None
            result.append(geo)

        return result

    def bbox_center_to_geodetic(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[GeodeticPoint]:
        x1, y1, x2, y2 = bbox
        return self.pixel_to_geodetic(((x1 + x2) / 2, (y1 + y2) / 2))

    def yolo_bbox_to_geodetic(
        self,
        cx_norm: float,
        cy_norm: float,
        w_norm: float,
        h_norm: float
    ) -> Optional[GeodeticPoint]:
        """YOLO 정규화 박스 → 박스 중심의 지리 좌표"""
        cx_px = cx_norm * self.metadata.width
        cy_px = cy_norm * self.metadata.height
        return self.pixel_to_geodetic((cx_px, cy_px))

    def compute_image_corners(self) -> List[GeodeticPoint]:
        if "corners" in self._cache:
            return self._cache["corners"]

        w = self.metadata.width
        h = self.metadata.height
        corners_pixel = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]

        result = []
        for px in corners_pixel:
            geo = self.pixel_to_geodetic(px)
            if geo is None:
                geo = GeodeticPoint(0, 0, 0)
            result.append(geo)

        self._cache["corners"] = result
        return result

    def compute_ground_sample_distance(self) -> float:
        """이미지 중심에서 1픽셀이 지상에서 차지하는 거리 (m)"""
        cx = self.metadata.width // 2
        cy = self.metadata.height // 2

        p1 = self.pixel_to_enu((cx, cy))
        p2 = self.pixel_to_enu((cx + 1, cy))
        p3 = self.pixel_to_enu((cx, cy + 1))

        if p1 is None or p2 is None or p3 is None:
            return float("nan")

        gsd_x = np.linalg.norm(p2 - p1)
        gsd_y = np.linalg.norm(p3 - p1)
        return (gsd_x + gsd_y) / 2

    def compute_coverage_area(self) -> float:
        """이미지가 지상에서 커버하는 면적 (m²)"""
        corners = self.compute_image_corners()
        enu_corners = [
            geodetic_to_enu(c, self.origin).to_array()[:2] for c in corners
        ]

        x1, y1 = enu_corners[0]
        x2, y2 = enu_corners[1]
        x3, y3 = enu_corners[2]
        x4, y4 = enu_corners[3]

        return 0.5 * abs(
            (x1 * y2 - x2 * y1) +
            (x2 * y3 - x3 * y2) +
            (x3 * y4 - x4 * y3) +
            (x4 * y1 - x1 * y4)
        )

    def georeference_full_image(self) -> GeoreferencingResult:
        """이미지 전체에 대한 georeferencing 결과 종합"""
        corners = self.compute_image_corners()
        enu_corners = [
            geodetic_to_enu(c, self.origin).to_array()[:2] for c in corners
        ]
        width_m = float(np.linalg.norm(enu_corners[1] - enu_corners[0]))
        height_m = float(np.linalg.norm(enu_corners[3] - enu_corners[0]))

        return GeoreferencingResult(
            image_corners_geo=corners,
            ground_sample_distance_m=self.compute_ground_sample_distance(),
            coverage_area_m2=self.compute_coverage_area(),
            nadir_check=verify_nadir_orientation(self.R_camera_to_enu),
            ground_height_used=self.ground_altitude,
            coverage_width_m=width_m,
            coverage_height_m=height_m,
            ground_plane_source=self.ground_plane_source,
        )

    def relief_displacement_m(self, object_height_m: float, radius_m: float) -> float:
        """
        투영 평면보다 object_height_m 만큼 높은 지점이,
        nadir 로부터 radius_m 떨어진 위치에서 겪는 방사형 변위.

            d = h * r / H

        평면을 패널 상면에 맞추면 패널에 대해서는 h≈0 이 되어 사라진다. (진단용)
        """
        return object_height_m * radius_m / self.agl_m

    def validate_with_lrf(self) -> Optional[dict]:
        """
        LRF(Laser Range Finder)로 측정한 타깃 좌표와 비교 검증

        DJI XMP의 LRFTarget* 필드는 이미지 중심에서 측정한 실제 거리/위치.

        주의: LRF 조준점은 패널 상면일 수도, 행간 지면일 수도 있다.
        vertical_gap_m 이 +1~2 m 면 LRF 가 행간 지면을, 0 근처면 패널 상면을
        맞은 것으로 볼 수 있다.
        """
        if not self.metadata.has_valid_lrf:
            return None

        lrf_distance, lrf_lat, lrf_lon, lrf_abs_alt = self.metadata.lrf
        if lrf_lat == 0.0 and lrf_lon == 0.0:
            return {
                "status": "no_lrf_position",
                "lrf_abs_alt_m": lrf_abs_alt,
                "plane_altitude_m": self.ground_altitude,
                "vertical_gap_m": float(self.ground_altitude - lrf_abs_alt),
            }

        cx = self.metadata.width // 2
        cy = self.metadata.height // 2
        computed_geo = self.pixel_to_geodetic((cx, cy))

        if computed_geo is None:
            return {"status": "computation_failed"}

        lrf_geo = GeodeticPoint(
            latitude=lrf_lat,
            longitude=lrf_lon,
            altitude=lrf_abs_alt or 0.0
        )

        computed_enu = geodetic_to_enu(computed_geo, self.origin).to_array()
        lrf_enu = geodetic_to_enu(lrf_geo, self.origin).to_array()

        error_horizontal_m = np.linalg.norm(computed_enu[:2] - lrf_enu[:2])
        gsd = self.nominal_gsd_m

        return {
            "status": "ok",
            "computed": {
                "lat": computed_geo.latitude,
                "lon": computed_geo.longitude,
            },
            "lrf_measured": {
                "lat": lrf_geo.latitude,
                "lon": lrf_geo.longitude,
            },
            "error_horizontal_m": float(error_horizontal_m),
            "error_in_pixels": float(error_horizontal_m / gsd) if gsd > 0 else None,
            "lrf_distance_m": lrf_distance,
            "lrf_abs_alt_m": lrf_abs_alt,
            "plane_altitude_m": self.ground_altitude,
            "ground_plane_source": self.ground_plane_source,
            "vertical_gap_m": float(self.ground_altitude - lrf_abs_alt),
        }
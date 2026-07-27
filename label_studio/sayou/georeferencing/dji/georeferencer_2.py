"""
src/georeferencing/dji/dji_georeferencer.py
DJI 이미지 georeferencing 메인 클래스

수정 이력(2026-07):
- LRF 지면고도 판정 버그 수정: _to_float 기본값이 0.0 이라
  `metadata.lrf[3] is not None` 은 항상 True → LRF 미측정 사진에서
  ground_altitude 가 0 m 로 잡혀 스케일이 폭주하던 문제. metadata의
  has_valid_lrf / lrf_target_abs_alt 프로퍼티를 사용하도록 교체.
- pixel_to_camera_ray 의 왜곡보정 경로가 cv2 를 import 없이 참조하던 문제.
  DewarpData 캘리브레이션이 붙으면 이 경로가 실제로 실행되므로 지연 import 추가.
- yolo_bbox_to_geodetic 의 metadata.image_width/height 오타(→ width/height) 수정.
- 내부 파라미터 K, D 는 등방 초점거리/DewarpData 기반으로 개선된
  sayou.image.metadata.estimate_intrinsics_from_metadata 를 그대로 사용.
- 레거시 DJIGeoreferencer 클래스는 ImageMetadata 에 존재하지 않는 필드
  (relative_altitude_m, latitude, absolute_altitude_m 등)를 참조하는
  실행 불가능한 죽은 코드라 제거. 필요 시 새 API 기준으로 복원 요망.
"""

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
    ):
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

        # ── 지면 절대고도 결정 (우선순위: 명시값 > LRF 실측 > AGL 추정) ──
        # 주의: metadata.lrf 는 _to_float 로 채워져 미측정 시 0.0 이 들어간다.
        #       따라서 `is not None` 이 아니라 has_valid_lrf 로 판정해야 한다.
        if ground_altitude is not None:
            self.ground_altitude = ground_altitude
        elif metadata.has_valid_lrf:
            # LRF 조준점의 실측 절대고도 (AGL 추정보다 정확)
            self.ground_altitude = metadata.lrf_target_abs_alt
        else:
            # 폴백: 절대고도 - 상대고도(이륙지점 기준 AGL)
            self.ground_altitude = metadata.gps.altitude - metadata.relative_height

        # ENU 원점(=드론)에서 본 지면 평면의 up 좌표 (보통 음수: 아래)
        self.ground_up_in_enu = self.ground_altitude - metadata.gps.altitude

        axes = compute_camera_axes_from_gimbal(
            gimbal_yaw_compass_deg=metadata.orientation[0],
            gimbal_pitch_deg=metadata.orientation[1],
            gimbal_roll_deg=metadata.orientation[2]
        )
        self.R_camera_to_enu = axes["R_camera_to_enu"]
        self.optical_axis_enu = axes["axis_z_enu"]

        self._cache = {}

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
        return GeoreferencingResult(
            image_corners_geo=self.compute_image_corners(),
            ground_sample_distance_m=self.compute_ground_sample_distance(),
            coverage_area_m2=self.compute_coverage_area(),
            nadir_check=verify_nadir_orientation(self.R_camera_to_enu),
            ground_height_used=self.ground_altitude
        )

    def validate_with_lrf(self) -> Optional[dict]:
        """
        LRF(Laser Range Finder)로 측정한 타깃 좌표와 비교 검증

        DJI XMP의 LRFTarget* 필드는 이미지 중심에서 측정한 실제 거리/위치.
        이 값과 우리 계산이 일치하면 georeferencing이 정확한 것.
        """
        # LRF 미탑재/미측정 시 lrf 값들은 0.0 으로 채워진다.
        # 유효한 실측 타깃이 있을 때만 검증한다.
        if not self.metadata.has_valid_lrf:
            return None

        lrf_distance, lrf_lat, lrf_lon, lrf_abs_alt = self.metadata.lrf
        if lrf_lat == 0.0 and lrf_lon == 0.0:
            # 절대고도(lrf[3])만 있고 위경도가 없으면 위치 검증은 불가.
            return {"status": "no_lrf_position"}

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
            "lrf_distance_m": lrf_distance,
        }
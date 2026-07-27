"""
src/georeferencing/dji/dji_georeferencer.py
DJI 이미지 georeferencing 메인 클래스
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
 
        if ground_altitude is not None:
            self.ground_altitude = ground_altitude
        elif metadata.lrf[3] is not None:
            self.ground_altitude = metadata.lrf[3]
        else:
            self.ground_altitude = metadata.gps.altitude - metadata.relative_height
 
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
        cx_px = cx_norm * self.metadata.image_width
        cy_px = cy_norm * self.metadata.image_height
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
        if self.metadata.lrf[1] is None:
            return None
 
        cx = self.metadata.width // 2
        cy = self.metadata.height // 2
        computed_geo = self.pixel_to_geodetic((cx, cy))
 
        if computed_geo is None:
            return {"status": "computation_failed"}
 
        lrf_geo = GeodeticPoint(
            latitude=self.metadata.lrf[1],
            longitude=self.metadata.lrf[2],
            altitude=self.metadata.lrf[3] or 0.0
        )
 
        computed_enu = geodetic_to_enu(computed_geo, self.origin).to_array()
        lrf_enu = geodetic_to_enu(lrf_geo, self.origin).to_array()
 
        error_horizontal_m = np.linalg.norm(computed_enu[:2] - lrf_enu[:2])
 
        lrf_distance, lrf_lat, lrf_lon, lrf_abs_alt = self.metadata.lrf
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

 
class DJIGeoreferencer:
    """DJI 드론 이미지 georeferencing"""
    
    def __init__(
        self,
        metadata: ImageMetadata,
        intrinsics: CameraIntrinsics,
        ground_height_below_drone_m: Optional[float] = None,
    ):
        self.meta = metadata
        self.intrinsics = intrinsics
        
        # 지면(또는 패널 평면)이 카메라 아래 얼마나 있나
        # XMP의 RelativeAltitude는 이륙지점 기준
        # 패널 평면이 이륙지점과 같은 높이라면 RelativeAltitude를 그대로 사용
        if ground_height_below_drone_m is None:
            self.ground_distance_m = metadata.relative_altitude_m
        else:
            self.ground_distance_m = ground_height_below_drone_m
        
        # 좌표계 원점 = 드론 위치
        self.origin = GeodeticPoint(
            latitude=metadata.latitude,
            longitude=metadata.longitude,
            altitude=metadata.absolute_altitude_m
        )
        
        # 카메라 위치 (ENU에서 원점)
        self.camera_pos_enu = np.array([0.0, 0.0, 0.0])
        
        # 카메라 회전 행렬
        self.R_cam_to_world = dji_gimbal_to_camera_rotation(
            metadata.gimbal_yaw_deg,
            metadata.gimbal_pitch_deg,
            metadata.gimbal_roll_deg
        )
        
        # 지면 평면 (수평, 카메라 아래 ground_distance_m)
        # ENU에서 z=-ground_distance (카메라 원점 기준 아래)
        self.ground_normal = np.array([0.0, 0.0, 1.0])
        self.ground_z = -self.ground_distance_m
    
    def pixel_to_camera_ray(self, u: float, v: float) -> np.ndarray:
        """픽셀 → 카메라 좌표 광선 방향"""
        K_inv = np.linalg.inv(self.intrinsics.K)
        normalized = K_inv @ np.array([u, v, 1.0])
        return normalized / np.linalg.norm(normalized)
    
    def pixel_to_enu(self, u: float, v: float) -> Optional[np.ndarray]:
        """픽셀 → ENU 월드 좌표"""
        ray_cam = self.pixel_to_camera_ray(u, v)
        ray_world = self.R_cam_to_world @ ray_cam
        ray_world = ray_world / np.linalg.norm(ray_world)
        
        # 광선: p(t) = camera_pos + t * ray_world
        # 평면: z = ground_z (카메라 원점 기준 아래)
        # camera_pos.z + t * ray_world.z = ground_z
        # t = (ground_z - camera_pos.z) / ray_world.z
        
        if abs(ray_world[2]) < 1e-9:
            return None  # 광선이 평면과 평행
        
        t = (self.ground_z - self.camera_pos_enu[2]) / ray_world[2]
        
        if t < 0:
            return None  # 광선이 위로 향함
        
        return self.camera_pos_enu + t * ray_world
    
    def pixel_to_geodetic(self, u: float, v: float) -> Optional[GeodeticPoint]:
        """픽셀 → 지리 좌표"""
        enu_array = self.pixel_to_enu(u, v)
        if enu_array is None:
            return None
        
        enu = ENUPoint(east=enu_array[0], north=enu_array[1], up=enu_array[2])
        return enu_to_geodetic(enu, self.origin)
    
    def bbox_to_geodetic(
        self, bbox: Tuple[float, float, float, float]
    ) -> Optional[List[GeodeticPoint]]:
        """박스 (x1,y1,x2,y2) → 4개 모서리 지리 좌표"""
        x1, y1, x2, y2 = bbox
        corners_pixel = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
        result = []
        for u, v in corners_pixel:
            geo = self.pixel_to_geodetic(u, v)
            if geo is None:
                return None
            result.append(geo)
        return result
    
    def bbox_center_to_geodetic(
        self, bbox: Tuple[float, float, float, float]
    ) -> Optional[GeodeticPoint]:
        """박스 중심 → 지리 좌표"""
        x1, y1, x2, y2 = bbox
        return self.pixel_to_geodetic((x1 + x2) / 2, (y1 + y2) / 2)
    
    def compute_image_coverage(self) -> GeoreferencingResult:
        """이미지 전체 커버리지 정보"""
        w = self.intrinsics.image_width
        h = self.intrinsics.image_height
        
        # 4개 모서리
        corners_geo = []
        corners_enu = []
        for u, v in [(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)]:
            enu = self.pixel_to_enu(u, v)
            geo = self.pixel_to_geodetic(u, v)
            corners_enu.append(enu)
            corners_geo.append(geo)
        
        # GSD 계산 (이미지 중앙)
        cx, cy = w // 2, h // 2
        p_center = self.pixel_to_enu(cx, cy)
        p_right = self.pixel_to_enu(cx + 1, cy)
        p_down = self.pixel_to_enu(cx, cy + 1)
        gsd = (np.linalg.norm(p_right - p_center) + 
               np.linalg.norm(p_down - p_center)) / 2
        
        # 커버 면적 (다각형 면적, ENU 평면 가정)
        x_coords = [c[0] for c in corners_enu]
        y_coords = [c[1] for c in corners_enu]
        area = 0.5 * abs(
            sum(x_coords[i] * y_coords[(i+1) % 4] - 
                x_coords[(i+1) % 4] * y_coords[i]
                for i in range(4))
        )
        
        # 너비·높이 (대략)
        width_m = np.linalg.norm(corners_enu[1] - corners_enu[0])
        height_m = np.linalg.norm(corners_enu[3] - corners_enu[0])
        
        return GeoreferencingResult(
            image_corners_geo=corners_geo,
            ground_sample_distance_m=gsd,
            coverage_area_m2=area,
            coverage_width_m=width_m,
            coverage_height_m=height_m,
        )

"""
src/georeferencing/georeferencer.py
단일 이미지 georeferencing 메인 클래스
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import logging

from .coordinates import (
    GeodeticPoint, ENUPoint,
    geodetic_to_enu, enu_to_geodetic
)
from .camera_pose import (
    DronePose, GimbalPose,
    compute_dji_camera_rotation
)
from .plane_intersection import (
    Plane, Ray,
    pixel_to_world_via_plane,
    pixel_to_camera_ray, transform_ray_to_world,
    intersect_ray_plane
)

logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    """카메라 내부 파라미터"""
    K: np.ndarray
    D: np.ndarray
    image_width: int
    image_height: int


@dataclass
class GeoreferencingInput:
    """Georeferencing 입력 데이터"""
    drone_position: GeodeticPoint
    drone_pose: DronePose
    gimbal_pose: GimbalPose
    
    intrinsics: CameraIntrinsics
    
    ground_height: float = 0.0
    ground_normal: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 1])
    )
    
    coordinate_origin: Optional[GeodeticPoint] = None


@dataclass
class GeoreferencedPoint:
    """결과 데이터"""
    pixel: Tuple[float, float]
    world_enu: ENUPoint
    geodetic: GeodeticPoint
    
    distance_from_camera_m: float


@dataclass
class GeoreferencingResult:
    """Georeferencing 결과 묶음"""
    image_corners_geodetic: List[GeodeticPoint]
    
    ground_sample_distance_m: float
    
    coverage_area_m2: float
    
    pixel_to_world_fn: callable
    world_to_pixel_fn: callable


class SingleImageGeoreferencer:
    """
    단일 이미지 georeferencing
    
    사용법:
        gr = SingleImageGeoreferencer(input_data)
        
        # 픽셀 → 지리 좌표
        geo_point = gr.pixel_to_geodetic((1024, 768))
        
        # 박스의 지리 좌표
        bbox_corners = gr.bbox_to_geodetic((100, 200, 500, 600))
        
        # 이미지 모서리 (커버리지)
        corners = gr.compute_image_corners()
    """
    
    def __init__(self, input_data: GeoreferencingInput):
        self.input = input_data
        
        if self.input.coordinate_origin is None:
            self.origin = self.input.drone_position
        else:
            self.origin = self.input.coordinate_origin
        
        drone_enu = geodetic_to_enu(self.input.drone_position, self.origin)
        self.camera_position_enu = drone_enu.to_array()
        
        self.R_camera_to_world = compute_dji_camera_rotation(
            self.input.drone_pose,
            self.input.gimbal_pose
        )
        
        self.ground_plane = Plane(
            normal=self.input.ground_normal / np.linalg.norm(self.input.ground_normal),
            d=self.input.ground_height
        )
        
        self._cache = {}
    
    def pixel_to_world_enu(self, pixel: Tuple[float, float]) -> Optional[np.ndarray]:
        """픽셀 → 월드 ENU 좌표"""
        pixel_array = np.array(pixel)
        
        return pixel_to_world_via_plane(
            pixel=pixel_array,
            K=self.input.intrinsics.K,
            D=self.input.intrinsics.D,
            R_camera_to_world=self.R_camera_to_world,
            camera_position=self.camera_position_enu,
            plane=self.ground_plane
        )
    
    def pixel_to_geodetic(
        self,
        pixel: Tuple[float, float]
    ) -> Optional[GeodeticPoint]:
        """픽셀 → 지리 좌표 (lat, lon, alt)"""
        world_enu_array = self.pixel_to_world_enu(pixel)
        
        if world_enu_array is None:
            logger.warning(f"Pixel {pixel} does not intersect ground plane")
            return None
        
        enu = ENUPoint(
            east=world_enu_array[0],
            north=world_enu_array[1],
            up=world_enu_array[2]
        )
        
        return enu_to_geodetic(enu, self.origin)
    
    def bbox_to_geodetic(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[List[GeodeticPoint]]:
        """
        박스 (x1, y1, x2, y2) → 4개 모서리의 지리 좌표
        
        Returns:
            [top_left, top_right, bottom_right, bottom_left]
        """
        x1, y1, x2, y2 = bbox
        corners_pixel = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2)
        ]
        
        corners_geo = []
        for px in corners_pixel:
            geo = self.pixel_to_geodetic(px)
            if geo is None:
                return None
            corners_geo.append(geo)
        
        return corners_geo
    
    def bbox_center_to_geodetic(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[GeodeticPoint]:
        """박스 중심점 → 지리 좌표"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return self.pixel_to_geodetic((cx, cy))
    
    def compute_image_corners(self) -> List[GeodeticPoint]:
        """이미지의 4개 모서리에 해당하는 지리 좌표 계산"""
        if "corners" in self._cache:
            return self._cache["corners"]
        
        w = self.input.intrinsics.image_width
        h = self.input.intrinsics.image_height
        
        corners_pixel = [
            (0, 0),
            (w - 1, 0),
            (w - 1, h - 1),
            (0, h - 1)
        ]
        
        corners_geo = []
        for px in corners_pixel:
            geo = self.pixel_to_geodetic(px)
            if geo is None:
                logger.warning(f"Image corner {px} does not project to ground")
                geo = GeodeticPoint(0, 0, 0)
            corners_geo.append(geo)
        
        self._cache["corners"] = corners_geo
        return corners_geo
    
    def compute_ground_sample_distance(self) -> float:
        """
        Ground Sample Distance (GSD) 계산
        
        한 픽셀이 지상에서 차지하는 거리(미터)
        
        이미지 중심 부근 픽셀들로 측정
        """
        w = self.input.intrinsics.image_width
        h = self.input.intrinsics.image_height
        
        cx, cy = w // 2, h // 2
        
        p1 = self.pixel_to_world_enu((cx, cy))
        p2 = self.pixel_to_world_enu((cx + 1, cy))
        p3 = self.pixel_to_world_enu((cx, cy + 1))
        
        if p1 is None or p2 is None or p3 is None:
            return float("nan")
        
        gsd_x = np.linalg.norm(p2 - p1)
        gsd_y = np.linalg.norm(p3 - p1)
        
        return (gsd_x + gsd_y) / 2
    
    def compute_coverage_area(self) -> float:
        """이미지가 지상에서 커버하는 면적 (m²)"""
        corners = self.compute_image_corners()
        
        enu_corners = [
            geodetic_to_enu(c, self.origin).to_array()[:2]
            for c in corners
        ]
        
        x1, y1 = enu_corners[0]
        x2, y2 = enu_corners[1]
        x3, y3 = enu_corners[2]
        x4, y4 = enu_corners[3]
        
        area = 0.5 * abs(
            (x1 * y2 - x2 * y1) +
            (x2 * y3 - x3 * y2) +
            (x3 * y4 - x4 * y3) +
            (x4 * y1 - x1 * y4)
        )
        
        return area
    
    def world_to_pixel(
        self,
        world_point: ENUPoint
    ) -> Optional[Tuple[float, float]]:
        """
        월드 좌표 → 픽셀 좌표 (역변환, 검증용)
        
        수식:
            point_camera = R_world_to_camera · (point_world - camera_pos)
            point_image = K · point_camera
            pixel = point_image / z
        """
        world_array = world_point.to_array()
        
        relative = world_array - self.camera_position_enu
        
        R_world_to_camera = self.R_camera_to_world.T
        point_camera = R_world_to_camera @ relative
        
        if point_camera[2] <= 0:
            return None
        
        K = self.input.intrinsics.K
        u = K[0, 0] * point_camera[0] / point_camera[2] + K[0, 2]
        v = K[1, 1] * point_camera[1] / point_camera[2] + K[1, 2]
        
        return (float(u), float(v))
    
    def geodetic_to_pixel(
        self,
        geo: GeodeticPoint
    ) -> Optional[Tuple[float, float]]:
        """지리 좌표 → 픽셀 좌표"""
        enu = geodetic_to_enu(geo, self.origin)
        return self.world_to_pixel(enu)

"""
src/georeferencing/plane_intersection.py
픽셀 광선과 지면 평면의 교차
"""
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class Ray:
    """3D 광선"""
    origin: np.ndarray
    direction: np.ndarray
    
    def point_at(self, t: float) -> np.ndarray:
        return self.origin + t * self.direction


@dataclass
class Plane:
    """3D 평면 (n · p = d)"""
    normal: np.ndarray
    d: float
    
    @classmethod
    def from_point_and_normal(cls, point: np.ndarray, normal: np.ndarray):
        normal = normal / np.linalg.norm(normal)
        d = np.dot(normal, point)
        return cls(normal=normal, d=d)
    
    @classmethod
    def horizontal_at_height(cls, height: float):
        return cls(normal=np.array([0, 0, 1]), d=height)


def pixel_to_camera_ray(
    pixel: np.ndarray,
    K: np.ndarray,
    D: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    픽셀 좌표 → 카메라 좌표계 광선 방향
    
    수식:
        [x, y, 1]^T = K^(-1) [u, v, 1]^T
        ray = normalize([x, y, 1])
    
    왜곡이 있으면 먼저 왜곡 보정
    """
    u, v = pixel
    
    if D is not None and np.any(D != 0):
        import cv2
        pts = np.array([[[u, v]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(pts, K, D)
        x, y = undistorted[0, 0]
    else:
        K_inv = np.linalg.inv(K)
        homogeneous = np.array([u, v, 1.0])
        normalized = K_inv @ homogeneous
        x, y = normalized[0], normalized[1]
    
    ray = np.array([x, y, 1.0])
    return ray / np.linalg.norm(ray)


def transform_ray_to_world(
    ray_camera: np.ndarray,
    R_camera_to_world: np.ndarray,
    camera_position: np.ndarray
) -> Ray:
    """
    카메라 좌표 광선 → 월드 좌표 광선
    """
    direction_world = R_camera_to_world @ ray_camera
    direction_world = direction_world / np.linalg.norm(direction_world)
    
    return Ray(origin=camera_position, direction=direction_world)


def intersect_ray_plane(ray: Ray, plane: Plane) -> Optional[np.ndarray]:
    """
    광선과 평면 교차점
    
    수식:
        ray: p(t) = origin + t · direction
        plane: n · p = d
        
        n · (origin + t · direction) = d
        t = (d - n · origin) / (n · direction)
    
    Returns:
        교차점 좌표 또는 None (광선이 평면과 평행하거나 뒤로 향함)
    """
    denom = np.dot(plane.normal, ray.direction)
    
    if abs(denom) < 1e-9:
        return None
    
    t = (plane.d - np.dot(plane.normal, ray.origin)) / denom
    
    if t < 0:
        return None
    
    intersection = ray.point_at(t)
    return intersection


def pixel_to_world_via_plane(
    pixel: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    R_camera_to_world: np.ndarray,
    camera_position: np.ndarray,
    plane: Plane
) -> Optional[np.ndarray]:
    """
    픽셀 좌표 → 월드 좌표 (평면 가정)
    
    전체 변환 체인을 한 번에 수행
    """
    ray_cam = pixel_to_camera_ray(pixel, K, D)
    
    ray_world = transform_ray_to_world(
        ray_cam, R_camera_to_world, camera_position
    )
    
    intersection = intersect_ray_plane(ray_world, plane)
    
    return intersection
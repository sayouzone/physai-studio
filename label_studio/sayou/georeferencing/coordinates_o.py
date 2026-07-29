"""
src/georeferencing/coordinates.py
좌표계 간 변환 유틸리티
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = 1 - (WGS84_B ** 2) / (WGS84_A ** 2)


@dataclass
class GeodeticPoint:
    """WGS84 지리 좌표"""
    latitude: float
    longitude: float
    altitude: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.latitude, self.longitude, self.altitude])


@dataclass
class ENUPoint:
    """East-North-Up 로컬 좌표"""
    east: float
    north: float
    up: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.east, self.north, self.up])


@dataclass
class ECEFPoint:
    """Earth-Centered Earth-Fixed 좌표"""
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


def geodetic_to_ecef(point: GeodeticPoint) -> ECEFPoint:
    """
    WGS84 지리 좌표 → ECEF 변환
    
    수식:
        N(φ) = a / sqrt(1 - e² sin²φ)
        X = (N + h) cosφ cosλ
        Y = (N + h) cosφ sinλ
        Z = (N(1-e²) + h) sinφ
    """
    lat_rad = np.radians(point.latitude)
    lon_rad = np.radians(point.longitude)
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat ** 2)
    
    x = (n + point.altitude) * cos_lat * cos_lon
    y = (n + point.altitude) * cos_lat * sin_lon
    z = (n * (1 - WGS84_E2) + point.altitude) * sin_lat
    
    return ECEFPoint(x, y, z)


def ecef_to_geodetic(point: ECEFPoint) -> GeodeticPoint:
    """
    ECEF → WGS84 지리 좌표 (Bowring 방법)
    """
    x, y, z = point.x, point.y, point.z
    
    lon = np.arctan2(y, x)
    
    p = np.sqrt(x ** 2 + y ** 2)
    
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    
    for _ in range(5):
        sin_lat = np.sin(lat)
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat ** 2)
        h = p / np.cos(lat) - n
        lat_new = np.arctan2(z, p * (1 - WGS84_E2 * n / (n + h)))
        if abs(lat - lat_new) < 1e-12:
            lat = lat_new
            break
        lat = lat_new
    
    sin_lat = np.sin(lat)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat ** 2)
    h = p / np.cos(lat) - n
    
    return GeodeticPoint(
        latitude=np.degrees(lat),
        longitude=np.degrees(lon),
        altitude=h
    )


def ecef_to_enu(
    point: ECEFPoint,
    origin: GeodeticPoint
) -> ENUPoint:
    """
    ECEF → ENU 변환 (origin 기준 로컬 좌표계)
    
    수식:
        ENU = R · (ECEF - ECEF_origin)
        R = [[-sinλ, cosλ, 0],
             [-sinφ cosλ, -sinφ sinλ, cosφ],
             [cosφ cosλ, cosφ sinλ, sinφ]]
    """
    origin_ecef = geodetic_to_ecef(origin)
    
    diff = np.array([
        point.x - origin_ecef.x,
        point.y - origin_ecef.y,
        point.z - origin_ecef.z
    ])
    
    lat_rad = np.radians(origin.latitude)
    lon_rad = np.radians(origin.longitude)
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    R = np.array([
        [-sin_lon,            cos_lon,           0      ],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat]
    ])
    
    enu_array = R @ diff
    
    return ENUPoint(
        east=enu_array[0],
        north=enu_array[1],
        up=enu_array[2]
    )


def enu_to_ecef(
    point: ENUPoint,
    origin: GeodeticPoint
) -> ECEFPoint:
    """
    ENU → ECEF 변환 (ecef_to_enu의 역)
    """
    origin_ecef = geodetic_to_ecef(origin)
    
    lat_rad = np.radians(origin.latitude)
    lon_rad = np.radians(origin.longitude)
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    R_inv = np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
        [ cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
        [ 0,        cos_lat,           sin_lat          ]
    ])
    
    diff = R_inv @ point.to_array()
    
    return ECEFPoint(
        x=origin_ecef.x + diff[0],
        y=origin_ecef.y + diff[1],
        z=origin_ecef.z + diff[2]
    )


def enu_to_geodetic(
    point: ENUPoint,
    origin: GeodeticPoint
) -> GeodeticPoint:
    """ENU → 지리 좌표 (체인 변환)"""
    ecef = enu_to_ecef(point, origin)
    return ecef_to_geodetic(ecef)


def geodetic_to_enu(
    point: GeodeticPoint,
    origin: GeodeticPoint
) -> ENUPoint:
    """지리 좌표 → ENU"""
    ecef = geodetic_to_ecef(point)
    return ecef_to_enu(ecef, origin)

"""
좌표계 변환: WGS84 ↔ ECEF ↔ ENU
"""
import numpy as np
from dataclasses import dataclass

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = 1 - (WGS84_B ** 2) / (WGS84_A ** 2)


@dataclass
class GeodeticPoint:
    latitude: float
    longitude: float
    altitude: float = 0.0


@dataclass
class ENUPoint:
    east: float
    north: float
    up: float

    def to_array(self) -> np.ndarray:
        return np.array([self.east, self.north, self.up])


@dataclass
class ECEFPoint:
    x: float
    y: float
    z: float


def geodetic_to_ecef(point: GeodeticPoint) -> ECEFPoint:
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
    x, y, z = point.x, point.y, point.z
    lon = np.arctan2(y, x)
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1 - WGS84_E2))

    for _ in range(8):
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

    return GeodeticPoint(np.degrees(lat), np.degrees(lon), h)


def _enu_rotation_matrix(origin: GeodeticPoint) -> np.ndarray:
    lat_rad = np.radians(origin.latitude)
    lon_rad = np.radians(origin.longitude)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    return np.array([
        [-sin_lon,            cos_lon,           0      ],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat]
    ])


def geodetic_to_enu(point: GeodeticPoint, origin: GeodeticPoint) -> ENUPoint:
    point_ecef = geodetic_to_ecef(point)
    origin_ecef = geodetic_to_ecef(origin)
    diff = np.array([
        point_ecef.x - origin_ecef.x,
        point_ecef.y - origin_ecef.y,
        point_ecef.z - origin_ecef.z
    ])
    enu = _enu_rotation_matrix(origin) @ diff
    return ENUPoint(enu[0], enu[1], enu[2])


def enu_to_geodetic(point: ENUPoint, origin: GeodeticPoint) -> GeodeticPoint:
    R_inv = _enu_rotation_matrix(origin).T
    diff = R_inv @ point.to_array()
    origin_ecef = geodetic_to_ecef(origin)
    point_ecef = ECEFPoint(
        origin_ecef.x + diff[0],
        origin_ecef.y + diff[1],
        origin_ecef.z + diff[2]
    )
    return ecef_to_geodetic(point_ecef)
"""
DJI 드론 사진의 EXIF + XMP 메타데이터 추출
태양광 패널 검사 georeferencing 입력 데이터 준비
"""
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import exifread
import numpy as np

from .camera_pose import compute_camera_axes_from_gimbal

# 35mm full-frame 센서 물리 크기 (mm) — FOV 환산 기준
SENSOR_35MM_W = 36.0
SENSOR_35MM_H = 24.0
SENSOR_35MM_DIAG = math.hypot(SENSOR_35MM_W, SENSOR_35MM_H)  # 43.2666
 
# 위경도 ↔ 미터 변환 (WGS84, 작은 패치 가정)
M_PER_DEG_LAT = 111_320.0

@dataclass
class DJIMetadata:
    """DJI 드론 사진의 georeferencing 필수 메타데이터"""
    image_path: str
    image_width: int
    image_height: int
    capture_time: str

    gps_latitude: float
    gps_longitude: float
    absolute_altitude: float
    relative_altitude: float

    gimbal_yaw_deg: float
    gimbal_pitch_deg: float
    gimbal_roll_deg: float

    focal_length_35mm: int
    focal_length_mm: Optional[float] = 0.

    flight_yaw_deg: Optional[float] = 0.
    flight_pitch_deg: Optional[float] = 0.
    flight_roll_deg: Optional[float] = 0.

    rtk_flag: Optional[int] = None
    lrf_distance: Optional[float] = None
    lrf_target_lat: Optional[float] = None
    lrf_target_lon: Optional[float] = None
    lrf_target_abs_alt: Optional[float] = None

    camera_model: Optional[str] = 'wide'                # 'wide' | 'zoom' | 'thermal'
    # RTK / 시간
    rtk_active: Optional[bool] = False
    # 캐시: 카메라 → ENU 회전 행렬
    R_cam_to_enu: np.ndarray = field(default=None) #, repr=False

    # GPS/RTK 측정 표준편차 (XMP에 포함될 경우)
    gps_std_xy: float = 0.10   # 기본 10cm
    gps_std_z: float = 0.15    # 기본 15cm

    # ----- 동적 FOV -----

    @property
    def focal_px(self) -> float:
        """35mm 환산 초점거리는 '대각선' 기준 정의 → 대각선으로 환산해야 등방 f가 나온다."""
        diag_px = math.hypot(self.width, self.height)
        return self.focal_length_in_35mm * diag_px / SENSOR_35MM_DIAG

    @property
    def hfov_deg(self) -> float:
        #return 2 * math.degrees(math.atan(SENSOR_35MM_W / (2 * self.focal_length_35mm)))
        return 2 * math.degrees(math.atan(self.width  / (2 * self.focal_px)))
 
    @property
    def vfov_deg(self) -> float:
        #return 2 * math.degrees(math.atan(SENSOR_35MM_H / (2 * self.focal_length_35mm)))
        return 2 * math.degrees(math.atan(self.height / (2 * self.focal_px)))
 
    @property
    def is_nadir(self) -> bool:
        return abs(self.gimbal_pitch_deg + 90.0) < 5.0
 
    # ----- 위경도 → 미터 환산 (해당 위도) -----
    @property
    def m_per_deg_lon(self) -> float:
        return M_PER_DEG_LAT * math.cos(math.radians(self.gps_latitude))

    def __repr__(self):
        rtk = "RTK" if self.rtk_flag and self.rtk_flag >= 50 else "GPS"
        lrf = f"LRF={self.lrf_distance:.1f}m" if self.lrf_distance else "no LRF"
        return (
            f"DJIMetadata({self.image_width}x{self.image_height}, "
            f"({self.gps_latitude:.6f}, {self.gps_longitude:.6f}), "
            f"alt={self.absolute_altitude:.1f}m (rel={self.relative_altitude:.1f}m), "
            f"gimbal_pitch={self.gimbal_pitch_deg:.1f}°, "
            f"gimbal_yaw={self.gimbal_yaw_deg:.1f}°, "
            f"{rtk}, {lrf})"
        )

def _ratio_to_float(ratio) -> float:
    if hasattr(ratio, 'num') and hasattr(ratio, 'den'):
        return ratio.num / ratio.den
    parts = str(ratio).split('/')
    if len(parts) == 2:
        return float(parts[0]) / float(parts[1])
    return float(ratio)

def _detect_camera(focal_35mm: float, w: int, h: int) -> str:
    """
    Zenmuse H20T:
      - Wide   : 24mm-eq (4056×3040 또는 5184×3888 high-res)
      - Zoom   : 31.7~127.7mm-eq (5184×3888)
      - Thermal: 640×512
    """
    if w == 640 and h == 512:
        return 'thermal'
    if 22 <= focal_35mm <= 26:
        return 'wide'
    return 'zoom'

def _dms_to_decimal(dms_value, ref: str) -> float:
    parts = dms_value.values
    degrees = _ratio_to_float(parts[0])
    minutes = _ratio_to_float(parts[1])
    seconds = _ratio_to_float(parts[2])
    decimal = degrees + minutes / 60 + seconds / 3600
    if ref in ('S', 'W'):
        decimal = -decimal
    return decimal

def extract_xmp(image_path: str) -> Optional[str]:
    """JPEG에서 XMP 메타데이터 블록 추출"""
    with open(image_path, 'rb') as f:
        data = f.read()

    start = data.find(b'<x:xmpmeta')
    if start == -1:
        return None
    end = data.find(b'</x:xmpmeta>', start)
    if end == -1:
        return None

    return data[start:end + len(b'</x:xmpmeta>')].decode('utf-8', errors='replace')


def parse_xmp_field(xmp: str, field_name: str) -> Optional[str]:
    """XMP에서 특정 필드 값 추출"""
    pattern = rf'{re.escape(field_name)}="([^"]*)"'
    match = re.search(pattern, xmp)
    return match.group(1) if match else None


def parse_xmp_float(xmp: str, field_name: str) -> Optional[float]:
    """XMP 필드를 float으로 파싱 (DJI는 +/- 부호 포함)"""
    value = parse_xmp_field(xmp, field_name)
    if value is None:
        return None
    return float(value)


def parse_xmp_int(xmp: str, field_name: str) -> Optional[int]:
    value = parse_xmp_field(xmp, field_name)
    if value is None:
        return None
    return int(value)


def extract_dji_metadata(image_path: str) -> DJIMetadata:
    """DJI 이미지의 georeferencing 메타데이터 종합 추출"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, 'rb') as f:
        exif = exifread.process_file(f, details=False)

    image_width = int(str(exif.get('EXIF ExifImageWidth', '0')))
    image_height = int(str(exif.get('EXIF ExifImageLength', '0')))

    gps_lat_dms = exif.get('GPS GPSLatitude')
    gps_lat_ref = str(exif.get('GPS GPSLatitudeRef', 'N'))
    gps_lon_dms = exif.get('GPS GPSLongitude')
    gps_lon_ref = str(exif.get('GPS GPSLongitudeRef', 'E'))

    if gps_lat_dms is None or gps_lon_dms is None:
        raise ValueError(f"GPS coordinates not found in {image_path}")

    gps_lat_exif = _dms_to_decimal(gps_lat_dms, gps_lat_ref)
    gps_lon_exif = _dms_to_decimal(gps_lon_dms, gps_lon_ref)

    focal_length_mm = _ratio_to_float(exif.get('EXIF FocalLength', '0/1'))
    focal_length_35mm = int(str(exif.get('EXIF FocalLengthIn35mmFilm', '0')))

    capture_time = str(exif.get('EXIF DateTimeOriginal', ''))

    xmp = extract_xmp(image_path)
    if xmp is None:
        raise ValueError(f"XMP metadata not found in {image_path}")

    gps_latitude = parse_xmp_float(xmp, 'drone-dji:GpsLatitude') or gps_lat_exif
    gps_longitude = parse_xmp_float(xmp, 'drone-dji:GpsLongitude') or gps_lon_exif
    absolute_altitude = parse_xmp_float(xmp, 'drone-dji:AbsoluteAltitude') or 0.0
    relative_altitude = parse_xmp_float(xmp, 'drone-dji:RelativeAltitude') or 0.0

    flight_yaw = parse_xmp_float(xmp, 'drone-dji:FlightYawDegree') or 0.0
    flight_pitch = parse_xmp_float(xmp, 'drone-dji:FlightPitchDegree') or 0.0
    flight_roll = parse_xmp_float(xmp, 'drone-dji:FlightRollDegree') or 0.0

    gimbal_yaw = parse_xmp_float(xmp, 'drone-dji:GimbalYawDegree') or 0.0
    gimbal_pitch = parse_xmp_float(xmp, 'drone-dji:GimbalPitchDegree') or 0.0
    gimbal_roll = parse_xmp_float(xmp, 'drone-dji:GimbalRollDegree') or 0.0

    rtk_flag = parse_xmp_int(xmp, 'drone-dji:RtkFlag')
    lrf_distance = parse_xmp_float(xmp, 'drone-dji:LRFTargetDistance')
    lrf_target_lat = parse_xmp_float(xmp, 'drone-dji:LRFTargetLat')
    lrf_target_lon = parse_xmp_float(xmp, 'drone-dji:LRFTargetLon')
    lrf_target_abs_alt = parse_xmp_float(xmp, 'drone-dji:LRFTargetAbsAlt')

    # 짐벌 자세 → 카메라 회전 행렬 (위임)
    axes = compute_camera_axes_from_gimbal(
        gimbal_yaw_compass_deg=gimbal_yaw,
        gimbal_pitch_deg=gimbal_pitch,
        gimbal_roll_deg=gimbal_roll,
    )
    R = axes['R_camera_to_enu']

    return DJIMetadata(
        image_path=str(path),
        image_width=image_width,
        image_height=image_height,
        gps_latitude=gps_latitude,
        gps_longitude=gps_longitude,
        absolute_altitude=absolute_altitude,
        relative_altitude=relative_altitude,
        flight_yaw_deg=flight_yaw,
        flight_pitch_deg=flight_pitch,
        flight_roll_deg=flight_roll,
        gimbal_yaw_deg=gimbal_yaw,
        gimbal_pitch_deg=gimbal_pitch,
        gimbal_roll_deg=gimbal_roll,
        focal_length_mm=focal_length_mm,
        focal_length_35mm=focal_length_35mm,
        capture_time=capture_time,
        rtk_flag=rtk_flag,
        lrf_distance=lrf_distance,
        lrf_target_lat=lrf_target_lat,
        lrf_target_lon=lrf_target_lon,
        lrf_target_abs_alt=lrf_target_abs_alt,
        camera_model=_detect_camera(focal_length_35mm, image_width, image_height),
        rtk_active=rtk_flag in (16, 50),
        R_cam_to_enu=R,
    )


def estimate_intrinsics_from_metadata(metadata: DJIMetadata) -> tuple:
    """
    EXIF 메타데이터에서 카메라 내부 파라미터 K, D 추정

    실제 캘리브레이션이 가장 정확하지만, 없을 때의 fallback.

    원리:
        FocalLengthIn35mmFilm = 35mm equivalent focal length
        실제 센서에서:
            f_pixels = (focal_length_35mm / 36) * image_width
        주점은 이미지 중심 가정.
    """
    import numpy as np

    f_pixels = (metadata.focal_length_35mm / 36.0) * metadata.image_width

    cx = metadata.image_width / 2.0
    cy = metadata.image_height / 2.0

    K = np.array([
        [f_pixels, 0,        cx],
        [0,        f_pixels, cy],
        [0,        0,        1.0]
    ])

    D = np.zeros(5)

    return K, D


def parse_xmp_attribute(xmp: str, attr: str) -> Optional[str]:
    """XMP 속성 값 파싱"""
    match = re.search(rf'{attr}="([^"]*)"', xmp)
    if match:
        return match.group(1)
    
    match = re.search(rf'<[^>]*:{attr}>([^<]*)</', xmp)
    if match:
        return match.group(1)
    
    return None


def parse_dji_metadata(filepath: Path) -> DJIMetadata:
    """DJI 이미지에서 메타데이터 추출 (XMP 우선)"""
    import exifread
    
    with open(filepath, 'rb') as f:
        exif_tags = exifread.process_file(f, details=False)
    
    xmp = extract_xmp(filepath)
    if xmp is None:
        raise ValueError(f"XMP not found in {filepath}")
    
    return DJIMetadata(
        latitude=float(parse_xmp_attribute(xmp, "GpsLatitude")),
        longitude=float(parse_xmp_attribute(xmp, "GpsLongitude")),
        absolute_altitude_m=float(parse_xmp_attribute(xmp, "AbsoluteAltitude")),
        relative_altitude_m=float(parse_xmp_attribute(xmp, "RelativeAltitude")),
        flight_yaw_deg=float(parse_xmp_attribute(xmp, "FlightYawDegree")),
        flight_pitch_deg=float(parse_xmp_attribute(xmp, "FlightPitchDegree")),
        flight_roll_deg=float(parse_xmp_attribute(xmp, "FlightRollDegree")),
        gimbal_yaw_deg=float(parse_xmp_attribute(xmp, "GimbalYawDegree")),
        gimbal_pitch_deg=float(parse_xmp_attribute(xmp, "GimbalPitchDegree")),
        gimbal_roll_deg=float(parse_xmp_attribute(xmp, "GimbalRollDegree")),
        image_width=int(str(exif_tags["EXIF ExifImageWidth"])),
        image_height=int(str(exif_tags["EXIF ExifImageLength"])),
        rtk_flag=int(parse_xmp_attribute(xmp, "RtkFlag") or "0"),
        rtk_std_lat_m=float(parse_xmp_attribute(xmp, "RtkStdLat") or "999"),
        rtk_std_lon_m=float(parse_xmp_attribute(xmp, "RtkStdLon") or "999"),
        rtk_std_hgt_m=float(parse_xmp_attribute(xmp, "RtkStdHgt") or "999"),
        camera_model=str(exif_tags.get("Image Model", "Unknown")),
        timestamp=str(exif_tags.get("EXIF DateTimeOriginal", "")),
    )

# ============================================================
# 2. 카메라 내부 파라미터 (캘리브레이션)
# ============================================================

@dataclass
class CameraIntrinsics:
    """카메라 내부 파라미터"""
    K: np.ndarray
    D: np.ndarray
    image_width: int
    image_height: int


def estimate_zh20t_zoom_intrinsics(width: int, height: int,
                                    focal_length_35mm: float = 47.0) -> CameraIntrinsics:
    """
    DJI ZH20T Zoom 카메라의 추정 내부 파라미터
    
    실제 운영에선 캘리브레이션 결과를 사용해야 하지만,
    여기선 35mm 등가 초점거리로부터 추정
    
    35mm equivalent focal length f_35mm = 47mm
    Sensor: 1/1.7" (실제 7.6 × 5.7mm)
    Diagonal_35mm = 43.27mm
    Diagonal_sensor = 9.5mm  (1/1.7" 대각선)
    
    f_x_pixel = (f_35mm / Diagonal_35mm) * Diagonal_sensor / sensor_width * image_width
    """
    # 35mm 등가 → 실제 픽셀 초점거리 계산
    # 간단 추정: f_pixel = (f_35mm / 36mm) * image_width
    fx = fy = (focal_length_35mm / 36.0) * width
    
    cx = width / 2
    cy = height / 2
    
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1.0]
    ])
    D = np.zeros(5)  # 왜곡 계수 (실제로는 캘리브레이션 필요)
    
    return CameraIntrinsics(K=K, D=D, image_width=width, image_height=height)


def estimate_zh20t_thermal_intrinsics(width: int, height: int) -> CameraIntrinsics:
    """
    DJI ZH20T Thermal 카메라 추정 내부 파라미터
    
    공식 사양: FOV 40.6° (대각선)
    해상도: 640 × 512
    """
    # FOV로부터 초점거리 계산
    # f = (width/2) / tan(FOV_h / 2)
    # FOV_diag 40.6° → FOV_h ≈ 32.5° (4:3.2 종횡비)
    fov_h_deg = 32.5
    fx = (width / 2) / np.tan(np.radians(fov_h_deg / 2))
    fy = fx  # 정사각 픽셀 가정
    
    cx = width / 2
    cy = height / 2
    
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1.0]
    ])
    D = np.zeros(5)
    
    return CameraIntrinsics(K=K, D=D, image_width=width, image_height=height)

# ============================================================
# 4. 카메라 회전 행렬 (DJI 자세 → 월드 회전)
# ============================================================

def dji_gimbal_to_camera_rotation(
    gimbal_yaw_deg: float,
    gimbal_pitch_deg: float,
    gimbal_roll_deg: float
) -> np.ndarray:
    """
    DJI 짐벌 자세 → 카메라 회전 행렬 (R_camera_to_world)
    
    DJI 관습:
    - GimbalYaw: 북에서 시계방향 (북=0°, 동=90°)
        XMP에선 -180~180. -90.5° = 거의 서쪽 향함.
    - GimbalPitch: 수평선 기준 (수평=0°, 직하방=-90°)
    - GimbalRoll: 카메라 광축 회전
    
    이 코드의 World 좌표계: ENU (East-North-Up)
    카메라 좌표계 (OpenCV): X=오른쪽, Y=아래, Z=앞 (광축)
    
    Nadir 촬영(pitch=-90°)일 때:
    - 카메라 Z축은 아래(-Up) 방향
    - 카메라 X축은 yaw에 따라 결정
    """
    yaw = np.radians(gimbal_yaw_deg)
    pitch = np.radians(gimbal_pitch_deg)
    roll = np.radians(gimbal_roll_deg)
    
    # ENU에서 짐벌의 forward 벡터 (카메라가 보는 방향)
    # gimbal pitch=0이면 수평 (yaw 방향), pitch=-90이면 down(-Up)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    
    # ENU 좌표계에서 yaw가 북에서 동으로 시계방향이라는 것은 
    # ENU의 N축에서 E축으로 회전. yaw=0 → +N, yaw=90° → +E
    # forward (cam Z): pitch와 yaw로 결정
    forward = np.array([
        cp * sy,    # East
        cp * cy,    # North
        sp,         # Up (pitch=-90이면 sp=-1, 즉 -Up = down)
    ])
    
    # right (cam X): yaw 방향으로 90° 시계방향
    right = np.array([
        cy,         # East
        -sy,        # North
        0,          # Up
    ])
    
    # down (cam Y): forward × right 또는 별도 계산
    # OpenCV 카메라: X=right, Y=down (image의 v 방향), Z=forward
    # down = forward × right (right-handed)는 아니고
    # 표준 OpenCV에서 Y는 Z×X (right와 forward의 외적의 음수)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    
    # roll 적용 (광축 기준 회전)
    # right와 down을 광축 기준으로 회전
    if abs(roll) > 1e-9:
        right_rolled = cr * right + sr * down
        down_rolled = -sr * right + cr * down
        right = right_rolled
        down = down_rolled
    
    # R_camera_to_world: 카메라 좌표 벡터를 월드 좌표로 변환
    # 카메라 X(right) → world right
    # 카메라 Y(down) → world down
    # 카메라 Z(forward) → world forward
    R_camera_to_world = np.column_stack([right, down, forward])
    
    return R_camera_to_world

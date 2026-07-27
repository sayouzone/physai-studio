import hashlib
import math
import numpy as np
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from datetime import datetime

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
# DJI RtkFlag 의미 (DJI SDK 문서 기준):
#   0  = None / GPS only
#   16 = RTK Float (수십 cm 정확도)
#   34 = RTK Single (저정밀)
#   50 = RTK Fixed (1~3 cm 정확도) ← 신뢰 가능
RTK_FIXED = 50

# 35mm full-frame 센서 물리 크기 (mm) — FOV 환산 기준
SENSOR_35MM_W = 36.0
SENSOR_35MM_H = 24.0
 
# 위경도 ↔ 미터 변환 (WGS84, 작은 패치 가정)
M_PER_DEG_LAT = 111_320.0

@dataclass
class GpsInfo:
    altitude: float
    lat: float
    lng: float


@dataclass
class GeoDesc:
    cs_type: str = "GEO_CS"
    geo_cs: str = "EPSG:4326"


@dataclass
class XmpInfo:
    bandName: str = ""
    captureUUID: str = ""
    droneID: str = ""
    cameraMaker: str = ""
    cameraModel: str = ""


@dataclass
class PosInfo:
    pos: list[float]
    pos_sigma: list[float]
    orientation: list[float]
    id: str


@dataclass
class ImageMetadata:
    """drone-dji 메타데이터 전체 구조 (제공된 JSON 스키마와 1:1 매핑)."""

    id: str
    thumbnailPath: str
    path: str
    origin_path: str
    gps: GpsInfo
    position: list[float]
    relative_height: float
    flight: list[float]
    orientation: list[float]
    orientation_type: str = "YPR"
    pos_sigma: list[float] = field(default_factory=lambda: [])
    geo_desc: GeoDesc = field(default_factory=GeoDesc)
    ppk: Any = None
    height: int = 0
    width: int = 0
    velocity: list[float] = field(default_factory=lambda: [])
    camera_model: str = ""
    camera_maker: str = ""
    rtk_flag: int = 0
    dewarp_flag: bool = True
    pre_calib_param: list[Any] = field(default_factory=lambda: [None] * 9)
    focal_length: float = 0.
    focal_length_in_35mm: int = 0.
    isImported: bool = True
    capture_time: int = 0
    xmp: XmpInfo = field(default_factory=lambda: {})
    aux_img: Any = None
    camera_sn: str = ""
    sub_camera_sn: str = ""
    lens_sn: str = ""
    rtk_std: list[float] = field(default_factory=lambda: [])
    lrf: list[float] = field(default_factory=lambda: [])
    lens_position: str = ""
    pre_calib_conf: int = 0
    drone_model: str = ""
    payload_model: str = ""
    pos_info: PosInfo = field(default_factory=lambda: {})

    # GPS/RTK 측정 표준편차 (XMP에 포함될 경우)
    gps_std_xy: float = 0.10   # 기본 10cm
    gps_std_z: float = 0.15    # 기본 15cm

    # RTK / 시간
    rtk_active: Optional[bool] = False
    # 캐시: 카메라 → ENU 회전 행렬
    R_cam_to_enu: np.ndarray = field(default=None, repr=False)

    @property
    def is_rtk_fixed(self) -> bool:
        return self.rtk_flag == RTK_FIXED

    # ----- 동적 FOV -----
    @property
    def hfov_deg(self) -> float:
        return 2 * math.degrees(math.atan(SENSOR_35MM_W / (2 * self.focal_length_in_35mm)))
 
    @property
    def vfov_deg(self) -> float:
        return 2 * math.degrees(math.atan(SENSOR_35MM_H / (2 * self.focal_length_in_35mm)))
 
    @property
    def is_nadir(self) -> bool:
        return abs(self.gimbal_pitch_deg + 90.0) < 5.0
 
    # ----- 위경도 → 미터 환산 (해당 위도) -----
    @property
    def m_per_deg_lon(self) -> float:
        return M_PER_DEG_LAT * math.cos(math.radians(self.gps.lat))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

# -----------------------------------------------------------------------------
# EXIF helpers
# -----------------------------------------------------------------------------


def _dms_to_decimal(dms: tuple, ref: str) -> float:
    """(deg, min, sec) + N/S/E/W → 십진 좌표."""
    deg, minutes, seconds = (float(x) for x in dms)
    value = deg + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        value = -value
    return value


def _parse_exif(img: Image.Image) -> dict[str, Any]:
    raw = img._getexif() or {}
    out: dict[str, Any] = {}
    for tag_id, value in raw.items():
        tag = TAGS.get(tag_id, tag_id)
        if tag == "GPSInfo":
            gps: dict[str, Any] = {}
            for k, v in value.items():
                gps[GPSTAGS.get(k, k)] = v
            out["GPSInfo"] = gps
        else:
            out[tag] = value
    return out


# -----------------------------------------------------------------------------
# XMP helpers
# -----------------------------------------------------------------------------


def _extract_xmp_block(image_path: Path) -> str:
    """JPG 바이너리에서 <x:xmpmeta ...> 블록만 잘라낸다."""
    data = image_path.read_bytes()
    start = data.find(b"<x:xmpmeta")
    end = data.find(b"</x:xmpmeta>")
    if start == -1 or end == -1:
        return ""
    return data[start : end + len(b"</x:xmpmeta>")].decode("utf-8", errors="ignore")


_ATTR_RE = re.compile(r'([\w\-]+:[\w\-]+)\s*=\s*"([^"]*)"')


def _parse_xmp_attrs(xmp_text: str) -> dict[str, str]:
    """drone-dji XMP는 속성(attribute) 형식이라 정규식으로 충분히 안전하게 파싱된다."""
    return {key: value for key, value in _ATTR_RE.findall(xmp_text)}


def _to_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value.lstrip("+"))
    except ValueError:
        return default


def _to_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value.lstrip("+")))
    except ValueError:
        return default


def estimate_intrinsics_from_metadata(metadata: ImageMetadata) -> tuple:
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

    f_pixels = (metadata.focal_length_in_35mm / 36.0) * metadata.width

    cx = metadata.width / 2.0
    cy = metadata.height / 2.0

    K = np.array([
        [f_pixels, 0,        cx],
        [0,        f_pixels, cy],
        [0,        0,        1.0]
    ])

    D = np.zeros(5)

    return K, D

# -----------------------------------------------------------------------------
# Core extractor
# -----------------------------------------------------------------------------


def _compute_id(image_path: Path) -> str:
    """파일 내용의 SHA-1 (DJI Terra/Smart Farm 계열에서 쓰는 컨벤션)."""
    h = hashlib.sha1()
    with image_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_metadata(
    image_path: str | Path,
    origin_path: str | None = None,
) -> ImageMetadata:
    """DJI JPG 한 장에서 표준 메타데이터를 추출한다.

    Parameters
    ----------
    image_path : 실제 디스크상의 파일 경로
    origin_path : 원본 캡처 경로 기록용 (없으면 image_path 그대로 사용)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    with Image.open(image_path) as img:
        width, height = img.size
        exif = _parse_exif(img)

    xmp_text = _extract_xmp_block(image_path)
    xmp = _parse_xmp_attrs(xmp_text)

    # ---- GPS (XMP 우선, fallback EXIF) -------------------------------------
    lat = _to_float(xmp.get("drone-dji:GpsLatitude"))
    lng = _to_float(xmp.get("drone-dji:GpsLongitude"))
    altitude = _to_float(xmp.get("drone-dji:AbsoluteAltitude"))

    if (lat == 0.0 or lng == 0.0) and "GPSInfo" in exif:
        gps = exif["GPSInfo"]
        lat = _dms_to_decimal(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
        lng = _dms_to_decimal(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
        altitude = float(gps.get("GPSAltitude", altitude))

    relative_height = _to_float(xmp.get("drone-dji:RelativeAltitude"))

    # ---- Flight (Yaw, Pitch, Roll) ------------------------------------
    yaw = _to_float(xmp.get("drone-dji:FlightYawDegree"))
    pitch = _to_float(xmp.get("drone-dji:FlightPitchDegree"))
    roll = _to_float(xmp.get("drone-dji:FlightRollDegree"))
    flight = [yaw, pitch, roll]

    # ---- Orientation (Yaw, Pitch, Roll) ------------------------------------
    yaw = _to_float(xmp.get("drone-dji:GimbalYawDegree"))
    pitch = _to_float(xmp.get("drone-dji:GimbalPitchDegree"))
    roll = _to_float(xmp.get("drone-dji:GimbalRollDegree"))
    orientation = [yaw, pitch, roll]

    # ---- RTK ----------------------------------------------------------------
    rtk_flag = _to_int(xmp.get("drone-dji:RtkFlag"))
    rtk_std = [
        _to_float(xmp.get("drone-dji:RtkStdLat")),
        _to_float(xmp.get("drone-dji:RtkStdLon")),
        _to_float(xmp.get("drone-dji:RtkStdHgt")),
    ]
    lrf = [
        _to_float(xmp.get("drone-dji:LRFTargetDistance")),
        _to_float(xmp.get("drone-dji:LRFTargetLat")),
        _to_float(xmp.get("drone-dji:LRFTargetLon")),
        _to_float(xmp.get("drone-dji:LRFTargetAbsAlt")),
    ]

    # pos_sigma 는 다른 단위로 들어가는 경우가 있어 별도 필드로 두지만,
    # H20T RTK 출력에서는 보통 [0.03, 0.03, 0.06] 처럼 고정 정밀도가 쓰인다.
    pos_sigma = [0.03, 0.03, 0.06]

    # ---- Velocity (m/s, body frame X/Y/Z) ----------------------------------
    velocity = [
        _to_float(xmp.get("drone-dji:FlightXSpeed")),
        _to_float(xmp.get("drone-dji:FlightYSpeed")),
        _to_float(xmp.get("drone-dji:FlightZSpeed")),
    ]

    # ---- Camera -------------------------------------------------------------
    camera_maker = str(exif.get("Make", xmp.get("tiff:Make", "")))
    base_model = str(exif.get("Model", xmp.get("tiff:Model", "")))
    image_source = xmp.get("drone-dji:ImageSource", "")
    camera_model = f"{base_model}_{image_source}" if image_source else base_model

    focal_length = float(exif.get("FocalLength", 0.0))
    focal_length_in_35mm = int(exif.get("FocalLengthIn35mmFilm", 0))
    camera_sn = str(exif.get("BodySerialNumber", ""))

    # ---- Capture time (epoch seconds, local tz from XMP) -------------------
    create_date = xmp.get("xmp:CreateDate") or xmp.get("xmp:ModifyDate")
    if create_date:
        # 예: "2025-12-17T13:02:00+09:00"
        capture_time = int(datetime.fromisoformat(create_date).timestamp())
    else:
        dt_str = exif.get("DateTimeOriginal") or exif.get("DateTime")
        capture_time = (
            int(datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S").timestamp())
            if dt_str else 0
        )

    image_id = _compute_id(image_path)
    final_origin_path = origin_path if origin_path else str(image_path)

    position = [lat, lng, altitude]

    return ImageMetadata(
        id=image_id,
        thumbnailPath="",
        path=f"./{image_id}.JPG",
        origin_path=final_origin_path,
        gps=GpsInfo(altitude=altitude, lat=lat, lng=lng),
        position=position,
        relative_height=relative_height,
        orientation=orientation,
        flight=flight,
        pos_sigma=pos_sigma,
        velocity=velocity,
        height=height,
        width=width,
        camera_model=camera_model,
        camera_maker=camera_maker,
        rtk_flag=rtk_flag,
        focal_length=focal_length,
        focal_length_in_35mm=focal_length_in_35mm,
        capture_time=capture_time,
        xmp=XmpInfo(
            cameraMaker=str(xmp.get("tiff:Make", camera_maker)),
            cameraModel=camera_model,
        ),
        camera_sn=camera_sn,
        lrf=lrf,
        rtk_std=rtk_std,
        pos_info=PosInfo(
            pos=position,
            pos_sigma=pos_sigma,
            orientation=orientation,
            id=image_id,
        ),
        rtk_active=rtk_flag in (16, 50),
    )

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
# 35mm 프레임 대각선 (mm).
#   FocalLengthIn35mmFilm(35mm 환산 초점거리)은 국제 표준(ISO 2720 / CIPA)상
#   "대각선 화각"을 기준으로 정의된 값이다. 따라서 가로·세로에 각각
#   36/24 를 독립적으로 적용하면 센서 종횡비가 3:2가 아닌 카메라
#   (H20T Zoom = 4:3)에서 fx ≠ fy 인 비물리적 이방성 초점거리가 나온다.
#   반드시 대각선 기준으로 픽셀 초점거리를 환산해야 fx = fy (등방)가 된다.
SENSOR_35MM_DIAG = math.hypot(SENSOR_35MM_W, SENSOR_35MM_H)  # ≈ 43.2666 mm

# 위경도 ↔ 미터 변환 (WGS84)
#   상수 M_PER_DEG_LAT 은 하위 호환을 위해 유지하되, 실제 계산은
#   위도별 자오선 곡률반경 기반 헬퍼(meters_per_deg_lat)를 쓰는 것을 권장한다.
#   적도~극 평균값 111_320 은 위도 34.7°에서 약 +0.35% 오차를 유발한다.
M_PER_DEG_LAT = 111_320.0

# WGS84 타원체 상수 (자오선/묘유선 곡률반경 계산용)
_WGS84_A = 6_378_137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2 - _WGS84_F)


def meters_per_deg_lat(lat_deg: float) -> float:
    """해당 위도에서 위도 1도당 미터 (자오선 곡률반경 M 기반)."""
    phi = math.radians(lat_deg)
    m = _WGS84_A * (1 - _WGS84_E2) / (1 - _WGS84_E2 * math.sin(phi) ** 2) ** 1.5
    return m * math.pi / 180.0


def meters_per_deg_lon(lat_deg: float) -> float:
    """해당 위도에서 경도 1도당 미터 (묘유선 곡률반경 N 기반)."""
    phi = math.radians(lat_deg)
    n = _WGS84_A / math.sqrt(1 - _WGS84_E2 * math.sin(phi) ** 2)
    return n * math.cos(phi) * math.pi / 180.0


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
    rtk_flag: int = 0
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

    # ----- 공장 캘리브레이션 (DJI XMP DewarpData) -----
    # DewarpData 가 있으면 fx, fy, cx, cy, 왜곡계수까지 실측값을 쓴다.
    # dewarp_fx/fy : 픽셀 초점거리 (native 해상도 기준)
    # dewarp_cx/cy : 주점의 '이미지 중심으로부터의 오프셋' (절대 좌표 아님!)
    # dewarp_dist  : [k1, k2, p1, p2, k3] (OpenCV 순서)
    dewarp_fx: Optional[float] = None
    dewarp_fy: Optional[float] = None
    dewarp_cx: Optional[float] = None   # offset from image center
    dewarp_cy: Optional[float] = None   # offset from image center
    dewarp_dist: list[float] = field(default_factory=lambda: [])

    # 캐시: 카메라 → ENU 회전 행렬
    R_cam_to_enu: np.ndarray = field(default=None, repr=False)

    @property
    def is_rtk_fixed(self) -> bool:
        return self.rtk_flag == RTK_FIXED

    # ----- 짐벌 자세 (orientation = [yaw, pitch, roll], degrees) -----
    # orientation 리스트 인덱싱을 곳곳에서 풀어 쓰지 않도록 프로퍼티로 노출.
    @property
    def gimbal_yaw_deg(self) -> float:
        return self.orientation[0]

    @property
    def gimbal_pitch_deg(self) -> float:
        return self.orientation[1]

    @property
    def gimbal_roll_deg(self) -> float:
        return self.orientation[2]

    # ----- LRF (Laser Range Finder) -----
    # lrf = [TargetDistance, TargetLat, TargetLon, TargetAbsAlt]
    # H20T 등 LRF 탑재 기종에서 촬영 시점 조준점의 실측 절대고도를 제공한다.
    # 평면 정사보정의 ground_z 추정에 altitude - relative_height 보다 훨씬 정확.
    @property
    def lrf_target_distance(self) -> float | None:
        return self.lrf[0] if len(self.lrf) >= 1 and self.lrf[0] > 0 else None

    @property
    def lrf_target_abs_alt(self) -> float | None:
        return self.lrf[3] if len(self.lrf) >= 4 and self.lrf[3] != 0 else None

    @property
    def has_valid_lrf(self) -> bool:
        return self.lrf_target_abs_alt is not None

    # ----- 캘리브레이션 유무 -----
    @property
    def has_dewarp(self) -> bool:
        """DewarpData 실측 캘리브레이션 값이 파싱되어 있는지."""
        return self.dewarp_fx is not None and self.dewarp_fy is not None

    # ----- 등방 픽셀 초점거리 (핵심 수정) -----
    @property
    def focal_px(self) -> float:
        """
        픽셀 단위 초점거리 (fx = fy, 등방).

        우선순위:
          1) DewarpData 실측값이 있으면 (fx + fy) / 2 사용
             (공장 캘리브레이션이므로 미세한 fx/fy 차이는 있으나 무시 가능한 수준)
          2) 없으면 FocalLengthIn35mmFilm 을 '대각선' 기준으로 픽셀 환산

               f_px = f35 * diag_px / diag_35mm

             이렇게 해야 4:3, 5:4 등 센서 종횡비와 무관하게 등방 f가 나온다.
        """
        if self.has_dewarp:
            return (self.dewarp_fx + self.dewarp_fy) / 2.0
        diag_px = math.hypot(self.width, self.height)
        return self.focal_length_in_35mm * diag_px / SENSOR_35MM_DIAG

    # ----- 주점 (principal point, 픽셀 절대 좌표) -----
    @property
    def principal_point_px(self) -> tuple[float, float]:
        """
        주점 (cx, cy) 픽셀 절대 좌표.
        DewarpData 가 있으면 '중심 + 오프셋', 없으면 이미지 중심으로 가정.
        """
        cx = self.width / 2.0
        cy = self.height / 2.0
        if self.has_dewarp:
            if self.dewarp_cx is not None:
                cx += self.dewarp_cx
            if self.dewarp_cy is not None:
                cy += self.dewarp_cy
        return cx, cy

    # ----- 동적 FOV (등방 f 기반) -----
    @property
    def hfov_deg(self) -> float:
        return 2 * math.degrees(math.atan(self.width / (2 * self.focal_px)))

    @property
    def vfov_deg(self) -> float:
        return 2 * math.degrees(math.atan(self.height / (2 * self.focal_px)))

    @property
    def is_nadir(self) -> bool:
        return abs(self.gimbal_pitch_deg + 90.0) < 5.0

    # ----- 위경도 → 미터 환산 (해당 위도, 곡률반경 기반) -----
    @property
    def m_per_deg_lat(self) -> float:
        return meters_per_deg_lat(self.gps.lat)

    @property
    def m_per_deg_lon(self) -> float:
        return meters_per_deg_lon(self.gps.lat)

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


def _parse_dewarp_data(raw: str | None) -> dict[str, Any]:
    """
    DJI XMP drone-dji:DewarpData 파싱.

    포맷 예 (H20T Zoom):
        "2025-12-17;3666.67,3666.67,12.34,-5.67,-0.12,0.03,0.0,0.0,0.0"
        날짜;fx,fy,cx,cy,k1,k2,p1,p2,k3

    - fx, fy : 픽셀 초점거리 (native 해상도 기준)
    - cx, cy : 주점의 '이미지 중심으로부터의 오프셋' (절대 좌표 아님)
    - k1,k2,p1,p2,k3 : 렌즈 왜곡계수 (OpenCV 순서)

    반환: {fx, fy, cx, cy, dist(list)} 또는 {} (파싱 실패 시).
    """
    if not raw or ";" not in raw:
        return {}
    try:
        payload = raw.split(";", 1)[1]
        vals = [float(v) for v in payload.split(",") if v.strip() != ""]
    except (ValueError, IndexError):
        return {}
    if len(vals) < 4:
        return {}
    fx, fy, cx, cy = vals[0], vals[1], vals[2], vals[3]
    dist = vals[4:9] if len(vals) >= 9 else vals[4:]
    # 왜곡계수는 최대 5개(k1,k2,p1,p2,k3)로 패딩
    dist = (dist + [0.0] * 5)[:5]
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "dist": dist}


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
    메타데이터에서 카메라 내부 파라미터 K, D 추정.

    우선순위:
      1) DewarpData 실측 캘리브레이션이 있으면 그대로 사용 (fx, fy, 주점 오프셋, 왜곡계수)
      2) 없으면 등방 초점거리(대각선 기준) + 주점=이미지 중심 + 무왜곡 가정

    원리 (fallback):
        FocalLengthIn35mmFilm 은 '대각선' 기준 35mm 환산 초점거리이므로
        가로폭(36mm)이 아니라 대각선(43.27mm)으로 픽셀 환산해야 등방 f가 나온다.

            f_px = f35 * diag_px / diag_35mm

        (기존 코드의 f_px = (f35/36)*width 는 4:3 센서에서 절대 스케일이
         약 +4% 커지는 버그가 있었다.)
    """
    if metadata.has_dewarp:
        cx, cy = metadata.principal_point_px
        K = np.array([
            [metadata.dewarp_fx, 0.0,               cx],
            [0.0,                metadata.dewarp_fy, cy],
            [0.0,                0.0,               1.0],
        ])
        D = np.array(metadata.dewarp_dist[:5], dtype=float)
        if D.size < 5:
            D = np.concatenate([D, np.zeros(5 - D.size)])
        return K, D

    f_px = metadata.focal_px  # 등방
    cx = metadata.width / 2.0
    cy = metadata.height / 2.0

    K = np.array([
        [f_px, 0.0,  cx],
        [0.0,  f_px, cy],
        [0.0,  0.0,  1.0],
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

    # ---- DewarpData (공장 캘리브레이션) -------------------------------------
    #   drone-dji:DewarpFlag 가 1이면 왜곡보정 정보가 유효.
    #   DewarpData 를 파싱해 fx, fy, 주점 오프셋, 왜곡계수를 얻는다.
    dewarp_flag_raw = _to_int(xmp.get("drone-dji:DewarpFlag"), default=1)
    dewarp = _parse_dewarp_data(xmp.get("drone-dji:DewarpData"))

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
        dewarp_flag=bool(dewarp_flag_raw),
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
        dewarp_fx=dewarp.get("fx"),
        dewarp_fy=dewarp.get("fy"),
        dewarp_cx=dewarp.get("cx"),
        dewarp_cy=dewarp.get("cy"),
        dewarp_dist=dewarp.get("dist", []),
        pos_info=PosInfo(
            pos=position,
            pos_sigma=pos_sigma,
            orientation=orientation,
            id=image_id,
        ),
        rtk_active=rtk_flag in (16, 50),
    )
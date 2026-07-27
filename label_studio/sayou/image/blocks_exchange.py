from dataclasses import dataclass
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

from pyproj import CRS, Transformer

from .metadata import ImageMetadata, extract_metadata

# -----------------------------------------------------------------------------
# Sensor specs (image-only photogrammetry로는 알 수 없으므로 카메라별 LUT)
# -----------------------------------------------------------------------------
 
# H20T Zoom 카메라: 1/2.3" CMOS, 5184x3888, sensor width ≈ 6.17mm → pixel size
# = 6.17 / 5184 ≈ 1.190 µm. 다만 ZoomCamera 실효 픽셀피치는 ImageSource 메타와
# 35mm 환산 비율(47/10.14 ≈ 4.635)로 역산하는 게 더 정확하다.
#   FocalLengthPixels = focal_mm / pixel_size_mm
#                     = focal_mm * (image_width / sensor_width_mm)
# 35mm 환산: equivalent_focal_35mm = focal_mm * (36mm / sensor_width_mm)
#  → sensor_width_mm = focal_mm * 36 / equivalent_focal_35mm
 
# 카메라별 실측 sensor width (mm) — DJI Terra/ContextCapture 캘리브레이션과
# 일치하도록 튜닝된 값. 35mm 환산 역산은 카메라 펌웨어가 보정 전 광학값을
# 보고하기 때문에 실제 픽셀피치와 어긋날 수 있어, LUT를 우선 사용한다.
_CAMERA_SENSOR_WIDTH_MM: dict[str, float] = {
    # H20T Zoom: 1/1.7" CMOS 실측, DJI Terra 캘리브레이션 결과(FLpx≈7715 @ f=10.14, w=5184)
    # 역산: sensor_width = 10.14 / (7715.6 / 5184) ≈ 6.812mm
    "ZH20T_ZoomCamera": 6.812,
    # H20T Wide: 1/2.3" CMOS, sensor width ≈ 6.17mm (참고값, 캘리브레이션 미검증)
    "ZH20T_WideCamera": 6.17,
    # H20T Infrared: 640x512 Vox microbolometer, 12µm 픽셀 → 7.68mm
    "ZH20T_InfraredCamera": 7.68,
}
 
_SENSOR_WIDTH_MM_CACHE: dict[tuple[float, int], float] = {}
 
 
def _sensor_width_mm(focal_mm: float, focal_35mm: int) -> float:
    """초점거리 + 35mm 환산값에서 센서 가로 길이를 역산 (LUT 폴백)."""
    if focal_35mm <= 0:
        return 6.17  # 1/2.3" 안전 폴백
    key = (focal_mm, focal_35mm)
    if key not in _SENSOR_WIDTH_MM_CACHE:
        _SENSOR_WIDTH_MM_CACHE[key] = focal_mm * 36.0 / focal_35mm
    return _SENSOR_WIDTH_MM_CACHE[key]
 
 
def compute_focal_length_pixels(meta: ImageMetadata) -> float:
    """카메라 메타데이터에서 ``FocalLengthPixels`` 계산.
 
    카메라 모델별 실측 센서폭(LUT) 을 우선 사용하고, 미등록 카메라는
    35mm 환산값으로 역산한다.
    """
    sensor_w = _CAMERA_SENSOR_WIDTH_MM.get(meta.camera_model)
    if sensor_w is None:
        sensor_w = _sensor_width_mm(meta.focal_length, meta.focal_length_in_35mm)
    pixel_size_mm = sensor_w / meta.width
    return meta.focal_length / pixel_size_mm
 
 
# -----------------------------------------------------------------------------
# WGS84 → UTM projection
# -----------------------------------------------------------------------------
 
 
def utm_epsg_from_wgs84(lat: float, lng: float) -> int:
    """위경도에서 UTM EPSG 코드를 자동 산출.
 
    북반구는 326xx, 남반구는 327xx. zone = floor((lng + 180) / 6) + 1.
    한국 (124~132°E): 51N (32651) ~ 52N (32652).
    """
    zone = int((lng + 180.0) / 6.0) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone
 
 
@dataclass
class SRSInfo:
    """BlocksExchange ``SpatialReferenceSystems`` 항목."""
 
    srs_id: int
    name: str
    definition: str  # "EPSG:32652" 형식
 
    @classmethod
    def from_epsg(cls, epsg: int, srs_id: int = 0) -> "SRSInfo":
        crs = CRS.from_epsg(epsg)
        return cls(srs_id=srs_id, name=crs.name, definition=f"EPSG:{epsg}")
 
 
# -----------------------------------------------------------------------------
# Gimbal YPR → Omega/Phi/Kappa
# -----------------------------------------------------------------------------
 
 
def gimbal_to_opk(yaw: float, pitch: float, roll: float) -> tuple[float, float, float]:
    """DJI Gimbal Yaw/Pitch/Roll → 사진측량 Omega/Phi/Kappa (degrees).
 
    DJI Gimbal 규약:
        - Yaw: 북쪽 기준 시계방향 (0=N, 90=E), 카메라 광축의 방위각
        - Pitch: 수평=0, nadir(직하방)=-90, 위쪽=+
        - Roll: 카메라 광축 기준 회전
 
    사진측량(ContextCapture/Pix4D) Omega/Phi/Kappa 규약:
        - 카메라 좌표계: x=오른쪽, y=아래, z=광축(앞)
        - 월드 좌표계: X=동, Y=북, Z=위
        - Omega: X축(동) 기준 회전, Phi: Y축(북) 기준, Kappa: Z축(위) 기준
 
    Nadir 캡처 (pitch=-90, roll=0) 의 경우 Omega≈180, Phi≈0, Kappa=yaw.
    """
    # 사진측량 표준식 (DJI Gimbal → OPK).
    # 정확한 변환은 회전행렬 R_camera_to_world 를 합성한 뒤 OPK로 분해해야 하지만,
    # nadir 부근 캡처(태양광 패널 조사)에서는 아래 근사식이 ContextCapture 결과와
    # 1° 이내로 일치한다. 더 큰 짐벌 기울기에서 정밀하게 쓰려면 전체 회전행렬
    # 분해(_opk_from_rotation_matrix) 경로를 사용한다.
    omega = 180.0 + roll
    phi = -(pitch + 90.0)
    kappa = yaw
 
    # [-180, 180] 정규화. ±180°의 경계값은 BlocksExchange 컨벤션상 +180°
    # 으로 출력하는 게 자연스러우므로 -180.0 이 나오면 +180.0 으로 뒤집는다.
    def _norm(angle: float) -> float:
        a = ((angle + 180.0) % 360.0) - 180.0
        return 180.0 if a == -180.0 else a
 
    return _norm(omega), _norm(phi), _norm(kappa)
 
 
def _opk_from_rotation_matrix(R: list[list[float]]) -> tuple[float, float, float]:
    """3x3 회전행렬에서 Omega/Phi/Kappa (deg) 분해.
 
    R = Rx(omega) * Ry(phi) * Rz(kappa) 규약 (ContextCapture 표준).
    근접 nadir가 아닌 일반각에 대비한 정밀 경로용 헬퍼.
    """
    phi = math.asin(-R[2][0])
    cos_phi = math.cos(phi)
    if abs(cos_phi) < 1e-9:  # gimbal lock
        omega = math.atan2(-R[1][2], R[1][1])
        kappa = 0.0
    else:
        omega = math.atan2(R[2][1], R[2][2])
        kappa = math.atan2(R[1][0], R[0][0])
    return math.degrees(omega), math.degrees(phi), math.degrees(kappa)
 
 
# -----------------------------------------------------------------------------
# BlocksExchange 데이터 모델
# -----------------------------------------------------------------------------
 
 
@dataclass
class PhotoPose:
    omega: float
    phi: float
    kappa: float
    x: float  # projected (e.g. UTM easting)
    y: float  # projected (e.g. UTM northing)
    z: float  # altitude (meters)
 
 
@dataclass
class PhotoEntry:
    photo_id: int
    image_path: str  # Photogroup에 기재될 상대/절대 경로
    pose: PhotoPose
 
 
@dataclass
class Photogroup:
    name: str
    width: int
    height: int
    focal_length_pixels: float
    principal_point_x: float
    principal_point_y: float
    aspect_ratio: float = 1.0
    skew: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    camera_model_type: str = "Perspective"
    photos: list[PhotoEntry] = None  # type: ignore[assignment]
 
    def __post_init__(self) -> None:
        if self.photos is None:
            self.photos = []
 
 
# -----------------------------------------------------------------------------
# Image → PhotoEntry
# -----------------------------------------------------------------------------
 
 
def _photogroup_key(meta: ImageMetadata) -> tuple:
    """동일한 내부 파라미터를 가진 사진들을 묶기 위한 키."""
    return (meta.camera_model, meta.width, meta.height,
            meta.focal_length, meta.focal_length_in_35mm)
 
def _image_path_for_xml(
    origin_path: str,
    mode: str,
    relative_to: Path | None,
) -> str:
    """XML 내부에 기록할 ImagePath 결정."""
    if mode == "absolute":
        return str(Path(origin_path).resolve())
    if mode == "relative" and relative_to is not None:
        return str(Path(origin_path).resolve().relative_to(relative_to.resolve()))
    # 기본: 파일명만 (DJI Terra 표준)
    return Path(origin_path).name

def image_to_pose(
    meta: ImageMetadata,
    transformer: Transformer,
) -> PhotoPose:
    """이미지 메타 → 투영 좌표 + OPK."""
    lat, lng, alt = meta.gps.lat, meta.gps.lng, meta.gps.altitude
    x, y = transformer.transform(lng, lat)
    yaw, pitch, roll = meta.orientation  # YPR
    omega, phi, kappa = gimbal_to_opk(yaw, pitch, roll)
    return PhotoPose(omega=omega, phi=phi, kappa=kappa, x=x, y=y, z=alt)
 
 
# -----------------------------------------------------------------------------
# XML serialization
# -----------------------------------------------------------------------------
 
 
def _make_element(parent: ET.Element, tag: str, text: object = None) -> ET.Element:
    el = ET.SubElement(parent, tag)
    if text is not None:
        el.text = str(text)
    return el
 
 
def build_blocks_exchange_xml(
    srs: SRSInfo,
    photogroups: list[Photogroup],
    block_name: str = "DJI AT Default: Block 1",
    block_description: str = "Result of Aero Triangulation of Block 1",
    version: str = "3.2",
) -> str:
    """``BlocksExchange`` XML 문자열 생성."""
    root = ET.Element("BlocksExchange", attrib={"version": version})
 
    # SpatialReferenceSystems
    srs_root = _make_element(root, "SpatialReferenceSystems")
    srs_el = _make_element(srs_root, "SRS")
    _make_element(srs_el, "Id", srs.srs_id)
    _make_element(srs_el, "Name", srs.name)
    _make_element(srs_el, "Definition", srs.definition)
 
    # Block
    block = _make_element(root, "Block")
    _make_element(block, "Name", block_name)
    _make_element(block, "Description", block_description)
    _make_element(block, "SRSId", srs.srs_id)
 
    pgs_root = _make_element(block, "Photogroups")
    for pg in photogroups:
        pg_el = _make_element(pgs_root, "Photogroup")
        _make_element(pg_el, "Name", pg.name)
 
        dims = _make_element(pg_el, "ImageDimensions")
        _make_element(dims, "Width", pg.width)
        _make_element(dims, "Height", pg.height)
 
        _make_element(pg_el, "CameraModelType", pg.camera_model_type)
        _make_element(pg_el, "FocalLengthPixels", repr(pg.focal_length_pixels))
 
        pp = _make_element(pg_el, "PrincipalPoint")
        _make_element(pp, "x", repr(pg.principal_point_x))
        _make_element(pp, "y", repr(pg.principal_point_y))
 
        dist = _make_element(pg_el, "Distortion")
        _make_element(dist, "K1", pg.k1)
        _make_element(dist, "K2", pg.k2)
        _make_element(dist, "K3", pg.k3)
        _make_element(dist, "P1", pg.p1)
        _make_element(dist, "P2", pg.p2)
 
        _make_element(pg_el, "AspectRatio", repr(pg.aspect_ratio))
        _make_element(pg_el, "Skew", pg.skew)
 
        for photo in pg.photos:
            ph_el = _make_element(pg_el, "Photo")
            _make_element(ph_el, "Id", photo.photo_id)
            _make_element(ph_el, "ImagePath", photo.image_path)
            pose_el = _make_element(ph_el, "Pose")
            rot_el = _make_element(pose_el, "Rotation")
            _make_element(rot_el, "Omega", repr(photo.pose.omega))
            _make_element(rot_el, "Phi", repr(photo.pose.phi))
            _make_element(rot_el, "Kappa", repr(photo.pose.kappa))
            ctr_el = _make_element(pose_el, "Center")
            _make_element(ctr_el, "x", repr(photo.pose.x))
            _make_element(ctr_el, "y", repr(photo.pose.y))
            _make_element(ctr_el, "z", repr(photo.pose.z))
 
    # Pretty-print with explicit XML declaration
    rough = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(rough).toprettyxml(indent="    ", encoding="utf-8")
    return pretty.decode("utf-8")
 
 #

def build_blocks_exchange(
    metas: list[tuple],
    epsg: int | None = None,
    block_name: str = "DJI AT Default: Block 1",
    block_description: str = "Result of Aero Triangulation of Block 1",
    image_path_in_xml: str = "name",  # "name" | "absolute" | "relative"
    relative_to: str | Path | None = None,
    include_ir: bool = False,
) -> str:
    if not metas:
        raise ValueError("처리할 사진이 없습니다 (include_ir=True 옵션 확인).")
 
    # 좌표계 결정
    if epsg is None:
        first_meta = metas[0][1]
        epsg = utm_epsg_from_wgs84(first_meta.gps.lat, first_meta.gps.lng)
    srs = SRSInfo.from_epsg(epsg)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    # Photogroup 묶기 + Photo 생성
    groups: dict[tuple, Photogroup] = {}
    photo_counter = 0
    rel_root = Path(relative_to) if relative_to else None
 
    for path, meta in metas:
        key = _photogroup_key(meta)
        if key not in groups:
            pg_index = len(groups) + 1
            groups[key] = Photogroup(
                name=str(pg_index),
                width=meta.width,
                height=meta.height,
                focal_length_pixels=compute_focal_length_pixels(meta),
                principal_point_x=meta.width / 2.0,
                principal_point_y=meta.height / 2.0,
                aspect_ratio=1.0,
            )
 
        pose = image_to_pose(meta, transformer)
        groups[key].photos.append(PhotoEntry(
            photo_id=photo_counter,
            #image_path=_image_path_for_xml(meta.origin_path, image_path_in_xml, rel_root),
            image_path=_image_path_for_xml(path, image_path_in_xml, rel_root),
            pose=pose,
        ))
        photo_counter += 1
 
    return build_blocks_exchange_xml(
        srs=srs,
        photogroups=list(groups.values()),
        block_name=block_name,
        block_description=block_description,
    )
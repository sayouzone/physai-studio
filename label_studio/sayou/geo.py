"""
python sayou/geo.py \
  --rgb-path ~/Development/sayouzone/solar-thermal/data/solar/에스엘에너지_사천시/RGB \
  --rgb-label-path ~/Development/sayouzone/solar-thermal/workspace/predict_sl_sacheon_l/predicted_labels \
  --output-dir ~/Development/sayouzone/solar-thermal/workspace/georeferencing/sl_sacheon
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict, Any

import numpy as np

from image.metadata import ImageMetadata, extract_metadata, estimate_intrinsics_from_metadata

from georeferencing.dji.camera_pose import compute_camera_axes_from_gimbal, verify_nadir_orientation
from georeferencing.dji.metadata import DJIMetadata, M_PER_DEG_LAT, extract_dji_metadata#, estimate_intrinsics_from_metadata
from georeferencing.dji.georeferencer import DJIImageGeoreferencer
from georeferencing.dji.coordinates import geodetic_to_enu, GeodeticPoint
from georeferencing.yolo_to_geo import (
    YOLODetection,
    GeoreferencedDetection,
    parse_yolo_label_file,
    export_to_geojson,
    export_to_csv,
    _compute_polygon_area_m2,
)

SOLAR_RGB_CLASSES = {
    0: "pv",
    1: "anomaly",
}

SOLAR_IR_CLASSES = {
    0: "panel",
    1: "hotspot",
}

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

@dataclass
class LabelImageScale:
    """
    YOLO 라벨이 사용한 정규화 기준 이미지 크기 vs 실제 이미지 크기
 
    예: 라벨은 640×640 (stretch)에서 만들었지만 실제 IR은 640×512
    """
    label_ref_width: int
    label_ref_height: int
    actual_width: int
    actual_height: int
 
    @property
    def needs_correction(self) -> bool:
        return (
            self.label_ref_width != self.actual_width
            or self.label_ref_height != self.actual_height
        )

# --------------------------------------------------------------------------- #
# 3. 픽셀 → 지면 투영 (광선-평면 교차)
# --------------------------------------------------------------------------- #

def _pixel_to_camera_ray(
        px: float, 
        py: float, 
        metadata: ImageMetadata
    ) -> np.ndarray:
    """
    픽셀(좌상단 원점) → 카메라 좌표계 단위 광선 (X=Right, Y=Down, Z=Forward).

    초점거리(픽셀 단위) f_px = (W/2) / tan(HFOV/2)
    image center를 (cx, cy) = (W/2, H/2)로 가정.
    """

    """
    초점거리(픽셀 단위) f_px = (W/2) / tan(HFOV/2)
    FOV 대신 EXIF FocalLengthIn35mmFilm 기반으로 계산.
    DJI H20T Zoom 카메라는 매 사진마다 줌 비율이 달라 FOV도 달라짐.
    따라서 pose.hfov_deg, pose.vfov_deg는 EXIF 기반으로 계산된 동적 FOV입니다.
    """
    fx_px = (metadata.width / 2.0) / math.tan(math.radians(metadata.hfov_deg / 2.0))
    fy_px = (metadata.height / 2.0) / math.tan(math.radians(metadata.vfov_deg / 2.0))

    """
    image center를 (cx, cy) = (W/2, H/2)로 가정.
    OpenCV pinhole: x_cam = (px - cx)/fx, y_cam = (py - cy)/fy, z_cam = 1
    따라서 cx, cy는 이미지 중심의 픽셀 좌표입니다.
    """
    cx_px = metadata.width / 2.0
    cy_px = metadata.height / 2.0

    # OpenCV pinhole: x_cam = (px - cx)/fx, y_cam = (py - cy)/fy, z_cam = 1
    ray_cam = np.array([
        (px - cx_px) / fx_px,
        (py - cy_px) / fy_px,
        1.0,
    ])
    return ray_cam / np.linalg.norm(ray_cam)

def _pixel_to_ground_enu(
    px: float, py: float,
    metadata: DJIMetadata,
    ground_z_below_drone: float | None = None,
) -> tuple[float, float]:
    """
    픽셀 → 카메라 nadir 발끝 기준 (East, North) 미터 오프셋.

    ground_z_below_drone : 드론에서 본 지면의 z 값(ENU에서 -AGL).
                           기본 None이면 -pose.rel_alt_m.
                           DEM 쓰면 픽셀별로 달리 줄 수 있음.
    """
    if ground_z_below_drone is None:
        """드론에서 본 지면의 z 값(ENU에서 -AGL)"""
        ground_z_below_drone = -metadata.relative_height   # 드론보다 AGL만큼 아래

    # (1) 카메라 좌표 광선 → ENU 광선
    ray_cam = _pixel_to_camera_ray(px, py, metadata)
    ray_enu = metadata.R_cam_to_enu @ ray_cam        # 3-vec in ENU

    # (2) 평면 z = ground_z_below_drone 와의 교차
    #     drone 위치를 원점(0,0,0)이라 두면 광선식: P = t * ray_enu
    #     ground_z_below_drone = t * ray_enu[2]
    if ray_enu[2] >= -1e-9:
        # 광선이 지면을 만나지 않거나 위로 향함 (잘못된 입력)
        raise ValueError(
            f"광선이 지면을 향하지 않음 (gimbal pitch={metadata.gimbal_pitch_deg}°). "
            f"ray_enu={ray_enu.tolist()}"
        )
    t = ground_z_below_drone / ray_enu[2]
    east  = t * ray_enu[0]
    north = t * ray_enu[1]
    return east, north


def _enu_to_lonlat(east_m: float, north_m: float, metadata: DJIMetadata) -> list[float]:
    """ENU 오프셋 → (lon, lat) 좌표. pose의 (lon, lat)을 기준으로 미터 단위 오프셋을 도 단위로 변환하여 더한다."""
    return [
        metadata.gps.lng + east_m / metadata.m_per_deg_lon,
        metadata.gps.lat + north_m / M_PER_DEG_LAT,
    ]

# --------------------------------------------------------------------------- #
# 4. Public API: bbox / YOLO label → GeoJSON
# --------------------------------------------------------------------------- #

def pixel_bbox_to_polygon(
    bbox_xyxy: Sequence[float], metadata: DJIMetadata,
) -> list[list[float]]:
    """픽셀 bbox(x1,y1,x2,y2) → GeoJSON Polygon ring [[lon,lat], ...]"""
    x1, y1, x2, y2 = bbox_xyxy
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    ring = []
    for px, py in corners:
        e, n = _pixel_to_ground_enu(px, py, metadata)
        ring.append(_enu_to_lonlat(e, n, metadata))
    return ring

def correct_yolo_for_stretch(
    detection: YOLODetection,
    scale: LabelImageScale,
) -> YOLODetection:
    """
    Stretch된 라벨을 원본 이미지 정규화 좌표로 변환
 
    핵심 인사이트:
        라벨 정규화 좌표는 stretch 이미지(label_ref) 기준으로 만들어졌음.
        정규화 좌표는 stretch 이미지에서 "객체가 차지하는 비율"을 표현.
 
        같은 객체가 native 이미지에서는 다른 비율을 차지함.
 
        예: 640x640 stretch에서 객체 cy=0.218 (픽셀 y=139.5)
            이건 native 640x512에서 픽셀 y=111.6에 해당
            native에서 정규화: 111.6 / 512 = 0.218 ← 같은 값!
 
        하지만 객체 height (h)는 다름:
            stretch 640x640에서 h=0.436 (픽셀 height=279)
            native 640x512에서 같은 객체 픽셀 height=279 × (512/640)=223
            native 정규화: 223/512 = 0.436 ← 같은 값!
 
        결론: stretch가 단순 비율 stretching이면 정규화 좌표는 같다.
        실제 보정은 픽셀 공간으로 변환할 때 일어난다.
 
    실제로 차이가 발생하는 경우:
        Roboflow의 'fit' 모드 (letterbox, padding) 사용 시
        crop 또는 rotate 적용 시
        일부만 stretch 적용 시
    """
    cx_px_label = detection.cx_norm * scale.label_ref_width
    cy_px_label = detection.cy_norm * scale.label_ref_height
    w_px_label = detection.w_norm * scale.label_ref_width
    h_px_label = detection.h_norm * scale.label_ref_height
 
    scale_x = scale.actual_width / scale.label_ref_width
    scale_y = scale.actual_height / scale.label_ref_height
 
    cx_px_native = cx_px_label * scale_x
    cy_px_native = cy_px_label * scale_y
    w_px_native = w_px_label * scale_x
    h_px_native = h_px_label * scale_y
 
    cx_norm_native = cx_px_native / scale.actual_width
    cy_norm_native = cy_px_native / scale.actual_height
    w_norm_native = w_px_native / scale.actual_width
    h_norm_native = h_px_native / scale.actual_height
 
    cx_norm_native = max(0, min(1, cx_norm_native))
    cy_norm_native = max(0, min(1, cy_norm_native))
    w_norm_native = max(0, min(1, w_norm_native))
    h_norm_native = max(0, min(1, h_norm_native))
 
    return YOLODetection(
        class_id=detection.class_id,
        cx_norm=cx_norm_native,
        cy_norm=cy_norm_native,
        w_norm=w_norm_native,
        h_norm=h_norm_native,
        confidence=detection.confidence,
    )
 
 
def convert_ir_yolo_to_geo(
    detection: YOLODetection,
    georeferencer: DJIImageGeoreferencer,
    label_scale: Optional[LabelImageScale] = None,
    class_names: Optional[Dict[int, str]] = None,
    edge_threshold: float = 0.05,
) -> Optional[GeoreferencedDetection]:
    """
    IR YOLO 검출 → Georeferenced 검출
 
    label_scale가 주어지면 stretch 보정 적용
    """
    if label_scale and label_scale.needs_correction:
        detection = correct_yolo_for_stretch(detection, label_scale)
 
    metadata = georeferencer.metadata
    W, H = metadata.width, metadata.height
 
    pixel_bbox = detection.to_pixel_xyxy(W, H)
    pixel_corners = detection.to_pixel_corners(W, H)
    pixel_center = detection.to_pixel_center(W, H)
 
    geo_corners = []
    for px in pixel_corners:
        geo = georeferencer.pixel_to_geodetic(px)
        if geo is None:
            return None
        geo_corners.append(geo)
 
    geo_center = georeferencer.pixel_to_geodetic(pixel_center)
    if geo_center is None:
        return None
 
    origin = georeferencer.origin
    enu_corners = [geodetic_to_enu(c, origin).to_array()[:2] for c in geo_corners]
 
    width_m = (
        np.linalg.norm(enu_corners[1] - enu_corners[0])
        + np.linalg.norm(enu_corners[2] - enu_corners[3])
    ) / 2
    height_m = (
        np.linalg.norm(enu_corners[3] - enu_corners[0])
        + np.linalg.norm(enu_corners[2] - enu_corners[1])
    ) / 2
    area_m2 = _compute_polygon_area_m2(geo_corners, origin)
 
    cx_norm, cy_norm = detection.cx_norm, detection.cy_norm
    dist_from_center = np.sqrt((cx_norm - 0.5) ** 2 + (cy_norm - 0.5) ** 2)
 
    x_edge = min(cx_norm, 1 - cx_norm)
    y_edge = min(cy_norm, 1 - cy_norm)
    near_edge = (
        x_edge < edge_threshold
        or y_edge < edge_threshold
        or detection.w_norm > 1 - 2 * edge_threshold
        or detection.h_norm > 1 - 2 * edge_threshold
    )
 
    class_name = (
        class_names.get(detection.class_id, f"class_{detection.class_id}")
        if class_names
        else f"class_{detection.class_id}"
    )
 
    return GeoreferencedDetection(
        class_id=detection.class_id,
        class_name=class_name,
        confidence=detection.confidence,
        image_width=W,
        image_height=H,
        pixel_bbox_xyxy=pixel_bbox,
        pixel_center=pixel_center,
        geo_corners=geo_corners,
        geo_center=geo_center,
        width_meters=float(width_m),
        height_meters=float(height_m),
        area_m2=float(area_m2),
        avg_distance_from_image_center_norm=float(dist_from_center),
        near_image_edge=near_edge,
    )

def extract_rgb_geo(image_path: str, yolo_label: str, output_path: str) -> DJIMetadata:
    print_section("1. EXIF + XMP 메타데이터 추출 (Georeferencer 준비)")
    #metadata = extract_dji_metadata(image_path)
    metadata = extract_metadata(image_path)

    # 짐벌 자세 → 카메라 회전 행렬 (위임)
    axes = compute_camera_axes_from_gimbal(
        gimbal_yaw_compass_deg=metadata.gimbal_yaw_deg,
        gimbal_pitch_deg=metadata.gimbal_pitch_deg,
        gimbal_roll_deg=metadata.gimbal_roll_deg,
    )
    R = axes['R_camera_to_enu']
    metadata.R_cam_to_enu=R

    print()
    print(f"  이미지: {Path(image_path).name}")
    print(f"  크기: {metadata.width} × {metadata.height}")
    print(f"  촬영 시각: {metadata.capture_time}")
    print(f"  카메라: ZH20T (focal {metadata.focal_length:.2f}mm, 35mm 환산 {metadata.focal_length_in_35mm}mm)")
    print(f"  드론 자세: yaw={metadata.flight[0]:.2f}°, pitch={metadata.flight[1]:.2f}°, roll={metadata.flight[2]:.2f}°")
    print(f"  짐벌 자세: yaw={metadata.gimbal_yaw_deg:.2f}°, pitch={metadata.gimbal_pitch_deg:.2f}°, roll={metadata.gimbal_roll_deg:.2f}°")
    print(f"  GPS: ({metadata.gps.lat:.7f}, {metadata.gps.lng:.7f})")
    print(f"  고도: 절대 {metadata.gps.altitude:.2f}m / 상대 {metadata.relative_height:.2f}m")
    if metadata.rtk_flag and metadata.rtk_flag >= 50:
        print(f"  RTK-GPS 활성: (정확도 매우 높음)")
    if metadata.lrf_target_distance:
        print(f"  LRF 측정: 거리 {metadata.lrf_target_distance:.2f}m, 타깃 고도 {metadata.lrf_target_abs_alt:.2f}m")

    print_section("2. 카메라 내부 파라미터 추정")
    K, D = estimate_intrinsics_from_metadata(metadata)
    print(f"  K =\n{K}")
    print(f"  D = {D}")
    print(f"  주의: EXIF 기반 추정값. 정확한 값은 캘리브레이션 필요")

    print_section("3. Georeferencer 초기화")
    gr = DJIImageGeoreferencer(metadata)
    print(f"  좌표 원점: ({gr.origin.latitude:.7f}, {gr.origin.longitude:.7f})")
    print(f"  카메라 위치 (ENU): {gr.camera_position_enu}")
    print(f"  지면 고도: {gr.ground_altitude:.2f}m")
    print(f"  지면이 카메라 아래로: {-gr.ground_up_in_enu:.2f}m")
    print(f"  R_camera_to_enu =\n{gr.R_camera_to_enu}")

    print_section("2. YOLO 라벨 파일 파싱")
    detections = parse_yolo_label_file(yolo_label, has_confidence=False)
    print(f"  파일: {Path(yolo_label).name}")
    print(f"  총 객체 수: {len(detections)}")
    print()
    print(f"  {'#':<3} {'class':<20} {'cx':<10} {'cy':<10} {'w':<10} {'h':<10}")
    print(f"  {'-'*3} {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for i, det in enumerate(detections):
        cls_name = SOLAR_RGB_CLASSES.get(det.class_id, f"class_{det.class_id}")
        print(f"  {i:<3} {cls_name:<20} {det.cx_norm:<10.4f} {det.cy_norm:<10.4f} "
              f"{det.w_norm:<10.4f} {det.h_norm:<10.4f}")
    
    print_section("3. YOLO → 픽셀 좌표 변환")
    print(f"  {'#':<3} {'class':<20} {'pixel bbox (x1,y1,x2,y2)':<40}")
    print(f"  {'-'*3} {'-'*20} {'-'*40}")
    extracted_bboxes = []
    for i, det in enumerate(detections):
        cls_name = SOLAR_RGB_CLASSES.get(det.class_id, f"class_{det.class_id}")
        x1, y1, x2, y2 = det.to_pixel_xyxy(metadata.width, metadata.height)
        print(f"  {i:<3} {cls_name:<20} ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        extracted_bboxes.append((f"({cls_name} ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})", int(x1), int(y1), int(x2), int(y2)))

    print_section("4. Nadir 정렬 검증")
    nadir = gr.georeference_full_image().nadir_check
    print(f"  카메라 광축의 ENU 방향: {nadir['z_axis_in_enu']}")
    print(f"  Nadir(직하방)에서 벗어난 각도: {nadir['angle_from_nadir_deg']:.3f}°")
    print(f"  Nadir 촬영 여부: {nadir['is_nadir']}")

    print_section("5. 이미지 모서리의 지리 좌표")
    corners = gr.compute_image_corners()
    labels = ["좌상단", "우상단", "우하단", "좌하단"]
    for label, corner in zip(labels, corners):
        print(f"  {label}: lat={corner.latitude:.7f}, lon={corner.longitude:.7f}")

    print_section("6. 이미지 중심 = 드론 바로 아래?")
    cx = metadata.width // 2
    cy = metadata.height // 2
    center_geo = gr.pixel_to_geodetic((cx, cy))
    print(f"  이미지 중심 픽셀: ({cx}, {cy})")
    print(f"  계산된 지리 좌표: ({center_geo.latitude:.7f}, {center_geo.longitude:.7f})")
    print(f"  드론 GPS:        ({metadata.gps.lat:.7f}, {metadata.gps.lng:.7f})")

    drone_pos = GeodeticPoint(metadata.gps.lat, metadata.gps.lng, gr.ground_altitude)
    center_enu = geodetic_to_enu(center_geo, drone_pos).to_array()
    horizontal_offset = np.linalg.norm(center_enu[:2])
    print(f"  수평 오프셋: {horizontal_offset:.3f}m")
    print(f"  (Nadir 촬영이면 0에 가까워야 함. 짐벌 미세 기울기로 약간 발생 정상)")

    print_section("7. LRF 측정값과 검증")
    lrf_check = gr.validate_with_lrf()
    if lrf_check and lrf_check.get("status") == "ok":
        print(f"  계산된 중심: ({lrf_check['computed']['lat']:.7f}, {lrf_check['computed']['lon']:.7f})")
        print(f"  LRF 측정값:  ({lrf_check['lrf_measured']['lat']:.7f}, {lrf_check['lrf_measured']['lon']:.7f})")
        print(f"  수평 오차: {lrf_check['error_horizontal_m']:.3f}m")
        print(f"  LRF 거리: {lrf_check['lrf_distance_m']:.2f}m")
        print(f"  → 오차가 1m 이내면 georeferencing 정확도 우수")
    else:
        print("  LRF 데이터 없음")

    print_section("8. Ground Sample Distance와 커버리지")
    result = gr.georeference_full_image()
    print(f"  GSD: {result.ground_sample_distance_m * 1000:.2f} mm/pixel")
    print(f"  커버리지: {result.coverage_area_m2:.1f} m²")
    print(f"  → 한 셀(156mm)이 약 {156 / (result.ground_sample_distance_m * 1000):.1f} 픽셀")

    print_section("9. 검출 박스 georeferencing 예시")

    for name, x, y, w, h in extracted_bboxes:
        bbox = (x, y, x + w, y + h)
        center = gr.bbox_center_to_geodetic(bbox)
        corners = gr.bbox_to_geodetic(bbox)
        if center and corners:
            print(f"\n  {name}")
            print(f"    중심: ({center.latitude:.7f}, {center.longitude:.7f})")
            print(f"    물리 크기: {gr.compute_ground_sample_distance() * w * 100:.1f} cm × "
                  f"{gr.compute_ground_sample_distance() * h * 100:.1f} cm")

    print_section("10. GeoJSON 출력")
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)
    #geojson_path = output_dir / "rgb_georeferenced.geojson"
    geojson_path = output_dir / f"{Path(image_path).stem}.geojson"

    features = []

    # nadir 검증 (경고만, 차단은 안 함)
    nadir_check = verify_nadir_orientation(metadata.R_cam_to_enu, tolerance_deg=10.0)
    is_oblique = not nadir_check['is_nadir']

    # (a) image footprint
    footprint_ring = pixel_bbox_to_polygon(
        (0, 0, metadata.width, metadata.height), metadata,
    )

    features.append({
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [footprint_ring]},
        "properties": {
            "name": "image_coverage",
            "image_path": metadata.origin_path,
            "camera_model": metadata.camera_model,
            "focal_35mm_eq": metadata.focal_length_in_35mm,
            "hfov_deg": round(metadata.hfov_deg, 2),
            "vfov_deg": round(metadata.vfov_deg, 2),
            "rel_alt_m": metadata.relative_height,
            "gimbal_pitch_deg": metadata.gimbal_pitch_deg,
            "gimbal_yaw_compass_deg": metadata.gimbal_yaw_deg,
            "gimbal_roll_deg": metadata.gimbal_roll_deg,
            "oblique_view": is_oblique,
            "angle_from_nadir_deg": round(nadir_check["angle_from_nadir_deg"], 2),
        }
    })

    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [metadata.gps.lng, metadata.gps.lat]
        },
        "properties": {
            "name": "drone_position",
            "rel_altitude_m": metadata.relative_height,
            "abs_altitude_m": metadata.gps.altitude,
            "rtk_active": metadata.rtk_active,
        }
    })

    for i, det in enumerate(detections):
        cls_name = SOLAR_RGB_CLASSES.get(det.class_id, f"class_{det.class_id}")
        x1, y1, x2, y2 = det.to_pixel_xyxy(metadata.width, metadata.height)
        ring = pixel_bbox_to_polygon((x1, y1, x2, y2), metadata)
        print(ring)

        # 지면상 크기 (참고용)
        e_corners = [_pixel_to_ground_enu(px, py, metadata)
                     for px, py in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]
        es = [c[0] for c in e_corners]; ns = [c[1] for c in e_corners]
        width_m  = max(es) - min(es)
        height_m = max(ns) - min(ns)

        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "class_id": det.class_id,
                "class_name": cls_name,
                "pixel_bbox": [round(x1, 1), round(y1, 1),
                               round(x2, 1), round(y2, 1)],
                "width_m": round(width_m, 3),
                "height_m": round(height_m, 3),
                "area_m2": round(width_m * height_m, 3),
                "type": "detection_box",
            },
        }

        features.append(feature)

    geojson = {
        "type": "FeatureCollection", 
        "features": features,
        "metadata": {
            "image_path": image_path,
            "modality": "rgb",
            "camera": f"ZH20T_{metadata.camera_model.capitalize()}",
            "image_native_size": [metadata.width, metadata.height],
            "focal_35mm_eq": metadata.focal_length_in_35mm,
            "hfov_deg": round(metadata.hfov_deg, 4),
            "vfov_deg": round(metadata.vfov_deg, 4),
            "capture_time": metadata.capture_time,
            "rtk_active": metadata.rtk_active,
            "gimbal": {
                "yaw_compass_deg": metadata.gimbal_yaw_deg,
                "pitch_deg": metadata.gimbal_pitch_deg,
                "roll_deg": metadata.gimbal_roll_deg,
            },
        },
    }
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"  GeoJSON 저장: {geojson_path}")
    print(f"  → geojson.io 또는 QGIS에서 열어서 시각 확인 가능")

    return metadata

def get_rgb_geo(image_path: str, yolo_label: str) -> Tuple[DJIMetadata, dict]:
    print_section("1. EXIF + XMP 메타데이터 추출 (Georeferencer 준비)")
    #metadata = extract_dji_metadata(image_path)
    metadata = extract_metadata(image_path)

    # 짐벌 자세 → 카메라 회전 행렬 (위임)
    axes = compute_camera_axes_from_gimbal(
        gimbal_yaw_compass_deg=metadata.gimbal_yaw_deg,
        gimbal_pitch_deg=metadata.gimbal_pitch_deg,
        gimbal_roll_deg=metadata.gimbal_roll_deg,
    )
    R = axes['R_camera_to_enu']
    metadata.R_cam_to_enu=R

    print()
    print(f"  이미지: {Path(image_path).name}")
    print(f"  크기: {metadata.width} × {metadata.height}")
    print(f"  촬영 시각: {metadata.capture_time}")
    print(f"  카메라: ZH20T (focal {metadata.focal_length:.2f}mm, 35mm 환산 {metadata.focal_length_in_35mm}mm)")
    print(f"  드론 자세: yaw={metadata.flight[0]:.2f}°, pitch={metadata.flight[1]:.2f}°, roll={metadata.flight[2]:.2f}°")
    print(f"  짐벌 자세: yaw={metadata.gimbal_yaw_deg:.2f}°, pitch={metadata.gimbal_pitch_deg:.2f}°, roll={metadata.gimbal_roll_deg:.2f}°")
    print(f"  GPS: ({metadata.gps.lat:.7f}, {metadata.gps.lng:.7f})")
    print(f"  고도: 절대 {metadata.gps.altitude:.2f}m / 상대 {metadata.relative_height:.2f}m")
    if metadata.rtk_flag and metadata.rtk_flag >= 50:
        print(f"  RTK-GPS 활성: (정확도 매우 높음)")
    if metadata.lrf_target_distance:
        print(f"  LRF 측정: 거리 {metadata.lrf_target_distance:.2f}m, 타깃 고도 {metadata.lrf_target_abs_alt:.2f}m")

    print_section("2. 카메라 내부 파라미터 추정")
    K, D = estimate_intrinsics_from_metadata(metadata)
    print(f"  K =\n{K}")
    print(f"  D = {D}")
    print(f"  주의: EXIF 기반 추정값. 정확한 값은 캘리브레이션 필요")

    print_section("3. Georeferencer 초기화")
    gr = DJIImageGeoreferencer(metadata)
    print(f"  좌표 원점: ({gr.origin.latitude:.7f}, {gr.origin.longitude:.7f})")
    print(f"  카메라 위치 (ENU): {gr.camera_position_enu}")
    print(f"  지면 고도: {gr.ground_altitude:.2f}m")
    print(f"  지면이 카메라 아래로: {-gr.ground_up_in_enu:.2f}m")
    print(f"  R_camera_to_enu =\n{gr.R_camera_to_enu}")

    print_section("2. YOLO 라벨 파일 파싱")
    detections = parse_yolo_label_file(yolo_label, has_confidence=False)
    print(f"  파일: {Path(yolo_label).name}")
    print(f"  총 객체 수: {len(detections)}")
    print()
    print(f"  {'#':<3} {'class':<20} {'cx':<10} {'cy':<10} {'w':<10} {'h':<10}")
    print(f"  {'-'*3} {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for i, det in enumerate(detections):
        cls_name = SOLAR_RGB_CLASSES.get(det.class_id, f"class_{det.class_id}")
        print(f"  {i:<3} {cls_name:<20} {det.cx_norm:<10.4f} {det.cy_norm:<10.4f} "
              f"{det.w_norm:<10.4f} {det.h_norm:<10.4f}")
    
    print_section("3. YOLO → 픽셀 좌표 변환")
    print(f"  {'#':<3} {'class':<20} {'pixel bbox (x1,y1,x2,y2)':<40}")
    print(f"  {'-'*3} {'-'*20} {'-'*40}")
    extracted_bboxes = []
    for i, det in enumerate(detections):
        cls_name = SOLAR_RGB_CLASSES.get(det.class_id, f"class_{det.class_id}")
        x1, y1, x2, y2 = det.to_pixel_xyxy(metadata.width, metadata.height)
        print(f"  {i:<3} {cls_name:<20} ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        extracted_bboxes.append((f"({cls_name} ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})", int(x1), int(y1), int(x2), int(y2)))

    print_section("4. Nadir 정렬 검증")
    nadir = gr.georeference_full_image().nadir_check
    print(f"  카메라 광축의 ENU 방향: {nadir['z_axis_in_enu']}")
    print(f"  Nadir(직하방)에서 벗어난 각도: {nadir['angle_from_nadir_deg']:.3f}°")
    print(f"  Nadir 촬영 여부: {nadir['is_nadir']}")

    print_section("5. 이미지 모서리의 지리 좌표")
    corners = gr.compute_image_corners()
    labels = ["좌상단", "우상단", "우하단", "좌하단"]
    for label, corner in zip(labels, corners):
        print(f"  {label}: lat={corner.latitude:.7f}, lon={corner.longitude:.7f}")

    print_section("6. 이미지 중심 = 드론 바로 아래?")
    cx = metadata.width // 2
    cy = metadata.height // 2
    center_geo = gr.pixel_to_geodetic((cx, cy))
    print(f"  이미지 중심 픽셀: ({cx}, {cy})")
    print(f"  계산된 지리 좌표: ({center_geo.latitude:.7f}, {center_geo.longitude:.7f})")
    print(f"  드론 GPS:        ({metadata.gps.lat:.7f}, {metadata.gps.lng:.7f})")

    drone_pos = GeodeticPoint(metadata.gps.lat, metadata.gps.lng, gr.ground_altitude)
    center_enu = geodetic_to_enu(center_geo, drone_pos).to_array()
    horizontal_offset = np.linalg.norm(center_enu[:2])
    print(f"  수평 오프셋: {horizontal_offset:.3f}m")
    print(f"  (Nadir 촬영이면 0에 가까워야 함. 짐벌 미세 기울기로 약간 발생 정상)")

    print_section("7. LRF 측정값과 검증")
    lrf_check = gr.validate_with_lrf()
    if lrf_check and lrf_check.get("status") == "ok":
        print(f"  계산된 중심: ({lrf_check['computed']['lat']:.7f}, {lrf_check['computed']['lon']:.7f})")
        print(f"  LRF 측정값:  ({lrf_check['lrf_measured']['lat']:.7f}, {lrf_check['lrf_measured']['lon']:.7f})")
        print(f"  수평 오차: {lrf_check['error_horizontal_m']:.3f}m")
        print(f"  LRF 거리: {lrf_check['lrf_distance_m']:.2f}m")
        print(f"  → 오차가 1m 이내면 georeferencing 정확도 우수")
    else:
        print("  LRF 데이터 없음")

    print_section("8. Ground Sample Distance와 커버리지")
    result = gr.georeference_full_image()
    print(f"  GSD: {result.ground_sample_distance_m * 1000:.2f} mm/pixel")
    print(f"  커버리지: {result.coverage_area_m2:.1f} m²")
    print(f"  → 한 셀(156mm)이 약 {156 / (result.ground_sample_distance_m * 1000):.1f} 픽셀")

    print_section("9. 검출 박스 georeferencing 예시")

    for name, x, y, w, h in extracted_bboxes:
        bbox = (x, y, x + w, y + h)
        center = gr.bbox_center_to_geodetic(bbox)
        corners = gr.bbox_to_geodetic(bbox)
        if center and corners:
            print(f"\n  {name}")
            print(f"    중심: ({center.latitude:.7f}, {center.longitude:.7f})")
            print(f"    물리 크기: {gr.compute_ground_sample_distance() * w * 100:.1f} cm × "
                  f"{gr.compute_ground_sample_distance() * h * 100:.1f} cm")

    print_section("10. GeoJSON 생성")
    features = []

    # nadir 검증 (경고만, 차단은 안 함)
    nadir_check = verify_nadir_orientation(metadata.R_cam_to_enu, tolerance_deg=10.0)
    is_oblique = not nadir_check['is_nadir']

    # (a) image footprint
    footprint_ring = pixel_bbox_to_polygon(
        (0, 0, metadata.width, metadata.height), metadata,
    )

    features.append({
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [footprint_ring]},
        "properties": {
            "name": "image_coverage",
            "image_path": metadata.origin_path,
            "camera_model": metadata.camera_model,
            "focal_35mm_eq": metadata.focal_length_in_35mm,
            "hfov_deg": round(metadata.hfov_deg, 2),
            "vfov_deg": round(metadata.vfov_deg, 2),
            "rel_alt_m": metadata.relative_height,
            "gimbal_pitch_deg": metadata.gimbal_pitch_deg,
            "gimbal_yaw_compass_deg": metadata.gimbal_yaw_deg,
            "gimbal_roll_deg": metadata.gimbal_roll_deg,
            "oblique_view": is_oblique,
            "angle_from_nadir_deg": round(nadir_check["angle_from_nadir_deg"], 2),
        }
    })

    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [metadata.gps.lng, metadata.gps.lat]
        },
        "properties": {
            "name": "drone_position",
            "rel_altitude_m": metadata.relative_height,
            "abs_altitude_m": metadata.gps.altitude,
            "rtk_active": metadata.rtk_active,
        }
    })

    for i, det in enumerate(detections):
        cls_name = SOLAR_RGB_CLASSES.get(det.class_id, f"class_{det.class_id}")
        x1, y1, x2, y2 = det.to_pixel_xyxy(metadata.width, metadata.height)
        ring = pixel_bbox_to_polygon((x1, y1, x2, y2), metadata)
        print(ring)

        # 지면상 크기 (참고용)
        e_corners = [_pixel_to_ground_enu(px, py, metadata)
                     for px, py in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]
        es = [c[0] for c in e_corners]; ns = [c[1] for c in e_corners]
        width_m  = max(es) - min(es)
        height_m = max(ns) - min(ns)

        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "class_id": det.class_id,
                "class_name": cls_name,
                "pixel_bbox": [round(x1, 1), round(y1, 1),
                               round(x2, 1), round(y2, 1)],
                "width_m": round(width_m, 3),
                "height_m": round(height_m, 3),
                "area_m2": round(width_m * height_m, 3),
                "type": "detection_box",
            },
        }

        features.append(feature)

    geojson = {
        "type": "FeatureCollection", 
        "features": features,
        "metadata": {
            "image_path": image_path,
            "modality": "rgb",
            "camera": f"ZH20T_{metadata.camera_model.capitalize()}",
            "image_native_size": [metadata.width, metadata.height],
            "focal_35mm_eq": metadata.focal_length_in_35mm,
            "hfov_deg": round(metadata.hfov_deg, 4),
            "vfov_deg": round(metadata.vfov_deg, 4),
            "capture_time": metadata.capture_time,
            "rtk_active": metadata.rtk_active,
            "gimbal": {
                "yaw_compass_deg": metadata.gimbal_yaw_deg,
                "pitch_deg": metadata.gimbal_pitch_deg,
                "roll_deg": metadata.gimbal_roll_deg,
            },
        },
    }

    return metadata, geojson

def _to_pixel(image_width: int, image_height: int) -> Tuple[float, float, float, float]:
    """정규화 (cx, cy, w, h) → 픽셀 (x1, y1, x2, y2)"""
    cx = 0.5 * image_width
    cy = 0.5 * image_height
    w = 1.0 * image_width
    h = 1.0 * image_height
    
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)
    
    return (x1, y1, x2, y2)

def _to_pixel_corners(image_width: int, image_height: int) -> List[Tuple[float, float]]:
    """4개 모서리 픽셀 좌표 (좌상→우상→우하→좌하)"""
    x1, y1, x2, y2 = _to_pixel(image_width, image_height)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def extract_ir_geo(image_path: str, yolo_label: str, output_path: str) -> ImageMetadata:
    print_section("1. IR 이미지 메타데이터 추출")
    #metadata = extract_dji_metadata(image_path)
    metadata = extract_metadata(image_path)

    # 짐벌 자세 → 카메라 회전 행렬 (위임)
    axes = compute_camera_axes_from_gimbal(
        gimbal_yaw_compass_deg=metadata.gimbal_yaw_deg,
        gimbal_pitch_deg=metadata.gimbal_pitch_deg,
        gimbal_roll_deg=metadata.gimbal_roll_deg,
    )
    R = axes['R_camera_to_enu']
    metadata.R_cam_to_enu=R

    lrf_distance, lrf_lat, lrf_lon, lrf_abs_alt = metadata.lrf

    print(f"  파일: {Path(image_path).name}")
    print(f"  네이티브 크기: {metadata.width} × {metadata.height}")
    print(f"  카메라: ZH20T Thermal (35mm 환산 {metadata.focal_length_in_35mm}mm)")
    print(f"  GPS: ({metadata.gps.lat:.7f}, {metadata.gps.lng:.7f})")
    print(f"  짐벌 자세: yaw={metadata.gimbal_yaw_deg:.2f}°, pitch={metadata.gimbal_pitch_deg:.2f}°, roll={metadata.gimbal_roll_deg:.2f}°")
    print(f"  RTK: {metadata.rtk_flag and metadata.rtk_flag >= 50}")
    print(f"  LRF 거리: {lrf_distance:.2f}m")
    print(f"  지면 고도: {lrf_abs_alt:.2f}m")
 
    print_section("2. 라벨 좌표계 분석")
    print(f"  라벨 정규화 기준: 640 × 640 (Roboflow stretch)")
    print(f"  원본 IR 이미지:   {metadata.width} × {metadata.height} (native)")
 
    label_scale = LabelImageScale(
        label_ref_width=640,
        label_ref_height=640,
        actual_width=metadata.width,
        actual_height=metadata.height,
    )
    print(f"  스케일 보정 필요: {label_scale.needs_correction}")
    print(f"  Y 압축 비율: {label_scale.actual_height / label_scale.label_ref_height:.4f}")

    print_section("2. YOLO 라벨 파일 파싱")
    detections = parse_yolo_label_file(yolo_label, has_confidence=False)
    print(f"  파일: {Path(yolo_label).name}")
    print(f"  총 객체 수: {len(detections)}")
    print()
    print(f"  {'#':<3} {'class':<20} {'cx':<10} {'cy':<10} {'w':<10} {'h':<10}")
    print(f"  {'-'*3} {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for i, det in enumerate(detections):
        cls_name = SOLAR_RGB_CLASSES.get(det.class_id, f"class_{det.class_id}")
        print(f"  {i:<3} {cls_name:<20} {det.cx_norm:<10.4f} {det.cy_norm:<10.4f} "
              f"{det.w_norm:<10.4f} {det.h_norm:<10.4f}")
 
    print_section("3. Stretch 보정 전후 좌표 비교")
    print(f"  {'#':<3} {'class':<20} {'before (640×640)':<25} {'after (native)':<25}")
    print(f"  {'-'*3} {'-'*20} {'-'*25} {'-'*25}")
 
    corrected_labels = []
    for i, det in enumerate(detections):
        cls_name = SOLAR_IR_CLASSES.get(det.class_id, f"c{det.class_id}")
        corrected = correct_yolo_for_stretch(det, label_scale)
        corrected_labels.append(corrected)
        before = f"({det.cx_norm:.3f},{det.cy_norm:.3f},{det.w_norm:.3f},{det.h_norm:.3f})"
        after = f"({corrected.cx_norm:.3f},{corrected.cy_norm:.3f},{corrected.w_norm:.3f},{corrected.h_norm:.3f})"
        print(f"  {i:<3} {cls_name:<20} {before:<25} {after:<25}")
 
    print_section("4. Georeferencer 초기화")
    gr = DJIImageGeoreferencer(metadata)
    K, _ = estimate_intrinsics_from_metadata(metadata)
    print(f"  K[0,0]: {K[0,0]:.1f} pixels")
    print(f"  GSD: {gr.compute_ground_sample_distance() * 1000:.2f} mm/pixel")
    print(f"  커버리지: {gr.compute_coverage_area():.1f} m²")

    # nadir 검증 (경고만, 차단은 안 함)
    nadir_check = verify_nadir_orientation(metadata.R_cam_to_enu, tolerance_deg=10.0)
    is_oblique = not nadir_check['is_nadir']

    # (a) image footprint
    pixel_corners = _to_pixel_corners(metadata.width, metadata.height)

    geo_corners = []
    for px in pixel_corners:
        geo = gr.pixel_to_geodetic(px)
        if geo is None:
            return None
        geo_corners.append([geo.longitude, geo.latitude])
    geo_corners.append(geo_corners[0])

    cover_detection = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon", 
            "coordinates": [geo_corners]
        },
        "properties": {
            "name": "image_coverage",
            "image_path": metadata.origin_path,
            "camera_model": metadata.camera_model,
            "focal_35mm_eq": metadata.focal_length_in_35mm,
            "hfov_deg": round(metadata.hfov_deg, 2),
            "vfov_deg": round(metadata.vfov_deg, 2),
            "rel_alt_m": metadata.relative_height,
            "gimbal_pitch_deg": metadata.gimbal_pitch_deg,
            "gimbal_yaw_compass_deg": metadata.gimbal_yaw_deg,
            "gimbal_roll_deg": metadata.gimbal_roll_deg,
            "oblique_view": is_oblique,
            "angle_from_nadir_deg": round(nadir_check["angle_from_nadir_deg"], 2),
        }
    }

    drone_position = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [metadata.gps.lng, metadata.gps.lat]
        },
        "properties": {
            "name": "drone_position",
            "rel_altitude_m": metadata.relative_height,
            "abs_altitude_m": metadata.gps.altitude,
            "rtk_active": metadata.rtk_active,
        }
    }

    print_section("5. IR YOLO → Georeferenced 변환")
    geo_detections = []
    
    for i, det in enumerate(detections):
        geo_det = convert_ir_yolo_to_geo(
            det,
            georeferencer=gr,
            label_scale=label_scale,
            class_names=SOLAR_IR_CLASSES,
        )
        if geo_det:
            geo_detections.append(geo_det)
 
            print(f"\n  [{i}] {geo_det.class_name}")
            print(f"      픽셀 bbox (native): "
                  f"({geo_det.pixel_bbox_xyxy[0]:.0f}, {geo_det.pixel_bbox_xyxy[1]:.0f}, "
                  f"{geo_det.pixel_bbox_xyxy[2]:.0f}, {geo_det.pixel_bbox_xyxy[3]:.0f})")
            print(f"      지리 중심: ({geo_det.geo_center.latitude:.7f}, "
                  f"{geo_det.geo_center.longitude:.7f})")
            print(f"      4개 모서리:")
            for lbl, c in zip(["TL", "TR", "BR", "BL"], geo_det.geo_corners):
                print(f"        {lbl}: ({c.latitude:.7f}, {c.longitude:.7f})")
            print(f"      물리 크기: {geo_det.width_meters:.2f}m × {geo_det.height_meters:.2f}m")
            print(f"      면적: {geo_det.area_m2:.2f} m²")
 

    print_section("6. 결과 검증: 패널 박스 일관성")
    panel_dets = [d for d in geo_detections if d.class_id == 1]
    if panel_dets:
        widths = [d.width_meters for d in panel_dets]
        heights = [d.height_meters for d in panel_dets]
        print(f"  패널 박스 {len(panel_dets)}개:")
        print(f"    너비 (m): min={min(widths):.2f}, max={max(widths):.2f}, mean={np.mean(widths):.2f}")
        print(f"    높이 (m): min={min(heights):.2f}, max={max(heights):.2f}, mean={np.mean(heights):.2f}")
        if max(widths) - min(widths) < 0.5:
            print(f"  ✓ 너비가 일관됨 (편차 < 50cm) → 변환이 올바름")
 
    print_section("7. 출력 파일 생성")
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)
 
    geojson_path = output_dir / f"{Path(image_path).stem}.geojson"
    csv_path = output_dir / "ir_yolo_georeferenced.csv"
 
    export_to_geojson(
        cover_detection,
        drone_position,
        geo_detections,
        str(geojson_path),
        image_metadata={
            "image_path": image_path,
            "modality": "infrared",
            "camera": "ZH20T_Thermal",
            "label_source_size": [label_scale.label_ref_width, label_scale.label_ref_height],
            "image_native_size": [metadata.width, metadata.height],
            "stretch_correction_applied": label_scale.needs_correction,
            "capture_time": metadata.capture_time,
            "rtk_active": metadata.rtk_flag and metadata.rtk_flag >= 50,
        },
    )
    export_to_csv(geo_detections, str(csv_path))
 
    print(f"  GeoJSON: {geojson_path}")
    print(f"  CSV: {csv_path}")

    return metadata

def main():
    ap = argparse.ArgumentParser(description="라벨 데이터를 지도에 매핑")
    ap.add_argument("--rgb-path",         default="data/solar/환경관리/RGB")
    ap.add_argument("--rgb-label-path",   default="workspace/labels_s200_l_2_d")
    ap.add_argument("--ir-path",          default="data/solar/환경관리/TM")
    ap.add_argument("--ir-label-path",    default="workspace/ir_rf")
    ap.add_argument("--output-dir",  default="workspace/claude/output")

    args = ap.parse_args()

    rgb_image_dir = Path(args.rgb_path)
    rgb_label_dir = Path(args.rgb_label_path)
    ir_image_dir = Path(args.ir_path)
    ir_label_dir = Path(args.ir_label_path)

    rgb_path = args.rgb_path + "/DJI_20251217130217_0007_Z.JPG"
    ir_path = args.ir_path + "/DJI_20251217130217_0007_T.JPG"
    rgb_yolo_label = args.rgb_label_path + "/DJI_20251217130217_0007_Z.txt"
    ir_yolo_label = args.ir_label_path + "/DJI_20251217130217_0007_T_JPG.rf.a9de1d4093d48a9aff0dcb47ad3fb589.txt"
    output_path = args.output_dir
    image_path = rgb_path
    yolo_label = rgb_yolo_label

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # RGB Georeferencing 추출
    rgb_images = sorted(rgb_image_dir.glob("*.JPG"))

    print(rgb_images)
    for p in rgb_images:
        label_filename = p.stem + ".txt"
        print(label_filename)
        rgb_labels = sorted(rgb_label_dir.glob(label_filename))
        if len(rgb_labels) >= 1:
            #extract_rgb_geo(image_path = str(p), yolo_label = str(rgb_labels[0]), output_path = output_path)
            metadata, geojson = get_rgb_geo(image_path = str(p), yolo_label = str(rgb_labels[0]))
            print(metadata, geojson)

    """
    # IR Georeferencing 추출
    ir_images = sorted(ir_image_dir.glob("*.JPG"))

    for p in ir_images:
        label_filename = p.stem + "*.txt"
        ir_labels = sorted(ir_label_dir.glob(label_filename))
        if len(ir_labels) >= 1:
            extract_ir_geo(image_path = str(p), yolo_label = str(ir_labels[0]), output_path = output_path)
    """

    #rgb_metadata = extract_rgb_geo(image_path = rgb_path, yolo_label = rgb_yolo_label, output_path = output_path)
    #ir_metadata = extract_ir_geo(image_path = ir_path, yolo_label = ir_yolo_label, output_path = output_path)

if __name__ == "__main__":
    main()

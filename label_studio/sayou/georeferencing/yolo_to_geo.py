"""
YOLO bbox 데이터 → Georeferenced bbox 변환

YOLO 형식:
    class_id cx_norm cy_norm w_norm h_norm [confidence]
    - 모든 좌표는 [0, 1] 범위 (이미지 크기로 정규화)
    - 박스는 축 정렬 (axis-aligned)
    - 한 줄에 한 객체

Georeferenced 형식:
    - 4개 모서리의 (lat, lon) — 일반적으로 사다리꼴
    - 중심점 (lat, lon)
    - 운영 메타데이터 (패널 ID, 물리 크기, 신뢰도 등)
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import json

import numpy as np

from sayou.georeferencing.coordinates import GeodeticPoint, geodetic_to_enu
from sayou.georeferencing.dji.georeferencer import DJIImageGeoreferencer


@dataclass
class YOLODetection:
    """YOLO 형식의 검출 결과 또는 라벨"""
    region_id: str
    class_id: int
    cx_norm: float
    cy_norm: float
    w_norm: float
    h_norm: float
    confidence: Optional[float] = None
    
    def __post_init__(self):
        for name, value in [
            ("cx_norm", self.cx_norm), ("cy_norm", self.cy_norm),
            ("w_norm", self.w_norm), ("h_norm", self.h_norm)
        ]:
            if not 0 <= value <= 1:
                raise ValueError(
                    f"YOLO {name} must be in [0,1], got {value}. "
                    f"이미 픽셀 좌표일 가능성이 있습니다."
                )
    
    def to_pixel_xyxy(self, image_width: int, image_height: int) -> Tuple[float, float, float, float]:
        """정규화 (cx, cy, w, h) → 픽셀 (x1, y1, x2, y2)"""
        cx = self.cx_norm * image_width
        cy = self.cy_norm * image_height
        w = self.w_norm * image_width
        h = self.h_norm * image_height
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)
        
        return (x1, y1, x2, y2)
    
    def to_pixel_corners(self, image_width: int, image_height: int) -> List[Tuple[float, float]]:
        """4개 모서리 픽셀 좌표 (좌상→우상→우하→좌하)"""
        x1, y1, x2, y2 = self.to_pixel_xyxy(image_width, image_height)
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    def to_pixel_center(self, image_width: int, image_height: int) -> Tuple[float, float]:
        return (self.cx_norm * image_width, self.cy_norm * image_height)


def parse_yolo_label_file(
    label_path: str,
    has_confidence: bool = False
) -> List[YOLODetection]:
    """
    YOLO 라벨 파일 파싱 (파일 읽기 + 파싱을 조합하는 진입점)
    
    has_confidence:
        False: 학습용 라벨 (class cx cy w h)
        True: 추론 결과 (class cx cy w h confidence)
    """
    lines = read_label_lines(label_path)
    detections = parse_yolo_label_lines(lines, has_confidence=has_confidence)
    
    return detections

def read_label_lines(label_path: str) -> List[str]:
    """
    라벨 파일을 읽어서 주석/빈 줄을 제외한 원본 라인 리스트를 반환
    (파일 존재 여부 체크 및 I/O만 담당)
    """
    path = Path(label_path)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with open(path) as f:
        lines = f.readlines()

    return lines


def parse_yolo_label_lines(
    lines: List[str],
    has_confidence: bool = False
) -> List[YOLODetection]:
    """
    라인 리스트를 YOLODetection 객체 리스트로 파싱
    (파일 I/O에 의존하지 않는 순수 파싱 로직)

    has_confidence:
        False: 학습용 라벨 (class cx cy w h)
        True: 추론 결과 (class cx cy w h confidence)
    """
    detections = []

    for line_num, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) < 5:
            raise ValueError(
                f"{line_num}: 최소 5개 값이 필요합니다 (got {len(parts)})"
            )

        try:
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 and has_confidence else None

            detections.append(YOLODetection(
                region_id=None,
                class_id=class_id,
                cx_norm=cx, cy_norm=cy,
                w_norm=w, h_norm=h,
                confidence=conf
            ))
        except (ValueError, IndexError) as e:
            raise ValueError(f"{line_num} 파싱 실패: {e}") from e

    return detections

@dataclass
class GeoreferencedDetection:
    """지리 좌표로 변환된 검출 결과"""
    class_id: int
    class_name: str
    confidence: Optional[float]
    
    image_width: int
    image_height: int
    pixel_bbox_xyxy: Tuple[float, float, float, float]
    pixel_center: Tuple[float, float]
    
    geo_corners: List[GeodeticPoint]
    geo_center: GeodeticPoint
    
    width_meters: float
    height_meters: float
    area_m2: float
    
    avg_distance_from_image_center_norm: float
    near_image_edge: bool
    
    panel_id: Optional[str] = None
    cell_position: Optional[Tuple[int, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["geo_corners"] = [
            {"lat": p.latitude, "lon": p.longitude, "alt": p.altitude}
            for p in self.geo_corners
        ]
        result["geo_center"] = {
            "lat": self.geo_center.latitude,
            "lon": self.geo_center.longitude,
            "alt": self.geo_center.altitude,
        }
        return result
    
    def to_geojson_feature(self) -> Dict[str, Any]:
        """GeoJSON Feature (Polygon)로 변환"""
        coords = [[p.longitude, p.latitude] for p in self.geo_corners]
        coords.append([self.geo_corners[0].longitude, self.geo_corners[0].latitude])
        
        properties = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "width_m": round(self.width_meters, 3),
            "height_m": round(self.height_meters, 3),
            "area_m2": round(self.area_m2, 3),
            "near_edge": self.near_image_edge,
            "pixel_bbox": [round(v, 1) for v in self.pixel_bbox_xyxy],
        }
        if self.panel_id:
            properties["panel_id"] = self.panel_id
        if self.cell_position:
            properties["cell_row"] = self.cell_position[0]
            properties["cell_col"] = self.cell_position[1]
        
        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": properties,
        }


def _compute_polygon_area_m2(corners_geo: List[GeodeticPoint], origin: GeodeticPoint) -> float:
    """4개 지리 좌표로 구성된 다각형 면적 (m²) - ENU 평면 근사"""
    enu = [geodetic_to_enu(c, origin).to_array()[:2] for c in corners_geo]
    x1, y1 = enu[0]
    x2, y2 = enu[1]
    x3, y3 = enu[2]
    x4, y4 = enu[3]
    return 0.5 * abs(
        (x1 * y2 - x2 * y1) +
        (x2 * y3 - x3 * y2) +
        (x3 * y4 - x4 * y3) +
        (x4 * y1 - x1 * y4)
    )


def convert_yolo_to_geo(
    detection: YOLODetection,
    georeferencer: DJIImageGeoreferencer,
    class_names: Optional[Dict[int, str]] = None,
    edge_threshold: float = 0.05,
) -> Optional[GeoreferencedDetection]:
    """
    YOLO 검출 → Georeferenced 검출
    
    Returns:
        GeoreferencedDetection 또는 None (변환 실패 시)
    """
    metadata = georeferencer.metadata
    W, H = metadata.image_width, metadata.image_height
    
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
        np.linalg.norm(enu_corners[1] - enu_corners[0]) +
        np.linalg.norm(enu_corners[2] - enu_corners[3])
    ) / 2
    height_m = (
        np.linalg.norm(enu_corners[3] - enu_corners[0]) +
        np.linalg.norm(enu_corners[2] - enu_corners[1])
    ) / 2
    area_m2 = _compute_polygon_area_m2(geo_corners, origin)
    
    cx_norm, cy_norm = detection.cx_norm, detection.cy_norm
    dist_from_center = np.sqrt((cx_norm - 0.5) ** 2 + (cy_norm - 0.5) ** 2)
    
    x_edge = min(cx_norm, 1 - cx_norm)
    y_edge = min(cy_norm, 1 - cy_norm)
    near_edge = (
        x_edge < edge_threshold or
        y_edge < edge_threshold or
        detection.w_norm > 1 - 2 * edge_threshold or
        detection.h_norm > 1 - 2 * edge_threshold
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


def convert_yolo_file_to_geo(
    label_path: str,
    georeferencer: DJIImageGeoreferencer,
    class_names: Optional[Dict[int, str]] = None,
    has_confidence: bool = False,
) -> List[GeoreferencedDetection]:
    """YOLO 라벨 파일 전체 변환"""
    detections = parse_yolo_label_file(label_path, has_confidence=has_confidence)
    
    results = []
    for det in detections:
        geo_det = convert_yolo_to_geo(det, georeferencer, class_names)
        if geo_det:
            results.append(geo_det)
    
    return results


def export_to_geojson(
    cover: dict,
    drone: dict,
    detections: List[GeoreferencedDetection],
    output_path: str,
    image_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """검출 결과를 GeoJSON으로 export"""
    #features = [d.to_geojson_feature() for d in detections]
    features = [cover, drone]
    for d in detections:
        features.append(d.to_geojson_feature())
    
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    
    if image_metadata:
        geojson["metadata"] = image_metadata
    
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)


def export_to_csv(
    detections: List[GeoreferencedDetection],
    output_path: str,
) -> None:
    """검출 결과를 CSV로 export (작업지시서용)"""
    import csv
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id", "class_name", "confidence",
            "center_lat", "center_lon",
            "tl_lat", "tl_lon", "tr_lat", "tr_lon",
            "br_lat", "br_lon", "bl_lat", "bl_lon",
            "width_m", "height_m", "area_m2",
            "panel_id", "cell_row", "cell_col",
            "near_edge",
        ])
        
        for d in detections:
            writer.writerow([
                d.class_id, d.class_name, d.confidence,
                d.geo_center.latitude, d.geo_center.longitude,
                d.geo_corners[0].latitude, d.geo_corners[0].longitude,
                d.geo_corners[1].latitude, d.geo_corners[1].longitude,
                d.geo_corners[2].latitude, d.geo_corners[2].longitude,
                d.geo_corners[3].latitude, d.geo_corners[3].longitude,
                round(d.width_meters, 3),
                round(d.height_meters, 3),
                round(d.area_m2, 3),
                d.panel_id or "",
                d.cell_position[0] if d.cell_position else "",
                d.cell_position[1] if d.cell_position else "",
                d.near_image_edge,
            ])

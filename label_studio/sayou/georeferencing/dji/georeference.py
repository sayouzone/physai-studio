"""
DJI H20T 실제 이미지로 georeferencing
=====================================

업로드 이미지: DJI_20251217130206_0003_Z.JPG (RGB)
              DJI_20251217130206_0003_T.JPG (IR)

촬영지: 전라남도 (위도 34.71°N, 경도 126.92°E)
촬영시: 2025-12-17 13:02:06
드론 고도: 지상 44.9m
짐벌 자세: 거의 정확한 nadir (-89.9°)
GPS 시스템: RTK Fixed (위경도 ~1cm, 고도 ~2.5cm)
"""
import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple

import numpy as np

from .metadata import (
    DJIMetadata,
    parse_dji_metadata,
    estimate_zh20t_zoom_intrinsics, 
    dji_gimbal_to_camera_rotation, estimate_zh20t_thermal_intrinsics
)
from .georeferencer import (
    DJIGeoreferencer, GeoreferencingResult
)

# ============================================================
# 3. 좌표계 변환 (간단 버전, 작은 영역용)
# ============================================================

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


WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def geodetic_to_ecef(p: GeodeticPoint) -> np.ndarray:
    lat_rad = np.radians(p.latitude)
    lon_rad = np.radians(p.longitude)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat ** 2)
    
    x = (n + p.altitude) * cos_lat * cos_lon
    y = (n + p.altitude) * cos_lat * sin_lon
    z = (n * (1 - WGS84_E2) + p.altitude) * sin_lat
    
    return np.array([x, y, z])


def ecef_to_geodetic(ecef: np.ndarray) -> GeodeticPoint:
    x, y, z = ecef
    lon = np.arctan2(y, x)
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    
    for _ in range(5):
        sin_lat = np.sin(lat)
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat ** 2)
        h = p / np.cos(lat) - n
        lat_new = np.arctan2(z, p * (1 - WGS84_E2 * n / (n + h)))
        if abs(lat - lat_new) < 1e-12:
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


def geodetic_to_enu(p: GeodeticPoint, origin: GeodeticPoint) -> ENUPoint:
    p_ecef = geodetic_to_ecef(p)
    o_ecef = geodetic_to_ecef(origin)
    diff = p_ecef - o_ecef
    
    lat_rad = np.radians(origin.latitude)
    lon_rad = np.radians(origin.longitude)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    
    R = np.array([
        [-sin_lon,            cos_lon,           0      ],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat]
    ])
    
    enu = R @ diff
    return ENUPoint(east=enu[0], north=enu[1], up=enu[2])


def enu_to_geodetic(p: ENUPoint, origin: GeodeticPoint) -> GeodeticPoint:
    lat_rad = np.radians(origin.latitude)
    lon_rad = np.radians(origin.longitude)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    
    R_inv = np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
        [ cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
        [ 0,        cos_lat,           sin_lat          ]
    ])
    
    diff = R_inv @ p.to_array()
    o_ecef = geodetic_to_ecef(origin)
    
    return ecef_to_geodetic(o_ecef + diff)

# ============================================================
# 5. Georeferencer
# ============================================================

# ============================================================
# 6. GeoJSON 출력 (시각화용)
# ============================================================

def export_image_footprint_geojson(
    georeferencer: DJIGeoreferencer,
    image_name: str,
    output_path: Path,
):
    """이미지 footprint를 GeoJSON으로 export"""
    coverage = georeferencer.compute_image_coverage()
    
    coords = [[c.longitude, c.latitude] for c in coverage.image_corners_geo]
    coords.append(coords[0])  # 다각형 닫기
    
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords]
        },
        "properties": {
            "image": image_name,
            "drone_lat": georeferencer.meta.latitude,
            "drone_lon": georeferencer.meta.longitude,
            "drone_alt_m": georeferencer.meta.relative_altitude_m,
            "gimbal_pitch_deg": georeferencer.meta.gimbal_pitch_deg,
            "gimbal_yaw_deg": georeferencer.meta.gimbal_yaw_deg,
            "gsd_mm": coverage.ground_sample_distance_m * 1000,
            "coverage_m2": coverage.coverage_area_m2,
            "rtk_flag": georeferencer.meta.rtk_flag,
            "timestamp": georeferencer.meta.timestamp,
        }
    }
    
    geojson = {
        "type": "FeatureCollection",
        "features": [feature]
    }
    
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)


# ============================================================
# 7. 메인 실행
# ============================================================

def process_image(
    image_path: Path,
    is_thermal: bool,
    output_dir: Path,
) -> dict:
    """단일 이미지 georeferencing"""
    print(f"\n{'='*70}")
    print(f"처리: {image_path.name}")
    print('='*70)
    
    # 1. 메타데이터 추출
    meta = parse_dji_metadata(image_path)
    print(f"\n[메타데이터]")
    print(f"  GPS: ({meta.latitude:.7f}, {meta.longitude:.7f})")
    print(f"  절대 고도: {meta.absolute_altitude_m:.2f}m")
    print(f"  상대 고도 (이륙지점 기준): {meta.relative_altitude_m:.2f}m")
    print(f"  드론 자세: yaw={meta.flight_yaw_deg:.1f}°, "
          f"pitch={meta.flight_pitch_deg:.1f}°, roll={meta.flight_roll_deg:.1f}°")
    print(f"  짐벌 자세: yaw={meta.gimbal_yaw_deg:.1f}°, "
          f"pitch={meta.gimbal_pitch_deg:.1f}°, roll={meta.gimbal_roll_deg:.1f}°")
    print(f"  RTK Flag: {meta.rtk_flag} ({'Fixed (1cm 정확도)' if meta.rtk_flag == 50 else 'Other'})")
    print(f"  RTK σ: lat={meta.rtk_std_lat_m*100:.2f}cm, "
          f"lon={meta.rtk_std_lon_m*100:.2f}cm, "
          f"hgt={meta.rtk_std_hgt_m*100:.2f}cm")
    print(f"  이미지: {meta.image_width} × {meta.image_height}")
    
    # 2. 카메라 내부 파라미터
    if is_thermal:
        intrinsics = estimate_zh20t_thermal_intrinsics(
            meta.image_width, meta.image_height
        )
        print(f"\n[카메라 (Thermal)]")
    else:
        intrinsics = estimate_zh20t_zoom_intrinsics(
            meta.image_width, meta.image_height
        )
        print(f"\n[카메라 (RGB Zoom)]")
    
    print(f"  fx={intrinsics.K[0,0]:.1f}, fy={intrinsics.K[1,1]:.1f}")
    print(f"  cx={intrinsics.K[0,2]:.1f}, cy={intrinsics.K[1,2]:.1f}")
    
    # 3. Georeferencer
    gr = DJIGeoreferencer(meta, intrinsics)
    
    # 4. 커버리지 계산
    coverage = gr.compute_image_coverage()
    print(f"\n[커버리지]")
    print(f"  GSD (Ground Sample Distance): {coverage.ground_sample_distance_m*100:.2f} cm/pixel "
          f"({coverage.ground_sample_distance_m*1000:.1f} mm/pixel)")
    print(f"  지상 너비: {coverage.coverage_width_m:.2f}m")
    print(f"  지상 높이: {coverage.coverage_height_m:.2f}m")
    print(f"  면적: {coverage.coverage_area_m2:.1f} m²")
    
    print(f"\n[이미지 4개 모서리 좌표]")
    for label, c in zip(["TL", "TR", "BR", "BL"], coverage.image_corners_geo):
        print(f"  {label}: ({c.latitude:.7f}, {c.longitude:.7f})")
    
    # 5. 가상 검출 박스 — 패널 위치 기준
    if not is_thermal:
        # RGB에서 패널이 우측 상단에 있음 (대략적 추정)
        defect_bbox = (3800, 200, 4200, 600)
    else:
        defect_bbox = (450, 50, 550, 200)
    
    print(f"\n[가상 결함 검출 박스 georeferencing]")
    print(f"  픽셀 박스: {defect_bbox}")
    
    center_geo = gr.bbox_center_to_geodetic(defect_bbox)
    if center_geo:
        print(f"  중심 좌표: ({center_geo.latitude:.7f}, {center_geo.longitude:.7f})")
    
    corners_geo = gr.bbox_to_geodetic(defect_bbox)
    if corners_geo:
        print(f"  4개 모서리:")
        for label, c in zip(["TL", "TR", "BR", "BL"], corners_geo):
            print(f"    {label}: ({c.latitude:.7f}, {c.longitude:.7f})")
    
    # 6. GeoJSON 저장
    output_path = output_dir / f"{image_path.stem}_footprint.geojson"
    export_image_footprint_geojson(gr, image_path.name, output_path)
    print(f"\n[GeoJSON 저장]")
    print(f"  {output_path}")
    
    return {
        "image": image_path.name,
        "metadata": asdict(meta),
        "coverage": {
            "gsd_m": coverage.ground_sample_distance_m,
            "area_m2": coverage.coverage_area_m2,
            "width_m": coverage.coverage_width_m,
            "height_m": coverage.coverage_height_m,
        },
        "image_corners": [
            {"lat": c.latitude, "lon": c.longitude}
            for c in coverage.image_corners_geo
        ],
    }


def main():
    output_dir = Path("/home/claude/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rgb_path = Path("/mnt/user-data/uploads/DJI_20251217130206_0003_Z.JPG")
    ir_path = Path("/mnt/user-data/uploads/DJI_20251217130206_0003_T.JPG")
    
    rgb_result = process_image(rgb_path, is_thermal=False, output_dir=output_dir)
    ir_result = process_image(ir_path, is_thermal=True, output_dir=output_dir)
    
    # 두 이미지 비교
    print(f"\n\n{'='*70}")
    print("RGB와 IR 이미지 커버리지 비교")
    print('='*70)
    print(f"\n{'항목':<30} {'RGB':>20} {'IR':>20}")
    print(f"{'-'*70}")
    print(f"{'GSD (mm/pixel)':<30} {rgb_result['coverage']['gsd_m']*1000:>18.1f} "
          f"{ir_result['coverage']['gsd_m']*1000:>18.1f}")
    print(f"{'지상 너비 (m)':<30} {rgb_result['coverage']['width_m']:>20.2f} "
          f"{ir_result['coverage']['width_m']:>20.2f}")
    print(f"{'지상 높이 (m)':<30} {rgb_result['coverage']['height_m']:>20.2f} "
          f"{ir_result['coverage']['height_m']:>20.2f}")
    print(f"{'면적 (m²)':<30} {rgb_result['coverage']['area_m2']:>20.1f} "
          f"{ir_result['coverage']['area_m2']:>20.1f}")
    
    # JSON 결과 저장
    summary_path = output_dir / "georeferencing_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "rgb": rgb_result,
            "ir": ir_result,
        }, f, indent=2, default=str)
    print(f"\n결과 요약 저장: {summary_path}")


if __name__ == "__main__":
    main()

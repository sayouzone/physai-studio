import logging
import math
import numpy as np
import os

from enum import Enum
from typing import List, Optional, Sequence, Tuple, Dict, Any

from sayou.georeferencing.dji.camera_pose import compute_camera_axes_from_gimbal, verify_nadir_orientation
from sayou.georeferencing.dji.metadata import DJIMetadata, M_PER_DEG_LAT
from sayou.georeferencing.dji.georeferencer import DJIImageGeoreferencer
from sayou.georeferencing.dji.coordinates import geodetic_to_enu
from sayou.georeferencing.yolo_to_geo import (
    YOLODetection,
    _compute_polygon_area_m2,
)
from sayou.image.metadata import ImageMetadata, extract_metadata
import math, inspect, sayou.image.metadata as MM

from tasks.models import Annotation

logger = logging.getLogger(__name__)

class AnnotationGeoreferencer:

    class SensorType(str, Enum):
        RGB = "RGB"
        IR = "IR"
        UNKNOWN = "UNKNOWN"

    SOLAR_RGB_CLASSES = {
        0: "pv",
        1: "anomaly",
    }

    SOLAR_IR_CLASSES = {
        0: "Panel",
        1: "Hotspot",
    }

    def _get_sensor_type(self, meta) -> SensorType:
        """
        DJI H20T 이미지 메타데이터에서 RGB/IR 센서 종류를 판별한다.
        1차: camera_model 문자열
        2차(폴백): 파일명 접미사(_W/_T), 해상도
        """
        model = (meta.camera_model or "").lower()

        if "infrared" in model or "thermal" in model:
            return AnnotationGeoreferencer.SensorType.IR
        if "wide" in model or "zoom" in model:
            return AnnotationGeoreferencer.SensorType.RGB

        # camera_model이 비어있거나 예상 밖 값일 때의 폴백
        origin = (meta.origin_path or "").upper()
        if origin.endswith("_T.JPG") or origin.endswith("_T.JPEG"):
            return AnnotationGeoreferencer.SensorType.IR
        if origin.endswith("_W.JPG") or origin.endswith("_W.JPEG"):
            return AnnotationGeoreferencer.SensorType.RGB

        # 최후 폴백: H20T 스펙상 IR은 항상 640x512
        if meta.width == 640 and meta.height == 512:
            return AnnotationGeoreferencer.SensorType.IR
        if meta.width == 4056 and meta.height == 3040:
            return AnnotationGeoreferencer.SensorType.RGB

        return AnnotationGeoreferencer.SensorType.UNKNOWN

    def pixel_bbox_to_polygon(
        self, bbox_xyxy: Sequence[float], metadata: DJIMetadata,
    ) -> list[list[float]]:
        """픽셀 bbox(x1,y1,x2,y2) → GeoJSON Polygon ring [[lon,lat], ...]"""
        x1, y1, x2, y2 = bbox_xyxy
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
        ring = []
        for px, py in corners:
            e, n = self._pixel_to_ground_enu(px, py, metadata)
            ring.append(self._enu_to_lonlat(e, n, metadata))
        return ring

    def _pixel_to_camera_ray(
        self,
        px: float,
        py: float,
        metadata: ImageMetadata
    ) -> np.ndarray:
        """
        픽셀(좌상단 원점) → 카메라 좌표계 단위 광선 (X=Right, Y=Down, Z=Forward).

        초점거리는 반드시 '등방'(fx = fy = metadata.focal_px)을 사용한다.

        metadata.focal_px 정의:
          - DewarpData(공장 캘리브레이션)가 있으면 실측 fx/fy 평균
          - 없으면 FocalLengthIn35mmFilm 을 '대각선' 기준으로 픽셀 환산
                f_px = f35 * diag_px / diag_35mm

        [수정 배경]
          예전 구현은 hfov/vfov 에서 fx, fy 를 각각 계산했는데,
          FocalLengthIn35mmFilm 은 대각선 기준 정의값이라 4:3 센서
          (H20T Zoom, 5184x3888)에서 fy/fx = 1.125 의 비물리적 이방성이
          발생했다. 이 때문에 x축 +4%, y축 -7.5% 스케일 오차가 생기고
          이미지 중심(nadir 발끝)에서 바깥으로 갈수록 오차가 방사형으로
          누적되어 "상단은 맞고 하단으로 갈수록 벌어지는" 증상이 나타났다.

        DewarpData 왜곡계수(k1,k2,p1,p2,k3)가 있으면 cv2.undistortPoints 로
        렌즈 왜곡까지 보정한다.
        """
        # 등방 초점거리 (DewarpData 있으면 실측 fx/fy 유지)
        if getattr(metadata, "has_dewarp", False):
            fx_px, fy_px = metadata.dewarp_fx, metadata.dewarp_fy
        else:
            fx_px = fy_px = metadata.focal_px

        # 주점: DewarpData 있으면 '중심 + 오프셋', 없으면 이미지 중심
        cx_px, cy_px = metadata.principal_point_px

        dist = np.asarray(getattr(metadata, "dewarp_dist", []) or [], dtype=float)
        if getattr(metadata, "has_dewarp", False) and dist.size and np.any(dist != 0):
            # 렌즈 왜곡계수가 있을 때만 실행되는 경로.
            # cv2 를 상단에서 항상 import 하면 폐쇄망/최소 환경에서 모듈 로딩이
            # 깨질 수 있으므로 필요한 시점에만 지연 import.
            import cv2
            K = np.array([
                [fx_px, 0.0,   cx_px],
                [0.0,   fy_px, cy_px],
                [0.0,   0.0,   1.0],
            ])
            pts = np.array([[[px, py]]], dtype=np.float32)
            undistorted = cv2.undistortPoints(pts, K, dist[:5])
            x, y = undistorted[0, 0]
            ray_cam = np.array([x, y, 1.0])
        else:
            # OpenCV pinhole: x_cam = (px - cx)/fx, y_cam = (py - cy)/fy, z_cam = 1
            ray_cam = np.array([
                (px - cx_px) / fx_px,
                (py - cy_px) / fy_px,
                1.0,
            ])
        return ray_cam / np.linalg.norm(ray_cam)

    def _pixel_to_ground_enu(
        self,
        px: float, py: float,
        metadata: DJIMetadata,
        ground_z_below_drone: float | None = None,
    ) -> tuple[float, float]:
        """
        픽셀 → 카메라 nadir 발끝 기준 (East, North) 미터 오프셋.

        ground_z_below_drone : 드론에서 본 지면의 z 값(ENU에서 -AGL).
                            기본 None이면 -relative_height.
                            LRF 실측 절대고도가 있으면 그것을 우선 쓴다.
                            DEM 쓰면 픽셀별로 달리 줄 수 있음.
        """
        if ground_z_below_drone is None:
            if getattr(metadata, "has_valid_lrf", False):
                # LRF 조준점 실측 절대고도 기반의 AGL (더 정확)
                ground_z_below_drone = metadata.lrf_target_abs_alt - metadata.gps.altitude
            else:
                # 드론에서 본 지면의 z 값(ENU에서 -AGL)
                ground_z_below_drone = -metadata.relative_height   # 드론보다 AGL만큼 아래

        # (1) 카메라 좌표 광선 → ENU 광선
        ray_cam = self._pixel_to_camera_ray(px, py, metadata)
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

    def _enu_to_lonlat(self, east_m: float, north_m: float, metadata: DJIMetadata) -> list[float]:
        """
        ENU 오프셋 → (lon, lat) 좌표.

        pose의 (lon, lat)을 기준으로 미터 오프셋을 도 단위로 변환하여 더한다.
        위경도↔미터 환산은 위도별 곡률반경 기반 프로퍼티(m_per_deg_lat/lon)를
        사용한다. (고정 상수 M_PER_DEG_LAT=111320 은 위도 34.7°에서 약 +0.35%
        남북 스케일 오차를 유발하므로 사용하지 않는다.)
        """
        return [
            metadata.gps.lng + east_m / metadata.m_per_deg_lon,
            metadata.gps.lat + north_m / metadata.m_per_deg_lat,
        ]

    def _annotation_to_yolo_labels(
        self,
        annotation,
        sensor_type: 'AnnotationGeoreferencer.SensorType',
    ) -> list[YOLODetection]:
        """
        Label Studio annotation.result (rectanglelabels 포맷)를
        YOLODetection 리스트로 변환한다. .txt 파일을 거치지 않고 메모리에서 바로 변환.

        Label Studio 좌표: 좌상단 기준 percent(0~100) x, y, width, height
        YOLO 좌표:         center 기준 0~1 정규화 cx, cy, w, h
        """
        if annotation is None or not annotation.result:
            logger.info('[AnnotationGeoreferencer] 사용 가능한 annotation result가 없음')
            return []

        label_to_class_id = self._label_to_class_id_map(sensor_type)
        labels: list[YOLODetection] = []

        for region in annotation.result:
            if region.get('type') != 'rectanglelabels':
                continue  # keypoint, polygon 등 다른 타입은 skip

            region_id = region.get('id')
            value = region.get('value', {})
            label_names = value.get('rectanglelabels') or []
            if not label_names:
                continue

            label = label_names[0]
            class_id = label_to_class_id.get(label)
            if class_id is None:
                logger.warning(
                    f"[AnnotationGeoreferencer] annotation={annotation.id}: "
                    f"알 수 없는 라벨 '{label}' (sensor={sensor_type.value}) - skip"
                )
                continue

            x_pct = value.get('x', 0.0)
            y_pct = value.get('y', 0.0)
            w_pct = value.get('width', 0.0)
            h_pct = value.get('height', 0.0)

            labels.append(YOLODetection(
                region_id=region_id,
                class_id=class_id,
                cx_norm=(x_pct + w_pct / 2.0) / 100.0,
                cy_norm=(y_pct + h_pct / 2.0) / 100.0,
                w_norm=w_pct / 100.0,
                h_norm=h_pct / 100.0,
                confidence=None,
            ))

        return labels

    def _get_annotation(self, task, include_predictions: bool = True) -> Optional['Annotation']:
        """
        annotation, draft, prediction 중 가장 최근에 작성/수정된 것을 반환한다.

        우선순위 정책:
        - 세 소스 모두 시간(updated_at 또는 created_at) 기준으로 최신 것 선택
        - 단, prediction은 사람이 만든 annotation/draft가 하나도 없을 때의
          fallback으로만 쓰는 것이 안전하므로, prefer_human=True(기본) 동작을 유지

        Returns:
            Annotation | AnnotationDraft | Prediction | None
            (세 모델 모두 .result 필드를 가지므로 _annotation_to_yolo_labels에서 동일하게 처리 가능)
        """
        annotation = (
            task.annotations
            .filter(was_cancelled=False)
            .order_by('-updated_at')
            .first()
        )

        draft = task.drafts.order_by('-updated_at').first()

        # ── annotation vs draft: 시간 비교로 최신 선택 ──────────────
        latest_human = None
        if annotation and draft:
            annotation_time = annotation.updated_at or annotation.created_at
            draft_time = draft.updated_at or draft.created_at
            latest_human = draft if draft_time > annotation_time else annotation
        else:
            latest_human = annotation or draft

        # ── 사람이 만든 결과가 있으면 그것을 우선 반환 ────────────────
        if latest_human is not None:
            return latest_human

        # ── 둘 다 없으면 prediction fallback ─────────────────────────
        if include_predictions:
            prediction = (
                task.predictions
                .order_by('-updated_at')
                .first()
            )
            if prediction is not None:
                logger.info(
                    f'[AnnotationGeoreferencer] task={task.id}: annotation/draft 없음 → '
                    f'prediction(id={prediction.id}, model={prediction.model_version}) 사용'
                )
            return prediction

        return None

    def _label_to_class_id_map(self, sensor_type: 'AnnotationGeoreferencer.SensorType') -> dict:
        """센서 타입(RGB/IR)에 맞는 label name → class_id 매핑을 반환한다."""
        classes = (
            self.SOLAR_IR_CLASSES if sensor_type == self.SensorType.IR
            else self.SOLAR_RGB_CLASSES
        )
        return {name: class_id for class_id, name in classes.items()}

    def _ir_footprint_ring(self, gr, metadata) -> Optional[list]:
        """
        IR 이미지 4개 모서리 → 지리 좌표 ring (geo.py extract_ir_geo의 footprint와 동일).
        pixel_to_geodetic이 None을 반환하면 footprint 생략.
        """
        pixel_corners = [
            (0, 0),
            (metadata.width, 0),
            (metadata.width, metadata.height),
            (0, metadata.height),
        ]
        ring = []
        for px in pixel_corners:
            geo = gr.pixel_to_geodetic(px)
            if geo is None:
                logger.warning('[AnnotationGeoreferencer] IR footprint 계산 실패 (pixel_to_geodetic=None)')
                return None
            ring.append([geo.longitude, geo.latitude])
        ring.append(ring[0])  # ring 닫기
        return ring

    def _ir_detection_to_feature(
        self,
        detection: 'YOLODetection',
        gr: 'DJIImageGeoreferencer',
        class_names: dict,
        index: int,
        annotation,
    ) -> Optional[dict]:
        """
        IR YOLO 검출 1개 → GeoJSON Feature.
        geo.py의 convert_ir_yolo_to_geo와 동일한 계산 경로
        (pixel_to_geodetic + ENU 변 길이 평균 + polygon area).

        주의: Label Studio annotation은 native 이미지 기준 percent 좌표이므로
        geo.py의 stretch 보정(640×640 → 640×512)은 불필요하다.
        """
        metadata = gr.metadata
        W, H = metadata.width, metadata.height

        pixel_bbox = detection.to_pixel_xyxy(W, H)
        pixel_corners = detection.to_pixel_corners(W, H)   # TL, TR, BR, BL

        # 4개 모서리 → 지리 좌표
        geo_corners = []
        for px in pixel_corners:
            geo = gr.pixel_to_geodetic(px)
            if geo is None:
                logger.warning(
                    f'[AnnotationGeoreferencer] IR det#{index} georeferencing 실패 '
                    f'(pixel={px}) - skip'
                )
                return None
            geo_corners.append(geo)

        # GeoJSON ring [[lon,lat], ...] (닫힌 폴리곤)
        ring = [[c.longitude, c.latitude] for c in geo_corners]
        ring.append(ring[0])

        # 물리 크기: ENU corner 변 길이 평균 (geo.py 방식)
        origin = gr.origin
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

        # 이미지 가장자리 근접 여부 (IR은 시야가 좁아 edge 검출 신뢰도 낮음)
        edge_threshold = 0.05
        x_edge = min(detection.cx_norm, 1 - detection.cx_norm)
        y_edge = min(detection.cy_norm, 1 - detection.cy_norm)
        near_edge = (
            x_edge < edge_threshold
            or y_edge < edge_threshold
            or detection.w_norm > 1 - 2 * edge_threshold
            or detection.h_norm > 1 - 2 * edge_threshold
        )

        cls_name = class_names.get(detection.class_id, f"class_{detection.class_id}")
        x1, y1, x2, y2 = pixel_bbox

        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "detection_index": index,
                "annotation_id": annotation.id if annotation else None,
                "region_id": detection.region_id,
                "class_id": detection.class_id,
                "class_name": cls_name,
                "sensor_type": self.SensorType.IR.value,
                "pixel_bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "width_m": round(float(width_m), 3),
                "height_m": round(float(height_m), 3),
                "area_m2": round(float(area_m2), 3),
                "near_image_edge": near_edge,
                "type": "detection_box",
            },
        }

    def _rgb_detection_to_feature(
        self,
        detection: 'YOLODetection',
        metadata: DJIMetadata,
        class_names: dict,
        index: int,
        annotation,
    ) -> Optional[dict]:
        """RGB YOLO 검출 1개 → GeoJSON Feature (광선-평면 교차 방식)."""
        x1, y1, x2, y2 = detection.to_pixel_xyxy(metadata.width, metadata.height)

        try:
            ring = self.pixel_bbox_to_polygon((x1, y1, x2, y2), metadata)
        except ValueError as e:
            logger.warning(f'[AnnotationGeoreferencer] RGB det#{index} georeferencing 실패: {e}')
            return None

        # 물리 크기: ENU corner 변 길이 평균 (등방 스케일이므로 축별 편차 없음)
        e_corners = [
            self._pixel_to_ground_enu(px, py, metadata)
            for px, py in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        ]
        es = [c[0] for c in e_corners]
        ns = [c[1] for c in e_corners]
        width_m = (
            math.hypot(e_corners[1][0] - e_corners[0][0], e_corners[1][1] - e_corners[0][1])
            + math.hypot(e_corners[2][0] - e_corners[3][0], e_corners[2][1] - e_corners[3][1])
        ) / 2
        height_m = (
            math.hypot(e_corners[3][0] - e_corners[0][0], e_corners[3][1] - e_corners[0][1])
            + math.hypot(e_corners[2][0] - e_corners[1][0], e_corners[2][1] - e_corners[1][1])
        ) / 2

        cls_name = class_names.get(detection.class_id, f"class_{detection.class_id}")

        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "detection_index": index,
                "annotation_id": annotation.id if annotation else None,
                "region_id": detection.region_id,
                "class_id": detection.class_id,
                "class_name": cls_name,
                "sensor_type": self.SensorType.RGB.value,
                "pixel_bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "width_m": round(float(width_m), 3),
                "height_m": round(float(height_m), 3),
                "area_m2": round(float(width_m * height_m), 3),
                "type": "detection_box",
            },
        }

    def __init__(self, image_path: str, annotation, include_footprint: bool = True, include_drone_position: bool = True):
        self.image_path = image_path
        self.annotation = annotation
        self.include_footprint = include_footprint
        self.include_drone_position = include_drone_position

    def get_geojson(
        self,
        image_path: str = None,
        annotation: Optional['Annotation'] = None,
    ):
        if image_path is not None:
            self.image_path = image_path
        if annotation is not None:
            self.annotation = annotation

        metadata = extract_metadata(self.image_path)

        print("모듈 실제 경로 :", MM.__file__)              # ← 어느 파일이 로드됐나
        print("focal_px 존재  :", hasattr(metadata, "focal_px"))   # ← False면 구 버전
        print(f"focal_px      = {metadata.focal_px:.1f}  (기대 7039.1)")
        print(f"hfov/vfov     = {metadata.hfov_deg:.4f} / {metadata.vfov_deg:.4f}  (기대 40.4301 / 30.8771)")
        print("has_dewarp    =", metadata.has_dewarp, "| fx,fy =", metadata.dewarp_fx, metadata.dewarp_fy)
        print("dewarp_dist   =", metadata.dewarp_dist)
        print(f"주점          = {metadata.principal_point_px}")
        print("lrf           =", metadata.lrf, "| has_valid_lrf =", metadata.has_valid_lrf)
        print(f"rel_height    = {metadata.relative_height}  abs_alt = {metadata.gps.altitude}")
        print(f"GSD           = {metadata.relative_height/metadata.focal_px*1000:.3f} mm/px")
        print(f"footprint     = {metadata.width*metadata.relative_height/metadata.focal_px:.2f} x {metadata.height*metadata.relative_height/metadata.focal_px:.2f} m")

        # 짐벌 자세 → 카메라 회전 행렬 (위임)
        axes = compute_camera_axes_from_gimbal(
            gimbal_yaw_compass_deg=metadata.gimbal_yaw_deg,
            gimbal_pitch_deg=metadata.gimbal_pitch_deg,
            gimbal_roll_deg=metadata.gimbal_roll_deg,
        )
        metadata.R_cam_to_enu = axes['R_camera_to_enu']

        sensor_type = self._get_sensor_type(metadata)

        detections = self._annotation_to_yolo_labels(self.annotation, sensor_type)

        # ── annotation → GeoJSON 계산 (분리된 함수 호출) ─────────────
        geojson = self.annotation_to_geojson(metadata, self.annotation, sensor_type, detections, self.include_footprint, self.include_drone_position)

        return metadata, geojson

    def annotation_to_geojson(
        self,
        metadata,
        annotation=None,
        sensor_type=None,
        detections=None,
        include_footprint: bool = True,
        include_drone_position: bool = True,
    ) -> dict:
        """
        Task의 annotation을 georeferencing하여 GeoJSON FeatureCollection으로 변환.
        센서 타입(RGB/IR)에 따라 변환 경로가 다르다:
        - RGB: pixel_bbox_to_polygon (광선-평면 교차, AGL 기반)
        - IR : DJIImageGeoreferencer.pixel_to_geodetic (LRF 지면고도 기반, geo.py extract_ir_geo와 동일)

        기존 get()의 하드코딩된 테스트 라벨 대신, 실제 annotation.result를
        _annotation_to_yolo_labels()로 변환해 사용하는 정식 계산 함수.

        Args:
            metadata: extract_metadata() + R_cam_to_enu 설정이 완료된 ImageMetadata
            annotation: 사용할 annotation (None이면 최신 유효 annotation 자동 선택)
            include_footprint: 이미지 커버리지 폴리곤 포함 여부
            include_drone_position: 드론 위치 Point 포함 여부

        Returns:
            GeoJSON FeatureCollection dict
        """
        is_ir = sensor_type == self.SensorType.IR

        # 센서 타입에 맞는 클래스 이름 맵
        class_names = self.SOLAR_IR_CLASSES if is_ir else self.SOLAR_RGB_CLASSES

        # ── IR은 DJIImageGeoreferencer 사용 (geo.py 방식) ─────────────
        gr = DJIImageGeoreferencer(metadata) if is_ir else None

        nadir_check = verify_nadir_orientation(metadata.R_cam_to_enu, tolerance_deg=10.0)
        is_oblique = not nadir_check['is_nadir']

        features = []

        # ── footprint ────────────────────────────────────────────────
        if include_footprint:
            if is_ir:
                footprint_ring = self._ir_footprint_ring(gr, metadata)
            else:
                footprint_ring = self.pixel_bbox_to_polygon(
                    (0, 0, metadata.width, metadata.height), metadata,
                )

            if footprint_ring is not None:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [footprint_ring]},
                    "properties": {
                        "name": "image_coverage",
                        "image_path": metadata.origin_path,
                        "camera_model": metadata.camera_model,
                        "sensor_type": sensor_type.value,
                        "focal_35mm_eq": metadata.focal_length_in_35mm,
                        "hfov_deg": round(metadata.hfov_deg, 2),
                        "vfov_deg": round(metadata.vfov_deg, 2),
                        "rel_alt_m": metadata.relative_height,
                        "gimbal_pitch_deg": metadata.gimbal_pitch_deg,
                        "gimbal_yaw_compass_deg": metadata.gimbal_yaw_deg,
                        "gimbal_roll_deg": metadata.gimbal_roll_deg,
                        "oblique_view": is_oblique,
                        "angle_from_nadir_deg": round(nadir_check["angle_from_nadir_deg"], 2),
                    },
                })

        # ── drone position ───────────────────────────────────────────
        if include_drone_position:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [metadata.gps.lng, metadata.gps.lat],
                },
                "properties": {
                    "name": "drone_position",
                    "rel_altitude_m": metadata.relative_height,
                    "abs_altitude_m": metadata.gps.altitude,
                    "rtk_active": metadata.rtk_active,
                },
            })

        # ── detection boxes (센서별 분기) ────────────────────────────
        for i, det in enumerate(detections):
            if is_ir:
                feature = self._ir_detection_to_feature(det, gr, class_names, i, annotation)
            else:
                feature = self._rgb_detection_to_feature(det, metadata, class_names, i, annotation)

            if feature is not None:
                features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "task_id": None,
                "annotation_id": annotation.id if annotation else None,
                "source_type": type(annotation).__name__ if annotation else None,
                "num_detections": len(detections),
                "image_path": metadata.origin_path,
                "modality": "infrared" if is_ir else "rgb",
                "camera": "ZH20T_Thermal" if is_ir else f"ZH20T_{(metadata.camera_model or '').capitalize()}",
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
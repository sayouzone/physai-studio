"""
YOLOv11 백엔드.

이 모듈은 '모델을 돌리는 일'만 담당한다. 판정 의미론(어떤 클래스가 어떤 분기에서
관측 가능한가, 신뢰도를 어떻게 매기는가)은 `yolo_branch.py` 에 있고, 융합은
`fusion.py` 에 있다. 이 분리를 유지해야 모델을 갈아끼워도 융합 계층이 안 흔들린다.

포함 기능
  - `YoloRuntime` 추상화: 기본은 ultralytics, 테스트/온프렘에서는 주입 가능
  - 타일 추론 + 타일 경계 NMS  (오소모자이크는 한 장이 수만 픽셀이라 필수)
  - `ThermalNormalizer`: **학습과 추론의 정규화를 강제로 일치**시키는 장치
  - 모델 클래스명 -> 파이프라인 택소노미 매핑
  - `ConfidenceCalibrator`: YOLO conf 는 확률이 아니다. 융합 전에 보정해야 한다.

설치:
    pip install ultralytics        # YOLOv11 (models: yolo11n/s/m/l/x, -seg)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import CLASS_INDEX, DEFECT_CLASSES, NORMAL, PanelROI


# --------------------------------------------------------------------------- #
# 검출 결과
# --------------------------------------------------------------------------- #

@dataclass
class Detection:
    """단일 검출. 좌표는 항상 '입력 원본 영상' 픽셀 좌표계."""
    cls_name: str                      # 모델이 낸 원래 클래스명
    taxon: Optional[str]               # 파이프라인 택소노미로 매핑된 이름 (없으면 None)
    conf: float                        # 원시 confidence
    calibrated: float                  # 보정 후 확률
    xyxy: np.ndarray                   # (4,) float
    polygon: Optional[np.ndarray] = None   # (N,2) seg 마스크 폴리곤
    source: str = ""

    @property
    def center(self) -> np.ndarray:
        x0, y0, x1, y1 = self.xyxy
        return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.xyxy
        return float(max(0.0, x1 - x0) * max(0.0, y1 - y0))


# --------------------------------------------------------------------------- #
# 열화상 정규화 (학습/추론 일관성)
# --------------------------------------------------------------------------- #

@dataclass
class ThermalNormalizer:
    """
    열화상 -> 8bit 변환 규칙.

    **여기가 IR YOLO 의 1번 실패 원인이다.** 학습 데이터는 퍼센타일 스트레치로
    만들어 놓고 추론에서는 고정 범위를 쓰거나(그 반대), 프레임마다 다른 범위를
    쓰면, 같은 ΔT 가 프레임마다 다른 화소값이 되어 모델이 학습한 대비가 무너진다.
    그래서 정규화 규칙을 **객체로 고정해 학습 스크립트와 추론 파이프라인이
    같은 인스턴스를 공유**하도록 만든다.

    mode="fixed"      : [vmin, vmax] °C 고정. 사이트 온도 범위를 알면 가장 안정적.
    mode="percentile" : 프레임별 퍼센타일. 편하지만 프레임 간 일관성이 없다.
    mode="relative"   : 패널 중앙값 기준 ±span K. 기온 변동에 가장 강인하며
                        ΔT 로 학습·추론하므로 계절/시간대 일반화가 가장 좋다.
    """
    mode: str = "relative"
    vmin: float = 10.0
    vmax: float = 80.0
    p_lo: float = 1.0
    p_hi: float = 99.0
    span: float = 20.0          # relative 모드의 ±범위 [K]

    def to_u8(self, thermal_c: np.ndarray,
              reference_c: Optional[float] = None) -> np.ndarray:
        x = np.asarray(thermal_c, dtype=np.float32)
        finite = np.isfinite(x)
        if not finite.any():
            return np.zeros(x.shape, np.uint8)

        if self.mode == "fixed":
            lo, hi = self.vmin, self.vmax
        elif self.mode == "percentile":
            lo, hi = np.percentile(x[finite], [self.p_lo, self.p_hi])
        elif self.mode == "relative":
            ref = reference_c if reference_c is not None else float(np.median(x[finite]))
            lo, hi = ref - self.span * 0.35, ref + self.span * 0.65
        else:
            raise ValueError(f"알 수 없는 mode: {self.mode}")

        if hi - lo < 1e-6:
            return np.zeros(x.shape, np.uint8)
        return np.clip((x - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    def to_3ch(self, thermal_c: np.ndarray,
               reference_c: Optional[float] = None) -> np.ndarray:
        """YOLO 입력용 3채널. gray 를 복제하되 colormap 은 쓰지 않는다.

        colormap(inferno 등)을 씌우면 사람 눈에는 좋지만 색상 LUT 가 비단조라
        모델이 온도 순서를 배우기 어려워진다. 학습·추론 모두 gray 복제로 통일한다.
        """
        g = self.to_u8(thermal_c, reference_c)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    def to_dict(self) -> Dict:
        return {"mode": self.mode, "vmin": self.vmin, "vmax": self.vmax,
                "p_lo": self.p_lo, "p_hi": self.p_hi, "span": self.span}

    @classmethod
    def from_dict(cls, d: Dict) -> "ThermalNormalizer":
        fields = {"mode", "vmin", "vmax", "p_lo", "p_hi", "span"}
        return cls(**{k: v for k, v in d.items() if k in fields})

    @classmethod
    def load(cls, path: Optional[str]) -> "ThermalNormalizer":
        """
        학습 시 저장해 둔 `normalizer.json` 을 읽는다.

        **IR 모델을 배포할 때 가중치와 반드시 함께 다녀야 하는 파일이다.**
        이 설정이 학습과 다르면 같은 ΔT 가 다른 화소값이 되어, 모델이 학습한
        대비가 통째로 어긋난다. 가중치는 멀쩡한데 성능만 조용히 무너지는
        전형적인 배포 사고다.
        """
        if not path:
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# --------------------------------------------------------------------------- #
# 신뢰도 보정
# --------------------------------------------------------------------------- #

class ConfidenceCalibrator:
    """
    클래스별 온도 스케일링 + 바이어스.  p' = sigmoid((logit(p) - b) / T)

    YOLO 의 conf 는 확률이 아니다. objectness x cls 점수에 NMS 가 얹힌 값이라
    보통 과신(overconfident)한다. 보정하지 않고 Dempster-Shafer 에 넣으면
    한쪽 분기가 거의 항상 이겨서 융합이 무의미해진다.

    **파라미터는 학습/검증 단계에서 만들어 JSON 으로 넘겨받는다.** 이 패키지는
    추론만 담당하므로 여기서는 읽어서 적용만 한다. 파일 형식:

        { "hotspot": [T, b], "substring": [T, b], ... }

    파일이 없으면 항등 동작(보정 없음)이며, 그 상태로 융합하면 YOLO 분기가
    과신한다는 점을 기억해 둘 것.
    """

    def __init__(self, params: Optional[Dict[str, Tuple[float, float]]] = None,
                 default: Tuple[float, float] = (1.0, 0.0)):
        self.params: Dict[str, Tuple[float, float]] = dict(params or {})
        self.default = default

    @staticmethod
    def _logit(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p)))

    def __call__(self, cls_name: str, conf: float) -> float:
        T, b = self.params.get(cls_name, self.default)
        z = (self._logit(conf) - b) / max(T, 1e-3)
        return float(1.0 / (1.0 + np.exp(-z)))

    @property
    def is_identity(self) -> bool:
        return not self.params and self.default == (1.0, 0.0)

    @classmethod
    def load(cls, path: Optional[str]) -> "ConfidenceCalibrator":
        if not path:
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls({k: (float(v[0]), float(v[1])) for k, v in d.items()})


# --------------------------------------------------------------------------- #
# 런타임 추상화
# --------------------------------------------------------------------------- #

class YoloRuntime:
    """
    추론 백엔드 인터페이스.

    predict(images) -> List[List[dict]]
        dict = {"cls": str, "conf": float, "xyxy": (4,) array, "polygon": (N,2) or None}

    주입 가능하게 둔 이유:
      - ultralytics 없이도 파이프라인 통합 로직을 테스트할 수 있어야 한다
      - 온프렘에서 TensorRT / ONNXRuntime 로 갈아끼우는 경우가 흔하다
    """

    def predict(self, images: Sequence[np.ndarray]) -> List[List[Dict]]:
        raise NotImplementedError

    @property
    def names(self) -> List[str]:
        return []


class UltralyticsRuntime(YoloRuntime):
    """ultralytics YOLOv11 래퍼. import 는 실제 사용 시점까지 지연시킨다."""

    def __init__(self, weights: str, device: str = "cpu", imgsz: int = 960,
                 conf: float = 0.15, iou: float = 0.5, half: bool = False,
                 max_det: int = 300, retina_masks: bool = True):
        self.weights = weights
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.half = half
        self.max_det = max_det
        self.retina_masks = retina_masks
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as e:                       # pragma: no cover
                raise ImportError(
                    "ultralytics 가 필요합니다: pip install ultralytics") from e
            self._model = YOLO(self.weights)
        return self._model

    @property
    def names(self) -> List[str]:
        m = self._ensure()
        n = getattr(m, "names", {})
        return [n[i] for i in sorted(n)] if isinstance(n, dict) else list(n)

    def predict(self, images: Sequence[np.ndarray]) -> List[List[Dict]]:
        m = self._ensure()
        results = m.predict(list(images), imgsz=self.imgsz, conf=self.conf,
                            iou=self.iou, device=self.device, half=self.half,
                            max_det=self.max_det, retina_masks=self.retina_masks,
                            verbose=False)
        out: List[List[Dict]] = []
        for r in results:
            dets: List[Dict] = []
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                out.append(dets)
                continue
            names = r.names if isinstance(r.names, dict) else dict(enumerate(r.names))
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            clsi = boxes.cls.cpu().numpy().astype(int)
            polys: List[Optional[np.ndarray]] = [None] * len(clsi)
            masks = getattr(r, "masks", None)
            if masks is not None and getattr(masks, "xy", None) is not None:
                for i, p in enumerate(masks.xy):
                    if i < len(polys) and p is not None and len(p) >= 3:
                        polys[i] = np.asarray(p, dtype=np.float64)
            for i in range(len(clsi)):
                dets.append({"cls": str(names.get(int(clsi[i]), str(clsi[i]))),
                             "conf": float(conf[i]),
                             "xyxy": xyxy[i].astype(np.float64),
                             "polygon": polys[i]})
            out.append(dets)
        return out


# --------------------------------------------------------------------------- #
# NMS
# --------------------------------------------------------------------------- #

def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def nms_per_class(dets: List[Detection], iou_thresh: float = 0.55) -> List[Detection]:
    """타일 경계에서 잘린 같은 결함이 두 번 잡히는 것을 정리한다."""
    keep: List[Detection] = []
    by_cls: Dict[str, List[Detection]] = {}
    for d in dets:
        by_cls.setdefault(d.cls_name, []).append(d)
    for _, group in by_cls.items():
        group = sorted(group, key=lambda d: -d.conf)
        boxes = np.stack([g.xyxy for g in group])
        alive = np.ones(len(group), bool)
        for i in range(len(group)):
            if not alive[i]:
                continue
            keep.append(group[i])
            if i + 1 < len(group):
                ious = _iou_matrix(boxes[i:i + 1], boxes[i + 1:])[0]
                alive[i + 1:][ious > iou_thresh] = False
    return keep


# --------------------------------------------------------------------------- #
# 타일 추론기
# --------------------------------------------------------------------------- #

class YoloDetector:
    """
    타일 추론 + 좌표 복원 + NMS + 택소노미 매핑 + 보정 을 묶은 검출기.

    오소모자이크는 한 변이 2만 픽셀을 넘기도 한다. 통째로 imgsz=960 에
    리사이즈해 넣으면 셀 핫스팟은 1픽셀 미만이 되어 사라진다. 반드시 타일로 썰되,
    타일 경계에 걸친 결함을 위해 overlap 을 준다.
    """

    def __init__(self,
                 runtime: YoloRuntime,
                 class_map: Optional[Dict[str, str]] = None,
                 calibrator: Optional[ConfidenceCalibrator] = None,
                 tile: int = 1280,
                 overlap: float = 0.2,
                 min_conf: float = 0.15,
                 nms_iou: float = 0.55,
                 batch: int = 8,
                 source: str = ""):
        self.runtime = runtime
        self.class_map = dict(class_map or {})
        self.calibrator = calibrator or ConfidenceCalibrator()
        self.tile = tile
        self.overlap = float(np.clip(overlap, 0.0, 0.6))
        self.min_conf = min_conf
        self.nms_iou = nms_iou
        self.batch = batch
        self.source = source

    # -- 타일 좌표 --------------------------------------------------------- #
    def _tiles(self, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        if max(h, w) <= self.tile * 1.15:
            return [(0, 0, w, h)]
        step = int(self.tile * (1.0 - self.overlap))
        xs = list(range(0, max(w - self.tile, 0) + 1, step)) or [0]
        ys = list(range(0, max(h - self.tile, 0) + 1, step)) or [0]
        if xs[-1] + self.tile < w:
            xs.append(max(0, w - self.tile))
        if ys[-1] + self.tile < h:
            ys.append(max(0, h - self.tile))
        return [(x, y, min(x + self.tile, w), min(y + self.tile, h))
                for y in ys for x in xs]

    # -- 추론 -------------------------------------------------------------- #
    def detect(self, image: np.ndarray) -> List[Detection]:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        tiles = self._tiles(h, w)

        out: List[Detection] = []
        for i in range(0, len(tiles), self.batch):
            chunk = tiles[i:i + self.batch]
            crops = [image[y0:y1, x0:x1] for (x0, y0, x1, y1) in chunk]
            preds = self.runtime.predict(crops)
            for (x0, y0, _, _), dets in zip(chunk, preds):
                for d in dets:
                    conf = float(d["conf"])
                    if conf < self.min_conf:
                        continue
                    name = str(d["cls"])
                    box = np.asarray(d["xyxy"], dtype=np.float64).copy()
                    box[[0, 2]] += x0
                    box[[1, 3]] += y0
                    poly = d.get("polygon")
                    if poly is not None:
                        poly = np.asarray(poly, dtype=np.float64) + np.array([x0, y0])
                    taxon = self.class_map.get(name, name if name in CLASS_INDEX else None)
                    out.append(Detection(
                        cls_name=name, taxon=taxon, conf=conf,
                        calibrated=self.calibrator(name, conf),
                        xyxy=box, polygon=poly, source=self.source))

        return nms_per_class(out, self.nms_iou)

    # -- 진단 -------------------------------------------------------------- #
    def unmapped_classes(self, dets: Iterable[Detection]) -> List[str]:
        """택소노미에 매핑되지 않은 모델 클래스. 설정 실수를 조기에 잡는다."""
        return sorted({d.cls_name for d in dets if d.taxon is None})

    def preflight(self) -> Dict:
        """
        추론 시작 전 자기 점검. 모델이 실제로 가진 클래스명과 매핑 테이블을 대조한다.

        오타 하나로 클래스가 통째로 버려져도 파이프라인은 아무 에러 없이
        "정상"만 뱉는다. 조용한 실패라 배포 직후에 반드시 한 번 확인해야 한다.
        가중치 로딩이 필요하므로 런타임이 지원하지 않으면 빈 결과를 돌려준다.
        """
        try:
            names = list(self.runtime.names)
        except Exception as e:                                  # pragma: no cover
            return {"error": str(e)}
        mapped, unmapped = {}, []
        for n in names:
            t = self.class_map.get(n, n if n in CLASS_INDEX else None)
            if t is None:
                unmapped.append(n)
            else:
                mapped[n] = t
        return {"model_classes": names, "mapped": mapped, "unmapped": unmapped,
                "calibrated": not self.calibrator.is_identity}


def load_class_map(path: Optional[str],
                   default: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    모델 클래스명 -> 파이프라인 택소노미 매핑을 JSON 에서 읽는다.

        { "hot_cell": "hotspot_cell", "bypass": "hotspot_substring" }

    학습 때 쓴 클래스 이름은 팀마다 다르므로, 이 매핑을 모델 옆에 같이 두는 것이
    가장 사고가 적다. 경로가 없으면 기본 매핑을 쓴다.
    """
    if not path:
        return dict(default or {})
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    merged = dict(default or {})
    merged.update({str(k): str(v) for k, v in d.items()})
    return merged


# --------------------------------------------------------------------------- #
# 패널 검출 (YOLO-seg)
# --------------------------------------------------------------------------- #

class YoloPanelDetector:
    """
    YOLOv11-seg 로 모듈 폴리곤을 뽑아 PanelROI 로 변환한다.

    규칙 기반 `detect_panels` 의 약점(완전 포화된 글레어 패널을 놓침, 맞붙은
    테이블 분할이 추정에 의존함)을 정면으로 해결하는 경로다.
    seg 모델이 없으면 detect 모델의 박스를 사각 폴리곤으로 써도 동작은 한다.
    """

    def __init__(self, detector: YoloDetector,
                 panel_classes: Sequence[str] = ("panel", "module", "pv_module"),
                 min_conf: float = 0.35,
                 array_grouping_px: Optional[float] = None,
                 approx_eps_frac: float = 0.02):
        self.detector = detector
        self.panel_classes = {c.lower() for c in panel_classes}
        self.min_conf = min_conf
        self.array_grouping_px = array_grouping_px
        self.approx_eps_frac = approx_eps_frac

    def __call__(self, rgb: np.ndarray) -> List[PanelROI]:
        from .panels import _assign_arrays, dominant_grid_angle

        dets = [d for d in self.detector.detect(rgb)
                if d.cls_name.lower() in self.panel_classes and d.conf >= self.min_conf]
        angle = dominant_grid_angle(rgb)
        rois: List[PanelROI] = []
        for i, d in enumerate(sorted(dets, key=lambda x: (x.center[1], x.center[0]))):
            if d.polygon is not None and len(d.polygon) >= 4:
                c = d.polygon.astype(np.float32)
                eps = self.approx_eps_frac * cv2.arcLength(c, True)
                poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)
                if len(poly) > 6:                       # 지나치게 복잡하면 회전사각형으로
                    poly = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float64)
            else:
                x0, y0, x1, y1 = d.xyxy
                poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
            rois.append(PanelROI(panel_id=f"P{i:05d}", polygon_rgb=poly,
                                 grid_angle_deg=angle,
                                 meta={"det_conf": float(d.conf), "src": "yolo-seg"}))
        if self.array_grouping_px:
            _assign_arrays(rois, self.array_grouping_px)
        return rois


# --------------------------------------------------------------------------- #
# 기본 클래스 매핑
# --------------------------------------------------------------------------- #

DEFAULT_RGB_CLASS_MAP: Dict[str, str] = {
    "soiling": "soiling", "dust": "soiling", "dirt": "soiling", "bird_drop": "soiling",
    "shadow": "shading", "shading": "shading", "obstruction": "shading",
    "crack": "cell_crack", "cell_crack": "cell_crack", "snail_trail": "cell_crack",
    "microcrack": "cell_crack",
    "discoloration": "discoloration", "yellowing": "discoloration", "browning": "discoloration",
    "delamination": "delamination", "bubble": "delamination",
    "glass_break": "glass_breakage", "broken_glass": "glass_breakage",
    "shattered": "glass_breakage",
    "panel": "__panel__", "module": "__panel__",
}

DEFAULT_IR_CLASS_MAP: Dict[str, str] = {
    "hotspot": "hotspot_cell", "hot_cell": "hotspot_cell", "single_cell": "hotspot_cell",
    "multi_cell": "hotspot_cell",
    "substring": "hotspot_substring", "bypass_diode": "hotspot_substring",
    "diode": "hotspot_substring", "string_hot": "hotspot_substring",
    "module_hot": "module_open", "open_circuit": "module_open", "offline": "module_open",
    "junction_box": "junction_box", "jbox": "junction_box",
    "pid": "pid",
    "soiling": "soiling", "dirt_thermal": "soiling",
    "shadow": "shading", "shading": "shading",
    "panel": "__panel__", "module": "__panel__",
}

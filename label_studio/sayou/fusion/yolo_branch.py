"""
YOLOv11 기반 분기 분석기.

규칙 기반 분기(`rgb_branch.py`, `ir_branch.py`)와 **완전히 동일한 인터페이스**를
구현한다. 따라서 `fusion.py` 와 `pipeline.py` 는 한 줄도 바뀌지 않아도 되고,
분기만 교체하거나 두 방식을 혼합할 수 있다.

    analyze(image, roi, array_ref) -> BranchEvidence

설계에서 놓치기 쉬운 점 세 가지.

1. **YOLO 의 conf 를 그대로 확률로 쓰면 안 된다.**
   objectness x cls 에 NMS 가 얹힌 값이라 과신한다. 보정하지 않은 채
   Dempster-Shafer 에 넣으면 한쪽 분기가 거의 항상 이겨서 융합이 무의미해진다.
   `ConfidenceCalibrator` 를 반드시 검증셋으로 fit 해서 쓴다.

2. **"검출 없음"은 "정상"이 아니라 "이 모델이 못 봤음"이다.**
   입력 조건이 나쁘면(정반사 포화, 저일사량, GSD 부족) 미검출은 정보가 아니다.
   그래서 클래스 확률은 YOLO 가 만들고, **신뢰도(reliability)는 물리 특징이
   만든다.** 이 역할 분담이 late fusion 이 제대로 동작하는 조건이다.

3. **관측 가능 클래스 마스크는 모델이 아니라 모달리티가 정한다.**
   RGB 모델이 우연히 'hotspot' 클래스를 갖고 있어도 RGB 는 셀 발열을 볼 수 없다.
   마스크는 RGB_MASK / IR_MASK 로 고정한다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .ir_branch import IRPanelAnalyzer
from .rgb_branch import RGBPanelAnalyzer, evidence_to_probs
from .types import (CLASS_INDEX, IR_MASK, N_CLASSES, NORMAL, RGB_MASK,
                    BranchEvidence, PanelROI)
from .yolo_backend import Detection, ThermalNormalizer, YoloDetector


# --------------------------------------------------------------------------- #
# 검출 -> 패널 귀속
# --------------------------------------------------------------------------- #

def _poly_mask_bbox(polygon: np.ndarray) -> np.ndarray:
    return np.array([polygon[:, 0].min(), polygon[:, 1].min(),
                     polygon[:, 0].max(), polygon[:, 1].max()], dtype=np.float64)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / max(aa + bb - inter, 1e-9))


def assign_detections(dets: Sequence[Detection], polygon: np.ndarray,
                      small_ratio: float = 0.35,
                      iou_thresh: float = 0.30,
                      pad: float = 2.0) -> List[Detection]:
    """
    검출을 하나의 패널 ROI 에 귀속시킨다.

    결함 스케일이 두 갈래라 단일 기준으로는 안 된다.
      - 셀 핫스팟/크랙처럼 **작은** 검출: 중심이 패널 폴리곤 안에 있으면 귀속.
        (IoU 로 걸면 면적비가 작아 절대 임계를 못 넘는다)
      - 모듈 전체 발열처럼 **큰** 검출: 박스 IoU 로 귀속.
    """
    if polygon is None or len(polygon) < 3:
        return []
    pbox = _poly_mask_bbox(polygon)
    parea = max((pbox[2] - pbox[0]) * (pbox[3] - pbox[1]), 1e-9)
    cnt = polygon.astype(np.float32).reshape(-1, 1, 2)

    out: List[Detection] = []
    for d in dets:
        if d.taxon is None or d.taxon.startswith("__"):
            continue
        ratio = d.area / parea
        if ratio <= small_ratio:
            c = d.center
            if cv2.pointPolygonTest(cnt, (float(c[0]), float(c[1])), True) >= -pad:
                out.append(d)
        else:
            if _iou(d.xyxy, pbox) >= iou_thresh:
                out.append(d)
    return out


def detections_to_scores(dets: Sequence[Detection], mask: np.ndarray,
                         multi_instance_boost: float = 0.35) -> Dict[str, float]:
    """
    귀속된 검출들을 클래스별 '증거 강도'로 집계.

        s_c = -ln(1 - p_max)  ->  evidence_to_probs(gain=1) 이 p_max 를 그대로 복원
        같은 클래스가 여러 개면 noisy-OR 로 소폭 가산 (셀 핫스팟 3개 = 더 확실)
    """
    best: Dict[str, List[float]] = {}
    for d in dets:
        i = CLASS_INDEX.get(d.taxon)
        if i is None or not mask[i]:
            continue
        best.setdefault(d.taxon, []).append(float(np.clip(d.calibrated, 0.0, 0.999)))

    scores: Dict[str, float] = {}
    for cls, ps in best.items():
        ps = sorted(ps, reverse=True)
        p = ps[0]
        for extra in ps[1:4]:
            p = 1.0 - (1.0 - p) * (1.0 - multi_instance_boost * extra)
        scores[cls] = float(-np.log(max(1.0 - p, 1e-6)))
    return scores


# --------------------------------------------------------------------------- #
# RGB 분기
# --------------------------------------------------------------------------- #

class YoloRGBAnalyzer:
    """
    YOLOv11(RGB 학습) 분기.

    클래스 확률은 모델이, 신뢰도는 물리 특징(정반사 포화·노출·선명도·GSD)이 만든다.
    물리 특징은 `RGBPanelAnalyzer` 의 추출기를 그대로 재사용하므로
    `glare_suppress` 같은 교차 중재 규칙이 그대로 동작한다.
    """

    def __init__(self,
                 detector: YoloDetector,
                 min_pixels: int = 400,
                 glare_frac_warn: float = 0.05,
                 feature_extractor: Optional[RGBPanelAnalyzer] = None,
                 gain: float = 1.0):
        self.detector = detector
        self.min_pixels = min_pixels
        self.glare_frac_warn = glare_frac_warn
        self.features = feature_extractor or RGBPanelAnalyzer(min_pixels=min_pixels)
        self.gain = gain
        self._dets: List[Detection] = []
        self._orphans: List[Detection] = []

    # 프레임 단위 1회 추론 (파이프라인이 호출)
    def prepare(self, image: np.ndarray, rois: Sequence[PanelROI]) -> None:
        self._dets = self.detector.detect(image)
        assigned = set()
        for r in rois:
            for d in assign_detections(self._dets, r.polygon_rgb):
                assigned.add(id(d))
        self._orphans = [d for d in self._dets
                         if id(d) not in assigned and d.taxon
                         and not d.taxon.startswith("__")]

    def array_reference(self, image, rois):
        return self.features.array_reference(image, rois)

    def analyze(self, image: np.ndarray, roi: PanelROI,
                array_ref: Optional[Tuple[float, float]] = None) -> BranchEvidence:
        f = self.features.extract_features(image, roi, array_ref)
        notes: List[str] = []

        if f.get("n_pixels", 0) < self.min_pixels:
            p = np.zeros(N_CLASSES)
            p[CLASS_INDEX[NORMAL]] = 1.0
            notes.append("ROI 화소 부족 - RGB 분기 무효")
            return BranchEvidence("rgb", p, RGB_MASK, 0.0, f, notes)

        dets = assign_detections(self._dets, roi.polygon_rgb)
        scores = detections_to_scores(dets, RGB_MASK)
        probs = evidence_to_probs(scores, RGB_MASK, gain=self.gain)

        # -- 신뢰도: 입력이 판정을 지지하는가 -------------------------------- #
        rel = 1.0
        if f["glare_frac"] > self.glare_frac_warn:
            rel *= float(np.exp(-6.0 * (f["glare_frac"] - self.glare_frac_warn)))
            notes.append(f"정반사 포화 {f['glare_frac']*100:.1f}% - RGB 신뢰도 하락")
        if f["dark_frac"] > 0.35:
            rel *= 0.6
            notes.append("과소노출 영역 과다")
        rel *= float(np.clip(f["sharpness"] / 600.0, 0.3, 1.0))
        rel *= float(np.clip(f["n_pixels"] / (4.0 * self.min_pixels), 0.3, 1.0))

        if dets:
            f["yolo_n_det"] = float(len(dets))
            f["yolo_max_conf"] = float(max(d.conf for d in dets))
            notes.append("YOLO: " + ", ".join(
                f"{d.taxon}@{d.calibrated:.2f}" for d in sorted(
                    dets, key=lambda x: -x.calibrated)[:3]))
        else:
            f["yolo_n_det"] = 0.0
            f["yolo_max_conf"] = 0.0

        return BranchEvidence("rgb", probs, RGB_MASK, float(np.clip(rel, 0, 1)), f, notes)


# --------------------------------------------------------------------------- #
# IR 분기
# --------------------------------------------------------------------------- #

class YoloIRAnalyzer:
    """
    YOLOv11(열화상 학습) 분기.

    입력은 **정규화된 8bit 열화상**이고, 신뢰도와 교차 규칙용 특징(ΔT_inter,
    ΔT_intra)은 **radiometric 원본**에서 계산한다. 이 둘을 섞으면 안 된다.
    ΔT 는 온도 배열에서만 의미가 있고, 모델은 정규화된 화소만 본다.

    `normalizer` 는 학습 스크립트와 반드시 동일한 인스턴스를 써야 한다.
    """

    def __init__(self,
                 detector: YoloDetector,
                 normalizer: Optional[ThermalNormalizer] = None,
                 radiometric: bool = True,
                 min_pixels: int = 120,
                 feature_extractor: Optional[IRPanelAnalyzer] = None,
                 gain: float = 1.0,
                 iec_severity_check: bool = True):
        self.detector = detector
        self.normalizer = normalizer or ThermalNormalizer()
        self.radiometric = radiometric
        self.min_pixels = min_pixels
        self.features = feature_extractor or IRPanelAnalyzer(
            radiometric=radiometric, min_pixels=min_pixels)
        self.gain = gain
        self.iec_severity_check = iec_severity_check
        self._dets: List[Detection] = []
        self._orphans: List[Detection] = []
        self._u8: Optional[np.ndarray] = None

    def prepare(self, thermal: np.ndarray, rois: Sequence[PanelROI]) -> None:
        ref = None
        if self.radiometric:
            vals = thermal[np.isfinite(thermal)]
            ref = float(np.median(vals)) if vals.size else None
        self._u8 = self.normalizer.to_3ch(thermal, ref) if self.radiometric \
            else cv2.cvtColor(np.asarray(thermal, np.uint8), cv2.COLOR_GRAY2BGR)
        self._dets = self.detector.detect(self._u8)
        assigned = set()
        for r in rois:
            if r.polygon_ir is None:
                continue
            for d in assign_detections(self._dets, r.polygon_ir):
                assigned.add(id(d))
        self._orphans = [d for d in self._dets
                         if id(d) not in assigned and d.taxon
                         and not d.taxon.startswith("__")]

    def array_reference(self, thermal, rois):
        return self.features.array_reference(thermal, rois)

    def analyze(self, thermal: np.ndarray, roi: PanelROI,
                array_ref_c: Optional[float] = None) -> BranchEvidence:
        f = self.features.extract_features(thermal, roi, array_ref_c)
        notes: List[str] = []

        if roi.polygon_ir is None or f.get("n_pixels", 0) < self.min_pixels:
            p = np.zeros(N_CLASSES)
            p[CLASS_INDEX[NORMAL]] = 1.0
            notes.append("IR 미대응 ROI 또는 화소 부족 - IR 분기 무효")
            return BranchEvidence("ir", p, IR_MASK, 0.0, f, notes)

        dets = assign_detections(self._dets, roi.polygon_ir)
        scores = detections_to_scores(dets, IR_MASK)

        # IEC TS 62446-3 정합성 점검: 모델이 핫스팟이라 했는데 ΔT 가 따라오지 않으면
        # 감쇄한다. 열화상 모델은 반사·주변 구조물을 핫스팟으로 오검출하기 쉽고,
        # 그 오류는 온도값으로 직접 반증할 수 있다.
        if self.iec_severity_check and self.radiometric:
            dt_intra = float(f.get("dt_intra", 0.0))
            dt_inter = float(f.get("dt_inter", 0.0))
            gates = {
                "hotspot_cell": dt_intra / max(self.features.dt_cell, 1e-6),
                "hotspot_substring": max(f.get("substring_contrast", 0.0), dt_intra)
                / max(self.features.dt_substring, 1e-6),
                "module_open": dt_inter / max(self.features.dt_module, 1e-6),
                "junction_box": f.get("jbox_dt", 0.0) / max(self.features.dt_jbox, 1e-6),
            }
            for cls, g in gates.items():
                if cls in scores and g < 0.45:
                    scores[cls] *= 0.3
                    notes.append(f"{cls}: ΔT 근거 부족(비 {g:.2f}) - 감쇄")

        probs = evidence_to_probs(scores, IR_MASK, gain=self.gain)

        # -- 신뢰도 ---------------------------------------------------------- #
        rel = 1.0
        if not self.radiometric:
            rel *= 0.75
            notes.append("비-radiometric 입력 - ΔT 검증 불가")
        if not self.features.min_irradiance_ok:
            rel *= 0.5
            notes.append("일사량 조건 미달")
        noise_floor = self.features.netd_k if self.radiometric else 0.02
        rel *= float(np.clip(f.get("std", 0.0) / (3.0 * max(noise_floor, 1e-6)), 0.45, 1.0))
        rel *= float(np.clip(f["n_pixels"] / (6.0 * self.min_pixels), 0.3, 1.0))
        sc = getattr(self.features, "_scene_contrast", float("nan"))
        if np.isfinite(sc) and sc < 5.0 and self.radiometric:
            rel *= 0.6
            notes.append(f"패널-배경 ΔT {sc:.1f}K - 일사량/가동 상태 재확인")
        if f["n_pixels"] < 300:
            notes.append("IR GSD 부족 - 셀 단위 판별 한계")

        f["yolo_n_det"] = float(len(dets))
        f["yolo_max_conf"] = float(max((d.conf for d in dets), default=0.0))
        if dets:
            notes.append("YOLO: " + ", ".join(
                f"{d.taxon}@{d.calibrated:.2f}" for d in sorted(
                    dets, key=lambda x: -x.calibrated)[:3]))

        return BranchEvidence("ir", probs, IR_MASK, float(np.clip(rel, 0, 1)), f, notes)


# --------------------------------------------------------------------------- #
# 하이브리드 (YOLO + 규칙, 분기 내부 융합)
# --------------------------------------------------------------------------- #

class HybridBranchAnalyzer:
    """
    같은 모달리티 안에서 YOLO 분기와 규칙 분기를 먼저 합친 뒤 하나의
    BranchEvidence 로 내보낸다. (분기 내부 융합 -> 분기 간 late fusion)

    모델 학습 초기에 특히 유용하다. 라벨이 적은 클래스는 규칙이 받쳐주고,
    데이터가 쌓이면 `w_yolo` 를 1.0 으로 올려 규칙을 걷어내면 된다.

    결합은 신뢰도 가중 로그 선형(기하평균)이다. 두 분기가 같은 모달리티를 보므로
    관측 가능 클래스 마스크가 동일하고, 따라서 Dempster-Shafer 의 집합값
    초점원소가 필요 없다.
    """

    def __init__(self, yolo, rules, w_yolo: float = 0.7, prior: float = 0.05):
        self.yolo = yolo
        self.rules = rules
        self.w_yolo = float(np.clip(w_yolo, 0.0, 1.0))
        self.prior = prior

    def prepare(self, image, rois):
        if hasattr(self.yolo, "prepare"):
            self.yolo.prepare(image, rois)

    def array_reference(self, image, rois):
        return self.rules.array_reference(image, rois)

    def analyze(self, image, roi, array_ref=None) -> BranchEvidence:
        a = self.yolo.analyze(image, roi, array_ref)
        b = self.rules.analyze(image, roi, array_ref)

        mask = a.mask
        u = mask.astype(float) / max(mask.sum(), 1)
        pa = (1 - self.prior) * a.probs + self.prior * u
        pb = (1 - self.prior) * b.probs + self.prior * u
        w = self.w_yolo
        logp = w * np.log(np.maximum(pa, 1e-12)) + (1 - w) * np.log(np.maximum(pb, 1e-12))
        logp[~mask] = -np.inf
        logp -= logp[mask].max()
        probs = np.where(mask, np.exp(logp), 0.0)
        probs /= max(probs.sum(), 1e-12)

        feats = dict(b.features)
        feats.update({k: v for k, v in a.features.items() if k.startswith("yolo_")})
        rel = float(np.clip(w * a.reliability + (1 - w) * b.reliability, 0.0, 1.0))
        notes = [f"hybrid(w_yolo={w:.2f})"] + a.notes + b.notes
        return BranchEvidence(a.branch, probs, mask, rel, feats, notes)

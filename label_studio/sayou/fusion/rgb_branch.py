"""
RGB 분기 분석기 (late fusion 의 첫 번째 독립 분기).

이 분기는 IR 을 전혀 보지 않는다. Late fusion 의 정의상 각 분기는 독립적으로
결론을 내야 하고, 교차 참조는 fusion 단계에서만 일어난다.

산출물: BranchEvidence(probs over RGB_OBSERVABLE, reliability, features)

reliability 하향 요인
  - 정반사(specular) 포화: 패널 표면 글레어는 외관 판정을 무력화한다.
    (오소모자이크에서 프레임 경계의 밝기 스텝을 만드는 그 현상과 동일한 원인)
  - 저조도 / 과소노출, ROI 화소 수 부족, 모션 블러
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .panels import crop_polygon, deskew_patch
from .types import (CLASS_INDEX, N_CLASSES, NORMAL, RGB_MASK, BranchEvidence,
                    PanelROI)


# --------------------------------------------------------------------------- #
# 유틸
# --------------------------------------------------------------------------- #

def evidence_to_probs(scores: Dict[str, float], mask: np.ndarray,
                      gain: float = 1.0) -> np.ndarray:
    """
    결함별 '증거 강도'(0 이상) -> 확률 분포.

    softmax 는 점수가 0 인 클래스들이 많을수록 정상 클래스를 희석시킨다.
    (클래스 12개 중 11개가 0점이어도 정상 확률이 0.5 를 넘지 못한다)
    대신 noisy-OR 형태를 쓴다.

        P(c)      ∝ 1 - exp(-gain * s_c)
        P(normal) ∝ prod_c (1 - P(c))

    증거가 전혀 없으면 P(normal)=1 로 수렴하고, 증거가 하나라도 강하면
    해당 클래스가 지배한다. 클래스 개수에 대해 불변이라는 것이 핵심 장점이다.
    """
    p = np.zeros(N_CLASSES)
    prod = 1.0
    for cls, s in scores.items():
        i = CLASS_INDEX[cls]
        if cls == NORMAL or not mask[i]:
            continue
        pc = 1.0 - float(np.exp(-gain * max(float(s), 0.0)))
        p[i] = pc
        prod *= (1.0 - pc)
    p[CLASS_INDEX[NORMAL]] = prod
    total = p.sum()
    if total <= 1e-12:
        p = mask.astype(float)
        return p / max(p.sum(), 1e-12)
    return p / total


def _blur_score(gray: np.ndarray, mask: np.ndarray) -> float:
    """라플라시안 분산 기반 선명도. 낮으면 크랙 판정 신뢰도가 떨어진다."""
    if mask.sum() < 50:
        return 0.0
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(np.var(lap[mask]))


def _line_density(gray: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """
    패널 내부 선형 구조 밀도. 셀 경계·버스바는 규칙적이므로, 이를 제거한 뒤
    남는 비규칙 선분(크랙/스네일 트레일) 길이를 면적으로 정규화한다.
    """
    if mask.sum() < 200:
        return 0.0, 0.0
    g = gray.astype(np.float32)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g = cv2.createCLAHE(3.0, (4, 4)).apply(g)
    edges = cv2.Canny(g, 40, 120, L2gradient=True)
    edges[~mask] = 0

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=18,
                            minLineLength=max(8, int(0.08 * max(gray.shape))),
                            maxLineGap=3)
    if lines is None:
        return 0.0, 0.0

    total, irregular = 0.0, 0.0
    for x1, y1, x2, y2 in lines[:, 0]:
        L = float(np.hypot(x2 - x1, y2 - y1))
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        total += L
        # 축 정렬(0/90도) 근처는 셀 격자로 간주하고 제외
        d = min(abs(ang - 0), abs(ang - 90), abs(ang - 180))
        if d > 12.0:
            irregular += L
    area = float(mask.sum())
    return irregular / area * 1e3, total / area * 1e3


def _detrend_grid(L: np.ndarray, mask: np.ndarray, k: int = 5) -> np.ndarray:
    """
    셀 격자·버스바 같은 주기적 선 구조를 median 필터로 제거한 저주파 성분.
    오염 얼룩은 저주파, 격자는 고주파 선형 구조이므로 이 분리가 핵심이다.
    """
    x = L.astype(np.float32).copy()
    if mask.any():
        x[~mask] = float(np.median(x[mask]))
    k = max(3, min(5, k | 1))          # float32 medianBlur 는 ksize<=5 만 지원
    med = cv2.medianBlur(x, k)
    med = cv2.medianBlur(med, k)       # 2회 적용으로 더 굵은 선 구조까지 제거
    return cv2.GaussianBlur(med, (0, 0), max(2.0, k / 2.0))


def _dark_blob_frac(L: np.ndarray, mask: np.ndarray, ref_level: float,
                    drop: float = 10.0) -> Tuple[float, float]:
    """
    '주변 모듈 기준선보다 어두운' 연결 영역의 최대 면적비와 그 낙차.

    두 가지 함정을 피한다.
      - 이봉성(bimodality)만 보면 셀 격자선 때문에 모든 패널이 이봉으로 나온다.
        -> 공간적 연결성을 요구한다.
      - 패널 내부 Otsu 로 자르면 '정상 패널의 어두운 셀 vs 밝은 버스바'가
        갈라져 모든 패널이 90% 그림자로 판정된다.
        -> 임계값을 패널 내부가 아니라 어레이 기준선에서 가져온다.
    """
    if mask.sum() < 200:
        return 0.0, 0.0
    dark = ((L <= ref_level - drop) & mask).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, ker, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    if n <= 1:
        return 0.0, 0.0
    i = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    blob = labels == i
    frac = float(stats[i, cv2.CC_STAT_AREA]) / float(mask.sum())
    bright = mask & ~blob
    drop = float(np.median(L[bright]) - np.median(L[blob])) if bright.sum() > 50 else 0.0
    return frac, drop


# --------------------------------------------------------------------------- #
# 메인 분석기
# --------------------------------------------------------------------------- #

class RGBPanelAnalyzer:
    """
    규칙 기반 RGB 분기.

    설계 원칙은 IR 분기와 동일하다: **절대값이 아니라 주변 모듈 대비 상대값으로
    판정한다.** 시간대·노출·화이트밸런스가 절대 밝기를 통째로 바꾸기 때문에
    "이 패널이 밝다"가 아니라 "같은 어레이의 다른 패널보다 밝다"가 근거가 된다.

    학습 모델로 교체할 때는 analyze() 시그니처만 맞추면 이하 파이프라인은 그대로다.

        class MyCNN(RGBPanelAnalyzer):
            def analyze(self, image, roi, array_ref=None) -> BranchEvidence: ...
    """

    def __init__(self,
                 glare_sat_thresh: int = 245,
                 glare_frac_warn: float = 0.05,
                 dL_soiling: float = 8.0,       # 어레이 대비 L* 상승 [0~100]
                 db_discolor: float = 4.0,      # 어레이 대비 b* 상승
                 dL_shading: float = 10.0,      # 어레이 대비 L* 하락
                 crack_density_thresh: float = 0.8,
                 min_pixels: int = 400,
                 gain: float = 1.1):
        self.glare_sat_thresh = glare_sat_thresh
        self.glare_frac_warn = glare_frac_warn
        self.dL_soiling = dL_soiling
        self.db_discolor = db_discolor
        self.dL_shading = dL_shading
        self.crack_density_thresh = crack_density_thresh
        self.min_pixels = min_pixels
        self.gain = gain

    # -- 어레이 기준값 ---------------------------------------------------- #
    def array_reference(self, image: np.ndarray,
                        rois: Sequence[PanelROI]) -> Dict[Optional[str], Tuple[float, float]]:
        """어레이별 (L* 중앙값, b* 중앙값). 글레어 패널은 기준 산출에서 제외한다."""
        buckets: Dict[Optional[str], List[Tuple[float, float]]] = {}
        for r in rois:
            patch, mask = crop_polygon(image, r.polygon_rgb)
            if patch.size == 0 or mask.sum() < self.min_pixels:
                continue
            bgr = patch if patch.ndim == 3 else cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            lab = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2LAB)
            L = lab[..., 0].astype(np.float32)[mask] * 100.0 / 255.0
            b = lab[..., 2].astype(np.float32)[mask] - 128.0
            if float((L > 96.0).mean()) > 0.15:       # 포화 패널 제외
                continue
            buckets.setdefault(r.array_id, []).append(
                (float(np.median(L)), float(np.median(b))))
        ref: Dict[Optional[str], Tuple[float, float]] = {}
        for k, v in buckets.items():
            arr = np.asarray(v)
            ref[k] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
        if ref:
            arr = np.asarray(list(ref.values()))
            ref.setdefault(None, (float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))))
        return ref

    # -- 특징 추출 -------------------------------------------------------- #
    def extract_features(self, image: np.ndarray, roi: PanelROI,
                         array_ref: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
        patch, mask = crop_polygon(image, roi.polygon_rgb)
        if patch.size == 0 or mask.sum() < self.min_pixels:
            return {"n_pixels": float(mask.sum() if mask.size else 0)}

        patch, mask = deskew_patch(patch, mask, roi.grid_angle_deg)
        bgr = patch if patch.ndim == 3 else cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        bgr8 = bgr.astype(np.uint8)
        hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr8, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)

        V = hsv[..., 2].astype(np.float32)
        S = hsv[..., 1].astype(np.float32)
        Lf = lab[..., 0].astype(np.float32) * 100.0 / 255.0     # L* 0~100
        Bf = lab[..., 2].astype(np.float32) - 128.0
        Af = lab[..., 1].astype(np.float32) - 128.0

        L_med = float(np.median(Lf[mask]))
        b_med = float(np.median(Bf[mask]))
        refL, refb = array_ref if array_ref is not None else (L_med, b_med)
        dL = L_med - refL
        db = b_med - refb

        glare = float((V[mask] >= self.glare_sat_thresh).mean())
        dark = float((V[mask] <= 45).mean())

        # 격자 제거 후 저주파 얼룩 강도 (오염의 공간적 서명)
        low = _detrend_grid(Lf, mask)
        mottling = float(np.std(low[mask]))

        dark_frac, dark_drop = _dark_blob_frac(Lf, mask, refL, self.dL_shading)

        # 박리: 밝고 채도 낮은 국소 얼룩
        vm, vs = float(V[mask].mean()), float(V[mask].std())
        delam = ((V > vm + 2.0 * vs) & (S < 60) & mask)
        delam_frac = float(delam.sum()) / float(mask.sum())

        crack_density, line_total = _line_density(gray, mask)
        sharpness = _blur_score(gray, mask)
        hf = float(np.mean(np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F))[mask]))

        return {
            "n_pixels": float(mask.sum()),
            "glare_frac": glare,
            "dark_frac": dark,
            "L_median": L_med,
            "b_median": b_med,
            "dL": float(dL),
            "db": float(db),
            "mottling": mottling,
            "dark_blob_frac": dark_frac,
            "dark_blob_drop": dark_drop,
            "sat_mean": float(S[mask].mean()),
            "a_mean": float(Af[mask].mean()),
            "delam_frac": delam_frac,
            "crack_density": crack_density,
            "line_total": line_total,
            "sharpness": sharpness,
            "hf_energy": hf,
        }

    # -- 점수화 ----------------------------------------------------------- #
    def analyze(self, image: np.ndarray, roi: PanelROI,
                array_ref: Optional[Tuple[float, float]] = None) -> BranchEvidence:
        f = self.extract_features(image, roi, array_ref)
        notes: List[str] = []

        if f.get("n_pixels", 0) < self.min_pixels:
            probs = np.zeros(N_CLASSES)
            probs[CLASS_INDEX[NORMAL]] = 1.0
            notes.append("ROI 화소 부족 - RGB 분기 무효")
            return BranchEvidence("rgb", probs, RGB_MASK, 0.0, f, notes)

        s: Dict[str, float] = {}

        # soiling : 주변보다 밝고(퇴적층 산란) + 얼룩 형태의 저주파 불균일
        soil = 1.4 * max(0.0, f["dL"] / self.dL_soiling - 1.0) \
            + 0.55 * max(0.0, f["mottling"] - 2.0)
        s["soiling"] = float(np.clip(soil, 0, 5))

        # shading : 주변보다 어둡고 + 연결된 어두운 영역이 넓고 낙차가 큼
        shade = 1.6 * max(0.0, (-f["dL"]) / self.dL_shading - 1.0) \
            + 2.2 * max(0.0, f["dark_blob_frac"] - 0.15)
        s["shading"] = float(np.clip(shade, 0, 5))

        # cell_crack / snail trail : 비규칙 선분 밀도
        crack = 1.6 * max(0.0, f["crack_density"] - self.crack_density_thresh)
        if f["sharpness"] < 400.0:
            crack *= 0.4
            notes.append("저선명도 - 크랙 근거 약화")
        s["cell_crack"] = float(np.clip(crack, 0, 5))

        # discoloration : b* 황변이 뚜렷하되 얼룩지지 않고 균일할 때
        uniformity = float(np.clip(1.0 - (f["mottling"] - 2.0) / 6.0, 0.1, 1.0))
        disc = 1.3 * max(0.0, f["db"] / self.db_discolor - 1.0) * uniformity
        s["discoloration"] = float(np.clip(disc, 0, 5))

        # delamination
        s["delamination"] = float(np.clip(7.0 * max(0.0, f["delam_frac"] - 0.05), 0, 5))

        # glass_breakage : 고주파 에너지 + 다방향 선분 동반
        brk = 0.04 * max(0.0, f["hf_energy"] - 25.0) + 0.5 * max(0.0, f["crack_density"] - 3.0)
        s["glass_breakage"] = float(np.clip(brk, 0, 5))

        probs = evidence_to_probs(s, RGB_MASK, gain=self.gain)

        # -- 신뢰도 -------------------------------------------------------- #
        rel = 1.0
        if f["glare_frac"] > self.glare_frac_warn:
            rel *= float(np.exp(-6.0 * (f["glare_frac"] - self.glare_frac_warn)))
            notes.append(f"정반사 포화 {f['glare_frac']*100:.1f}% - 외관 판정 신뢰도 하락")
        if f["dark_frac"] > 0.35:
            rel *= 0.6
            notes.append("과소노출 영역 과다")
        if array_ref is None:
            rel *= 0.7
            notes.append("어레이 기준값 없음 - 상대 판정 불가")
        rel *= float(np.clip(f["sharpness"] / 600.0, 0.3, 1.0))
        rel *= float(np.clip(f["n_pixels"] / (4.0 * self.min_pixels), 0.3, 1.0))

        return BranchEvidence("rgb", probs, RGB_MASK, float(np.clip(rel, 0, 1)), f, notes)

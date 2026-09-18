"""
IR(열화상) 분기 분석기 (late fusion 의 두 번째 독립 분기).

핵심 개념: 절대 온도가 아니라 ΔT(패널 내 편차, 어레이 기준 편차)로 판정한다.
   ΔT_intra = T_max(패널 내) - T_median(패널 내)
   ΔT_inter = T_median(패널) - T_median(어레이 전체)

IEC TS 62446-3 스타일 패턴 분류:
   - 단일 셀 핫스팟   : 작은 국소 블롭, ΔT_intra 큼
   - 서브스트링/다이오드 : 패널을 3(또는 2)분할했을 때 한 구획 전체가 상승
   - 모듈 전체 발열   : ΔT_inter 큼, 내부 균일
   - 정션박스        : 패널 장축 끝단 좁은 대역 발열
   - PID             : 프레임(테두리)에서 중앙으로 향하는 온도 구배
   - 그림자/오염      : 온도 상승이지만 경계가 외부 요인 형상과 일치, 보통 ΔT 작음

radiometric 입력이 아닌 8bit 정규화 영상이면 ΔT 단위가 [K]가 아니므로
`radiometric=False` 로 두고 상대 판정만 수행한다(신뢰도 자동 하향).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .panels import crop_polygon, deskew_patch
from .types import (CLASS_INDEX, IR_MASK, N_CLASSES, NORMAL, BranchEvidence,
                    PanelROI)
from .rgb_branch import evidence_to_probs


# --------------------------------------------------------------------------- #
# 라디오메트릭 변환
# --------------------------------------------------------------------------- #

def dn_to_celsius(dn: np.ndarray, scale: float = 0.04, offset: float = -273.15) -> np.ndarray:
    """
    16bit DN -> 섭씨. DJI R-JPEG 를 TSDK 로 뽑은 경우 보통 0.1 K/DN 또는
    Kelvin*100 형식이다. 기본값은 FLIR 계열 0.04 K/DN 관례.
    실제 값은 반드시 카메라 메타데이터로 확인할 것.
    """
    return dn.astype(np.float32) * scale + offset


def apply_emissivity_correction(t_c: np.ndarray, emissivity: float = 0.85,
                                t_reflected_c: float = 20.0) -> np.ndarray:
    """
    단순 방사율 보정 (Stefan-Boltzmann 근사).
    유리 표면 방사율은 통상 0.85~0.90. 보정 없으면 ΔT 가 과소평가된다.
    """
    T = t_c + 273.15
    Tr = t_reflected_c + 273.15
    e = float(np.clip(emissivity, 0.05, 1.0))
    obj4 = (T ** 4 - (1.0 - e) * Tr ** 4) / e
    obj4 = np.clip(obj4, 1.0, None)
    return obj4 ** 0.25 - 273.15


# --------------------------------------------------------------------------- #
# 패턴 특징
# --------------------------------------------------------------------------- #

def _substring_profile(t: np.ndarray, mask: np.ndarray, n_sub: int = 3
                       ) -> Tuple[float, int, np.ndarray]:
    """
    패널 장축을 n_sub 등분하여 각 구획 중앙값을 계산.
    반환: (최대구획 - 중앙값구획, argmax, 구획 중앙값 배열)
    """
    h, w = t.shape[:2]
    axis = 1 if w >= h else 0
    n = t.shape[axis]
    med = []
    for k in range(n_sub):
        sl = slice(int(k * n / n_sub), int((k + 1) * n / n_sub))
        sub = t[:, sl] if axis == 1 else t[sl, :]
        m = mask[:, sl] if axis == 1 else mask[sl, :]
        med.append(np.median(sub[m]) if m.sum() > 20 else np.nan)
    med = np.asarray(med, dtype=np.float64)
    if np.all(np.isnan(med)):
        return 0.0, -1, med
    contrast = float(np.nanmax(med) - np.nanmedian(med))
    return contrast, int(np.nanargmax(med)), med


def _blob_stats(t: np.ndarray, mask: np.ndarray, delta_thresh: float
                ) -> Tuple[float, float, float]:
    """
    기준 온도 대비 +delta_thresh 이상인 연결 블롭 중 최대 것의
    (면적비, 최대ΔT, 원형도) 를 반환. 셀 핫스팟 판별용.
    """
    if mask.sum() < 50:
        return 0.0, 0.0, 0.0
    base = float(np.median(t[mask]))
    hot = ((t - base) >= delta_thresh) & mask
    hot_u8 = hot.astype(np.uint8)
    hot_u8 = cv2.morphologyEx(hot_u8, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hot_u8, 8)
    if n <= 1:
        return 0.0, 0.0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = int(np.argmax(areas)) + 1
    area = float(areas[i - 1])
    blob = labels == i
    dmax = float((t[blob] - base).max())
    w_, h_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    circ = float(area / max(w_ * h_, 1))
    return area / float(mask.sum()), dmax, circ


def _edge_gradient(t: np.ndarray, mask: np.ndarray, band_frac: float = 0.15) -> float:
    """PID 지표: 프레임 인접 대역 평균 - 중앙부 평균."""
    h, w = t.shape[:2]
    by, bx = max(1, int(h * band_frac)), max(1, int(w * band_frac))
    edge = np.zeros_like(mask)
    edge[:by, :] = edge[-by:, :] = True
    edge[:, :bx] = edge[:, -bx:] = True
    edge &= mask
    core = mask & ~edge
    if edge.sum() < 20 or core.sum() < 20:
        return 0.0
    return float(np.median(t[edge]) - np.median(t[core]))


def _junction_box_score(t: np.ndarray, mask: np.ndarray) -> float:
    """장축 끝단 10% 대역의 국소 발열 여부."""
    h, w = t.shape[:2]
    axis = 1 if w >= h else 0
    n = t.shape[axis]
    k = max(2, int(0.10 * n))
    base = float(np.median(t[mask]))
    ends = []
    for sl in (slice(0, k), slice(n - k, n)):
        sub = t[:, sl] if axis == 1 else t[sl, :]
        m = mask[:, sl] if axis == 1 else mask[sl, :]
        if m.sum() > 10:
            ends.append(float(np.percentile(sub[m], 98) - base))
    return max(ends) if ends else 0.0


# --------------------------------------------------------------------------- #
# 메인 분석기
# --------------------------------------------------------------------------- #

class IRPanelAnalyzer:
    """
    규칙 기반 IR 분기. 임계값은 IEC TS 62446-3 및 현장 관행 기준 기본값이며,
    사이트 조건(일사량 > 600 W/m², 풍속 < 4 m/s)에서 유효하다.
    """

    def __init__(self,
                 radiometric: bool = True,
                 dt_cell: float = 10.0,        # 단일 셀 핫스팟 ΔT [K]
                 dt_substring: float = 5.0,    # 서브스트링 ΔT [K]
                 dt_module: float = 4.0,       # 모듈 전체 ΔT_inter [K]
                 dt_jbox: float = 6.0,
                 dt_pid: float = 3.0,
                 min_irradiance_ok: bool = True,
                 min_pixels: int = 120,
                 netd_k: float = 0.08,
                 gain: float = 1.1):
        self.radiometric = radiometric
        self.dt_cell = dt_cell
        self.dt_substring = dt_substring
        self.dt_module = dt_module
        self.dt_jbox = dt_jbox
        self.dt_pid = dt_pid
        self.min_irradiance_ok = min_irradiance_ok
        self.min_pixels = min_pixels
        self.netd_k = netd_k
        self.gain = gain
        self._scene_contrast: float = float("nan")

    # ------------------------------------------------------------------ #
    def array_reference(self, thermal: np.ndarray, rois: List[PanelROI]
                        ) -> Dict[Optional[str], float]:
        """
        어레이별 기준 온도(중앙값)를 미리 계산.
        ΔT_inter 는 '주변 정상 모듈 대비' 여야 하므로 전체 화면 중앙값이 아니라
        같은 어레이/스트링 내부 중앙값을 기준으로 삼는다.
        """
        buckets: Dict[Optional[str], List[float]] = {}
        for r in rois:
            if r.polygon_ir is None:
                continue
            patch, mask = crop_polygon(thermal, r.polygon_ir)
            if patch.size == 0 or mask.sum() < self.min_pixels:
                continue
            buckets.setdefault(r.array_id, []).append(float(np.median(patch[mask])))
        ref = {k: float(np.median(v)) for k, v in buckets.items() if v}
        if ref:
            ref[None] = ref.get(None, float(np.median(list(ref.values()))))

        # 장면 열 대비 = (모듈 중앙값) - (비패널 배경 중앙값).
        # 주의: '모듈 간 편차'로 정의하면 안 된다. 건전한 어레이는 모듈 간 편차가
        # 작은 것이 정상이므로, 그것을 신뢰도 하락 사유로 쓰면 정상 사이트가
        # 통째로 저신뢰 처리된다. 여기서 보고 싶은 것은 '발전 중이고 일사가 충분한가'
        # 이며, 그 지표는 패널이 배경보다 확실히 뜨거운가이다.
        allmed = [x for v in buckets.values() for x in v]
        if allmed:
            bg = np.ones(thermal.shape[:2], np.uint8)
            for r in rois:
                if r.polygon_ir is not None:
                    cv2.fillPoly(bg, [np.round(r.polygon_ir).astype(np.int32)], 0)
            bgvals = thermal[bg.astype(bool)]
            bg_med = float(np.median(bgvals)) if bgvals.size > 100 else float("nan")
            self._scene_contrast = float(np.median(allmed)) - bg_med
        else:
            self._scene_contrast = float("nan")
        return ref

    # ------------------------------------------------------------------ #
    def extract_features(self, thermal: np.ndarray, roi: PanelROI,
                         array_ref_c: Optional[float]) -> Dict[str, float]:
        if roi.polygon_ir is None:
            return {"n_pixels": 0.0}
        patch, mask = crop_polygon(thermal, roi.polygon_ir)
        if patch.size == 0 or mask.sum() < self.min_pixels:
            return {"n_pixels": float(mask.sum() if mask.size else 0)}

        patch, mask = deskew_patch(patch.astype(np.float32), mask, roi.grid_angle_deg)
        t = patch.astype(np.float32)
        vals = t[mask]

        med = float(np.median(vals))
        p98 = float(np.percentile(vals, 98))
        p02 = float(np.percentile(vals, 2))
        dt_intra = p98 - med
        dt_inter = med - array_ref_c if array_ref_c is not None else 0.0

        blob_frac, blob_dt, blob_circ = _blob_stats(t, mask, max(2.0, 0.5 * self.dt_cell))
        sub_contrast, sub_idx, sub_med = _substring_profile(t, mask, 3)
        sub2_contrast, _, _ = _substring_profile(t, mask, 2)
        pid_grad = _edge_gradient(t, mask)
        jbox = _junction_box_score(t, mask)

        uniformity = float(np.std(vals))
        # 균일 발열도: 내부 편차 대비 모듈 전체 편차 비율
        homog = float(abs(dt_inter) / max(uniformity, 1e-3))

        return {
            "n_pixels": float(mask.sum()),
            "t_median": med,
            "t_p98": p98,
            "t_p02": p02,
            "dt_intra": float(dt_intra),
            "dt_inter": float(dt_inter),
            "blob_area_frac": blob_frac,
            "blob_dt": blob_dt,
            "blob_circularity": blob_circ,
            "substring_contrast": sub_contrast,
            "substring_index": float(sub_idx),
            "half_contrast": sub2_contrast,
            "pid_gradient": pid_grad,
            "jbox_dt": jbox,
            "std": uniformity,
            "homogeneity": homog,
        }

    # ------------------------------------------------------------------ #
    def analyze(self, thermal: np.ndarray, roi: PanelROI,
                array_ref_c: Optional[float] = None) -> BranchEvidence:
        f = self.extract_features(thermal, roi, array_ref_c)
        notes: List[str] = []

        if f.get("n_pixels", 0) < self.min_pixels:
            probs = np.zeros(N_CLASSES)
            probs[CLASS_INDEX[NORMAL]] = 1.0
            reason = "IR 미대응 ROI" if roi.polygon_ir is None else "ROI 화소 부족"
            notes.append(f"{reason} - IR 분기 무효")
            return BranchEvidence("ir", probs, IR_MASK, 0.0, f, notes)

        # 비-라디오메트릭이면 ΔT 스케일을 std 기반으로 표준화
        k = 1.0
        if not self.radiometric:
            k = 1.0 / max(f["std"], 1e-3)
            notes.append("비-라디오메트릭 입력 - ΔT 는 상대 단위")

        dt_intra = f["dt_intra"] * (k if not self.radiometric else 1.0)
        sub = f["substring_contrast"] * (k if not self.radiometric else 1.0)
        dt_inter = f["dt_inter"] * (k if not self.radiometric else 1.0)
        c_cell = self.dt_cell if self.radiometric else 3.0
        c_sub = self.dt_substring if self.radiometric else 1.5
        c_mod = self.dt_module if self.radiometric else 1.2
        c_jbox = self.dt_jbox if self.radiometric else 2.0
        c_pid = self.dt_pid if self.radiometric else 1.0

        s: Dict[str, float] = {}

        # 단일 셀 핫스팟: 작고(면적 5% 이하) 뚜렷한 블롭
        cell = 0.0
        if f["blob_area_frac"] < 0.12 and f["blob_dt"] > 0:
            cell = 2.0 * (f["blob_dt"] / max(c_cell, 1e-6)) * (1.0 + f["blob_circularity"]) / 2.0
        s["hotspot_cell"] = float(np.clip(cell, 0, 5))

        # 서브스트링: 구획 대비가 크고 블롭이 넓다
        ss = 2.2 * max(0.0, sub / max(c_sub, 1e-6) - 0.6)
        if f["blob_area_frac"] > 0.18:
            ss *= 1.3
        s["hotspot_substring"] = float(np.clip(ss, 0, 5))

        # 모듈 전체: ΔT_inter 크고 내부는 균일
        mo = 2.4 * max(0.0, dt_inter / max(c_mod, 1e-6) - 0.7)
        if f["homogeneity"] < 0.8:
            mo *= 0.5
        s["module_open"] = float(np.clip(mo, 0, 5))

        # 정션박스
        s["junction_box"] = float(np.clip(
            2.0 * max(0.0, f["jbox_dt"] / max(c_jbox, 1e-6) - 0.8), 0, 5))

        # PID: 테두리 고온 구배
        s["pid"] = float(np.clip(
            1.8 * max(0.0, f["pid_gradient"] / max(c_pid, 1e-6) - 0.8), 0, 5))

        # 오염 / 그림자: 모듈 전체가 '약하게, 균일하게' 더 뜨거운 상태.
        #   ΔT_inter 가 1~3K 수준이면서 내부 편차가 작다 = 표면 요인(오염/차폐)
        #   ΔT_inter 가 그보다 크면 위의 module_open 이 가져간다.
        # IR 만으로는 오염과 그림자를 구분할 수 없다. 두 클래스에 비슷한 점수를
        # 주고 판단을 보류하는 것이 정직하며, 이 모호성은 RGB 분기가 해소한다.
        # (late fusion 이 실제로 값을 만들어내는 지점이 바로 여기다)
        mild = float(np.clip((dt_inter - 0.6) / max(0.6 * c_mod, 1e-6), 0.0, 1.5))
        uniform_f = 1.0 - float(np.clip(dt_intra / max(c_cell, 1e-6), 0.0, 1.0))
        base = 1.4 * mild * uniform_f
        if dt_inter > 1.2 * c_mod:
            base *= 0.3
        s["soiling"] = float(np.clip(base, 0, 3))
        s["shading"] = float(np.clip(0.8 * base, 0, 3))
        if base > 0.5:
            notes.append("표면 요인(오염/차폐) 의심 - IR 단독으로는 구분 불가")

        probs = evidence_to_probs(s, IR_MASK, gain=self.gain)

        # -- 신뢰도 -------------------------------------------------------- #
        rel = 1.0
        if not self.radiometric:
            rel *= 0.75
        if not self.min_irradiance_ok:
            rel *= 0.5
            notes.append("일사량 조건 미달 - ΔT 판정 신뢰도 하락")
        # 센서 잡음(NETD) 대비 신호가 너무 작으면 판정 불가.
        # 주의: '패널이 균일하다'는 것은 정상의 근거이지 신뢰도 하락 사유가 아니다.
        # 따라서 패널 내부 std 가 아니라 NETD 기준 잡음 여유로만 할인한다.
        noise_floor = self.netd_k if self.radiometric else 0.02
        rel *= float(np.clip(f["std"] / (3.0 * max(noise_floor, 1e-6)), 0.45, 1.0))
        if np.isfinite(self._scene_contrast) and self._scene_contrast < 5.0 and self.radiometric:
            rel *= 0.6
            notes.append(f"패널-배경 ΔT {self._scene_contrast:.1f}K - 일사량/가동 상태 재확인")
        rel *= float(np.clip(f["n_pixels"] / (6.0 * self.min_pixels), 0.3, 1.0))
        if f["n_pixels"] < 300:
            notes.append("IR GSD 부족 - 셀 단위 판별 한계")

        return BranchEvidence("ir", probs, IR_MASK, float(np.clip(rel, 0, 1)), f, notes)

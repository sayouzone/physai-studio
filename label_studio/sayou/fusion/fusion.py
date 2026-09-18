"""
Late Fusion (결정 수준 융합).

Early fusion(채널 concat)과 달리, 각 분기는 이미 '결론'을 냈고 여기서는
그 결론들을 결합한다. 이 방식의 장점이 태양광 점검에서 특히 크다.

  - IR 프레임이 일부 없거나 정합이 나빠도 RGB 단독으로 판정이 가능하다 (graceful degradation)
  - 각 분기를 독립적으로 교체/재학습할 수 있다 (규칙 -> CNN)
  - 정반사 포화, 저일사량 같은 '분기별 신뢰 불능 조건'을 가중치로 명시할 수 있다

구현된 융합기
  1) DempsterShaferFusion : 관측 불가 클래스를 Θ(모름) 질량으로 남기는 증거이론.
                            상충(conflict)을 명시적으로 측정한다. 기본값.
  2) LogOpinionPoolFusion : 신뢰도 가중 로그 선형 결합 (기하평균). 단순·안정적.
  3) NoisyOrFusion        : 상보적 증거(한쪽만 봐도 결함) 강조용.
  4) LearnedFusion        : 두 분기 확률+신뢰도를 입력으로 하는 로지스틱 결합.
                            라벨이 쌓이면 이걸로 갈아탄다.

그 위에 도메인 중재 규칙(cross-modal arbitration)을 얹는다. 이 규칙들이
현장 오탐의 대부분을 잡는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import (CLASS_INDEX, DEFECT_CLASSES, N_CLASSES, NORMAL,
                    SEVERITY_BY_CLASS, BranchEvidence, PanelVerdict)


# --------------------------------------------------------------------------- #
# 기본 융합기
# --------------------------------------------------------------------------- #

class BaseFusion:
    def fuse(self, evidences: Sequence[BranchEvidence]) -> Tuple[np.ndarray, float, Dict]:
        """반환: (fused_probs[N_CLASSES], unknown_mass, diagnostics)"""
        raise NotImplementedError


FULL_SET = (1 << N_CLASSES) - 1


def _bits(mask: np.ndarray) -> int:
    v = 0
    for i, on in enumerate(mask):
        if on:
            v |= (1 << i)
    return v


def _popcount(x: int) -> int:
    return bin(x).count("1")


class DempsterShaferFusion(BaseFusion):
    """
    집합값 초점원소(set-valued focal element)를 쓰는 증거이론 결합.

    이 파이프라인에서 가장 중요한 설계 결정이 여기 있다.

      **어떤 분기가 '정상'이라고 말했을 때 그것은 "그 분기가 볼 수 있는 범위에서
      정상"이라는 뜻이지, "모듈이 건전하다"는 뜻이 아니다.**

    RGB 는 셀 핫스팟이나 바이패스 다이오드 이상을 원리적으로 볼 수 없다.
    RGB 의 normal 질량을 단일톤 {normal} 에 주면, RGB 가 "정상"이라고 말하는
    순간 IR 이 찾은 핫스팟과 정면 충돌(conflict)이 생기고 Dempster 결합에서
    질량이 큰 쪽이 이겨 열 결함이 통째로 지워진다.
    late fusion 구현에서 가장 흔한 실패 모드이며, 조용히 재현율만 깎아먹는다.

    따라서 분기 b 의 초점원소를 다음과 같이 둔다.

        m_b({c})   = r_b * P_b(c)                       (b 가 판별 가능한 결함 c)
        m_b(A_b)   = r_b * P_b(normal),
                     A_b = {normal} ∪ {b 가 관측 불가능한 클래스}
        m_b(Theta) = 1 - r_b

    A_rgb ∩ {hotspot_cell} = {hotspot_cell} 이므로 충돌 없이 IR 의 발견이 살아남고,
    A_rgb ∩ A_ir = {normal} 이므로 두 분기가 모두 정상일 때만 정상으로 수렴한다.
    """

    def __init__(self, conflict_warn: float = 0.6, discount_on_conflict: bool = True):
        self.conflict_warn = conflict_warn
        self.discount_on_conflict = discount_on_conflict

    @staticmethod
    def _bpa(ev: BranchEvidence) -> Dict[int, float]:
        p = np.where(ev.mask, ev.probs, 0.0)
        ssum = p.sum()
        p = p / ssum if ssum > 1e-12 else np.zeros_like(p)
        r = float(np.clip(ev.reliability, 0.0, 1.0))

        m: Dict[int, float] = {}
        n_idx = CLASS_INDEX[NORMAL]
        for i in range(N_CLASSES):
            if i == n_idx or not ev.mask[i] or p[i] <= 1e-6:
                continue
            m[1 << i] = m.get(1 << i, 0.0) + r * float(p[i])

        A = (1 << n_idx) | (FULL_SET & ~_bits(ev.mask))
        m[A] = m.get(A, 0.0) + r * float(p[n_idx])
        m[FULL_SET] = m.get(FULL_SET, 0.0) + (1.0 - r)
        return {k: v for k, v in m.items() if v > 1e-12}

    @staticmethod
    def _combine(m1: Dict[int, float], m2: Dict[int, float]) -> Tuple[Dict[int, float], float]:
        out: Dict[int, float] = {}
        conflict = 0.0
        for a, va in m1.items():
            for b, vb in m2.items():
                inter = a & b
                w = va * vb
                if inter == 0:
                    conflict += w
                else:
                    out[inter] = out.get(inter, 0.0) + w
        denom = max(1.0 - conflict, 1e-9)
        return {k: v / denom for k, v in out.items()}, float(np.clip(conflict, 0.0, 1.0))

    def fuse(self, evidences):
        active = [e for e in evidences if e is not None and e.reliability > 1e-6]
        if not active:
            p = np.zeros(N_CLASSES)
            p[CLASS_INDEX[NORMAL]] = 1.0
            return p, 1.0, {"conflict": 0.0, "n_branches": 0}

        m = self._bpa(active[0])
        conflict_total = 0.0
        for ev in active[1:]:
            m, k = self._combine(m, self._bpa(ev))
            conflict_total = max(conflict_total, k)

        diag = {"conflict": conflict_total, "n_branches": len(active)}

        if self.discount_on_conflict and conflict_total > self.conflict_warn:
            # 상충이 크면 결론을 Theta 쪽으로 되돌린다 (과신 방지)
            d = float(np.clip((conflict_total - self.conflict_warn) /
                              (1.0 - self.conflict_warn), 0.0, 1.0))
            m = {k: v * (1.0 - d) for k, v in m.items()}
            m[FULL_SET] = m.get(FULL_SET, 0.0) + d
            diag["discounted"] = d

        # pignistic 변환: 각 초점원소의 질량을 그 원소들에 균등 배분
        probs = np.zeros(N_CLASSES)
        for S, v in m.items():
            n = _popcount(S)
            if n == 0:
                continue
            share = v / n
            for i in range(N_CLASSES):
                if S & (1 << i):
                    probs[i] += share
        total = probs.sum()
        if total > 1e-12:
            probs = probs / total

        return probs, float(m.get(FULL_SET, 0.0)), diag


class LogOpinionPoolFusion(BaseFusion):
    """신뢰도 가중 기하평균. P ∝ prod_b P_b^{w_b}. 관측 불가 클래스는 균등 사전분포로 대체."""

    def __init__(self, prior_strength: float = 0.15, normal_keep: float = 0.55):
        self.prior_strength = prior_strength
        self.normal_keep = normal_keep

    def fuse(self, evidences):
        active = [e for e in evidences if e is not None and e.reliability > 1e-6]
        if not active:
            p = np.zeros(N_CLASSES)
            p[CLASS_INDEX[NORMAL]] = 1.0
            return p, 1.0, {"n_branches": 0}

        logp = np.zeros(N_CLASSES)
        wsum = 0.0
        uniform = np.full(N_CLASSES, 1.0 / N_CLASSES)
        n_idx = CLASS_INDEX[NORMAL]
        for ev in active:
            p = np.where(ev.mask, ev.probs, 0.0)
            p = p / max(p.sum(), 1e-12)
            # DS 와 같은 의미론: 이 분기의 'normal' 은 {normal} ∪ {관측 불가} 에 대한
            # 진술이므로 그 집합에 균등 분배한다. 그래야 RGB 의 '정상'이 IR 전용
            # 결함 클래스를 억누르지 않는다.
            # 다만 DS 와 달리 균등 배분하면 '정상'이 관측 불가 클래스 수만큼
            # 희석돼 멀쩡한 모듈이 전부 유보 처리된다. 절반은 normal 에 남긴다
            # (= 결함보다 정상이 흔하다는 사전분포).
            unobs = ~ev.mask.copy()
            unobs[n_idx] = False
            spread, p[n_idx] = p[n_idx], 0.0
            p[n_idx] += self.normal_keep * spread
            if unobs.any():
                p = p + unobs.astype(float) * ((1.0 - self.normal_keep) * spread / unobs.sum())
            else:
                p[n_idx] += (1.0 - self.normal_keep) * spread
            # 남은 관측 불가 클래스에는 균등 사전분포를 섞어 0 로그를 피한다
            q = (1.0 - self.prior_strength) * p + self.prior_strength * uniform
            w = float(np.clip(ev.reliability, 0, 1))
            logp += w * np.log(np.maximum(q, 1e-12))
            wsum += w
        logp /= max(wsum, 1e-9)
        logp -= logp.max()
        probs = np.exp(logp)
        probs /= probs.sum()
        unknown = float(np.clip(1.0 - wsum / max(len(active), 1), 0.0, 1.0))
        return probs, unknown, {"n_branches": len(active), "weight_sum": wsum}


class NoisyOrFusion(BaseFusion):
    """
    상보적 증거용. 결함 c 가 존재할 확률을 1 - prod(1 - w_b * P_b(c)) 로 본다.
    "한 모달리티에서만 보이는 결함"이 많은 경우(크랙은 RGB, 다이오드는 IR)에 적합하나
    오탐이 늘어나므로 리콜 우선 운용에서만 권장한다.
    """

    def fuse(self, evidences):
        active = [e for e in evidences if e is not None and e.reliability > 1e-6]
        if not active:
            p = np.zeros(N_CLASSES)
            p[CLASS_INDEX[NORMAL]] = 1.0
            return p, 1.0, {"n_branches": 0}
        neg = np.ones(N_CLASSES)
        for ev in active:
            w = float(np.clip(ev.reliability, 0, 1))
            neg *= (1.0 - w * np.where(ev.mask, ev.probs, 0.0))
        pos = 1.0 - neg
        pos[CLASS_INDEX[NORMAL]] = float(np.prod(
            [e.probs[CLASS_INDEX[NORMAL]] for e in active]))
        probs = pos / max(pos.sum(), 1e-12)
        return probs, 0.0, {"n_branches": len(active)}


# --------------------------------------------------------------------------- #
# 학습형 융합 (라벨이 생긴 뒤)
# --------------------------------------------------------------------------- #

class LearnedFusion(BaseFusion):
    """
    입력 피처 = [P_rgb(관측가능), r_rgb, P_ir(관측가능), r_ir, r_rgb*r_ir, 1]
    다항 로지스틱 회귀. 외부 의존성 없이 간단한 경사하강으로 학습한다.
    """

    def __init__(self, l2: float = 1e-3, lr: float = 0.25, epochs: int = 600):
        self.W: Optional[np.ndarray] = None
        self.l2, self.lr, self.epochs = l2, lr, epochs

    @staticmethod
    def featurize(evidences: Sequence[BranchEvidence]) -> np.ndarray:
        by = {e.branch: e for e in evidences if e is not None}
        parts = []
        for name in ("rgb", "ir"):
            e = by.get(name)
            if e is None:
                parts.append(np.zeros(N_CLASSES + 1))
            else:
                p = np.where(e.mask, e.probs, 0.0)
                parts.append(np.concatenate([p, [float(e.reliability)]]))
        r = [parts[0][-1], parts[1][-1]]
        return np.concatenate(parts + [[r[0] * r[1], 1.0]])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LearnedFusion":
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Y = np.zeros((len(y), N_CLASSES))
        Y[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
        self.W = np.zeros((X.shape[1], N_CLASSES))
        for _ in range(self.epochs):
            Z = X @ self.W
            Z -= Z.max(axis=1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(axis=1, keepdims=True)
            G = X.T @ (P - Y) / len(X) + self.l2 * self.W
            self.W -= self.lr * G
        return self

    def fuse(self, evidences):
        if self.W is None:
            return LogOpinionPoolFusion().fuse(evidences)
        x = self.featurize(evidences)
        z = x @ self.W
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        return p, 0.0, {"model": "learned"}


# --------------------------------------------------------------------------- #
# 교차 모달 중재 규칙
# --------------------------------------------------------------------------- #

@dataclass
class Rule:
    name: str
    apply: Callable[[np.ndarray, Optional[BranchEvidence], Optional[BranchEvidence]],
                    Optional[np.ndarray]]
    description: str = ""


def _boost(probs: np.ndarray, cls: str, factor: float) -> np.ndarray:
    q = probs.copy()
    q[CLASS_INDEX[cls]] *= factor
    return q / max(q.sum(), 1e-12)


def _get(ev: Optional[BranchEvidence], cls: str) -> float:
    return 0.0 if ev is None else float(ev.probs[CLASS_INDEX[cls]])


def _feat(ev: Optional[BranchEvidence], k: str, default: float = 0.0) -> float:
    return default if ev is None else float(ev.features.get(k, default))


def default_rules() -> List[Rule]:
    rules: List[Rule] = []

    def r_glare_suppress(p, rgb, ir):
        """RGB 정반사 포화 + IR 발열 -> 반사에 의한 가짜 외관결함 억제, IR 우선."""
        if _feat(rgb, "glare_frac") > 0.12:
            q = p.copy()
            # 글레어는 밝기를 통째로 올려 '오염'으로도 오검출된다.
            for c in ("cell_crack", "discoloration", "delamination",
                      "glass_breakage", "soiling"):
                q[CLASS_INDEX[c]] *= 0.35
            return q / max(q.sum(), 1e-12)
        return None

    rules.append(Rule("glare_suppress", r_glare_suppress,
                      "정반사 포화 구간에서 RGB 외관 결함 판정을 감쇄"))

    def r_shadow_explains_heat(p, rgb, ir):
        """RGB 에서 그림자가 확실한데 IR 이 발열을 보고하면 '결함'이 아닌 차폐로 귀결."""
        if _get(rgb, "shading") > 0.45 and _feat(rgb, "dark_blob_frac") > 0.2:
            if _get(ir, "hotspot_substring") + _get(ir, "hotspot_cell") > 0.3:
                q = p.copy()
                q[CLASS_INDEX["hotspot_substring"]] *= 0.4
                q[CLASS_INDEX["hotspot_cell"]] *= 0.4
                q[CLASS_INDEX["shading"]] *= 3.0
                return q / max(q.sum(), 1e-12)
        return None

    rules.append(Rule("shadow_explains_heat", r_shadow_explains_heat,
                      "그림자로 설명되는 발열은 결함에서 제외"))

    def r_crack_plus_hotspot(p, rgb, ir):
        """RGB 크랙 + IR 셀 핫스팟이 같은 패널에서 동시 검출 -> 강한 확증."""
        if _get(rgb, "cell_crack") > 0.3 and _get(ir, "hotspot_cell") > 0.3:
            return _boost(p, "hotspot_cell", 2.5)
        return None

    rules.append(Rule("crack_plus_hotspot", r_crack_plus_hotspot,
                      "외관 크랙과 셀 핫스팟 동시 검출 시 확증 강화"))

    def r_soiling_confirmed(p, rgb, ir):
        """RGB 오염 + IR 광범위 약한 발열 -> 오염 확증(세정 대상)."""
        if _get(rgb, "soiling") > 0.30 and _feat(ir, "dt_inter") > 1.0 \
                and _feat(ir, "dt_intra") < 6.0:
            return _boost(p, "soiling", 2.2)
        return None

    rules.append(Rule("soiling_confirmed", r_soiling_confirmed,
                      "외관 오염 + 광범위 저강도 발열 = 오염 확증"))

    def r_clean_surface_vs_heat(p, rgb, ir):
        """
        RGB 는 깨끗한데(고신뢰) IR 만 모듈 전체 발열 -> 전기적 결함 가능성 상승.
        외관에 원인이 없다는 '음성 증거'를 적극적으로 쓰는 규칙.
        """
        if rgb is not None and rgb.reliability > 0.6 and _get(rgb, NORMAL) > 0.6:
            if _get(ir, "module_open") > 0.3:
                return _boost(p, "module_open", 2.0)
        return None

    rules.append(Rule("clean_surface_vs_heat", r_clean_surface_vs_heat,
                      "외관 정상 + 모듈 전체 발열 = 전기적 결함 의심 강화"))

    return rules


# --------------------------------------------------------------------------- #
# 오케스트레이터
# --------------------------------------------------------------------------- #

class LateFusionEngine:
    """
    분기 증거 -> (융합기) -> (중재 규칙) -> PanelVerdict.

    registration_reliability 는 두 분기 사이 '대응 신뢰도'다.
    이 값이 낮으면 교차 규칙을 끄고, IR 분기 가중치도 낮춘다.
    (정합이 어긋난 상태의 융합은 융합이 아니라 오염이다)
    """

    def __init__(self,
                 fusion: Optional[BaseFusion] = None,
                 rules: Optional[List[Rule]] = None,
                 rule_min_registration: float = 0.45,
                 abstain_threshold: float = 0.30,
                 abstain_margin: float = 0.10):
        self.fusion = fusion or DempsterShaferFusion()
        self.rules = default_rules() if rules is None else rules
        self.rule_min_registration = rule_min_registration
        self.abstain_threshold = abstain_threshold
        self.abstain_margin = abstain_margin

    def decide(self,
               panel_id: str,
               rgb_ev: Optional[BranchEvidence],
               ir_ev: Optional[BranchEvidence],
               registration_reliability: float = 1.0) -> PanelVerdict:

        reg = float(np.clip(registration_reliability, 0.0, 1.0))

        # 정합 신뢰도로 IR 분기를 할인한다 (ROI 대응이 IR 쪽에서만 성립하므로)
        ir_use = ir_ev
        if ir_ev is not None and reg < 1.0:
            ir_use = BranchEvidence(ir_ev.branch, ir_ev.probs, ir_ev.mask,
                                    ir_ev.reliability * (0.35 + 0.65 * reg),
                                    ir_ev.features, list(ir_ev.notes))

        evidences = [e for e in (rgb_ev, ir_use) if e is not None]
        probs, unknown, diag = self.fusion.fuse(evidences)

        fired: List[str] = []
        if reg >= self.rule_min_registration:
            for rule in self.rules:
                try:
                    out = rule.apply(probs, rgb_ev, ir_use)
                except Exception:
                    out = None
                if out is not None:
                    probs = out
                    fired.append(rule.name)
        else:
            fired.append("cross_modal_rules_disabled(low_registration)")

        order = np.argsort(probs)[::-1]
        idx = int(order[0])
        label = DEFECT_CLASSES[idx]
        conf = float(probs[idx])
        margin = conf - float(probs[order[1]]) if len(order) > 1 else conf

        # 유보(abstain): 최상위 확률이 낮고 '동시에' 2위와의 격차도 작을 때만.
        # 격차가 뚜렷하면 절대 확률이 낮아도 판정 자체는 유효하다.
        if conf < self.abstain_threshold and margin < self.abstain_margin:
            fired.append("abstain(low_confidence)")
            label = "uncertain"
            severity = 1
        else:
            severity = SEVERITY_BY_CLASS.get(label, 1)

        if diag.get("conflict", 0.0) > 0.6:
            fired.append(f"high_conflict({diag['conflict']:.2f})")

        return PanelVerdict(
            panel_id=panel_id,
            label=label,
            confidence=conf,
            severity=severity,
            fused_probs=probs,
            unknown_mass=unknown,
            rgb=rgb_ev,
            ir=ir_use,
            rules_fired=fired,
            registration_reliability=reg,
        )

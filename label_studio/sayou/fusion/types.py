"""
공통 자료구조 및 결함 택소노미.

Late fusion 의 핵심은 "두 분기가 서로 다른 것을 볼 수 있다"는 점을 명시적으로
모델링하는 것이다. 따라서 클래스 목록은 공유하되, 각 분기가 실제로 판별 가능한
클래스 집합(observability mask)을 함께 선언한다. 판별 불가능한 클래스에 대해서는
확률을 0 으로 두지 않고 '모름(Theta)' 질량으로 남긴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# 택소노미
# --------------------------------------------------------------------------- #

NORMAL = "normal"

DEFECT_CLASSES: Tuple[str, ...] = (
    NORMAL,
    "soiling",            # 오염 / 표면 퇴적
    "shading",            # 그림자 / 차폐 (결함 아님, 운영 이슈)
    "cell_crack",         # 셀 크랙 / 스네일 트레일
    "discoloration",      # EVA 변색 / 황변
    "delamination",       # 박리
    "glass_breakage",     # 유리 파손
    "hotspot_cell",       # 단일 셀 핫스팟
    "hotspot_substring",  # 바이패스 다이오드 / 서브스트링 이상
    "module_open",        # 모듈 전체 발열 (개방 / 스트링 탈락)
    "junction_box",       # 정션박스 발열
    "pid",                # PID 패턴
)

CLASS_INDEX: Dict[str, int] = {c: i for i, c in enumerate(DEFECT_CLASSES)}
N_CLASSES = len(DEFECT_CLASSES)

# 각 분기가 '판별 가능한' 클래스. 나머지는 unknown 으로 남긴다.
RGB_OBSERVABLE: Tuple[str, ...] = (
    NORMAL, "soiling", "shading", "cell_crack",
    "discoloration", "delamination", "glass_breakage",
)

IR_OBSERVABLE: Tuple[str, ...] = (
    NORMAL, "soiling", "shading", "hotspot_cell", "hotspot_substring",
    "module_open", "junction_box", "pid",
)

# IEC TS 62446-3 기반 심각도 등급 (1: 관찰, 2: 단기 조치, 3: 즉시 조치)
SEVERITY_BY_CLASS: Dict[str, int] = {
    NORMAL: 0,
    "soiling": 1,
    "shading": 1,
    "discoloration": 1,
    "pid": 2,
    "delamination": 2,
    "cell_crack": 2,
    "hotspot_cell": 2,
    "hotspot_substring": 3,
    "junction_box": 3,
    "module_open": 3,
    "glass_breakage": 3,
}


def observability_mask(names: Sequence[str]) -> np.ndarray:
    m = np.zeros(N_CLASSES, dtype=bool)
    for n in names:
        m[CLASS_INDEX[n]] = True
    return m


RGB_MASK = observability_mask(RGB_OBSERVABLE)
IR_MASK = observability_mask(IR_OBSERVABLE)


# --------------------------------------------------------------------------- #
# 카메라 / 정합
# --------------------------------------------------------------------------- #

@dataclass
class CameraIntrinsics:
    """핀홀 내부 파라미터. dist 는 OpenCV 순서 (k1,k2,p1,p2,k3)."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: Optional[np.ndarray] = None

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]], dtype=np.float64
        )

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float,
                 dist: Optional[np.ndarray] = None) -> "CameraIntrinsics":
        f = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        return cls(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height, dist=dist)


@dataclass
class StereoExtrinsics:
    """RGB 카메라 좌표계 -> IR 카메라 좌표계 강체 변환.  X_ir = R @ X_rgb + t"""
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))  # meter


@dataclass
class GeoTransform:
    """GDAL 스타일 6-요소 affine. (x, y) = f(col, row)."""
    coeffs: Tuple[float, float, float, float, float, float]

    def to_matrix(self) -> np.ndarray:
        c = self.coeffs
        return np.array([[c[1], c[2], c[0]],
                         [c[4], c[5], c[3]],
                         [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass
class RegistrationResult:
    """RGB 픽셀 -> IR 픽셀 사상 호모그래피와 품질 지표."""
    H_rgb_to_ir: np.ndarray
    method: str
    residual_px: float           # IR 픽셀 기준 잔차 RMSE (추정치)
    overlap_ratio: float         # RGB 프레임 중 IR FOV 에 들어오는 비율
    converged: bool
    ncc: float = float("nan")    # 정합 후 gradient 도메인 상관계수
    stages: List[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return self.converged and self.overlap_ratio > 0.2 and np.isfinite(self.residual_px)

    def reliability(self, tol_px: float = 2.0) -> float:
        """정합 신뢰도 0~1. late fusion 의 분기 가중치에 그대로 곱해 쓴다."""
        if not self.is_usable:
            return 0.0
        geo = float(np.exp(-0.5 * (self.residual_px / max(tol_px, 1e-6)) ** 2))
        pho = float(np.clip((self.ncc + 1.0) / 2.0, 0.0, 1.0)) if np.isfinite(self.ncc) else 0.7
        return float(np.clip(geo * (0.5 + 0.5 * pho) * self.overlap_ratio ** 0.25, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# 패널 ROI
# --------------------------------------------------------------------------- #

@dataclass
class PanelROI:
    """단일 모듈(패널) 영역. polygon 은 RGB 픽셀 좌표계의 (N,2) 배열."""
    panel_id: str
    polygon_rgb: np.ndarray
    polygon_ir: Optional[np.ndarray] = None
    grid_angle_deg: float = 0.0     # 패널 장축 회전각
    array_id: Optional[str] = None  # 어레이/스트링 그룹 키
    meta: Dict = field(default_factory=dict)

    def bbox(self, which: str = "rgb") -> Tuple[int, int, int, int]:
        poly = self.polygon_rgb if which == "rgb" else self.polygon_ir
        if poly is None:
            raise ValueError(f"polygon_{which} 가 없습니다: {self.panel_id}")
        x0, y0 = np.floor(poly.min(axis=0)).astype(int)
        x1, y1 = np.ceil(poly.max(axis=0)).astype(int)
        return int(x0), int(y0), int(x1), int(y1)


# --------------------------------------------------------------------------- #
# 분기 출력 / 융합 결과
# --------------------------------------------------------------------------- #

@dataclass
class BranchEvidence:
    """
    한 분기(RGB 또는 IR)가 한 패널에 대해 내놓은 결과.

    probs      : N_CLASSES 길이. 관측 가능 클래스에 대해서만 합이 1 이 되도록 정규화.
    mask       : 이 분기가 판별 가능한 클래스 마스크.
    reliability: 이 패널·이 프레임에서 분기 자체를 얼마나 믿을지 (0~1).
                 (예: RGB 는 정반사 포화 시 하락, IR 은 ΔT 대비 낮으면 하락)
    """
    branch: str
    probs: np.ndarray
    mask: np.ndarray
    reliability: float
    features: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def top(self) -> Tuple[str, float]:
        idx = int(np.argmax(np.where(self.mask, self.probs, -np.inf)))
        return DEFECT_CLASSES[idx], float(self.probs[idx])


@dataclass
class PanelVerdict:
    panel_id: str
    label: str
    confidence: float
    severity: int
    fused_probs: np.ndarray
    unknown_mass: float
    rgb: Optional[BranchEvidence] = None
    ir: Optional[BranchEvidence] = None
    rules_fired: List[str] = field(default_factory=list)
    registration_reliability: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "panel_id": self.panel_id,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "severity": int(self.severity),
            "unknown_mass": round(float(self.unknown_mass), 4),
            "registration_reliability": round(float(self.registration_reliability), 4),
            "probs": {c: round(float(p), 4)
                      for c, p in zip(DEFECT_CLASSES, self.fused_probs) if p > 1e-3},
            "rgb": None if self.rgb is None else {
                "top": self.rgb.top()[0],
                "score": round(self.rgb.top()[1], 4),
                "reliability": round(self.rgb.reliability, 4),
                "features": {k: round(float(v), 4) for k, v in self.rgb.features.items()},
                "notes": self.rgb.notes,
            },
            "ir": None if self.ir is None else {
                "top": self.ir.top()[0],
                "score": round(self.ir.top()[1], 4),
                "reliability": round(self.ir.reliability, 4),
                "features": {k: round(float(v), 4) for k, v in self.ir.features.items()},
                "notes": self.ir.notes,
            },
            "rules_fired": self.rules_fired,
        }

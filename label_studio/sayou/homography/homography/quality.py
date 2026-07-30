"""RTK 품질 검증 및 BA 가중치 공분산 산출 (파이프라인 3단계).

왜 필요한가
-----------
GCP-free 파이프라인에서 절대 좌표의 유일한 근거는 RTK 관측값이다. 그런데
RTK 는 조용히 열화한다 — FIX 를 잃고 FLOAT 로 떨어져도 EXIF 에는 좌표가
그대로 들어 있고, 오차만 cm → dm~m 로 커진다. 이 프레임을 다른 프레임과
같은 가중치로 BA 에 넣으면 **한 장이 블록 전체를 끌고 간다**.

이 모듈은 두 가지를 한다.

1. **게이트**: RtkFlag 와 σ 임계값으로 명백한 불량 프레임을 제외.
2. **가중치**: 살아남은 프레임에 ``w = 1/σ²`` 를 부여해
   ``sfm.bundle_adjustment.rtk_constrained_bundle_adjustment`` 의
   ``rtk_weights`` 로 넘긴다. σ 가 큰 프레임은 자동으로 발언권이 줄어든다.

σ 축 순서 주의
--------------
``meta.rtk_std = [σ_lat, σ_lon, σ_hgt]`` (미터). 투영좌표계는 (X=Easting,
Y=Northing) 이므로 **σ_lon → X, σ_lat → Y** 로 뒤바꿔 넣어야 한다.
이 축 교차는 눈에 잘 띄지 않는 버그의 단골이다. 한국 위도에서 σ_lat 과
σ_lon 은 보통 비슷해서 틀려도 결과가 그럴듯해 보이지만, RTK 가 열화하는
구간에서는 두 값이 3~5 배까지 벌어진다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "RTK_FIXED", "RTK_FLOAT", "RTK_SINGLE",
    "RTKQualityConfig", "RTKQuality",
    "assess_rtk_quality", "build_rtk_priors_and_weights",
]

# DJI RtkFlag 값. 50 = FIXED (cm 급). 그 외는 열화 상태.
RTK_FIXED = 50
RTK_FLOAT = 34
RTK_SINGLE = 16


@dataclass
class RTKQualityConfig:
    """RTK 게이트 설정.

    Attributes
    ----------
    require_fixed : ``True`` 면 ``RtkFlag != 50`` 인 프레임을 전부 제외.
        시험비행/부분 열화 데이터를 살려야 하면 ``False`` + σ 임계값에 의존.
    max_sigma_xy_m, max_sigma_z_m : σ 상한 (m). 초과 시 제외.
    sigma_floor_m : σ 하한 (m). RTK 가 σ=0 을 보고하는 사례가 있는데
        ``1/σ²`` 가 발산해 그 프레임이 hard constraint 로 굳어버린다.
        물리적으로도 RTK 가 0 오차일 수 없으므로 바닥값을 깐다.
    default_sigma_xy_m, default_sigma_z_m : σ 필드 자체가 비어 있을 때의
        보수적 대체값.
    max_nadir_deg : 광축이 연직에서 이만큼 이상 벗어난 프레임 제외.
        평면 호모그래피는 경사가 커질수록 지형기복 오차가 급증한다.
    """

    require_fixed: bool = True
    max_sigma_xy_m: float = 0.10
    max_sigma_z_m: float = 0.20
    sigma_floor_m: float = 0.005
    default_sigma_xy_m: float = 0.10
    default_sigma_z_m: float = 0.15
    max_nadir_deg: float = 20.0


@dataclass
class RTKQuality:
    """프레임 하나의 RTK 품질 판정 결과."""

    index: int
    accepted: bool
    rtk_flag: int
    sigma_xyz_m: np.ndarray                       # (3,) [σ_X, σ_Y, σ_Z] 투영좌표축
    nadir_deg: float = float("nan")
    reasons: list[str] = field(default_factory=list)

    @property
    def weights(self) -> np.ndarray:
        """BA 용 ``1/σ²`` 가중치 (3,)."""
        return 1.0 / (self.sigma_xyz_m ** 2)


def assess_rtk_quality(metas: list,
                       rotations: list[np.ndarray] | None = None,
                       config: RTKQualityConfig | None = None) -> list[RTKQuality]:
    """메타데이터 리스트 → 프레임별 품질 판정.

    Parameters
    ----------
    metas : ``ImageMetadata`` 리스트.
    rotations : 프레임별 R (``pose.rotation_from_metadata``). 주면 nadir 각도
        필터도 적용. ``None`` 이면 자세 검사 생략.
    """
    cfg = config or RTKQualityConfig()
    results: list[RTKQuality] = []

    for i, meta in enumerate(metas):
        reasons: list[str] = []
        flag = int(getattr(meta, "rtk_flag", 0) or 0)

        # ---- σ 추출: [σ_lat, σ_lon, σ_hgt] → [σ_X(=lon), σ_Y(=lat), σ_Z] ----
        std = list(getattr(meta, "rtk_std", None) or [])
        if len(std) >= 3 and all(np.isfinite(std[:3])) and any(v > 0 for v in std[:3]):
            s_lat, s_lon, s_hgt = float(std[0]), float(std[1]), float(std[2])
            sigma = np.array([s_lon, s_lat, s_hgt])      # ★ 축 교차
        else:
            reasons.append("rtk_std 없음 — 기본 σ 사용")
            sigma = np.array([cfg.default_sigma_xy_m,
                              cfg.default_sigma_xy_m,
                              cfg.default_sigma_z_m])

        sigma = np.maximum(sigma, cfg.sigma_floor_m)

        accepted = True
        if cfg.require_fixed and flag != RTK_FIXED:
            accepted = False
            reasons.append(f"RtkFlag={flag} (FIXED 아님)")
        if sigma[0] > cfg.max_sigma_xy_m or sigma[1] > cfg.max_sigma_xy_m:
            accepted = False
            reasons.append(f"σ_xy={sigma[:2].max()*100:.1f}cm > "
                           f"{cfg.max_sigma_xy_m*100:.1f}cm")
        if sigma[2] > cfg.max_sigma_z_m:
            accepted = False
            reasons.append(f"σ_z={sigma[2]*100:.1f}cm > {cfg.max_sigma_z_m*100:.1f}cm")

        nadir = float("nan")
        if rotations is not None and rotations[i] is not None:
            from .pose import angle_from_nadir_deg
            nadir = angle_from_nadir_deg(rotations[i])
            if nadir > cfg.max_nadir_deg:
                accepted = False
                reasons.append(f"연직편차 {nadir:.1f}° > {cfg.max_nadir_deg:.1f}°")

        results.append(RTKQuality(index=i, accepted=accepted, rtk_flag=flag,
                                  sigma_xyz_m=sigma, nadir_deg=nadir,
                                  reasons=reasons))

    n_ok = sum(r.accepted for r in results)
    logger.info("RTK 품질 검증: %d/%d 통과 (%.1f%%)",
                n_ok, len(results), 100.0 * n_ok / max(len(results), 1))
    if n_ok < len(results):
        from collections import Counter
        tally = Counter(r.reasons[0] if r.reasons else "?"
                        for r in results if not r.accepted)
        for reason, cnt in tally.most_common():
            logger.info("  제외 %4d장: %s", cnt, reason)
    return results


def build_rtk_priors_and_weights(camera_xyz: np.ndarray,
                                 qualities: list[RTKQuality]) -> tuple[np.ndarray, np.ndarray]:
    """BA 입력용 ``(rtk_priors, rtk_weights)`` 배열 생성.

    ``rtk_constrained_bundle_adjustment`` 는 잔차에
    ``sqrt(w) · (camera_pos − rtk_prior)`` 를 더한다. 즉 ``w = 1/σ²`` 이면
    잔차가 σ 로 정규화되어 재투영 잔차(픽셀)와 같은 척도로 비교된다.

    Returns
    -------
    priors : (n_cam, 3) — 입력 ``camera_xyz`` 그대로.
    weights : (n_cam, 3) — ``1/σ²``.
    """
    priors = np.asarray(camera_xyz, dtype=np.float64).reshape(-1, 3)
    if len(priors) != len(qualities):
        raise ValueError(
            f"카메라 수 불일치: priors={len(priors)}, qualities={len(qualities)}"
        )
    weights = np.vstack([q.weights for q in qualities])
    return priors, weights

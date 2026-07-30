"""RTK 측정 품질 검증 및 BA prior 가중치 계산.

DJI RtkFlag 의미 (DJI SDK 문서 기준)
------------------------------------
* 0  = None / GPS only
* 16 = RTK Float  (수십 cm 정확도)
* 34 = RTK Single (저정밀)
* 50 = RTK Fixed  (1~3 cm 정확도) ← 신뢰 가능
"""

from __future__ import annotations

import logging

import numpy as np

from sayou.image.metadata import ImageMetadata

logger = logging.getLogger(__name__)

RTK_FIXED = 50


def validate_rtk_quality(metas: list[ImageMetadata],
                         min_fixed_ratio: float = 0.9) -> bool:
    """RTK Fixed 비율이 임계값 이상인지 검사 + 로깅.

    Parameters
    ----------
    metas : 메타데이터 리스트
    min_fixed_ratio : Fixed 비율 하한 (기본 90%). 미만이면 GCP-free 비추천.

    Returns
    -------
    bool : 임계값 통과 여부.
    """
    if not metas:
        return False
    fixed_count = sum(m.is_rtk_fixed for m in metas)
    ratio = fixed_count / len(metas)
    logger.info("RTK Fixed 비율: %d/%d (%.1f%%)",
                fixed_count, len(metas), ratio * 100)
    if ratio < min_fixed_ratio:
        logger.warning(
            "RTK Fixed 비율이 %.1f%% 미만입니다. "
            "GCP 없이 진행 시 정확도가 떨어질 수 있습니다.",
            min_fixed_ratio * 100,
        )
    return ratio >= min_fixed_ratio


def estimate_ground_z(meta: ImageMetadata) -> float:
    """평면 정사보정용 지표면 절대고도(m) 추정.

    우선순위:
      1) LRF 실측 (lrf[3] = LRFTargetAbsAlt) — H20T 등 LRF 탑재 기종.
         촬영 시점 조준점의 실측 절대고도라 가장 정확.
      2) gps.altitude - relative_height — RelativeAltitude 는 이륙지점
         기준이라 촬영지 지형이 이륙지와 다르면 오차가 크다.
    """
    if meta.has_valid_lrf:
        return float(meta.lrf_target_abs_alt)
    if meta.relative_height:
        return meta.gps.altitude - meta.relative_height
    logger.warning("ground_z 추정 정보 부족 → altitude - 100m 사용")
    return meta.gps.altitude - 100.0


def compute_rtk_prior_weights(metas: list[ImageMetadata]) -> np.ndarray:
    """RTK 측정 표준편차 기반 ``(N, 3)`` BA prior 가중치 (1/σ²).

    RTK Fixed 가 아니면 패널티 σ_xy=20cm, σ_z=30cm 를 강제 적용해서
    Float/Single 측정값이 해를 끌어당기지 못하게 한다.

    RTK 표준편차로 가중치 산출 (Fixed=1cm 가정, Float=20cm 패널티)
    """
    rtk_weights = np.zeros((len(metas), 3))
    for i, m in enumerate(metas):
        sigma_xy = m.gps_std_xy if m.is_rtk_fixed else max(m.gps_std_xy, 0.20)
        sigma_z = m.gps_std_z if m.is_rtk_fixed else max(m.gps_std_z, 0.30)
        rtk_weights[i] = [1.0 / sigma_xy**2, 1.0 / sigma_xy**2, 1.0 / sigma_z**2]
    return rtk_weights


__all__ = [
    "RTK_FIXED",
    "validate_rtk_quality",
    "estimate_ground_z",
    "compute_rtk_prior_weights",
]
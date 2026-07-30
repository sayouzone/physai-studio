"""RTK 측정 품질 검증 및 BA prior 가중치 계산.

DJI RtkFlag 의미 (DJI SDK 문서 기준)
------------------------------------
* 0  = None / GPS only
* 16 = RTK Float  (수십 cm 정확도)
* 34 = RTK Single (저정밀)
* 50 = RTK Fixed  (1~3 cm 정확도) ← 신뢰 가능

★ 이 버전에서 고친 것
---------------------
1. **실측 σ 사용.** ``image.metadata.extract_metadata`` 는 XMP 에서
   ``RtkStdLat / RtkStdLon / RtkStdHgt`` 를 파싱해 ``meta.rtk_std`` 에
   넣는데, 원본 ``compute_rtk_prior_weights`` 는 이 값을 한 번도 읽지 않고
   고정 상수 ``gps_std_xy=0.10``, ``gps_std_z=0.15`` 만 썼다. 즉 프레임별
   실제 측위 정확도가 BA 가중치에 전혀 반영되지 않았다. RTK 가 열화하는
   구간(비행 반환점, 수목 근처)에서 σ 는 3~5 배까지 벌어지는데, 같은
   가중치를 주면 그 프레임이 블록 전체를 끌고 간다.

2. **σ 축 순서.** ``rtk_std = [σ_lat, σ_lon, σ_hgt]`` 인데 투영좌표는
   (X=Easting, Y=Northing) 이다. → **σ_lon 이 X, σ_lat 이 Y**. 뒤집으면
   한국 위도에서는 두 값이 비슷해 결과가 그럴듯해 보이지만, 정확히
   그 "열화 구간"에서만 틀린다.

3. **σ 하한.** RTK 가 σ=0 을 보고하는 프레임이 실제로 존재한다. ``1/σ²``
   이 발산해 그 프레임이 사실상 hard constraint 가 되어 BA 를 고정시킨다.

4. **``estimate_ground_z`` 폴백 상수 제거.** 원본은 정보가 없을 때
   ``altitude - 100.0`` 을 반환했다. 100 m 는 아무 근거가 없는 값이고,
   이 값이 그대로 정사영상 평면 고도가 되어 GSD 와 지상범위 전체를
   틀리게 만든다. 조용히 틀린 값을 내느니 ``None`` 을 반환해 호출자가
   그 사진을 건너뛰게 하는 편이 낫다.
"""

from __future__ import annotations

import logging

import numpy as np

from sayou.image.metadata import ImageMetadata

logger = logging.getLogger(__name__)

RTK_FIXED = 50
RTK_FLOAT = 16
RTK_SINGLE = 34

# σ 하한 (m). RTK 가 0 을 보고해도 물리적으로 0 오차일 수 없다.
SIGMA_FLOOR_M = 0.005
# RTK Fixed 가 아닌 프레임에 강제하는 패널티 σ.
PENALTY_SIGMA_XY_M = 0.20
PENALTY_SIGMA_Z_M = 0.30


def validate_rtk_quality(metas: list[ImageMetadata],
                         min_fixed_ratio: float = 0.9) -> bool:
    """RTK Fixed 비율이 임계값 이상인지 검사 + 로깅."""
    if not metas:
        return False
    fixed_count = sum(m.is_rtk_fixed for m in metas)
    ratio = fixed_count / len(metas)
    logger.info("RTK Fixed 비율: %d/%d (%.1f%%)",
                fixed_count, len(metas), ratio * 100)

    # 플래그 분포도 함께 남긴다 — Float 인지 GPS-only 인지에 따라 대응이 다르다.
    from collections import Counter
    tally = Counter(int(m.rtk_flag or 0) for m in metas)
    names = {RTK_FIXED: "Fixed", RTK_FLOAT: "Float",
             RTK_SINGLE: "Single", 0: "None/GPS"}
    logger.info("  RtkFlag 분포: %s",
                ", ".join(f"{names.get(k, k)}={v}" for k, v in tally.most_common()))

    if ratio < min_fixed_ratio:
        logger.warning(
            "RTK Fixed 비율이 %.1f%% 미만입니다. "
            "GCP 없이 진행 시 정확도가 떨어질 수 있습니다.",
            min_fixed_ratio * 100,
        )
    return ratio >= min_fixed_ratio


def sigma_xyz(meta: ImageMetadata) -> np.ndarray:
    """프레임의 측위 표준편차 ``[σ_X, σ_Y, σ_Z]`` (m, 투영좌표축).

    우선순위: XMP 실측 ``rtk_std`` → ``gps_std_xy/z`` 기본값.
    RTK Fixed 가 아니면 패널티 σ 를 하한으로 강제한다.
    """
    std = list(getattr(meta, "rtk_std", None) or [])
    if len(std) >= 3 and all(np.isfinite(std[:3])) and any(v > 0 for v in std[:3]):
        s_lat, s_lon, s_hgt = float(std[0]), float(std[1]), float(std[2])
        sigma = np.array([s_lon, s_lat, s_hgt])   # ★ lon→X, lat→Y 축 교차
    else:
        sigma = np.array([meta.gps_std_xy, meta.gps_std_xy, meta.gps_std_z],
                         dtype=float)

    if not meta.is_rtk_fixed:
        sigma = np.maximum(sigma, [PENALTY_SIGMA_XY_M,
                                   PENALTY_SIGMA_XY_M,
                                   PENALTY_SIGMA_Z_M])
    return np.maximum(sigma, SIGMA_FLOOR_M)


def estimate_ground_z(meta: ImageMetadata) -> float | None:
    """평면 정사보정용 지표면 절대고도(m) 추정.

    우선순위:
      1) LRF 실측 (``lrf[3]`` = LRFTargetAbsAlt) — H20T 등 LRF 탑재 기종.
         촬영 시점 조준점의 실측 절대고도라 가장 정확.
      2) ``gps.altitude - relative_height`` — RelativeAltitude 는 이륙지점
         기준이라 촬영지 지형이 이륙지와 다르면 오차가 크다.

    Returns
    -------
    추정 고도 (m), 또는 근거가 전혀 없으면 ``None``.
    호출자는 ``None`` 인 사진을 **건너뛰어야** 한다. 임의 상수로 채우면
    그 사진의 GSD 와 지상범위가 통째로 틀어지고, 모자이크에서는 그
    프레임이 엉뚱한 위치에 합성된다.
    """
    if meta.has_valid_lrf:
        return float(meta.lrf_target_abs_alt)
    if meta.relative_height:
        return float(meta.gps.altitude) - float(meta.relative_height)
    logger.warning("ground_z 추정 근거 없음 (LRF/RelativeAltitude 둘 다 없음): %s",
                   meta.origin_path)
    return None


def compute_rtk_prior_weights(metas: list[ImageMetadata]) -> np.ndarray:
    """RTK 실측 표준편차 기반 ``(N, 3)`` BA prior 가중치 (``1/σ²``).

    ``rtk_constrained_bundle_adjustment`` 는 잔차에
    ``sqrt(w) · (camera_pos − rtk_prior)`` 를 더한다. ``w = 1/σ²`` 이면
    이 항이 σ 단위로 정규화되어 픽셀 단위 재투영 잔차와 같은 척도에서
    비교된다 (둘 다 "표준편차 몇 배" 가 된다).
    """
    sigmas = np.vstack([sigma_xyz(m) for m in metas])
    weights = 1.0 / sigmas ** 2

    logger.info("RTK σ: X %.1f~%.1f cm, Y %.1f~%.1f cm, Z %.1f~%.1f cm "
                "(중앙값 %.1f / %.1f / %.1f cm)",
                sigmas[:, 0].min() * 100, sigmas[:, 0].max() * 100,
                sigmas[:, 1].min() * 100, sigmas[:, 1].max() * 100,
                sigmas[:, 2].min() * 100, sigmas[:, 2].max() * 100,
                np.median(sigmas[:, 0]) * 100,
                np.median(sigmas[:, 1]) * 100,
                np.median(sigmas[:, 2]) * 100)

    spread = sigmas[:, :2].max() / max(np.median(sigmas[:, :2]), 1e-9)
    if spread > 5:
        logger.warning("프레임 간 σ_xy 편차가 %.1f 배입니다. 일부 구간에서 "
                       "RTK 가 열화했을 수 있으니 해당 프레임을 확인하세요.", spread)
    return weights


__all__ = [
    "RTK_FIXED", "RTK_FLOAT", "RTK_SINGLE",
    "validate_rtk_quality",
    "sigma_xyz",
    "estimate_ground_z",
    "compute_rtk_prior_weights",
]

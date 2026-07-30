"""RTK 위치 기반 인접쌍 선택 (파이프라인 4단계).

문제
----
N 장을 전수 매칭하면 ``N(N−1)/2`` 쌍이다. 165 장이면 13,530 쌍, 1,000 장이면
499,500 쌍. 그런데 드론 사진의 대부분은 서로 **겹치지 않는다** — 겹치지 않는
쌍의 매칭 시도는 전부 낭비이고, 게다가 태양광 패널처럼 반복 텍스처가 강한
장면에서는 겹치지 않는 쌍이 **가짜 매칭을 만들어 낸다**.

RTK 가 cm 급 위치를 이미 알려주므로, 겹칠 수 없는 쌍은 매칭 전에 잘라낸다.

기하학적 판정
-------------
두 사진의 footprint 는 이미 계산할 수 있다 (``homography`` 모듈). 그러나
매칭 단계는 평면 추정 전이므로, 여기서는 **보수적 근사**를 쓴다:

    r_i = (촬영고도) × tan(화각 반각) × sqrt(2)      # 외접원 반경
    겹침 가능 ⟺ |C_i − C_j| < (r_i + r_j) × margin

``sqrt(2)`` 는 직사각형 footprint 의 대각 반경, ``margin`` 은 자세 기울기와
지형 기복에 대한 여유. 이 판정은 실제 겹침의 **상위집합**이라 안전하다
(진짜 겹치는 쌍을 버리지 않는다).

인접쌍 필터가 없으면 뭐가 잘못되나
----------------------------------
겹치지 않는 두 프레임이 매칭되면 그 track 은 전혀 다른 두 지상점을 하나로
묶는다. ``tracks.build_tracks`` 의 충돌 검사는 *같은 이미지의 두 keypoint* 만
잡아내지 이 경우는 통과시킨다. 결과적으로 BA 가 존재하지 않는 지상점을
만족시키려고 카메라 자세를 왜곡한다.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["footprint_radius_m", "select_gps_neighbor_pairs"]


def footprint_radius_m(altitude_agl_m: float,
                       f_px: float,
                       width_px: int,
                       height_px: int) -> float:
    """nadir 가정 footprint 외접원 반경 (m).

    ``(w/2, h/2)`` 픽셀의 화각 tangent 를 그대로 쓴다. 초점거리가 픽셀 단위라
    센서 크기가 필요 없다.
    """
    if f_px <= 0 or altitude_agl_m <= 0:
        return 0.0
    half_diag_px = 0.5 * float(np.hypot(width_px, height_px))
    return float(altitude_agl_m * half_diag_px / f_px)


def select_gps_neighbor_pairs(camera_xyz: np.ndarray,
                              radii_m: np.ndarray | float,
                              *,
                              margin: float = 1.15,
                              max_pairs_per_image: int | None = 12,
                              require_min_baseline_m: float = 0.5) -> list[tuple[int, int]]:
    """겹칠 가능성이 있는 이미지 쌍 목록.

    Parameters
    ----------
    camera_xyz : (N, 3) 투영좌표계 카메라 중심.
    radii_m : (N,) 프레임별 footprint 반경, 또는 스칼라 (전 프레임 동일).
    margin : 반경 합에 곱하는 여유 계수. 자세 기울기/지형 기복 대비.
    max_pairs_per_image : 이미지당 최대 이웃 수 (가까운 순). 호버링이나
        중복 촬영으로 한 지점에 수십 장이 몰릴 때 조합 폭발을 막는다.
        ``None`` 이면 제한 없음.
    require_min_baseline_m : 베이스라인이 이보다 짧은 쌍은 제외. 거의 같은
        위치의 두 장은 삼각측량 시선각이 0 에 가까워 깊이가 발산한다.

    Returns
    -------
    ``[(i, j), ...]`` — 항상 ``i < j``, 정렬됨.
    """
    C = np.asarray(camera_xyz, dtype=np.float64).reshape(-1, 3)
    n = len(C)
    if n < 2:
        return []

    r = (np.full(n, float(radii_m)) if np.isscalar(radii_m)
         else np.asarray(radii_m, dtype=np.float64).reshape(n))

    # 수평거리만 본다 (고도차는 겹침에 거의 영향 없음).
    xy = C[:, :2]

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(xy)
        max_reach = float((r.max() * 2) * margin)
        neighbor_lists = tree.query_ball_point(xy, r=max_reach)
        candidates = ((i, j) for i, nbrs in enumerate(neighbor_lists)
                      for j in nbrs if j > i)
    except ImportError:  # pragma: no cover
        logger.info("scipy 없음 — 전수 조합으로 인접쌍 선택 (느림)")
        candidates = itertools.combinations(range(n), 2)

    scored: dict[int, list[tuple[float, int]]] = {i: [] for i in range(n)}
    for i, j in candidates:
        d = float(np.hypot(*(xy[i] - xy[j])))
        if d < require_min_baseline_m:
            continue
        if d >= (r[i] + r[j]) * margin:
            continue
        scored[i].append((d, j))
        scored[j].append((d, i))

    pairs: set[tuple[int, int]] = set()
    for i, lst in scored.items():
        lst.sort()
        keep = lst if max_pairs_per_image is None else lst[:max_pairs_per_image]
        for _, j in keep:
            pairs.add((min(i, j), max(i, j)))

    out = sorted(pairs)
    total = n * (n - 1) // 2
    logger.info("인접쌍 선택: %d쌍 (전수 %d쌍의 %.1f%%), 이미지당 평균 %.1f개",
                len(out), total, 100.0 * len(out) / max(total, 1),
                2.0 * len(out) / n)
    isolated = [i for i in range(n) if not any(i in p for p in out)]
    if isolated:
        logger.warning("이웃이 없는 고립 프레임 %d장 — 이 프레임들은 BA 에서 "
                       "RTK prior 로만 고정된다 (index 예: %s)",
                       len(isolated), isolated[:8])
    return out

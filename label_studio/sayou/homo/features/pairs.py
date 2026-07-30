"""KD-Tree 기반 인접 페어 탐색 + tie-point 빌드.

수천 장 처리 시 풀 매칭 O(N²) 을 피하기 위해 RTK 좌표로 KD-Tree 를 만들어
각 이미지의 최근접 ``k`` 개 이웃과만 매칭. RANSAC fundamental matrix 로
outlier 제거 후 inlier 만 보존.

견고성
------
* SIFT 추출이 실패한 이미지 (desc=None, kp=[]) 는 매칭에서 제외.
* ``cv2.findFundamentalMat`` 는 다음 케이스에서 OpenCV assertion 으로
  죽을 수 있어 try/except 로 감싸고, 추가로 사전 조건을 검사한다:
    - 입력 점이 8 개 미만 (7-point algorithm 최소)
    - 점들이 동일선상에 있어 행렬이 퇴화
    - dtype 또는 shape 불일치
* RANSAC 실패한 페어는 단순히 건너뛰고, 메인 결과에서 제외.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import time
from typing import Iterable, Sequence
from scipy.spatial import cKDTree

from sayou.image.metadata import ImageMetadata

from ..crs import CRSConverter
from ..gpu_backend import free_gpu_memory
from .extract import extract_all_features, extract_features
from .match import match_pair

from ..utils import fmt_elapsed

logger = logging.getLogger(__name__)

# findFundamentalMat 의 8-point algorithm 최소 점 개수.
# OpenCV 의 7-point 모드도 있지만 RANSAC 안정성을 위해 충분한 여유를 둔다.
_MIN_POINTS_FOR_F_MATRIX = 20

# Lowe ratio test 통과한 매칭이 이 미만이면 페어 자체를 건너뜀.
_MIN_RAW_MATCHES = 30

# RANSAC inlier 가 이 미만이면 페어를 신뢰하지 않음.
_MIN_INLIER_MATCHES = 20


def find_neighbor_pairs(metas: list[ImageMetadata],
                        crs: CRSConverter,
                        k_neighbors: int = 8) -> list[tuple[int, int]]:
    """RTK 좌표 KD-Tree 로 인접 페어 후보 생성.

    O(N²) 풀 매칭 대신 O(N log N) 으로 단축. 수천 장 처리 시 필수.
    """
    xy = np.array([crs.forward(m.gps.lng, m.gps.lat) for m in metas])
    if len(metas) < 2:
        logger.warning("이미지가 2장 미만 → 페어 매칭 불가")
        return []
    tree = cKDTree(xy)
    pairs = set()
    for i in range(len(metas)):
        _, idxs = tree.query(xy[i], k=min(k_neighbors + 1, len(metas)))
        idxs = np.atleast_1d(idxs)  # k=1 이면 스칼라가 반환되므로 배열로 강제
        for j in idxs:
            j = int(j)
            if j != i:  # 자기 자신 제외
                pairs.add((min(i, j), max(i, j)))
    logger.info("인접 페어 %d개 생성 (k=%d)", len(pairs), k_neighbors)
    return sorted(pairs)


def _safe_find_fundamental(pts_i: np.ndarray, pts_j: np.ndarray) -> np.ndarray | None:
    """``cv2.findFundamentalMat`` 호출을 사전 검사 + 예외 방어로 감싸기.

    OpenCV 의 내부 assertion (예: matrix.cpp row range 체크) 은 ``cv2.error`` 로
    발생하므로 명시적으로 잡는다. 모든 점이 동일선상이거나, 8 점 미만이거나,
    pts shape 가 (N, 2) 가 아니면 ``None`` 반환.

    Returns
    -------
    inlier mask : (N,) bool array 또는 None
    """
    # ---- 1) 사전 조건 검사 ------------------------------------------------
    if pts_i is None or pts_j is None:
        return None
    if pts_i.shape != pts_j.shape:
        return None
    if pts_i.ndim != 2 or pts_i.shape[1] != 2:
        return None
    if pts_i.shape[0] < _MIN_POINTS_FOR_F_MATRIX:
        return None

    # ---- 2) 동일선상 (degenerate) 사전 차단 --------------------------------
    # 점들이 모두 같은 좌표거나 한 직선 위에 있으면 findFundamentalMat 가
    # SVD 분해 단계에서 죽는다. 분산이 0 에 가까우면 사전에 거른다.
    if np.std(pts_i, axis=0).min() < 1e-6:
        return None
    if np.std(pts_j, axis=0).min() < 1e-6:
        return None

    # ---- 3) dtype 보장 ----------------------------------------------------
    # OpenCV 는 float32/float64 만 받음. float16 등으로 오면 죽음.
    pts_i = np.ascontiguousarray(pts_i, dtype=np.float32)
    pts_j = np.ascontiguousarray(pts_j, dtype=np.float32)

    # ---- 4) findFundamentalMat 호출 + 예외 방어 ----------------------------
    try:
        F, mask = cv2.findFundamentalMat(
            pts_i, pts_j, cv2.FM_RANSAC, 1.0, 0.99,
        )
    except cv2.error as e:
        # OpenCV 의 -215 assertion 등이 ``cv2.error`` 로 raise 됨.
        logger.debug("findFundamentalMat 실패: %s", e)
        return None
    if mask is None or F is None or F.size == 0:
        return None
    return mask.ravel().astype(bool)


def _collect_needed_indices(pairs: Iterable[tuple[int, int]]) -> list[int]:
    """
    이미지 쌍 리스트에서 등장하는 모든 고유 이미지 인덱스를 수집.

    Args:
        pairs: (i, j) 형태의 이미지 인덱스 쌍 시퀀스.
               예: [(0, 1), (0, 2), (1, 2), (2, 3)]

    Returns:
        정렬된 고유 인덱스 리스트. 위 예시 → [0, 1, 2, 3]

    이 함수의 목적: pairs에 등장하는 이미지에 대해서만 feature를 추출하기 위함.
    전체 metas를 다 추출하면 매칭에 쓰이지 않는 이미지까지 처리하게 됨.
    """
    needed: set[int] = set()
    for pair in pairs:
        needed.update(pair)
    return sorted(needed)

def build_tie_points(metas: list[ImageMetadata],
                     pairs: list[tuple[int, int]]):
    """SIFT 특징점 추출 + 페어별 매칭 + RANSAC outlier 제거.

    견고성
    ------
    * 특징점이 너무 적은 이미지는 사전 제외 (전체 페어에서 빠짐).
    * ``cv2.findFundamentalMat`` 의 OpenCV assertion 을 ``_safe_find_fundamental``
      이 흡수 — 동일선상/저점수/dtype 문제로 죽지 않는다.
    * 각 페어가 어떤 이유로 제외됐는지 통계 로그 출력.

    Returns
    -------
    matches : dict[(i, j) -> (pts_i, pts_j, idx_i, idx_j)]
        pts_i, pts_j : (N, 2) 매칭된 픽셀 좌표
        idx_i, idx_j : (N,)   각 매칭의 keypoint 인덱스
                       (track 생성 시 어느 keypoint끼리 연결됐는지 추적용)
    features : dict[i -> (keypoints, descriptors, shape)]
    """
    if not pairs:
        logger.warning("매칭할 페어가 없음 → tie point 생성 생략")
        return {}, {}

    # 페어에 등장하는 이미지만 SIFT 추출 (불필요한 추출/메모리 절약)
    t0 = time.perf_counter()
    needed = sorted({i for pair in pairs for i in pair})
    features = {i: extract_features(metas[i].origin_path) for i in needed}
    logger.info("- SIFT 추출: %s", fmt_elapsed(time.perf_counter() - t0))

    # 진단: 추출 실패 / 특징점 부족 이미지.
    empty_imgs = [
        i for i, (kp, desc, _shape) in features.items()
        if desc is None or len(kp) < _MIN_POINTS_FOR_F_MATRIX
    ]
    if empty_imgs:
        logger.warning(
            "특징점이 부족한 이미지 %d장 (전체 %d장): 인덱스 예시 %s ...",
            len(empty_imgs), len(features), empty_imgs[:5],
        )

    matches = {}
    dropped_no_desc = 0
    dropped_few_matches = 0
    dropped_ransac = 0
    dropped_few_inliers = 0

    for i, j in pairs:
        kp_i, desc_i, _ = features[i]
        kp_j, desc_j, _ = features[j]

        # 빈 descriptor 차단.
        if desc_i is None or desc_j is None:
            dropped_no_desc += 1
            continue
        if (len(kp_i) < _MIN_POINTS_FOR_F_MATRIX
                or len(kp_j) < _MIN_POINTS_FOR_F_MATRIX):
            dropped_no_desc += 1
            continue

        # 매칭 (BFMatcher 예외 흡수).
        try:
            good = match_pair(desc_i, desc_j)
        except cv2.error as e:
            logger.debug("매칭 실패 (pair %d-%d): %s", i, j, e)
            dropped_ransac += 1
            continue

        if len(good) < 30:
            dropped_few_matches += 1
            continue

        try:
            pts_i = np.float32([kp_i[m.queryIdx].pt for m in good])
            pts_j = np.float32([kp_j[m.trainIdx].pt for m in good])
        except (IndexError, AttributeError) as e:
            logger.warning("매칭 인덱스 오류 (pair %d-%d): %s", i, j, e)
            dropped_ransac += 1
            continue

        idx_i = np.array([m.queryIdx for m in good], dtype=np.int64)
        idx_j = np.array([m.trainIdx for m in good], dtype=np.int64)

        # RANSAC (방어된 wrapper 사용).
        inliers = _safe_find_fundamental(pts_i, pts_j)
        if inliers is None:
            dropped_ransac += 1
            continue
        if inliers.sum() < 20:
            dropped_few_inliers += 1
            continue

        matches[(i, j)] = (pts_i[inliers], pts_j[inliers],
                           idx_i[inliers], idx_j[inliers])

    logger.info(
        "유효 매칭 페어: %d/%d (제거: desc없음 %d, 매칭부족 %d, "
        "RANSAC실패 %d, inlier부족 %d)",
        len(matches), len(pairs),
        dropped_no_desc, dropped_few_matches,
        dropped_ransac, dropped_few_inliers,
    )
    return matches, features


__all__ = ["find_neighbor_pairs", "build_tie_points"]

"""Descriptor 페어 매칭 — CUDA BFMatcher 또는 CPU BFMatcher.

Descriptor 종류에 따라 거리 함수가 다르다:
* SIFT (float32, dim 128) → ``NORM_L2``
* ORB  (uint8,   dim  32) → ``NORM_HAMMING``

Lowe ratio test 기본값
----------------------
* SIFT: 0.75 (논문 기본)
* ORB : 0.80 (binary descriptor 는 ratio 분포가 살짝 다름, 약간 완화)
"""

from __future__ import annotations

import cv2
import numpy as np

from ..gpu_backend import HAS_CV_CUDA


def _match_gpu_l2(desc1: np.ndarray, desc2: np.ndarray,
                  ratio: float = 0.75) -> list:
    """CUDA BFMatcher (L2 / SIFT 용). desc 는 float32."""
    gpu1 = cv2.cuda_GpuMat()
    gpu1.upload(desc1.astype(np.float32))
    gpu2 = cv2.cuda_GpuMat()
    gpu2.upload(desc2.astype(np.float32))
    matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(gpu1, gpu2, k=2)
    return [m for m, n in knn if m.distance < ratio * n.distance]


def _match_gpu_hamming(desc1: np.ndarray, desc2: np.ndarray,
                       ratio: float = 0.8) -> list:
    """CUDA BFMatcher (Hamming / ORB 용). desc 는 uint8."""
    gpu1 = cv2.cuda_GpuMat()
    gpu1.upload(desc1)
    gpu2 = cv2.cuda_GpuMat()
    gpu2.upload(desc2)
    matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(gpu1, gpu2, k=2)
    return [m for m, n in knn if m.distance < ratio * n.distance]


def _match_cpu(desc1, desc2, norm_type, ratio: float) -> list:
    bf = cv2.BFMatcher(norm_type)
    knn = bf.knnMatch(desc1, desc2, k=2)
    return [m for m, n in knn if m.distance < ratio * n.distance]


def match_pair(desc1: np.ndarray | None,
               desc2: np.ndarray | None,
               desc_type: str = "SIFT") -> list:
    """페어 descriptor 매칭. GPU 가용 + descriptor 종류에 따라 자동 분기.

    Parameters
    ----------
    desc1, desc2 : descriptor 행렬. ``None`` 이거나 길이가 2 미만이면 빈 리스트.
    desc_type : ``"SIFT"`` (L2) 또는 ``"ORB"`` (Hamming).

    Returns
    -------
    Lowe ratio test 통과한 ``cv2.DMatch`` 리스트.
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    if desc_type == "ORB":
        if HAS_CV_CUDA:
            return _match_gpu_hamming(desc1, desc2)
        return _match_cpu(desc1, desc2, cv2.NORM_HAMMING, ratio=0.8)
    # SIFT
    if HAS_CV_CUDA:
        return _match_gpu_l2(desc1, desc2)
    return _match_cpu(desc1, desc2, cv2.NORM_L2, ratio=0.75)


__all__ = ["match_pair"]
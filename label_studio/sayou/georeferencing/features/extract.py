"""특징점 추출 — CUDA SIFT / CUDA ORB / CPU SIFT(ProcessPool) 자동 선택.

가속 우선순위
-------------
1. **CUDA SIFT** (OpenCV 4.5.1+ contrib 빌드 필요)
2. **CUDA ORB** — SIFT 대비 5~10배 빠름, descriptor 가 binary
3. **CPU SIFT + ProcessPool** — GPU 모두 불가용일 때 ``n_workers`` 개
   프로세스로 병렬 추출. ``gpu_backend.PROCESSPOOL_SAFE`` 가 False 면
   자동으로 단일 프로세스 직렬 실행으로 fallback.

호출자는 단순히 ``extract_all_features(paths)`` 만 부르면 되고, 반환 dict
에 ``__desc_type__`` 키로 descriptor 종류("SIFT" or "ORB") 가 들어가서
매칭 단계가 거리 함수를 자동 선택할 수 있다.

Python 3.14 + opencv-python 4.13 호환성
----------------------------------------
opencv-python 4.13 빌드가 numpy 2 ABI 를 요구하는데 런타임에 numpy 1 이
있으면 spawn 워커에서 ``import cv2`` 시 segfault → BrokenProcessPool.
이 조합은 ``gpu_backend._diagnose_processpool_safety`` 가 자동 감지해서
``PROCESSPOOL_SAFE=False`` 로 표시하므로 본 모듈은 즉시 단일 프로세스로
전환한다.

추가 안전장치
-------------
* ``spawn`` context 명시 (가장 보수적).
* 워커 안 예외를 ``("err", path, traceback)`` 튜플로 직렬화 (segfault 가
  아닌 일반 예외는 메인 로그에 traceback 표시).
* 워커 안 OpenCV/BLAS 스레드 1 로 제한 (스레드 폭증 방지).
* 런타임 ``BrokenProcessPool`` 발생 시 자동 단일 프로세스 재시도.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import traceback
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..gpu_backend import (
    HAS_CV_CUDA_ORB,
    HAS_CV_CUDA_SIFT,
    PROCESSPOOL_SAFE,
)

from .sift import FeatureResult, KorniaSIFTExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker initializer
# ---------------------------------------------------------------------------
def _worker_init():
    """워커 프로세스 안에서 OpenCV/BLAS 스레드 수를 1 로 제한."""
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


# ---------------------------------------------------------------------------
# Single-image extractors
# ---------------------------------------------------------------------------
def _extract_cuda_sift(image_path: Path, max_features: int):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)
    if hasattr(cv2.cuda, "SIFT_CUDA"):
        sift = cv2.cuda.SIFT_CUDA.create(nfeatures=max_features)
    else:
        sift = cv2.cuda.SIFT_create(nfeatures=max_features)
    kp_gpu, desc_gpu = sift.detectAndComputeAsync(gpu_img, None)
    kp = sift.convert(kp_gpu)
    desc = desc_gpu.download() if desc_gpu is not None else None
    return kp, desc, img.shape


def _extract_cuda_orb(image_path: Path, max_features: int):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)
    orb = cv2.cuda.ORB_create(nfeatures=max_features)
    kp_gpu, desc_gpu = orb.detectAndComputeAsync(gpu_img, None)
    kp = orb.convert(kp_gpu)
    desc = desc_gpu.download() if desc_gpu is not None else None
    return kp, desc, img.shape


def _extract_cpu_sift_pickleable(args):
    """CPU SIFT 워커 함수 — ProcessPool 안에서 호출 (top-level).

    Returns
    -------
    ``("ok", kp_arr, desc, shape)`` — 성공.
    ``("err", path_str, traceback_str)`` — 일반 예외. 메인에서 로깅.

    워커 자체가 segfault 한 경우는 본 함수가 반환하지 못하고, 메인 쪽
    ``fut.result()`` 가 ``BrokenProcessPool`` 을 던지므로 상위에서 처리.
    """
    image_path, max_features = args
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ("ok",
                    np.zeros((0, 7), dtype=np.float32),
                    None,
                    (0, 0))
        sift = cv2.SIFT_create(nfeatures=max_features)
        kp, desc = sift.detectAndCompute(img, None)
        kp_arr = (
            np.array(
                [(k.pt[0], k.pt[1], k.size, k.angle,
                  k.response, k.octave, k.class_id) for k in kp],
                dtype=np.float32,
            )
            if kp else np.zeros((0, 7), dtype=np.float32)
        )
        return ("ok", kp_arr, desc, img.shape)
    except Exception:
        return ("err", str(image_path), traceback.format_exc())


def _unpack_kp_array(kp_arr: np.ndarray) -> list:
    """``(N, 7)`` 직렬화 배열 → ``cv2.KeyPoint`` 리스트 복원."""
    return [
        cv2.KeyPoint(
            x=float(r[0]), y=float(r[1]), size=float(r[2]),
            angle=float(r[3]), response=float(r[4]),
            octave=int(r[5]), class_id=int(r[6]),
        )
        for r in kp_arr
    ]


# ---------------------------------------------------------------------------
# CPU SIFT 추출 오케스트레이션
# ---------------------------------------------------------------------------
def _extract_cpu_sift_serial(image_paths: list[Path],
                             max_features: int) -> dict[int, Any]:
    """단일 프로세스 직렬 추출."""
    features: dict[int, Any] = {}
    for i, path in enumerate(image_paths):
        result = _extract_cpu_sift_pickleable((path, max_features))
        if result[0] == "err":
            logger.warning("SIFT 추출 실패: %s\n%s", result[1], result[2])
            features[i] = ([], None, (0, 0))
            continue
        _, kp_arr, desc, shape = result
        features[i] = (_unpack_kp_array(kp_arr), desc, shape)
    return features


def _extract_cpu_sift_parallel(image_paths: list[Path],
                               max_features: int,
                               n_workers: int) -> dict[int, Any]:
    """``ProcessPoolExecutor`` 로 병렬 추출 (spawn context).

    워커 segfault 시 ``BrokenProcessPool`` 이 raise 되므로 상위에서 catch.
    """
    features: dict[int, Any] = {}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers,
                             mp_context=ctx,
                             initializer=_worker_init) as pool:
        futs = {
            pool.submit(_extract_cpu_sift_pickleable, (p, max_features)): i
            for i, p in enumerate(image_paths)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            result = fut.result()  # BrokenProcessPool 은 여기서 raise.
            if result[0] == "err":
                logger.warning("SIFT 추출 실패: %s\n%s", result[1], result[2])
                features[i] = ([], None, (0, 0))
                continue
            _, kp_arr, desc, shape = result
            features[i] = (_unpack_kp_array(kp_arr), desc, shape)
    return features


def _resolve_n_workers(n_workers: int | None) -> int:
    """워커 수 결정. PROCESSPOOL_SAFE / 환경변수 / 인자 우선순위."""
    # 위험 조합 자동 감지 시 강제 1.
    if not PROCESSPOOL_SAFE:
        return 1

    # 환경변수 명시.
    env_workers = os.environ.get("GEOREF_NUM_WORKERS")
    if env_workers:
        try:
            return max(1, int(env_workers))
        except ValueError:
            pass

    # 인자 명시.
    if n_workers is not None:
        return max(1, n_workers)

    # 기본값.
    return min(8, (os.cpu_count() or 4))


# ------------------------------------------------------------------ ORB
def extract_features_orb(
    path: str | Path,
    max_features: int = 5000,
) -> FeatureResult:
    """OpenCV ORB. CPU 전용, 가장 빠르고 가벼움."""
    img = _load_grayscale(path)
    orb = cv2.ORB_create(nfeatures=max_features)
    kp, desc = orb.detectAndCompute(img, None)

    if desc is None or len(kp) == 0:
        return FeatureResult(
            keypoints=np.empty((0, 2), dtype=np.float32),
            descriptors=np.empty((0, 32), dtype=np.uint8),
            responses=np.empty((0,), dtype=np.float32),
        )

    pts = np.array([k.pt for k in kp], dtype=np.float32)
    resp = np.array([k.response for k in kp], dtype=np.float32)
    return FeatureResult(keypoints=pts, descriptors=desc, responses=resp)


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------
def extract_features(image_path: Path, max_features: int = 8000):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp, desc = sift.detectAndCompute(img, None)

    #orb = cv2.ORB_create(nfeatures=max_features)
    #kp, desc = orb.detectAndCompute(img, None)
    #print(kp, type(kp), desc, type(desc))
    
    return kp, desc, img.shape


def extract_features_device(
    path: str | Path,
    backend: Literal["orb", "sift"] = "orb",
    max_features: int = 5000,
    extractor: Optional[KorniaSIFTExtractor] = None,
) -> FeatureResult:
    """
    pairs.py에서 호출하는 단일 진입점.

    - backend='orb': CPU ORB (가벼움, 빠름)
    - backend='sift': Kornia SIFT (MPS/CUDA 가속, 더 풍부한 descriptor)

    SIFT를 여러 이미지에 쓸 때는 KorniaSIFTExtractor를 미리 만들어
    extractor 인자로 넘기면 모델 재초기화 비용을 피할 수 있음.
    """
    if backend == "orb":
        return extract_features_orb(path, max_features=max_features)

    if backend == "sift":
        if extractor is None:
            extractor = KorniaSIFTExtractor(max_features=max_features)
        return extractor.extract(path)

    raise ValueError(f"Unknown backend: {backend}")


def extract_all_features(image_paths: list[Path],
                         max_features: int = 8000,
                         n_workers: int | None = None) -> dict[Any, Any]:
    """이미지 경로 리스트 → ``{idx: (keypoints, descriptors, shape)}`` dict.

    반환 dict 에는 추가로 ``"__desc_type__"`` 키로 descriptor 종류
    (``"SIFT"`` 또는 ``"ORB"``) 가 들어간다.

    Parameters
    ----------
    image_paths : 추출할 이미지 경로 리스트.
    max_features : 한 장당 추출 상한 (기본 8000).
    n_workers : CPU SIFT fallback 시 프로세스 수. ``None`` 이면
        ``min(8, os.cpu_count())``. GPU 경로에서는 무시.
        ``gpu_backend.PROCESSPOOL_SAFE`` 가 False 면 자동으로 1.
    """
    features: dict[Any, Any] = {}

    if HAS_CV_CUDA_SIFT:
        logger.info("특징점 추출: CUDA SIFT (이미지 %d장)", len(image_paths))
        for i, path in enumerate(image_paths):
            features[i] = _extract_cuda_sift(path, max_features)
        features["__desc_type__"] = "SIFT"
        return features

    if HAS_CV_CUDA_ORB:
        logger.info("특징점 추출: CUDA ORB (이미지 %d장)", len(image_paths))
        for i, path in enumerate(image_paths):
            features[i] = _extract_cuda_orb(path, max_features)
        features["__desc_type__"] = "ORB"
        return features

    # ---- CPU SIFT fallback ------------------------------------------------
    workers = _resolve_n_workers(n_workers)

    if workers <= 1:
        logger.info("특징점 추출: CPU SIFT (단일 프로세스, 이미지 %d장)",
                    len(image_paths))
        feats = _extract_cpu_sift_serial(image_paths, max_features)
    else:
        logger.info("특징점 추출: CPU SIFT × %d workers (spawn, 이미지 %d장)",
                    workers, len(image_paths))
        try:
            feats = _extract_cpu_sift_parallel(
                image_paths, max_features, workers,
            )
        except BrokenExecutor as e:
            # 사전 진단을 통과했지만 런타임에 워커가 죽은 경우.
            logger.error(
                "ProcessPool 워커가 죽었습니다 (%s). 단일 프로세스로 재시도합니다.\n"
                "반복되면 GEOREF_DISABLE_MULTIPROCESS=1 로 강제하거나 "
                "OpenCV/Python/numpy 빌드 호환성을 확인하세요.",
                type(e).__name__,
            )
            feats = _extract_cpu_sift_serial(image_paths, max_features)

    features.update(feats)
    features["__desc_type__"] = "SIFT"
    return features


__all__ = ["extract_all_features", "extract_features"]
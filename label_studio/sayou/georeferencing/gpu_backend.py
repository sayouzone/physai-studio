"""GPU 가속 백엔드 + 버전 호환성 진단.

이 모듈은 사용 가능한 GPU 라이브러리를 import 단계에서 감지하여
런타임에 안전하게 분기할 수 있도록 plain 플래그/얇은 wrapper 만 제공한다.

추가로, 알려진 위험 조합(예: opencv-python 4.13 + Python 3.14 + numpy 2 미만)
을 감지해서 ProcessPool 자체를 자동 비활성화하여 ``BrokenProcessPool`` 을
사전에 방지한다.

핵심 원칙
---------
1. 어떤 라이브러리든 import 실패해도 절대 모듈 import 가 죽지 않게.
2. 호출부는 capability 플래그(``HAS_CUPY`` 등)만 보고 분기.
3. GPU 사용 후 메모리는 명시적으로 해제 (장기 실행 시 누수 방지).

환경변수
--------
* ``GEOREF_DISABLE_GPU=1`` — 모든 GPU 경로 비활성화
* ``GEOREF_DISABLE_MULTIPROCESS=1`` — ProcessPool 비활성, 단일 프로세스 강제
* ``GEOREF_NUM_WORKERS=N`` — ProcessPool 워커 수 명시 (extract.py 에서 읽음)
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CuPy
# ---------------------------------------------------------------------------
HAS_CUPY: bool = False
cp = None  # type: ignore
try:
    import cupy as _cp  # type: ignore

    _cp.cuda.runtime.getDeviceCount()
    cp = _cp
    HAS_CUPY = True
    logger.info("[GPU] CuPy 활성화 (CUDA device count=%d)",
                _cp.cuda.runtime.getDeviceCount())
except Exception as e:
    logger.info("[GPU] CuPy 비활성 (%s) — BA/triangulation/warp CPU 사용",
                type(e).__name__)


# ---------------------------------------------------------------------------
# OpenCV CUDA
# ---------------------------------------------------------------------------
HAS_CV_CUDA: bool = False
HAS_CV_CUDA_SIFT: bool = False
HAS_CV_CUDA_ORB: bool = False
HAS_CV_CUDA_REMAP: bool = False
_CV2_VERSION: str = "unknown"
try:
    import cv2 as _cv2

    _CV2_VERSION = getattr(_cv2, "__version__", "unknown")
    if _cv2.cuda.getCudaEnabledDeviceCount() > 0:
        HAS_CV_CUDA = True
        HAS_CV_CUDA_SIFT = hasattr(_cv2.cuda, "SIFT_CUDA") or hasattr(
            _cv2.cuda, "SIFT_create"
        )
        HAS_CV_CUDA_ORB = hasattr(_cv2.cuda, "ORB_create")
        HAS_CV_CUDA_REMAP = hasattr(_cv2.cuda, "remap")
        logger.info(
            "[GPU] OpenCV CUDA 활성화 (SIFT=%s, ORB=%s, remap=%s)",
            HAS_CV_CUDA_SIFT, HAS_CV_CUDA_ORB, HAS_CV_CUDA_REMAP,
        )
    else:
        logger.info("[GPU] OpenCV CUDA 비활성 (device 0개)")
except Exception as e:
    logger.info("[GPU] OpenCV CUDA 비활성 (%s)", type(e).__name__)


# ---------------------------------------------------------------------------
# PyTorch (LightGlue 등 학습 기반 매칭 옵션)
# ---------------------------------------------------------------------------
HAS_TORCH: bool = False
HAS_LIGHTGLUE: bool = False
torch = None  # type: ignore
try:
    import torch as _torch  # type: ignore

    if _torch.cuda.is_available():
        torch = _torch
        HAS_TORCH = True
        try:
            from lightglue import LightGlue, SuperPoint  # type: ignore  # noqa: F401

            HAS_LIGHTGLUE = True
            logger.info("[GPU] PyTorch + LightGlue 활성화")
        except Exception:
            logger.info("[GPU] PyTorch CUDA 활성, LightGlue 미설치")
    else:
        logger.info("[GPU] PyTorch CUDA 비활성")
except Exception as e:
    logger.info("[GPU] PyTorch 비활성 (%s)", type(e).__name__)


# ---------------------------------------------------------------------------
# 버전 호환성 진단 — ProcessPool 안전성 판정
# ---------------------------------------------------------------------------
# 알려진 위험 조합:
#   1. opencv-python 4.13.x + Python 3.14 + numpy < 2
#      → opencv-python 4.13 빌드가 numpy 2 ABI 를 요구하는데
#        런타임에 numpy 1 이 있으면 spawn 워커에서 cv2 import 시 segfault.
#        근거: https://github.com/opencv/opencv-python/issues/1201
#
# 이외에도 새 위험 조합이 발견되면 _known_bad_combo() 에 추가.
# 진단 결과에 따라 PROCESSPOOL_SAFE 플래그를 설정하고, extract.py 가
# 이 플래그를 보고 단일 프로세스 fallback 으로 즉시 전환한다.
PROCESSPOOL_SAFE: bool = True
_PROCESSPOOL_DIAGNOSIS: str = ""


def _check_numpy_version() -> tuple[int, int]:
    """numpy major.minor 버전 튜플 반환. import 실패 시 (0, 0)."""
    try:
        import numpy as _np
        parts = _np.__version__.split(".")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except Exception:
        return (0, 0)


def _diagnose_processpool_safety() -> tuple[bool, str]:
    """ProcessPool 사용이 안전한지 진단.

    Returns
    -------
    (safe, reason) : 안전 여부와 사유 메시지.
    """
    py_major, py_minor = sys.version_info.major, sys.version_info.minor
    np_major, np_minor = _check_numpy_version()
    cv_version = _CV2_VERSION

    # opencv-python 4.13.x + Python 3.14 + numpy < 2 → segfault in spawn worker
    if (cv_version.startswith("4.13.")
            and (py_major, py_minor) >= (3, 14)
            and (np_major, np_minor) > (0, 0)
            and np_major < 2):
        return (False,
                f"opencv-python {cv_version} + Python {py_major}.{py_minor} "
                f"+ numpy {np_major}.{np_minor} 조합은 spawn 워커에서 "
                f"BrokenProcessPool 을 일으킵니다. numpy>=2 로 업그레이드하거나 "
                f"opencv-python<4.13 으로 다운그레이드 권장. "
                f"임시 회피로 ProcessPool 자동 비활성화.")

    # 환경변수 강제.
    if os.environ.get("GEOREF_DISABLE_MULTIPROCESS", "").lower() in ("1", "true", "yes"):
        return (False, "GEOREF_DISABLE_MULTIPROCESS 환경변수로 비활성")

    return (True, "")


PROCESSPOOL_SAFE, _PROCESSPOOL_DIAGNOSIS = _diagnose_processpool_safety()
if not PROCESSPOOL_SAFE:
    logger.warning("[MP] ProcessPool 비활성: %s", _PROCESSPOOL_DIAGNOSIS)


# ---------------------------------------------------------------------------
# 환경변수로 강제 비활성
# ---------------------------------------------------------------------------
if os.environ.get("GEOREF_DISABLE_GPU", "").lower() in ("1", "true", "yes"):
    logger.warning("[GPU] GEOREF_DISABLE_GPU 설정 — 모든 GPU 경로 비활성화")
    HAS_CUPY = False
    HAS_CV_CUDA = False
    HAS_CV_CUDA_SIFT = False
    HAS_CV_CUDA_ORB = False
    HAS_CV_CUDA_REMAP = False
    HAS_TORCH = False
    HAS_LIGHTGLUE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def free_gpu_memory() -> None:
    """루프 사이에 호출해서 캐시된 GPU 메모리 풀을 비운다."""
    if HAS_CUPY:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    if HAS_TORCH:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def gpu_summary() -> str:
    """현재 활성화된 가속 경로 요약 (시작 로그용)."""
    parts: list[str] = []
    if HAS_CUPY:
        parts.append("CuPy")
    if HAS_CV_CUDA:
        bits = []
        if HAS_CV_CUDA_SIFT:
            bits.append("SIFT")
        if HAS_CV_CUDA_ORB:
            bits.append("ORB")
        if HAS_CV_CUDA_REMAP:
            bits.append("remap")
        parts.append(f"OpenCV-CUDA[{','.join(bits) or 'none'}]")
    if HAS_LIGHTGLUE:
        parts.append("LightGlue")
    elif HAS_TORCH:
        parts.append("PyTorch")
    mp_status = "MP=on" if PROCESSPOOL_SAFE else "MP=off"
    if not parts:
        return f"CPU only ({mp_status})"
    return f"{' + '.join(parts)} ({mp_status})"


__all__ = [
    "HAS_CUPY", "cp",
    "HAS_CV_CUDA", "HAS_CV_CUDA_SIFT", "HAS_CV_CUDA_ORB", "HAS_CV_CUDA_REMAP",
    "HAS_TORCH", "HAS_LIGHTGLUE", "torch",
    "PROCESSPOOL_SAFE",
    "free_gpu_memory", "gpu_summary",
]
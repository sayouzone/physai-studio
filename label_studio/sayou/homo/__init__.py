"""GCP-Free Georeferencing 패키지.

드론 RTK/PPK 좌표를 카메라 위치 제약조건으로 사용하는 GCP-free
SfM + Bundle Adjustment + Orthophoto 파이프라인.

공개 API
--------
* :func:`pipeline.run_pipeline` — 메인 진입점.
* :class:`crs.CRSConverter` — WGS84 ↔ 투영좌표 변환.
* :mod:`rtk` — RTK 품질 검증 및 prior 가중치.
* :mod:`features` — SIFT/ORB 추출, 매칭, tie-point 빌드.
* :mod:`sfm` — track 빌드, 삼각측량, RTK 제약 BA.
* :mod:`ortho` — 평면 가정 정사영상.
* :mod:`gpu_backend` — GPU capability 감지 + fallback 헬퍼.

가속 백엔드 — CuPy / OpenCV CUDA / PyTorch+LightGlue 를 런타임 감지.
모두 미가용 환경에서도 CPU fallback 으로 동작.
"""

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
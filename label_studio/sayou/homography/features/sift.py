"""Kornia SIFT 추출기 (선택적 GPU 경로) + 공통 헬퍼.

★ 이 버전에서 고친 것
---------------------
1. **torch / kornia 지연 import.** 원본은 모듈 최상단에서
   ``import torch`` / ``from kornia.feature import SIFTFeature`` 를 했다.
   그런데 ``features/__init__`` → ``extract`` → ``sift`` 체인 때문에
   **torch 가 없으면 파이프라인 전체가 import 되지 않는다**::

       >>> import solar_thermal.georeferencing
       ModuleNotFoundError: No module named 'torch'

   Kornia SIFT 는 ``backend="sift"`` 를 명시했을 때만 쓰는 선택 경로인데,
   CPU/CUDA-OpenCV 만으로 돌리려는 배포 환경까지 torch 를 강제했다.
   → 클래스 생성자 안으로 옮겨서, 실제로 쓸 때만 import 하고 없으면
   명확한 안내 메시지를 낸다.

2. **``Path`` import 누락.** 원본은 ``path: str | Path`` 를 타입힌트로
   쓰면서 ``pathlib.Path`` 를 import 하지 않았다.
   ``from __future__ import annotations`` 덕분에 정의 시점에는 안 죽지만,
   ``typing.get_type_hints`` 나 pydantic 류가 힌트를 평가하는 순간 죽는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class FeatureResult:
    """공통 feature 결과 컨테이너."""
    keypoints: np.ndarray                    # (N, 2) — (x, y) 픽셀 좌표
    descriptors: np.ndarray                  # (N, D) ORB uint8 D=32 / SIFT float32 D=128
    responses: Optional[np.ndarray] = None   # (N,) keypoint strength


def _load_grayscale(path: str | Path) -> np.ndarray:
    """이미지를 grayscale uint8 numpy array 로 로드."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _require_torch():
    """torch + kornia 를 지연 import. 없으면 안내와 함께 실패."""
    try:
        import torch
        from kornia.feature import SIFTFeature
    except ImportError as exc:
        raise ImportError(
            "Kornia SIFT 경로에는 torch 와 kornia 가 필요합니다 "
            "(pip install torch kornia). GPU 없이 쓰려면 "
            "backend='orb' 또는 OpenCV SIFT 경로를 사용하세요."
        ) from exc
    return torch, SIFTFeature


def _select_device(torch, prefer: str = "auto"):
    """MPS > CUDA > CPU 순으로 사용 가능한 디바이스 선택."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _numpy_to_tensor(torch, img: np.ndarray, device):
    """(H, W) uint8 → (1, 1, H, W) float32 [0, 1] tensor on device."""
    tensor = torch.from_numpy(img).float() / 255.0
    return tensor.unsqueeze(0).unsqueeze(0).to(device)


# ------------------------------------------------------------------ Kornia SIFT
class KorniaSIFTExtractor:
    """``SIFTFeature`` 를 한 번만 초기화해서 재사용.

    ``pairs.py`` 에서 이미지마다 새로 만들지 않도록 클래스로 감쌌다.
    torch/kornia 는 이 생성자에서 처음 import 된다.
    """

    def __init__(self, max_features: int = 5000, device: str = "auto"):
        torch, SIFTFeature = _require_torch()
        self._torch = torch
        self.device = _select_device(torch, device)
        self.max_features = max_features
        self.sift = SIFTFeature(num_features=max_features).to(self.device).eval()

    def extract(self, path: str | Path) -> FeatureResult:
        torch = self._torch
        with torch.no_grad():
            img = _load_grayscale(path)
            img_tensor = _numpy_to_tensor(torch, img, self.device)
            lafs, responses, descs = self.sift(img_tensor)

            # lafs: (1, N, 2, 3) — affine frame, 중심점은 [:, :, :, 2]
            if lafs.shape[1] == 0:
                return FeatureResult(
                    keypoints=np.empty((0, 2), dtype=np.float32),
                    descriptors=np.empty((0, 128), dtype=np.float32),
                    responses=np.empty((0,), dtype=np.float32),
                )
            return FeatureResult(
                keypoints=lafs[0, :, :, 2].cpu().numpy().astype(np.float32),
                descriptors=descs[0].cpu().numpy().astype(np.float32),
                responses=responses[0].cpu().numpy().astype(np.float32),
            )


__all__ = ["FeatureResult", "KorniaSIFTExtractor", "_load_grayscale"]

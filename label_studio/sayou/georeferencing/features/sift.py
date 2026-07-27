# src/solar_thermal/georeferencing/features/extract.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np
import torch
from kornia.feature import SIFTFeature


@dataclass
class FeatureResult:
    """공통 feature 결과 컨테이너."""
    keypoints: np.ndarray  # (N, 2) - (x, y) 픽셀 좌표
    descriptors: np.ndarray  # (N, D) - ORB: uint8 D=32, SIFT: float32 D=128
    responses: Optional[np.ndarray] = None  # (N,) keypoint strength


def _select_device(prefer: str = "auto") -> torch.device:
    """MPS > CUDA > CPU 순으로 사용 가능한 디바이스 선택."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_grayscale(path: str | Path) -> np.ndarray:
    """이미지를 grayscale uint8 numpy array로 로드."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _numpy_to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W) uint8 → (1, 1, H, W) float32 [0, 1] tensor on device."""
    tensor = torch.from_numpy(img).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (H, W) → (1, 1, H, W)
    return tensor.to(device)


# ------------------------------------------------------------------ Kornia SIFT
class KorniaSIFTExtractor:
    """
    SIFTFeature를 한 번만 초기화해서 재사용.
    pairs.py에서 이미지마다 새로 만들지 않도록 클래스로 감쌈.
    """

    def __init__(
        self,
        max_features: int = 5000,
        device: str = "auto",
    ):
        self.device = _select_device(device)
        self.max_features = max_features
        self.sift = SIFTFeature(num_features=max_features).to(self.device).eval()

    @torch.no_grad()
    def extract(self, path: str | Path) -> FeatureResult:
        img = _load_grayscale(path)
        img_tensor = _numpy_to_tensor(img, self.device)

        lafs, responses, descs = self.sift(img_tensor)

        # lafs: (1, N, 2, 3) — affine frame, 중심점은 [:, :, :, 2]
        # responses: (1, N)
        # descs: (1, N, 128)
        if lafs.shape[1] == 0:
            return FeatureResult(
                keypoints=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, 128), dtype=np.float32),
                responses=np.empty((0,), dtype=np.float32),
            )

        keypoints = lafs[0, :, :, 2].cpu().numpy().astype(np.float32)  # (N, 2) (x, y)
        descriptors = descs[0].cpu().numpy().astype(np.float32)        # (N, 128)
        resp = responses[0].cpu().numpy().astype(np.float32)           # (N,)

        return FeatureResult(
            keypoints=keypoints,
            descriptors=descriptors,
            responses=resp,
        )
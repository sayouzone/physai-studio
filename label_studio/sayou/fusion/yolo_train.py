"""
YOLOv11 학습 지원: 데이터셋 내보내기, data.yaml 생성, 학습 실행, 신뢰도 보정 fit.

    RGB 모델 / IR 모델 / 패널 세그 모델을 **각각 따로** 학습한다.
    한 모델에 두 모달리티를 섞어 넣지 않는다 — late fusion 의 전제가
    "분기가 독립적으로 결론을 낸다"이므로, 모델도 분리되어야 한다.

내보내기 규칙 (YOLO 표준 레이아웃)

    dataset/
      images/{train,val}/*.jpg
      labels/{train,val}/*.txt      # cls cx cy w h   (정규화) 또는 seg 폴리곤
      data.yaml
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .yolo_backend import ThermalNormalizer


# --------------------------------------------------------------------------- #
# 데이터셋 내보내기
# --------------------------------------------------------------------------- #

@dataclass
class LabeledBox:
    cls_name: str
    xyxy: Tuple[float, float, float, float]
    polygon: Optional[np.ndarray] = None     # (N,2) 절대 픽셀. seg 학습용


def _write_label(path: str, boxes: Sequence[LabeledBox], names: List[str],
                 w: int, h: int, segment: bool) -> None:
    lines = []
    for b in boxes:
        if b.cls_name not in names:
            continue
        ci = names.index(b.cls_name)
        if segment and b.polygon is not None and len(b.polygon) >= 3:
            pts = np.asarray(b.polygon, dtype=np.float64).copy()
            pts[:, 0] = np.clip(pts[:, 0] / w, 0, 1)
            pts[:, 1] = np.clip(pts[:, 1] / h, 0, 1)
            lines.append(f"{ci} " + " ".join(f"{v:.6f}" for v in pts.reshape(-1)))
        else:
            x0, y0, x1, y1 = b.xyxy
            cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
            bw, bh = abs(x1 - x0) / w, abs(y1 - y0) / h
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{ci} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_rgb_dataset(samples: Sequence[Tuple[np.ndarray, Sequence[LabeledBox]]],
                       out_dir: str, names: List[str],
                       val_ratio: float = 0.2, segment: bool = False,
                       seed: int = 0) -> str:
    """samples = [(BGR 이미지, 라벨들), ...] -> YOLO 데이터셋 디렉터리."""
    return _export(samples, out_dir, names, val_ratio, segment, seed, None)


def export_ir_dataset(samples: Sequence[Tuple[np.ndarray, Sequence[LabeledBox]]],
                      out_dir: str, names: List[str],
                      normalizer: ThermalNormalizer,
                      val_ratio: float = 0.2, segment: bool = False,
                      seed: int = 0) -> str:
    """
    samples 의 이미지는 **섭씨 float 배열**이어야 한다.

    정규화는 `normalizer` 가 하며, 그 설정이 `normalizer.json` 으로 함께 저장된다.
    추론 파이프라인은 이 파일을 읽어 같은 규칙을 쓴다.
    **이 파일을 잃어버리면 학습된 IR 모델은 사실상 못 쓴다.**
    """
    return _export(samples, out_dir, names, val_ratio, segment, seed, normalizer)


def _export(samples, out_dir, names, val_ratio, segment, seed, normalizer) -> str:
    rng = random.Random(seed)
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    for i, (img, boxes) in enumerate(samples):
        split = "val" if rng.random() < val_ratio else "train"
        if normalizer is not None:
            arr = np.asarray(img, dtype=np.float32)
            ref = float(np.median(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else None
            out_img = normalizer.to_3ch(arr, ref)
        else:
            out_img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = out_img.shape[:2]
        stem = f"{i:06d}"
        cv2.imwrite(os.path.join(out_dir, "images", split, stem + ".jpg"), out_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        _write_label(os.path.join(out_dir, "labels", split, stem + ".txt"),
                     boxes, names, w, h, segment)

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\nval: images/val\n\nnames:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")

    if normalizer is not None:
        with open(os.path.join(out_dir, "normalizer.json"), "w", encoding="utf-8") as f:
            json.dump(normalizer.to_dict(), f, indent=2)
    return yaml_path


# --------------------------------------------------------------------------- #
# 학습
# --------------------------------------------------------------------------- #

def train_yolo(data_yaml: str,
               model: str = "yolo11s.pt",
               epochs: int = 200,
               imgsz: int = 960,
               batch: int = 16,
               device: str = "0",
               project: str = "runs/pv",
               name: str = "exp",
               modality: str = "rgb",
               **overrides):
    """
    ultralytics 학습 래퍼. 모달리티별로 증강 기본값을 다르게 준다.

    IR 에서 색상 증강(hsv_h/hsv_s)을 켜면 안 된다. 열화상의 화소값은 온도이고,
    채널 3개는 같은 gray 의 복제다. 색조를 흔드는 것은 온도를 흔드는 것과 같아서
    ΔT 근거를 학습하지 못하게 만든다. 밝기(hsv_v) 역시 소폭만 허용한다.

    또한 모듈은 상하 뒤집혀도 모듈이므로 flipud 를 켜도 되지만, 정션박스처럼
    위치에 의미가 있는 클래스가 있으면 꺼야 한다.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:                                   # pragma: no cover
        raise ImportError("pip install ultralytics") from e

    if modality == "ir":
        aug = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.15,
                   degrees=8.0, translate=0.1, scale=0.35, shear=2.0,
                   fliplr=0.5, flipud=0.0, mosaic=0.6, mixup=0.0, erasing=0.2)
    else:
        aug = dict(hsv_h=0.012, hsv_s=0.5, hsv_v=0.4,
                   degrees=8.0, translate=0.1, scale=0.4, shear=2.0,
                   fliplr=0.5, flipud=0.0, mosaic=1.0, mixup=0.05, erasing=0.3)
    aug.update(overrides)

    m = YOLO(model)
    return m.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                   device=device, project=project, name=name,
                   patience=40, cos_lr=True, **aug)


# --------------------------------------------------------------------------- #
# 신뢰도 보정 fit
# --------------------------------------------------------------------------- #

def fit_calibration(detector, samples, iou_thresh: float = 0.4):
    """
    검증셋에서 (conf, 정답여부) 쌍을 모아 클래스별 보정 파라미터를 학습한다.

    samples = [(image, [LabeledBox, ...]), ...]

    보정 없이 YOLO conf 를 Dempster-Shafer 에 넣으면 과신 때문에 한 분기가
    거의 항상 이긴다. 융합 전에 반드시 한 번 돌려야 하는 단계다.
    """
    from .yolo_backend import ConfidenceCalibrator

    pool: Dict[str, Tuple[List[float], List[int]]] = {}
    for img, gts in samples:
        dets = detector.detect(img)
        gt_by_cls: Dict[str, List[np.ndarray]] = {}
        for g in gts:
            gt_by_cls.setdefault(g.cls_name, []).append(np.asarray(g.xyxy, float))
        for d in dets:
            boxes = gt_by_cls.get(d.cls_name, [])
            hit = 0
            for b in boxes:
                x0, y0 = max(d.xyxy[0], b[0]), max(d.xyxy[1], b[1])
                x1, y1 = min(d.xyxy[2], b[2]), min(d.xyxy[3], b[3])
                inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
                ua = d.area + (b[2] - b[0]) * (b[3] - b[1]) - inter
                if inter / max(ua, 1e-9) >= iou_thresh:
                    hit = 1
                    break
            c, y = pool.setdefault(d.cls_name, ([], []))
            c.append(d.conf)
            y.append(hit)

    calib = ConfidenceCalibrator()
    for cls, (confs, ys) in pool.items():
        calib.fit_class(cls, confs, ys)
    detector.calibrator = calib
    return calib

"""
패널(모듈) ROI 추출 및 RGB<->IR ROI 페어링.

Late fusion 은 "동일 좌표 패널"이 두 모달리티에서 같은 개체를 가리킨다는 전제
위에서만 성립한다. 따라서 ROI 는 한쪽(보통 해상도가 높은 RGB)에서 정의하고,
정합 호모그래피로 IR 좌표계에 사상하는 방식이 가장 안전하다.
반대로 IR 에서만 보이는 개체는 late fusion 대상이 아니라 IR 단독 판정 대상이다.

외부 라벨(Label Studio export, GeoJSON)이 있으면 그것을 1순위로 쓴다.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .registration import crossmodal_feature, normalize01, to_gray, warp_points
from .types import PanelROI


# --------------------------------------------------------------------------- #
# 격자 각도 (FFT)
# --------------------------------------------------------------------------- #

def dominant_grid_angle(img: np.ndarray, search_deg: float = 90.0) -> float:
    """
    패널 배열의 지배적 격자 회전각(도)을 FFT 파워 스펙트럼에서 추정.
    회전 보정 없이 ROI 통계를 내면 측정 아티팩트가 크게 생기므로 먼저 구한다.
    반환 범위: [-45, 45)
    """
    g = to_gray(img).astype(np.float32)
    g = normalize01(g)
    h, w = g.shape
    n = int(2 ** np.floor(np.log2(min(h, w))))
    n = max(n, 64)
    y0, x0 = (h - n) // 2, (w - n) // 2
    patch = g[y0:y0 + n, x0:x0 + n]
    patch = patch - patch.mean()
    win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    F = np.fft.fftshift(np.abs(np.fft.fft2(patch * win)))
    F = np.log1p(F)

    cy = cx = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - cy, xx - cx)
    band = (r > n * 0.02) & (r < n * 0.45)
    theta = np.degrees(np.arctan2(yy - cy, xx - cx))
    theta = (theta + 180.0) % 180.0          # 0~180 (선 방향은 180 주기)

    bins = np.arange(0.0, 180.0 + 1e-6, 0.5)
    idx = np.digitize(theta[band], bins) - 1
    power = np.bincount(idx, weights=F[band], minlength=len(bins) - 1)
    count = np.bincount(idx, minlength=len(bins) - 1).astype(np.float64)
    prof = power / np.maximum(count, 1.0)

    # 직교 격자는 90도 떨어진 두 피크를 만든다 -> 두 방향 에너지를 합산
    k = len(prof)
    half = k // 2
    comb = prof[:half] + np.roll(prof, -half)[:half]
    best = int(np.argmax(comb))
    ang = bins[best] + 0.25
    ang = ((ang + 45.0) % 90.0) - 45.0
    if abs(ang) > search_deg:
        return 0.0
    return float(ang)


# --------------------------------------------------------------------------- #
# 패널 검출
# --------------------------------------------------------------------------- #

def _dark_mask(rgb: np.ndarray, bg_sigma_frac: float = 1 / 14.0,
               min_contrast: float = 5.0) -> np.ndarray:
    """
    '주변보다 어두운' 영역 마스크.

    adaptive threshold 는 블록이 모듈보다 작으면 모듈 내부의 국소 평균이 곧
    모듈 자신이 되어 아무것도 못 잡는다. 대신 큰 커널 배경 추정치를 빼서
    (top-hat 과 같은 원리) 조도 변화에 강인한 대비 영상을 만든다.
    """
    g = to_gray(rgb)
    g8 = (normalize01(g) * 255).astype(np.uint8)
    g8 = cv2.createCLAHE(2.0, (8, 8)).apply(g8)
    sigma = max(8.0, bg_sigma_frac * max(g8.shape))
    bg = cv2.GaussianBlur(g8.astype(np.float32), (0, 0), sigma)
    diff = bg - g8.astype(np.float32)          # 어두울수록 큰 값
    d8 = np.clip(diff, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(float(t), min_contrast)
    m = (diff >= t).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return m


def _split_rect(rect, k: int) -> List[np.ndarray]:
    """회전 사각형을 장축 방향으로 k 등분한 폴리곤 리스트로 반환."""
    (cx, cy), (w, h), ang = rect
    long_is_w = w >= h
    L = max(w, h)
    S = min(w, h)
    step = L / k
    th = np.deg2rad(ang)
    ux = np.array([np.cos(th), np.sin(th)])            # 사각형 w 방향
    uy = np.array([-np.sin(th), np.cos(th)])
    along, across = (ux, uy) if long_is_w else (uy, ux)
    out = []
    for i in range(k):
        off = -L / 2 + step * (i + 0.5)
        c = np.array([cx, cy]) + along * off
        a = along * (step / 2) * 0.96
        b = across * (S / 2) * 0.98
        out.append(np.array([c - a - b, c + a - b, c + a + b, c - a + b]))
    return out


def detect_panels(rgb: np.ndarray,
                  min_area_px: int = 800,
                  max_area_frac: float = 0.30,
                  expected_aspect: float = 1.7,
                  aspect_tol: float = 0.55,
                  rectangularity_min: float = 0.62,
                  split_tables: bool = True,
                  array_grouping_px: Optional[float] = None) -> List[PanelROI]:
    """
    설명 가능한(디버깅 가능한) 패널 검출기.

      1) 배경 추정치를 뺀 대비 영상에서 '어두운 영역' 마스크 생성
      2) 연결 성분 -> minAreaRect
      3) 종횡비가 모듈 1장보다 크게 길면 테이블(여러 장이 맞붙은 구조)로 보고
         장축 방향으로 등분한다. 실제 어레이는 모듈이 맞닿아 있어 이 단계가 없으면
         한 덩어리로 잡힌다.

    운영 환경에서는 학습 기반 세그멘테이션으로 교체하는 것을 권장한다.
    반환 타입(List[PanelROI])만 유지하면 이하 파이프라인은 그대로 동작한다.
    """
    m = _dark_mask(rgb)
    H, W = m.shape[:2]
    max_area = max_area_frac * H * W
    global_angle = dominant_grid_angle(rgb)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    rois: List[PanelROI] = []

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px or area > max_area:
            continue
        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        w_r, h_r = rect[1]
        if min(w_r, h_r) < 4 or w_r * h_r <= 0:
            continue
        if cv2.contourArea(c) / (w_r * h_r) < rectangularity_min:
            continue

        aspect = max(w_r, h_r) / max(min(w_r, h_r), 1e-6)
        k = 1
        if split_tables:
            k = max(1, int(round(aspect / expected_aspect)))
        if k == 1:
            lo = expected_aspect * (1.0 - aspect_tol)
            hi = expected_aspect * (1.0 + aspect_tol) + 0.4
            if not (lo <= aspect <= hi):
                continue
            polys = [cv2.boxPoints(rect).astype(np.float64)]
        else:
            polys = _split_rect(rect, k)

        for poly in polys:
            if cv2.contourArea(poly.astype(np.float32)) < min_area_px * 0.5:
                continue
            rois.append(PanelROI(
                panel_id=f"P{len(rois):05d}",
                polygon_rgb=poly,
                grid_angle_deg=global_angle,
                meta={"area_px": float(area) / max(k, 1),
                      "aspect": float(aspect), "split_k": int(k)}))

    if array_grouping_px:
        _assign_arrays(rois, array_grouping_px)
    return rois


def _assign_arrays(rois: List[PanelROI], radius: float) -> None:
    """중심점 근접 연결 성분으로 어레이(스트링) 그룹 부여. ΔT 기준값 계산에 쓰인다."""
    if not rois:
        return
    c = np.array([r.polygon_rgb.mean(axis=0) for r in rois])
    parent = list(range(len(rois)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(rois)):
        d = np.linalg.norm(c - c[i], axis=1)
        for j in np.where((d < radius) & (np.arange(len(rois)) > i))[0]:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    for i, r in enumerate(rois):
        r.array_id = f"A{find(i):04d}"


# --------------------------------------------------------------------------- #
# 외부 라벨 로더
# --------------------------------------------------------------------------- #

def load_panels_from_labelstudio(path: str, image_width: int, image_height: int,
                                 label_filter: Optional[Sequence[str]] = None) -> List[PanelROI]:
    """
    Label Studio JSON export 에서 polygonlabels / rectanglelabels 를 읽어 ROI 로 변환.
    좌표는 퍼센트 단위이므로 픽셀로 환산한다.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = data if isinstance(data, list) else [data]
    rois: List[PanelROI] = []

    for task in tasks:
        for ann in task.get("annotations", []) or task.get("completions", []):
            for res in ann.get("result", []):
                v = res.get("value", {})
                labels = v.get("polygonlabels") or v.get("rectanglelabels") or []
                if label_filter and not set(labels) & set(label_filter):
                    continue
                if "points" in v:
                    pts = np.array(v["points"], dtype=np.float64)
                    pts[:, 0] *= image_width / 100.0
                    pts[:, 1] *= image_height / 100.0
                elif {"x", "y", "width", "height"} <= set(v):
                    x = v["x"] * image_width / 100.0
                    y = v["y"] * image_height / 100.0
                    w = v["width"] * image_width / 100.0
                    h = v["height"] * image_height / 100.0
                    pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                    if v.get("rotation"):
                        th = np.deg2rad(v["rotation"])
                        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                        pts = (pts - pts[0]) @ R.T + pts[0]
                else:
                    continue
                rois.append(PanelROI(panel_id=res.get("id", f"P{len(rois):05d}"),
                                     polygon_rgb=pts,
                                     meta={"labels": labels}))
    return rois


# --------------------------------------------------------------------------- #
# 페어링 및 ROI 추출
# --------------------------------------------------------------------------- #

def project_rois_to_ir(rois: Iterable[PanelROI], H_rgb_to_ir: np.ndarray,
                       ir_shape: Tuple[int, int],
                       min_inside_ratio: float = 0.8) -> List[PanelROI]:
    """
    ROI 를 IR 좌표계로 사상하고, IR 프레임 밖으로 상당 부분 벗어난 ROI 는
    polygon_ir=None 으로 남긴다(-> RGB 단독 판정 대상).
    """
    ih, iw = ir_shape[:2]
    out = []
    for r in rois:
        p = warp_points(H_rgb_to_ir, r.polygon_rgb)
        inside = ((p[:, 0] >= 0) & (p[:, 0] < iw) & (p[:, 1] >= 0) & (p[:, 1] < ih)).mean()
        r.polygon_ir = p if inside >= min_inside_ratio else None
        r.meta["ir_inside_ratio"] = float(inside)
        out.append(r)
    return out


def crop_polygon(img: np.ndarray, polygon: np.ndarray,
                 pad: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    폴리곤 bbox 로 crop 하고 폴리곤 내부 마스크를 함께 반환.
    반환 배열은 원본 dtype 유지(열화상 radiometric float 보존).
    """
    h, w = img.shape[:2]
    x0 = int(max(0, np.floor(polygon[:, 0].min()) - pad))
    y0 = int(max(0, np.floor(polygon[:, 1].min()) - pad))
    x1 = int(min(w, np.ceil(polygon[:, 0].max()) + pad))
    y1 = int(min(h, np.ceil(polygon[:, 1].max()) + pad))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return np.empty((0, 0), img.dtype), np.empty((0, 0), bool)
    patch = img[y0:y1, x0:x1]
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [np.round(polygon - [x0, y0]).astype(np.int32)], 1)
    return patch, mask.astype(bool)


def deskew_patch(patch: np.ndarray, mask: np.ndarray, angle_deg: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    패널 격자를 축 정렬시킨다. 셀/서브스트링 단위 분할(3분할 등)을 하려면 필수.
    """
    if patch.size == 0 or abs(angle_deg) < 0.2:
        return patch, mask
    h, w = patch.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    flags = cv2.INTER_NEAREST if patch.dtype == np.uint8 else cv2.INTER_LINEAR
    p = cv2.warpAffine(patch, M, (w, h), flags=flags, borderValue=0)
    m = cv2.warpAffine(mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0
    return p, m

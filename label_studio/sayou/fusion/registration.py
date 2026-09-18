"""
RGB - IR 정합 (registration).

RGB 와 LWIR 은 복사 물리가 달라서 밝기 대응이 단조롭지 않다. 따라서 밝기 기반
정합(SSD/NCC on raw intensity)은 실패하기 쉽다. 이 모듈은 세 단계로 접근한다.

  1) coarse : 메타데이터(내부/외부 파라미터 + 지면 평면) 또는 GeoTransform 으로
              초기 호모그래피 H0 를 해석적으로 계산한다.
              평면 유도 호모그래피:  H = K_ir (R + t nᵀ / d) K_rgbⁿ¹
  2) refine  : gradient 크기 / 구조 텐서 도메인으로 변환한 뒤 ECC 로 정밀화.
              (에지는 modality 에 불변한 편이므로 cross-modal 에서 잘 동작)
  3) fallback: 상호정보량(MI) 기반 similarity 탐색. ECC 가 발산할 때 사용.

마지막으로 격자 잔차(residual_px)와 겹침 비율을 계산해 late fusion 의
분기 가중치로 넘긴다. 정합이 나쁘면 "융합하지 않는" 판단도 정상 동작에 포함된다.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import minimize

from .types import (CameraIntrinsics, GeoTransform, RegistrationResult,
                    StereoExtrinsics)

# --------------------------------------------------------------------------- #
# 전처리: cross-modal 공통 특징 도메인
# --------------------------------------------------------------------------- #

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def normalize01(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """퍼센타일 스트레치. 열화상의 극단 이상치(태양 반사, 하늘)를 잘라낸다."""
    x = img.astype(np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x)
    lo, hi = np.percentile(finite, [p_lo, p_hi])
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def crossmodal_feature(img: np.ndarray, blur_sigma: float = 1.2,
                       clahe_clip: float = 2.0) -> np.ndarray:
    """
    RGB/IR 공통 정합 특징: CLAHE -> 가우시안 -> Sobel 크기 -> 퍼센타일 정규화.

    밝기 극성(polarity)에 의존하지 않도록 gradient '크기'만 쓴다.
    태양광 패널처럼 직선 격자가 강한 장면에서 특히 안정적이다.
    """
    g = to_gray(img)
    g8 = (normalize01(g) * 255.0).astype(np.uint8)
    g8 = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(g8)
    f = cv2.GaussianBlur(g8.astype(np.float32), (0, 0), blur_sigma)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return normalize01(mag, 2.0, 98.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# 1) Coarse: 해석적 초기해
# --------------------------------------------------------------------------- #

def plane_induced_homography(K_rgb: np.ndarray,
                             K_ir: np.ndarray,
                             ext: StereoExtrinsics,
                             plane_normal: np.ndarray,
                             plane_distance: float) -> np.ndarray:
    """
    평면 유도 호모그래피.  RGB 픽셀 -> IR 픽셀.

    plane_normal     : RGB 카메라 좌표계에서의 평면 법선 (단위벡터)
    plane_distance   : RGB 카메라 중심에서 평면까지 거리 d [m] (nᵀX = d)

    d 는 LRF(레이저 거리계) 또는 지면 평면 추정값을 쓴다. d 가 20 m 이상이고
    베이스라인이 수 cm 인 M3T 급 구성에서는 t·nᵀ/d 항이 작아 회전 항이 지배적이다.
    그래도 근거리(<10 m) 또는 경사 지형에서는 이 항이 수 픽셀 차이를 만든다.
    """
    n = np.asarray(plane_normal, dtype=np.float64).reshape(3, 1)
    n = n / (np.linalg.norm(n) + 1e-12)
    t = np.asarray(ext.t, dtype=np.float64).reshape(3, 1)
    d = float(max(plane_distance, 1e-3))
    M = ext.R + (t @ n.T) / d
    H = K_ir @ M @ np.linalg.inv(K_rgb)
    return H / H[2, 2]


def homography_from_geotransform(gt_rgb: GeoTransform, gt_ir: GeoTransform) -> np.ndarray:
    """
    두 정사영상(ortho)이 이미 지오레퍼런싱 되어 있을 때의 정합.
    RGB 픽셀 -> 지상좌표 -> IR 픽셀. 사실상 affine 이며 정합 오차의 하한을 제공한다.
    """
    A_rgb = gt_rgb.to_matrix()          # col,row -> map
    A_ir = gt_ir.to_matrix()
    H = np.linalg.inv(A_ir) @ A_rgb     # RGB 픽셀 -> IR 픽셀
    return H / H[2, 2]


def homography_from_fov_only(rgb: CameraIntrinsics, ir: CameraIntrinsics) -> np.ndarray:
    """외부 파라미터를 모를 때의 최소 가정: 동축(co-axial) + 무한원 평면."""
    return (ir.K @ np.linalg.inv(rgb.K)) / 1.0


# --------------------------------------------------------------------------- #
# 2) Refine: ECC (gradient 도메인)
# --------------------------------------------------------------------------- #

def refine_ecc(rgb: np.ndarray,
               ir: np.ndarray,
               H0: np.ndarray,
               motion: int = cv2.MOTION_HOMOGRAPHY,
               iters: int = 400,
               eps: float = 1e-7,
               pyramid: Tuple[float, ...] = (0.25, 0.5, 1.0)) -> Tuple[np.ndarray, bool, float]:
    """
    ECC 로 H 를 정밀화. 피라미드로 수렴 반경을 넓힌다.

    반환: (H, converged, ecc_score)
    주의: OpenCV ECC 의 warp 는 'template(=IR) 좌표 -> input(=RGB) 좌표' 방향이
    아니라 findTransformECC(templateImage, inputImage, warp) 에서
    warp 가 template -> input 이다. 여기서는 IR 을 template 으로 두고
    H_ir_to_rgb 를 최적화한 뒤 역행렬을 취한다.
    """
    f_rgb = crossmodal_feature(rgb)
    f_ir = crossmodal_feature(ir)

    H = np.asarray(H0, dtype=np.float32).copy()
    score = -1.0
    ok = False

    for scale in pyramid:
        r = cv2.resize(f_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        i = cv2.resize(f_ir, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        S = np.diag([scale, scale, 1.0]).astype(np.float32)
        # H 는 RGB->IR. 스케일 좌표계로 옮김.
        H_s = (S @ H @ np.linalg.inv(S)).astype(np.float32)
        W = np.linalg.inv(H_s).astype(np.float32)          # IR->RGB (template->input)
        W = (W / W[2, 2]).astype(np.float32)
        if motion != cv2.MOTION_HOMOGRAPHY:
            W = W[:2, :].astype(np.float32)

        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
        try:
            score, W_opt = cv2.findTransformECC(
                templateImage=i, inputImage=r, warpMatrix=W,
                motionType=motion, criteria=crit, inputMask=None, gaussFiltSize=5
            )
            ok = True
        except cv2.error:
            continue

        if motion != cv2.MOTION_HOMOGRAPHY:
            W_opt = np.vstack([W_opt, [0.0, 0.0, 1.0]]).astype(np.float32)
        H_s_opt = np.linalg.inv(W_opt.astype(np.float64))
        H = (np.linalg.inv(S.astype(np.float64)) @ H_s_opt @ S.astype(np.float64))
        H = (H / H[2, 2]).astype(np.float32)

    return H.astype(np.float64), ok, float(score)


# --------------------------------------------------------------------------- #
# 3) Fallback: 상호정보량(MI)
# --------------------------------------------------------------------------- #

def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 48,
                       mask: Optional[np.ndarray] = None) -> float:
    if mask is None:
        mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask].ravel()
    b = b[mask].ravel()
    if a.size < 64:
        return 0.0
    hist, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 1], [0, 1]])
    pxy = hist / max(hist.sum(), 1.0)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])))


def refine_mi_similarity(rgb: np.ndarray, ir: np.ndarray, H0: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    ECC 실패 시 상호정보량으로 similarity(스케일·회전·평행이동) 보정만 수행.
    Powell 은 미분 불가 목적함수에 견고하다.
    """
    f_rgb = normalize01(to_gray(rgb))
    f_ir = normalize01(to_gray(ir))
    h, w = f_ir.shape[:2]

    def build(params):
        s, th, tx, ty = params
        c, sn = np.cos(th) * s, np.sin(th) * s
        D = np.array([[c, -sn, tx], [sn, c, ty], [0, 0, 1]], dtype=np.float64)
        return D @ H0

    def cost(params):
        H = build(params)
        warped = cv2.warpPerspective(f_rgb, H, (w, h), flags=cv2.INTER_LINEAR,
                                     borderValue=np.nan)
        m = np.isfinite(warped)
        if m.mean() < 0.2:
            return 10.0
        return -mutual_information(warped, f_ir, mask=m)

    best, best_cost = np.array([1.0, 0.0, 0.0, 0.0]), cost([1.0, 0.0, 0.0, 0.0])
    for x0 in ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 6.0, 0.0], [1.0, 0.0, -6.0, 0.0],
               [1.0, 0.0, 0.0, 6.0], [1.02, 0.01, 0.0, 0.0]):
        res = minimize(cost, np.array(x0), method="Powell",
                       options={"xtol": 1e-3, "ftol": 1e-4, "maxiter": 400})
        if res.fun < best_cost:
            best_cost, best = float(res.fun), res.x
    return build(best), -best_cost


# --------------------------------------------------------------------------- #
# 품질 평가
# --------------------------------------------------------------------------- #

def _overlap_ratio(H: np.ndarray, rgb_shape, ir_shape) -> float:
    h, w = rgb_shape[:2]
    ih, iw = ir_shape[:2]
    ys, xs = np.mgrid[0:h:16, 0:w:16]
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=0)
    q = H @ pts
    q = q[:2] / np.where(np.abs(q[2]) < 1e-12, 1e-12, q[2])
    inside = (q[0] >= 0) & (q[0] < iw) & (q[1] >= 0) & (q[1] < ih)
    return float(inside.mean())


def _residual_px(rgb: np.ndarray, ir: np.ndarray, H: np.ndarray,
                 tile: int = 96, stride: int = 64, search: int = 8) -> Tuple[float, float]:
    """
    정합 후 잔차 추정. IR 좌표계에서 타일별 위상상관(subpixel)으로 잔여 시프트를
    측정하고 그 RMS 를 잔차로 본다. 동시에 gradient NCC 도 돌려준다.
    이 값이 late fusion 가중치를 직접 좌우한다.
    """
    ih, iw = ir.shape[:2]
    f_ir = crossmodal_feature(ir)
    f_rgb = crossmodal_feature(rgb)
    warped = cv2.warpPerspective(f_rgb, H, (iw, ih), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    valid = cv2.warpPerspective(np.ones(f_rgb.shape[:2], np.float32), H, (iw, ih)) > 0.99
    shifts = []
    for y in range(0, ih - tile, stride):
        for x in range(0, iw - tile, stride):
            if not valid[y:y + tile, x:x + tile].all():
                continue
            A = warped[y:y + tile, x:x + tile]
            B = f_ir[y:y + tile, x:x + tile]
            if A.std() < 1e-3 or B.std() < 1e-3:
                continue
            win = cv2.createHanningWindow((tile, tile), cv2.CV_32F)
            (dx, dy), resp = cv2.phaseCorrelate(np.ascontiguousarray(A, np.float32),
                                                np.ascontiguousarray(B, np.float32), win)
            if resp < 0.05 or abs(dx) > search or abs(dy) > search:
                continue
            shifts.append((dx, dy))

    if len(shifts) < 4:
        resid = float("nan")
    else:
        s = np.asarray(shifts)
        # 로버스트: 상위 20% 이상치 절단
        r = np.linalg.norm(s, axis=1)
        keep = r <= np.percentile(r, 80)
        resid = float(np.sqrt(np.mean(r[keep] ** 2)))

    m = valid & np.isfinite(warped) & np.isfinite(f_ir)
    if m.sum() > 1000:
        a, b = warped[m], f_ir[m]
        ncc = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    else:
        ncc = float("nan")
    return resid, ncc


# --------------------------------------------------------------------------- #
# 공개 API
# --------------------------------------------------------------------------- #

def register_pair(rgb: np.ndarray,
                  ir: np.ndarray,
                  H_init: Optional[np.ndarray] = None,
                  rgb_cam: Optional[CameraIntrinsics] = None,
                  ir_cam: Optional[CameraIntrinsics] = None,
                  extrinsics: Optional[StereoExtrinsics] = None,
                  plane_normal: Optional[np.ndarray] = None,
                  plane_distance: Optional[float] = None,
                  use_ecc: bool = True,
                  residual_tol_px: float = 3.0) -> RegistrationResult:
    """
    RGB 영상과 IR 영상을 정합해 RGB->IR 호모그래피를 만든다.

    초기해 우선순위: H_init > (내부+외부+평면) > (내부만) > 단위행렬 스케일.
    """
    stages = []

    if H_init is not None:
        H0 = np.asarray(H_init, dtype=np.float64)
        stages.append("init:given")
    elif rgb_cam is not None and ir_cam is not None and extrinsics is not None \
            and plane_normal is not None and plane_distance is not None:
        H0 = plane_induced_homography(rgb_cam.K, ir_cam.K, extrinsics,
                                      plane_normal, plane_distance)
        stages.append("init:plane_induced")
    elif rgb_cam is not None and ir_cam is not None:
        H0 = homography_from_fov_only(rgb_cam, ir_cam)
        stages.append("init:fov")
    else:
        sx = ir.shape[1] / rgb.shape[1]
        sy = ir.shape[0] / rgb.shape[0]
        H0 = np.diag([sx, sy, 1.0])
        stages.append("init:scale")

    H = H0 / H0[2, 2]
    converged = True
    method = stages[-1]

    if use_ecc:
        H_ecc, ok, _score = refine_ecc(rgb, ir, H)
        if ok and np.all(np.isfinite(H_ecc)):
            H, method = H_ecc, "ecc"
            stages.append("refine:ecc")
        else:
            stages.append("refine:ecc_failed")

    resid, ncc = _residual_px(rgb, ir, H)

    # ECC 결과가 나쁘면 MI 폴백
    if (not np.isfinite(resid)) or resid > residual_tol_px or (np.isfinite(ncc) and ncc < 0.15):
        H_mi, mi = refine_mi_similarity(rgb, ir, H0)
        r2, n2 = _residual_px(rgb, ir, H_mi)
        stages.append(f"fallback:mi(mi={mi:.3f})")
        better = (np.isfinite(r2) and (not np.isfinite(resid) or r2 < resid))
        if better:
            H, resid, ncc, method = H_mi, r2, n2, "mi"

    if not np.isfinite(resid):
        converged = False
        resid = float("inf")

    return RegistrationResult(
        H_rgb_to_ir=H,
        method=method,
        residual_px=resid,
        overlap_ratio=_overlap_ratio(H, rgb.shape, ir.shape),
        converged=converged and resid <= residual_tol_px * 2.0,
        ncc=ncc,
        stages=stages,
    )


def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(N,2) 점들을 H 로 사상."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H.astype(np.float64)).reshape(-1, 2)


def warp_rgb_to_ir(rgb: np.ndarray, H: np.ndarray, ir_shape) -> np.ndarray:
    ih, iw = ir_shape[:2]
    return cv2.warpPerspective(rgb, H, (iw, ih), flags=cv2.INTER_LINEAR)

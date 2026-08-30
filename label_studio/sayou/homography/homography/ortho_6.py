"""정사영상 모자이크 — 시임 최적화 + 노출 보정.

이 파일이 고치는 것
-------------------
이전 ``select`` (winner-take-all) 는 각 출력 픽셀에서 **기하 점수**(feather ×
연직근접) 가 가장 높은 프레임을 골랐다. 문제는 그 점수가 **이미지 내용을 전혀
보지 않는다**는 것이다. 결과적으로 시임이 카메라 연직점의 중간지점을 따라
직선/다각형으로 지나가며, 그 선이 **패널 위를 그대로 가로지른다**.

패널은 주변(잔디·그림자) 과 밝기 차가 극단적이라, 프레임 간 정합이 10 cm 로
좋아도 그 선이 패널을 자르는 순간 눈에 띈다. 실측 모자이크에서 시임 대부분은
0.01~0.16 m 로 잘 맞아 있었는데도 육안으로는 "패널이 끊긴" 것처럼 보였다.
즉 **정합 오차가 아니라 시임의 '위치'가 문제**였다.

세 가지를 넣었다.

1. **시임 최적화** (``seam_optimize``) — 새 프레임을 얹을 때, 이미 채워진
   캔버스와의 **차이 영상**을 만들어 점수에서 뺀다. 두 프레임이 일치하는
   곳(균질한 잔디, 그림자 안쪽) 은 벌점이 없고, 어긋나는 곳(패널 경계) 은
   큰 벌점을 받는다. 그 결과 시임이 **패널을 피해 잔디/그림자 쪽으로 흘러간다.**
   차이 영상을 크게 블러해서 시임이 국소 노이즈를 따라 너덜거리지 않게 한다.

2. **노출 보정** (``exposure_compensate``) — 프레임마다 노출/화이트밸런스가
   달라 시임에서 밝기 계단이 생긴다. 겹침 영역의 중앙값 비로 프레임별 이득을
   구해 맞춘다. 이득은 ``[0.7, 1.4]`` 로 제한해서 한 장이 튀어도 전체가
   망가지지 않게 한다.

3. **포화(정반사) 페널티** (``glint_penalty``) — 태양광 패널은 유리라 시야각에
   따라 하늘을 정반사해 완전히 날아간다(255 포화). 그런 픽셀은 결함 판독에
   쓸 수 없으므로 점수를 깎아, 같은 지점을 정상 노출로 찍은 다른 프레임이
   선택되게 한다.

이 셋은 모두 **원인 불문**으로 동작한다. 정합 오차가 relief 때문이든 자세
오차 때문이든, 시임이 차이가 작은 곳으로 가고 밝기가 맞으면 이음선은 보이지
않는다. 상용 정사모자이크 소프트웨어가 쓰는 방식과 같은 계열이다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .homography import (
    FrameHomography,
    ground_bounds_from_homography,
    recommend_gsd,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MosaicConfig",
    "build_source_weight",
    "warp_frame",
    "orthorectify_frame",
    "mosaic_frames",
]

MAX_OUT_PIXELS = 400_000_000
_BLEND_MODES = ("select", "average")


class MosaicConfig:
    """모자이크 설정.

    Attributes
    ----------
    blend_mode : ``"select"`` (기본) winner-take-all. ``"average"`` 는
        가중평균 — 정합 잔차가 있으면 ghosting 이 생기므로 비권장.
    feather_px : 경계 페더링 폭 (원본 픽셀).
    nadir_falloff : 주점에서 멀어질수록 점수 감쇠 (0~1).
    align_frames : 합성 직전에 각 프레임을 **이미 놓인 캔버스에 위상상관으로
        정렬**해서 프레임별 등록 오차(자세·BA 잔차로 인한 0.1~0.5 m 수준의
        평행이동)를 소거한다. 패널 상단 에지의 세로 단차와 톱니 끊김의
        직접 대책. 보정량은 로그와 stats 에 기록된다.
    align_max_m : 정렬 탐색 반경 (m). **반복 구조(패널 행 주기)의 절반보다
        작아야 한다** — 실측 행 주기 2.72 m 기준 1.0 m 이 안전하다. 등록
        오차가 이보다 크면 정렬로는 못 잡고 SfM/BA 를 고쳐야 한다.
    align_min_response : 최적 위치 ZNCC 하한 (0~1). 겹침에 텍스처가 없거나
        내용이 다르면(정반사 등) 정렬을 생략한다.
    seam_optimize : 차이 영상 벌점으로 시임을 저차이 영역(잔디·그림자) 으로
        유도. **패널이 시임에서 끊겨 보이는 문제의 주 대책.**
    seam_cost_weight : 벌점 강도. 0 이면 시임 최적화 없음. 크게 줄수록 시임이
        내용을 더 따라가지만, 지나치면 연직근접 점수를 무시해 기울어진 시야의
        프레임이 선택될 수 있다. 0.5~2.0 권장.
    seam_cost_blur_m : 차이 영상 블러 반경 (m). 시임이 국소 노이즈를 따라
        너덜거리지 않게 한다. 패널 한 장 폭 정도(1~2 m) 가 적당.
    exposure_compensate : 프레임별 이득으로 밝기 계단 제거.
    glint_penalty : 포화 픽셀 점수 감쇠 (0~1). 0 이면 미사용.
    glint_threshold : 이 값 이상을 포화로 본다 (8bit 기준).
    seam_blend_px : 시임 좌우 이 폭만 부드럽게 섞는다. 0 이면 하드 컷.
    band_count, dtype, max_out_pixels : 출력 형식.
    """

    def __init__(self,
                 blend_mode: str = "select",
                 feather_px: float = 120.0,
                 nadir_falloff: float = 0.6,
                 align_frames: bool = True,
                 align_max_m: float = 1.0,
                 align_min_response: float = 0.30,
                 seam_optimize: bool = True,
                 seam_cost_weight: float = 1.0,
                 seam_cost_blur_m: float = 1.5,
                 exposure_compensate: bool = True,
                 glint_penalty: float = 0.7,
                 glint_threshold: int = 250,
                 seam_blend_px: float = 0.0,
                 band_count: int = 3,
                 dtype=np.uint8,
                 max_out_pixels: int = MAX_OUT_PIXELS):
        if blend_mode not in _BLEND_MODES:
            raise ValueError(f"blend_mode={blend_mode!r} — {_BLEND_MODES} 중 하나")
        self.blend_mode = blend_mode
        self.align_frames = bool(align_frames)
        self.align_max_m = float(max(align_max_m, 0.0))
        self.align_min_response = float(align_min_response)
        self.feather_px = float(feather_px)
        self.nadir_falloff = float(np.clip(nadir_falloff, 0.0, 1.0))
        self.seam_optimize = bool(seam_optimize)
        self.seam_cost_weight = float(max(seam_cost_weight, 0.0))
        self.seam_cost_blur_m = float(max(seam_cost_blur_m, 0.0))
        self.exposure_compensate = bool(exposure_compensate)
        self.glint_penalty = float(np.clip(glint_penalty, 0.0, 1.0))
        self.glint_threshold = int(glint_threshold)
        self.seam_blend_px = float(max(seam_blend_px, 0.0))
        self.band_count = int(band_count)
        self.dtype = dtype
        self.max_out_pixels = int(max_out_pixels)


# ---------------------------------------------------------------------------
# 기하 점수 맵
# ---------------------------------------------------------------------------
_WEIGHT_CACHE: dict[tuple, np.ndarray] = {}


def build_source_weight(width: int, height: int, cx: float, cy: float,
                        cfg: MosaicConfig) -> np.ndarray:
    """(H, W) float32 기하 점수. feather × 연직근접. 카메라별 캐시."""
    key = (width, height, round(cx, 3), round(cy, 3),
           cfg.feather_px, cfg.nadir_falloff)
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    if cfg.feather_px > 0:
        mask = np.ones((height, width), dtype=np.uint8)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        w = np.clip(dist / cfg.feather_px, 0.0, 1.0).astype(np.float32)
    else:
        w = np.ones((height, width), dtype=np.float32)

    if cfg.nadir_falloff > 0:
        ys, xs = np.mgrid[0:height, 0:width]
        r2 = ((xs - cx) ** 2 + (ys - cy) ** 2).astype(np.float32)
        r2_max = float(max(cx, width - cx) ** 2 + max(cy, height - cy) ** 2)
        w *= (1.0 - cfg.nadir_falloff * (r2 / r2_max)).astype(np.float32)

    w = np.maximum(w, 1e-4).astype(np.float32)
    if len(_WEIGHT_CACHE) > 8:
        _WEIGHT_CACHE.clear()
    _WEIGHT_CACHE[key] = w
    return w


def _frame_window(fh, x_min, y_max, gsd_m, out_w, out_h):
    b = fh.footprint_bounds()
    if b is None:
        return None
    bx0, by0, bx1, by1 = b
    u0 = max(int(np.floor((bx0 - x_min) / gsd_m)) - 1, 0)
    u1 = min(int(np.ceil((bx1 - x_min) / gsd_m)) + 1, out_w)
    v0 = max(int(np.floor((y_max - by1) / gsd_m)) - 1, 0)
    v1 = min(int(np.ceil((y_max - by0) / gsd_m)) + 1, out_h)
    if u1 <= u0 or v1 <= v0:
        return None
    return u0, v0, u1 - u0, v1 - v0


def warp_frame(fh: FrameHomography, image: np.ndarray,
               x_min: float, y_max: float, gsd_m: float,
               out_w: int, out_h: int, cfg: MosaicConfig,
               undistort_maps=None):
    """프레임 하나를 자기 창 안에서 정사 warp. ``(u0, v0, warped, weight)``."""
    win = _frame_window(fh, x_min, y_max, gsd_m, out_w, out_h)
    if win is None:
        return None
    u0, v0, ww, wh = win

    if undistort_maps is not None:
        image = cv2.remap(image, undistort_maps[0], undistort_maps[1],
                          interpolation=cv2.INTER_LINEAR)

    H = fh.ortho_pixel_matrix(x_min + u0 * gsd_m, y_max - v0 * gsd_m, gsd_m)
    warped = cv2.warpPerspective(
        image, H, (ww, wh),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    src_w = build_source_weight(fh.intr.width, fh.intr.height,
                                fh.intr.cx, fh.intr.cy, cfg)
    weight = cv2.warpPerspective(
        src_w, H, (ww, wh),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    # 지평선 너머(동차 분모 부호 반전) 영역 제거.
    us = np.arange(ww, dtype=np.float32)
    vs = np.arange(wh, dtype=np.float32)[:, None]
    weight[(H[2, 0] * us + H[2, 1] * vs + H[2, 2]) <= 1e-9] = 0.0

    if warped.ndim == 2:
        warped = warped[:, :, None]

    # 포화(정반사) 페널티 — 날아간 픽셀은 판독 불가이므로 우선순위를 낮춘다.
    if cfg.glint_penalty > 0:
        sat = warped.max(axis=2) >= cfg.glint_threshold
        if sat.any():
            weight[sat] *= (1.0 - cfg.glint_penalty)
    return u0, v0, warped, weight


# ---------------------------------------------------------------------------
# 단일 프레임 GeoTIFF
# ---------------------------------------------------------------------------
def orthorectify_frame(fh, image_path, output_path, gsd_m=None, epsg=5186,
                       cfg=None, undistort=True) -> bool:
    """사진 한 장 → 정사영상 GeoTIFF."""
    import rasterio
    from rasterio.transform import from_origin

    cfg = cfg or MosaicConfig()
    b = fh.footprint_bounds()
    if b is None:
        logger.warning("정사영상 건너뜀 (footprint 실패): %s", image_path)
        return False
    x_min, y_min, x_max, y_max = b
    if gsd_m is None:
        gsd_m = fh.gsd_at_principal_point()
    if not np.isfinite(gsd_m) or gsd_m <= 0:
        logger.warning("정사영상 건너뜀 (GSD 비정상): %s", image_path)
        return False

    out_w = int(np.ceil((x_max - x_min) / gsd_m))
    out_h = int(np.ceil((y_max - y_min) / gsd_m))
    if out_w <= 0 or out_h <= 0 or out_w * out_h > cfg.max_out_pixels:
        logger.warning("정사영상 건너뜀 (출력 %d×%d): %s", out_w, out_h, image_path)
        return False

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.warning("정사영상 건너뜀 (읽기 실패): %s", image_path)
        return False
    if img.ndim == 3 and img.shape[2] >= 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)

    maps = fh.intr.undistort_maps() if undistort else None
    res = warp_frame(fh, img, x_min, y_max, gsd_m, out_w, out_h, cfg, maps)
    if res is None:
        return False
    u0, v0, warped, weight = res

    bands = warped.shape[2]
    canvas = np.zeros((out_h, out_w, bands), dtype=warped.dtype)
    canvas[v0:v0 + warped.shape[0], u0:u0 + warped.shape[1]] = warped
    m = np.zeros((out_h, out_w), dtype=bool)
    m[v0:v0 + weight.shape[0], u0:u0 + weight.shape[1]] = weight <= 0
    canvas[m] = 0

    with rasterio.open(output_path, "w", driver="GTiff",
                       height=out_h, width=out_w, count=bands,
                       dtype=canvas.dtype, crs=f"EPSG:{epsg}",
                       transform=from_origin(x_min, y_max, gsd_m, gsd_m),
                       compress="lzw") as dst:
        for i in range(bands):
            dst.write(canvas[:, :, i], i + 1)
    return True


# ---------------------------------------------------------------------------
# 모자이크
# ---------------------------------------------------------------------------
def _measure_canvas_shift(canvas_sub: np.ndarray,
                          wf_gray: np.ndarray,
                          overlap: np.ndarray,
                          max_shift_px: float,
                          max_side: int = 640):
    """겹침 영역에서 프레임 → 캔버스 정렬 이동량 측정 (탐색 반경 제한).

    ★ 위상상관(전역 FFT 피크)을 쓰면 안 된다. 태양광 패널 행은 주기
    구조라(실측 2.72 m), 전역 피크가 ±1~2 주기 어긋난 자리에 걸린다
    (실측: 주입 오차 0.3 m 에 대해 76 px·156 px 같은 주기 배수 오답).
    대신 **탐색 반경을 ``max_shift_px`` 로 제한한 마스크드 템플릿 매칭**
    을 쓴다. 반경이 주기의 절반보다 작으면 모호성이 원천적으로 없다.

    Returns
    -------
    ``(dx_px, dy_px, zncc)`` — 캔버스 대비 프레임 내용의 **어긋남**
    (출력픽셀). 정렬하려면 내용을 ``(−dx, −dy)`` 만큼 옮겨야 한다.
    부호는 단위검증으로 확정 (+12 px 어긋난 프레임 → 측정 +12.01).
    측정 불가 시 ``(0, 0, 0)``.
    """
    n_ov = int(np.count_nonzero(overlap))
    if n_ov < 4000:
        return 0.0, 0.0, 0.0
    x0, y0, bw, bh = cv2.boundingRect(overlap.astype(np.uint8))
    if bh < 96 or bw < 96:
        return 0.0, 0.0, 0.0

    a = canvas_sub[y0:y0 + bh, x0:x0 + bw].astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    b = wf_gray[y0:y0 + bh, x0:x0 + bw]
    m = overlap[y0:y0 + bh, x0:x0 + bw]

    scale = max(bh / max_side, bw / max_side, 1.0)
    if scale > 1.0:
        nh, nw = max(int(bh / scale), 32), max(int(bw / scale), 32)
        a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m.astype(np.uint8), (nw, nh),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    if m.mean() < 0.25:
        return 0.0, 0.0, 0.0

    r = int(np.ceil(max_shift_px / scale)) + 1
    H_, W_ = a.shape
    if W_ - 2 * r < 48 or H_ - 2 * r < 48:
        return 0.0, 0.0, 0.0

    mu_a, sd_a = float(a[m].mean()), float(a[m].std()) + 1e-6
    mu_b, sd_b = float(b[m].mean()), float(b[m].std()) + 1e-6
    a = (a - mu_a) / sd_a
    b = (b - mu_b) / sd_b

    tpl = b[r:H_ - r, r:W_ - r]
    tmask = m[r:H_ - r, r:W_ - r].astype(np.float32)
    if tmask.mean() < 0.25:
        return 0.0, 0.0, 0.0
    res = cv2.matchTemplate(a, tpl * tmask, cv2.TM_SQDIFF,
                            mask=tmask)
    _, _, minloc, _ = cv2.minMaxLoc(res)
    mx, my = minloc

    # 서브픽셀: 최소점 주변 포물선 적합.
    def _sub(v0, v1, v2):
        d = v0 - 2 * v1 + v2
        return 0.0 if abs(d) < 1e-9 else float(np.clip((v0 - v2) / (2 * d), -0.5, 0.5))
    fx = _sub(res[my, mx - 1], res[my, mx], res[my, mx + 1]) if 0 < mx < res.shape[1] - 1 else 0.0
    fy = _sub(res[my - 1, mx], res[my, mx], res[my + 1, mx]) if 0 < my < res.shape[0] - 1 else 0.0

    # 프레임 내용이 (+d) 어긋나 있으면 템플릿은 캔버스에서 (r − d) 에서 발견
    # → 보정(내용에 적용할 이동) = (r − minloc). 부호는 단위검증으로 확정.
    dx = (r - (mx + fx)) * scale
    dy = (r - (my + fy)) * scale

    # 품질: 최적 위치에서의 마스크드 ZNCC 를 직접 계산.
    sh_x, sh_y = int(round(mx - r)), int(round(my - r))
    A = a[r + sh_y:H_ - r + sh_y, r + sh_x:W_ - r + sh_x]
    if A.shape != tpl.shape:
        return 0.0, 0.0, 0.0
    mm = tmask > 0
    if mm.sum() < 1000:
        return 0.0, 0.0, 0.0
    va, vb = A[mm], tpl[mm]
    va = va - va.mean(); vb = vb - vb.mean()
    zncc = float(np.dot(va, vb) /
                 (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
    return dx, dy, zncc


def _frame_order(frames: list[FrameHomography]) -> list[int]:
    """중앙에 가까운 프레임부터 배치.

    시임 최적화는 '이미 놓인 것' 과의 차이를 보므로 순서에 의존한다. 블록
    중앙부터 바깥으로 자라게 하면 기준이 안정적이다 (임의 순서면 가장자리
    프레임이 기준이 되어 전체가 끌려간다).
    """
    C = np.array([f.camera_xyz[:2] for f in frames])
    ctr = C.mean(axis=0)
    return list(np.argsort(np.hypot(C[:, 0] - ctr[0], C[:, 1] - ctr[1])))


def mosaic_frames(frames: list[FrameHomography],
                  image_paths: list[str | Path],
                  output_path: str | Path,
                  gsd_m: float | None = None,
                  epsg: int = 5186,
                  cfg: MosaicConfig | None = None,
                  undistort: bool = True) -> dict:
    """여러 프레임을 하나의 정사 모자이크로 합성 (시임 최적화 + 노출 보정)."""
    import rasterio
    from rasterio.transform import from_origin

    cfg = cfg or MosaicConfig()
    if len(frames) != len(image_paths):
        raise ValueError(f"프레임 {len(frames)}개 vs 경로 {len(image_paths)}개")
    if not frames:
        raise ValueError("프레임이 없음")

    bounds = ground_bounds_from_homography(frames)
    if bounds is None:
        raise ValueError("모든 프레임의 footprint 계산 실패")
    x_min, y_min, x_max, y_max = bounds
    if gsd_m is None:
        gsd_m = recommend_gsd(frames)
    out_w = int(np.ceil((x_max - x_min) / gsd_m))
    out_h = int(np.ceil((y_max - y_min) / gsd_m))
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"출력 크기 비정상: {out_w}×{out_h}")
    if out_w * out_h > cfg.max_out_pixels:
        raise ValueError(
            f"출력 {out_w}×{out_h} = {out_w*out_h/1e6:.0f} Mpx 가 상한 초과. "
            f"GSD 를 키우거나 타일로 나눌 것 (현재 {gsd_m:.4f} m)")

    bands = cfg.band_count
    select_mode = cfg.blend_mode == "select"
    logger.info("모자이크 캔버스: %d×%d px, GSD=%.4f m, 범위 %.1f×%.1f m, "
                "mode=%s (시임최적화=%s, 노출보정=%s)",
                out_w, out_h, gsd_m, x_max - x_min, y_max - y_min,
                cfg.blend_mode, cfg.seam_optimize, cfg.exposure_compensate)

    if select_mode:
        canvas = np.zeros((out_h, out_w, bands), dtype=cfg.dtype)
        best_score = np.zeros((out_h, out_w), dtype=np.float32)
    else:
        acc = np.zeros((out_h, out_w, bands), dtype=np.float32)
        acc_w = np.zeros((out_h, out_w), dtype=np.float32)

    blur_px = max(int(cfg.seam_cost_blur_m / gsd_m), 1)
    maps_cache: dict[tuple, object] = {}
    n_ok = n_fail = 0
    gains: list[float] = []
    align_log: list[tuple] = []

    order = _frame_order(frames) if select_mode else list(range(len(frames)))

    for idx in order:
        fh, path = frames[idx], image_paths[idx]
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("읽기 실패, 건너뜀: %s", path)
            n_fail += 1
            continue
        if img.ndim == 3 and img.shape[2] >= 3:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        elif img.ndim == 2 and bands == 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        maps = None
        if undistort and fh.intr.has_distortion:
            key = (fh.intr.width, fh.intr.height, round(fh.intr.f_px, 3))
            if key not in maps_cache:
                maps_cache[key] = fh.intr.undistort_maps()
            maps = maps_cache[key]

        res = warp_frame(fh, img, x_min, y_max, gsd_m, out_w, out_h, cfg, maps)
        if res is None:
            n_fail += 1
            continue
        u0, v0, warped, weight = res
        h, w = weight.shape

        if not select_mode:
            sub = acc[v0:v0 + h, u0:u0 + w]
            sub += warped[:, :, :bands].astype(np.float32) * weight[:, :, None]
            acc_w[v0:v0 + h, u0:u0 + w] += weight
            n_ok += 1
            continue

        canvas_sub = canvas[v0:v0 + h, u0:u0 + w]
        score_sub = best_score[v0:v0 + h, u0:u0 + w]
        occupied = score_sub > 0
        valid = weight > 0
        overlap = occupied & valid

        wf = warped[:, :, :bands].astype(np.float32)

        # ---- 프레임 정렬: 캔버스 대비 등록 오차 보정 ----------------------
        # 자세/BA 잔차로 프레임이 0.1~0.5 m 씩 어긋나면 아무리 시임을 잘
        # 골라도 패널 에지에 단차가 남는다. 여기서 위상상관으로 이동량을
        # 재고, FrameHomography 의 국소 원점을 옮겨(= 기하 전체를 이동)
        # 다시 warp 한다. 재보간이 아니라 원본에서 한 번만 보간한다.
        if (cfg.align_frames and overlap.sum() > 4000
                and cfg.align_max_m > 0):
            dx_px, dy_px, resp = _measure_canvas_shift(
                canvas_sub, wf.mean(axis=2), overlap,
                max_shift_px=cfg.align_max_m / gsd_m)
            shift_m = np.hypot(dx_px, dy_px) * gsd_m
            if (resp >= cfg.align_min_response
                    and 0.5 * gsd_m < shift_m <= cfg.align_max_m):
                from dataclasses import replace as _dc_replace
                # 측정값은 '어긋남'이므로 보정은 그 반대 방향.
                # 출력 +x = 동, 출력 +y = 남 → 보정 ENU = (−dx·gsd, +dy·gsd).
                # 국소 원점을 ENU 로 +d 옮기면 내용이 +d 이동한다.
                corr_e, corr_n = -dx_px * gsd_m, +dy_px * gsd_m
                fh2 = _dc_replace(
                    fh, origin_xy=fh.origin_xy + np.array([corr_e, corr_n]))
                del warped, weight, wf          # 재warp 전 메모리 해제
                res2 = warp_frame(fh2, img, x_min, y_max, gsd_m,
                                  out_w, out_h, cfg, maps)
                if res2 is None:
                    res2 = warp_frame(fh, img, x_min, y_max, gsd_m,
                                      out_w, out_h, cfg, maps)
                    corr_e = corr_n = 0.0
                if res2 is not None:
                    u0, v0, warped, weight = res2
                    h, w = weight.shape
                    canvas_sub = canvas[v0:v0 + h, u0:u0 + w]
                    score_sub = best_score[v0:v0 + h, u0:u0 + w]
                    occupied = score_sub > 0
                    valid = weight > 0
                    overlap = occupied & valid
                    wf = warped[:, :, :bands].astype(np.float32)
                    if corr_e or corr_n:
                        # 검산: 보정 후 잔여 어긋남이 보정 전보다 크면 오탐
                        # → 원래 기하로 되돌린다 (greedy 체인 보호).
                        rdx, rdy, _ = _measure_canvas_shift(
                            canvas_sub, wf.mean(axis=2), overlap,
                            max_shift_px=cfg.align_max_m / gsd_m)
                        if np.hypot(rdx, rdy) > np.hypot(dx_px, dy_px):
                            res3 = warp_frame(fh, img, x_min, y_max, gsd_m,
                                              out_w, out_h, cfg, maps)
                            if res3 is not None:
                                del warped, weight, wf
                                u0, v0, warped, weight = res3
                                h, w = weight.shape
                                canvas_sub = canvas[v0:v0 + h, u0:u0 + w]
                                score_sub = best_score[v0:v0 + h, u0:u0 + w]
                                occupied = score_sub > 0
                                valid = weight > 0
                                overlap = occupied & valid
                                wf = warped[:, :, :bands].astype(np.float32)
                        else:
                            align_log.append((str(path), corr_e, corr_n, resp))

        # ---- 노출 보정 ---------------------------------------------------
        if cfg.exposure_compensate and overlap.sum() > 500:
            a = wf[overlap].mean(axis=1)
            b = canvas_sub.astype(np.float32)[overlap].mean(axis=1)
            keep = (a > 5) & (a < 250) & (b > 5) & (b < 250)
            if keep.sum() > 200:
                gain = float(np.median(b[keep]) / max(np.median(a[keep]), 1e-6))
                gain = float(np.clip(gain, 0.7, 1.4))
                if abs(gain - 1.0) > 0.01:
                    wf *= gain
                    gains.append(gain)

        # ---- 시임 최적화: 차이 벌점 ---------------------------------------
        eff = weight.copy()
        if cfg.seam_optimize and cfg.seam_cost_weight > 0 and overlap.sum() > 500:
            diff = np.zeros((h, w), dtype=np.float32)
            d = np.abs(wf.mean(axis=2) - canvas_sub.astype(np.float32).mean(axis=2))
            diff[overlap] = d[overlap]
            # 크게 블러해서 시임이 국소 노이즈를 따라 너덜거리지 않게.
            diff = cv2.blur(diff, (blur_px, blur_px))
            scale = float(np.percentile(diff[overlap], 75)) + 1e-3
            eff = weight * np.exp(-cfg.seam_cost_weight * diff / scale)
            eff[~valid] = 0.0

        better = eff > score_sub
        if cfg.seam_blend_px > 0:
            span = max(cfg.seam_blend_px * float(np.max(eff) + 1e-6) / max(w, h), 1e-6)
            alpha = np.clip(0.5 + 0.5 * (eff - score_sub) / span, 0.0, 1.0)
            alpha[~occupied] = np.where(valid[~occupied], 1.0, 0.0)
            alpha[~valid] = 0.0
            mix = alpha[:, :, None]
            blended = wf * mix + canvas_sub.astype(np.float32) * (1.0 - mix)
            touched = alpha > 0
            canvas_sub[touched] = np.clip(blended, 0, 255)[touched].astype(cfg.dtype)
        elif better.any():
            canvas_sub[better] = np.clip(wf[better], 0, 255).astype(cfg.dtype)
        score_sub[better] = eff[better]
        n_ok += 1

    if select_mode:
        filled = best_score > 0
        out = canvas
    else:
        filled = acc_w > 1e-6
        out = np.zeros((out_h, out_w, bands), dtype=cfg.dtype)
        info = np.iinfo(cfg.dtype) if np.issubdtype(cfg.dtype, np.integer) else None
        for i in range(bands):
            band = np.zeros((out_h, out_w), dtype=np.float32)
            band[filled] = acc[:, :, i][filled] / acc_w[filled]
            if info is not None:
                band = np.clip(band, info.min, info.max)
            out[:, :, i] = band.astype(cfg.dtype)
    fill_ratio = float(filled.mean())

    with rasterio.open(output_path, "w", driver="GTiff",
                       height=out_h, width=out_w, count=bands,
                       dtype=out.dtype, crs=f"EPSG:{epsg}",
                       transform=from_origin(x_min, y_max, gsd_m, gsd_m),
                       compress="lzw", tiled=True,
                       blockxsize=512, blockysize=512) as dst:
        for i in range(bands):
            dst.write(out[:, :, i], i + 1)

    if gains:
        logger.info("노출 보정 이득: 중앙값 %.3f, 범위 %.3f~%.3f (%d장)",
                    float(np.median(gains)), min(gains), max(gains), len(gains))
    if align_log:
        mags = [np.hypot(a[1], a[2]) for a in align_log]
        logger.info("프레임 정렬 보정: %d장, 중앙값 %.2f m, 최대 %.2f m",
                    len(align_log), float(np.median(mags)), float(max(mags)))
        if max(mags) > 1.0:
            worst = max(align_log, key=lambda a: np.hypot(a[1], a[2]))
            logger.warning("정렬 보정 1 m 초과 프레임 존재 (%s: %.2f m) — "
                           "해당 프레임의 자세/RTK 를 확인하세요.",
                           Path(worst[0]).name, np.hypot(worst[1], worst[2]))
    logger.info("모자이크 저장: %s (%d장 합성, %d장 실패, 충전율 %.1f%%)",
                output_path, n_ok, n_fail, fill_ratio * 100)

    return {
        "output_path": str(output_path),
        "width": out_w, "height": out_h, "gsd_m": gsd_m, "epsg": epsg,
        "blend_mode": cfg.blend_mode,
        "seam_optimize": cfg.seam_optimize,
        "exposure_compensate": cfg.exposure_compensate,
        "bounds": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
        "frames_ok": n_ok, "frames_failed": n_fail,
        "fill_ratio": fill_ratio,
        "frame_alignments": [
            {"path": p_, "de_m": round(de, 3), "dn_m": round(dn, 3),
             "response": round(r_, 3)}
            for p_, de, dn, r_ in align_log],
        "plane": {"a": frames[0].plane.a, "b": frames[0].plane.b,
                  "c": frames[0].plane.c,
                  "slope_deg": frames[0].plane.slope_deg,
                  "rmse_m": frames[0].plane.inlier_rmse_m},
    }

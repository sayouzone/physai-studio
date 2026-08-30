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

# 안전장치일 뿐 — 타일 스트리밍이라 메모리 제약이 아니다.
MAX_OUT_PIXELS = 8_000_000_000
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
    max_offnadir_ratio : 프레임에서 **실제로 쓸 영역의 off-nadir 상한**
        ``r/h = r_px/f_px``. 0 이면 제한 없음.

        모자이크 내부는 중복이 커서 ``select`` 가 알아서 연직 근처(작은 k)만
        고르지만, **가장자리는 그 지점을 찍은 프레임이 하나뿐이라 선택의 여지
        없이 프레임 최외곽(H20T 기준 k=0.479, 25.6°)이 쓰인다.** 그 자리에서
        기복변위 ``Δh·k`` 가 최대가 된다 — 기준면과 지면이 1.87 m 차이나면
        가장자리에서 0.90 m 다. 실측에서도 국소 어긋남이 안쪽 0.061 m →
        바깥 0.321 m 로 5배 커졌다.

        이 값으로 프레임의 바깥 고리를 아예 쓰지 않으면 그 열화를 막을 수
        있다. 내부는 중복 덕에 손실이 없고, 모자이크 가장자리만 조금 좁아진다.
    offnadir_fallback : ``True`` 면 제한 때문에 아무 프레임도 덮지 못하는
        픽셀에 한해 제한을 풀어 채운다 (구멍 방지). ``False`` 면 그 부분을
        비운다 — 품질을 우선할 때.
    tile_memory_mb : 타일 하나의 메모리 예산 (MB). 출력이 아무리 커도
        메모리는 이 값으로 묶인다.
    image_cache : 원본 이미지 LRU 캐시 장수. 한 프레임이 여러 타일에 걸치면
        재사용된다. 늘리면 디코딩이 줄고 메모리는 는다.
    band_count, dtype, max_out_pixels : 출력 형식.
    """

    def __init__(self,
                 blend_mode: str = "select",
                 feather_px: float = 120.0,
                 nadir_falloff: float = 0.6,
                 align_frames: bool = True,
                 align_max_m: float = 1.2,
                 align_min_response: float = 0.30,
                 seam_optimize: bool = True,
                 seam_cost_weight: float = 1.0,
                 seam_cost_blur_m: float = 1.5,
                 exposure_compensate: bool = True,
                 glint_penalty: float = 0.7,
                 glint_threshold: int = 250,
                 seam_blend_px: float = 0.0,
                 max_offnadir_ratio: float = 0.35,
                 offnadir_fallback: bool = True,
                 tile_memory_mb: float = 256.0,
                 image_cache: int = 8,
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
        self.max_offnadir_ratio = float(max(max_offnadir_ratio, 0.0))
        self.offnadir_fallback = bool(offnadir_fallback)
        self.tile_memory_mb = float(max(tile_memory_mb, 32.0))
        self.image_cache = int(max(image_cache, 1))
        self.band_count = int(band_count)
        self.dtype = dtype
        self.max_out_pixels = int(max_out_pixels)


# ---------------------------------------------------------------------------
# 기하 점수 맵
# ---------------------------------------------------------------------------
_WEIGHT_CACHE: dict[tuple, np.ndarray] = {}


def build_source_weight(width: int, height: int, cx: float, cy: float,
                        cfg: MosaicConfig, f_px: float = 0.0,
                        restrict: bool = True) -> np.ndarray:
    """(H, W) float32 기하 점수. feather × 연직근접 (× 연직 제한).

    ``restrict=True`` 이고 ``cfg.max_offnadir_ratio > 0`` 이면 off-nadir 이
    상한을 넘는 바깥 고리를 0 으로 만들어 그 영역이 선택되지 않게 한다.
    """
    key = (width, height, round(cx, 3), round(cy, 3),
           cfg.feather_px, cfg.nadir_falloff,
           round(f_px, 2), bool(restrict), cfg.max_offnadir_ratio)
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

    if restrict and cfg.max_offnadir_ratio > 0 and f_px > 0:
        ys, xs = np.mgrid[0:height, 0:width]
        r_lim = cfg.max_offnadir_ratio * f_px
        outside = ((xs - cx) ** 2 + (ys - cy) ** 2) > r_lim ** 2
        w = w.copy()
        w[outside] = 0.0

    w = np.maximum(w, 1e-4).astype(np.float32) if not (
        restrict and cfg.max_offnadir_ratio > 0 and f_px > 0) else np.where(
        w > 0, np.maximum(w, 1e-4), 0.0).astype(np.float32)
    if len(_WEIGHT_CACHE) > 16:
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
               undistort_maps=None, restrict: bool = True):
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
                                fh.intr.cx, fh.intr.cy, cfg,
                                f_px=fh.intr.f_px, restrict=restrict)
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


def _robust_bounds(frames: list[FrameHomography]) -> tuple:
    """전체 footprint 범위 + 이상 프레임 진단.

    BA 가 한두 프레임을 엉뚱한 곳으로 보내면 그 프레임 하나가 캔버스를
    몇 배로 부풀린다. 카메라 위치 중앙값에서 크게 벗어난 프레임을 찾아
    경고한다 (제외하지는 않는다 — 판단은 사용자 몫).
    """
    b = [f.footprint_bounds() for f in frames]
    ok = [x for x in b if x is not None]
    if not ok:
        return None, []
    arr = np.array(ok)
    x_min, y_min = arr[:, 0].min(), arr[:, 1].min()
    x_max, y_max = arr[:, 2].max(), arr[:, 3].max()

    C = np.array([f.camera_xyz[:2] for f in frames])
    med = np.median(C, axis=0)
    d = np.hypot(C[:, 0] - med[0], C[:, 1] - med[1])
    mad = np.median(np.abs(d - np.median(d))) + 1e-6
    outliers = np.nonzero(d > np.median(d) + 8.0 * mad)[0].tolist()
    return (float(x_min), float(y_min), float(x_max), float(y_max)), outliers


def _coarse_reference(frames, image_paths, order, bounds, cfg,
                      coarse_px: int = 4000, restrict: bool = True):
    """저해상도 사전 패스 — 프레임별 노출 이득 + **전역 기준 영상**.

    타일 단위 합성에서 두 가지가 타일마다 달라지면 경계에 단차가 생긴다:

    1. **노출 이득** — 타일마다 다시 구하면 밝기 계단이 생긴다.
    2. **시임 판정** — 시임 최적화는 '이미 놓인 캔버스' 와의 차이를 보는데,
       타일이 바뀌면 캔버스가 비어 있어 선택이 달라진다. 실측에서 타일
       경계의 행간 밝기 변화가 주변의 8~40 배로 튀었다.

    그래서 여기서 저해상도 기준 영상을 한 번 만들어 두고, 본 패스에서는
    **타일과 무관하게** 그 기준에 대해 시임 비용을 계산한다. 이득도 여기서
    확정한다. 결과가 타일 분할에 의존하지 않게 된다.

    시임 결정은 어차피 ``seam_cost_blur_m``(기본 1.5 m) 로 블러한 저주파
    정보다. 따라서 **저해상도에서 승자 라벨맵을 확정**해도 잃는 것이 없고,
    본 패스는 그 라벨을 적용하기만 하면 되므로 타일 분할과 완전히 무관해진다.

    Returns ``(gains, label_map, ref_gsd)``.
    """
    x_min, y_min, x_max, y_max = bounds
    g = max((x_max - x_min) / coarse_px, (y_max - y_min) / coarse_px)
    ow, oh = int((x_max - x_min) / g), int((y_max - y_min) / g)
    if ow < 32 or oh < 32:
        return {}, None, None

    canvas = np.zeros((oh, ow), dtype=np.float32)
    filled = np.zeros((oh, ow), dtype=bool)
    label = np.full((oh, ow), -1, dtype=np.int32)
    score = np.zeros((oh, ow), dtype=np.float32)
    blur_c = max(int(cfg.seam_cost_blur_m / g), 1)
    gains: dict[int, float] = {}

    for idx in order:
        img = cv2.imread(str(image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        small = cv2.resize(img, None, fx=0.25, fy=0.25,
                           interpolation=cv2.INTER_AREA)
        fh = frames[idx]
        H = fh.ortho_pixel_matrix(x_min, y_max, g)
        S = np.diag([0.25, 0.25, 1.0]).astype(np.float64)
        w = cv2.warpPerspective(
            small, (S @ H).astype(np.float64), (ow, oh),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        valid = w > 1
        ov = valid & filled
        gain = 1.0
        if ov.sum() > 200:
            a, b = w[ov], canvas[ov]
            keep = (a > 5) & (a < 250) & (b > 5) & (b < 250)
            if keep.sum() > 100:
                gain = float(np.clip(
                    np.median(b[keep]) / max(np.median(a[keep]), 1e-6),
                    0.7, 1.4))
        gains[idx] = gain if cfg.exposure_compensate else 1.0
        wg = w * gains[idx]

        # --- 승자 라벨 결정 (기하 점수 × 시임 비용) ---
        sw = build_source_weight(fh.intr.width, fh.intr.height,
                                 fh.intr.cx, fh.intr.cy, cfg,
                                 f_px=fh.intr.f_px, restrict=restrict)
        sw_s = cv2.resize(sw, None, fx=0.25, fy=0.25,
                          interpolation=cv2.INTER_AREA)
        if restrict and cfg.max_offnadir_ratio > 0:
            sw_s[sw_s <= 0] = 0.0
        wt = cv2.warpPerspective(
            sw_s, (S @ H).astype(np.float64), (ow, oh),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        wt[~valid] = 0.0
        if restrict and cfg.max_offnadir_ratio > 0:
            valid = valid & (wt > 0)
        if cfg.glint_penalty > 0:
            wt[wg >= cfg.glint_threshold] *= (1.0 - cfg.glint_penalty)

        eff = wt
        ov = valid & filled
        if cfg.seam_optimize and cfg.seam_cost_weight > 0 and ov.sum() > 200:
            diff = np.zeros((oh, ow), dtype=np.float32)
            diff[ov] = np.abs(wg[ov] - canvas[ov])
            diff = cv2.blur(diff, (blur_c, blur_c))
            sc = float(np.percentile(diff[ov], 75)) + 1e-3
            eff = wt * np.exp(-cfg.seam_cost_weight * diff / sc)
            eff[~valid] = 0.0

        better = eff > score
        label[better] = idx
        score[better] = eff[better]

        put = valid & ~filled
        canvas[put] = wg[put]
        filled |= valid

    if gains and cfg.exposure_compensate:
        v = np.array(list(gains.values()))
        logger.info("노출 이득 사전 산출: 중앙값 %.3f, 범위 %.3f~%.3f",
                    float(np.median(v)), float(v.min()), float(v.max()))
    logger.info("전역 라벨맵: %d×%d px (GSD %.3f m), 배정된 프레임 %d개 — "
                "타일 독립 시임 판정", ow, oh, g,
                int(len(np.unique(label[label >= 0]))))
    return gains, label, g


def mosaic_frames(frames: list[FrameHomography],
                  image_paths: list[str | Path],
                  output_path: str | Path,
                  gsd_m: float | None = None,
                  epsg: int = 5186,
                  cfg: MosaicConfig | None = None,
                  undistort: bool = True) -> dict:
    """여러 프레임을 하나의 정사 모자이크로 합성 — **타일 스트리밍**.

    ★ 이전 구현은 캔버스 전체를 메모리에 올렸다. 612 Mpx 출력이면 캔버스
      1.83 GB + 점수맵 2.45 GB = 4.3 GB 라서 ``max_out_pixels`` 상한에
      걸려 ``ValueError`` 로 죽었다. 상한은 알고리즘의 한계가 아니라
      **구현 제약**이었다.

      이 버전은 출력을 가로 띠(tile) 로 나눠 한 번에 하나씩만 메모리에
      두고 GeoTIFF 에 곧바로 써 넣는다. 메모리는 출력 크기와 무관하게
      타일 크기로 정해지므로 상한이 필요 없다 (안전장치로만 아주 크게 남김).

      노출 이득은 타일마다 새로 구하면 타일 경계에 밝기 단차가 생기므로,
      저해상도 사전 패스에서 **한 번만** 산출해 고정한다.
    """
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    cfg = cfg or MosaicConfig()
    if len(frames) != len(image_paths):
        raise ValueError(f"프레임 {len(frames)}개 vs 경로 {len(image_paths)}개")
    if not frames:
        raise ValueError("프레임이 없음")

    bounds, outliers = _robust_bounds(frames)
    if bounds is None:
        raise ValueError("모든 프레임의 footprint 계산 실패")
    x_min, y_min, x_max, y_max = bounds
    if outliers:
        logger.warning(
            "카메라 위치가 무리에서 크게 벗어난 프레임 %d장 감지 (index %s) — "
            "BA 가 이 프레임들을 엉뚱한 곳으로 보냈다면 캔버스가 불필요하게 "
            "커집니다. RTK prior 대비 이동량 로그를 확인하세요.",
            len(outliers), outliers[:8])

    if gsd_m is None:
        gsd_m = recommend_gsd(frames)
    out_w = int(np.ceil((x_max - x_min) / gsd_m))
    out_h = int(np.ceil((y_max - y_min) / gsd_m))
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"출력 크기 비정상: {out_w}×{out_h}")
    if out_w * out_h > cfg.max_out_pixels:
        raise ValueError(
            f"출력 {out_w}×{out_h} = {out_w*out_h/1e6:.0f} Mpx 가 안전 상한 "
            f"{cfg.max_out_pixels/1e6:.0f} Mpx 를 초과합니다. 자세 추정이 "
            f"발산했거나 GSD 가 너무 작습니다 (현재 {gsd_m:.4f} m). "
            f"cfg.max_out_pixels 를 올려 강제할 수도 있습니다.")

    bands = cfg.band_count
    select_mode = cfg.blend_mode == "select"
    order = _frame_order(frames) if select_mode else list(range(len(frames)))

    # 타일 높이: 메모리 예산에서 역산 (캔버스 uint8 + 점수 float32).
    bytes_per_row = out_w * (bands + 4)
    tile_h = int(np.clip(cfg.tile_memory_mb * 1e6 / max(bytes_per_row, 1),
                         256, 8192))
    n_tiles = int(np.ceil(out_h / tile_h))
    logger.info("모자이크: %d×%d px (%.0f Mpx), GSD=%.4f m, 범위 %.1f×%.1f m, "
                "타일 %d개 (높이 %d px, ~%.0f MB/타일)",
                out_w, out_h, out_w * out_h / 1e6, gsd_m,
                x_max - x_min, y_max - y_min, n_tiles, tile_h,
                tile_h * bytes_per_row / 1e6)

    gains, label_map, ref_gsd = _coarse_reference(
        frames, image_paths, order, bounds, cfg, restrict=True)

    # ★ 연직 제한으로 아무도 못 덮은 픽셀은 제한을 풀어 한 번 더 채운다.
    #   내부는 중복이 커서 손실이 없고, 모자이크 가장자리에만 영향을 준다.
    n_fallback = 0.0
    if (cfg.max_offnadir_ratio > 0 and cfg.offnadir_fallback
            and label_map is not None):
        hole = label_map < 0
        if hole.any():
            _, lab_full, _ = _coarse_reference(
                frames, image_paths, order, bounds, cfg, restrict=False)
            if lab_full is not None:
                fill = hole & (lab_full >= 0)
                n_fallback = float(fill.mean())
                label_map[fill] = lab_full[fill]
                logger.info("연직 제한 예외 적용: 전체의 %.1f%% 픽셀은 "
                            "k>%.2f 영역으로 채움 (그 부분은 기복변위가 크다)",
                            100.0 * fill.mean(), cfg.max_offnadir_ratio)

    # 프레임별 footprint 를 출력 픽셀 창으로 미리 계산 (타일 교차 판정용).
    wins = {}
    for idx in range(len(frames)):
        w_ = _frame_window(frames[idx], x_min, y_max, gsd_m, out_w, out_h)
        if w_ is not None:
            wins[idx] = w_

    maps_cache: dict[tuple, object] = {}
    img_cache: dict[int, np.ndarray] = {}

    def load(idx):
        if idx not in img_cache:
            im = cv2.imread(str(image_paths[idx]), cv2.IMREAD_UNCHANGED)
            if im is not None:
                if im.ndim == 3 and im.shape[2] >= 3:
                    im = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2RGB)
                elif im.ndim == 2 and bands == 3:
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            img_cache[idx] = im
            if len(img_cache) > cfg.image_cache:
                img_cache.pop(next(iter(img_cache)))
        return img_cache[idx]

    n_ok = set()
    filled_total = 0

    with rasterio.open(
            output_path, "w", driver="GTiff",
            height=out_h, width=out_w, count=bands,
            dtype=np.dtype(cfg.dtype).name, crs=f"EPSG:{epsg}",
            transform=from_origin(x_min, y_max, gsd_m, gsd_m),
            compress="lzw", tiled=True,
            blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER") as dst:

        for t in range(n_tiles):
            v0t = t * tile_h
            th = min(tile_h, out_h - v0t)
            canvas = np.zeros((th, out_w, bands), dtype=cfg.dtype)
            score = np.zeros((th, out_w), dtype=np.float32)

            # 이 타일에 해당하는 전역 기준 영상 조각 (업샘플).
            # ★ 전역 좌표를 그대로 쓰는 아핀 역매핑으로 라벨을 뽑는다.
            #   정수 잘라내기 후 resize 하면 타일 오프셋에 따라 매핑이 어긋나
            #   타일 분할만 바꿔도 결과가 달라진다 (실측 11.7% 불일치).
            lbl_tile = None
            if label_map is not None:
                k = gsd_m / ref_gsd
                M_ref = np.array([[k, 0.0, 0.0],
                                  [0.0, k, k * v0t]], dtype=np.float64)
                lbl_tile = cv2.warpAffine(
                    label_map, M_ref, (out_w, th),
                    flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE)

            for idx in order:
                w_ = wins.get(idx)
                if w_ is None:
                    continue
                fu0, fv0, fw, fh_ = w_
                if fv0 + fh_ <= v0t or fv0 >= v0t + th:
                    continue                      # 이 타일과 안 겹침
                img = load(idx)
                if img is None:
                    continue
                fh = frames[idx]

                maps = None
                if undistort and fh.intr.has_distortion:
                    key = (fh.intr.width, fh.intr.height, round(fh.intr.f_px, 3))
                    if key not in maps_cache:
                        maps_cache[key] = fh.intr.undistort_maps()
                    maps = maps_cache[key]

                res = warp_frame(fh, img, x_min, y_max - v0t * gsd_m, gsd_m,
                                 out_w, th, cfg, maps,
                                 restrict=(cfg.max_offnadir_ratio <= 0
                                           or not cfg.offnadir_fallback))
                if res is None:
                    continue
                u0, v0, warped, weight = res
                h, w = weight.shape
                wf = warped[:, :, :bands].astype(np.float32)
                g = gains.get(idx, 1.0)
                if g != 1.0:
                    wf *= g

                cs = canvas[v0:v0 + h, u0:u0 + w]
                ss = score[v0:v0 + h, u0:u0 + w]
                valid = weight > 0
                if not select_mode:
                    cs[valid] = np.clip(wf[valid], 0, 255).astype(cfg.dtype)
                    ss[valid] = np.maximum(ss[valid], weight[valid])
                    n_ok.add(idx)
                    continue

                # 승자는 전역 라벨맵이 이미 정해 뒀다 → 타일과 완전히 무관.
                if lbl_tile is not None:
                    better = valid & (lbl_tile[v0:v0 + h, u0:u0 + w] == idx)
                else:
                    better = valid & (weight > ss)
                if better.any():
                    cs[better] = np.clip(wf[better], 0, 255).astype(cfg.dtype)
                    ss[better] = weight[better]
                n_ok.add(idx)

            filled_total += int((score > 0).sum())
            for bi in range(bands):
                dst.write(canvas[:, :, bi], bi + 1,
                          window=Window(0, v0t, out_w, th))
            del canvas, score
            if (t + 1) % 5 == 0 or t == n_tiles - 1:
                logger.info("  타일 %d/%d 완료", t + 1, n_tiles)

    fill_ratio = filled_total / float(out_w * out_h)
    logger.info("모자이크 저장: %s (%d장 합성, 충전율 %.1f%%)",
                output_path, len(n_ok), fill_ratio * 100)
    if fill_ratio < 0.5:
        logger.warning("충전율 %.1f%% — 촬영 경로에 공백이 있거나 프레임 "
                       "footprint 가 크게 떨어져 있습니다.", fill_ratio * 100)

    return {
        "output_path": str(output_path),
        "width": out_w, "height": out_h, "gsd_m": gsd_m, "epsg": epsg,
        "blend_mode": cfg.blend_mode,
        "seam_optimize": cfg.seam_optimize,
        "exposure_compensate": cfg.exposure_compensate,
        "tiles": n_tiles,
        "max_offnadir_ratio": cfg.max_offnadir_ratio,
        "offnadir_fallback_ratio": n_fallback,
        "bounds": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
        "frames_ok": len(n_ok), "frames_failed": len(frames) - len(n_ok),
        "outlier_frames": outliers,
        "fill_ratio": fill_ratio,
        "plane": {"a": frames[0].plane.a, "b": frames[0].plane.b,
                  "c": frames[0].plane.c,
                  "slope_deg": frames[0].plane.slope_deg,
                  "rmse_m": frames[0].plane.inlier_rmse_m},
    }

"""호모그래피 기반 정사영상 생성 및 모자이크 (파이프라인 7~8단계).

``simple_orthophoto`` 대비 달라진 점
------------------------------------
* meshgrid → diff → einsum → 나눗셈 → ``cv2.remap`` 대신
  ``cv2.warpPerspective(..., WARP_INVERSE_MAP)`` **한 번**. 출력 픽셀당
  부동소수 연산이 사라지고, 좌표 맵 두 장 (out_h×out_w×float32 ×2) 을 만들지
  않으므로 메모리도 크게 준다. 5000×5000 출력 기준 맵만 200 MB 였다.
* 경사평면 지원 (``GroundPlane.a/b``). 수평 가정은 특수한 경우일 뿐이다.
* 다중 프레임 **가중 모자이크** — feather 블렌딩 + 연직 근접 가중.
* 프레임마다 전체 캔버스를 다루지 않고 자기 footprint 창(window) 안에서만
  warp 한다. 100장짜리 현장에서 이 차이가 수십 배다.

블렌딩 가중치
-------------
두 성분의 곱:

1. **feather** — 이미지 경계로부터의 거리. 인접 프레임 접합선에서 밝기
   불연속(계단)이 생기지 않게 한다.
2. **연직 근접도** — 주점에서 멀수록 시선이 기울어 지형기복 오차
   (``Δh · tan θ``) 가 커진다. 같은 지점이 두 프레임에 찍혔다면 **더 연직에
   가까운 쪽**을 신뢰해야 한다. 가장자리에 낮은 가중을 줘서 이를 구현.

이 가중은 픽셀값 평균이지 기하 보정이 아니다. 기복 오차 자체를 없애려면
DSM 이 필요하고, 그건 이 파이프라인이 의도적으로 포기한 것이다.
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

# 출력 래스터 상한. 이를 넘으면 자세 추정이 발산했거나 GSD 가 잘못된 것.
MAX_OUT_PIXELS = 400_000_000


class MosaicConfig:
    """모자이크 설정.

    Attributes
    ----------
    feather_px : 경계 페더링 폭 (원본 픽셀). 0 이면 페더링 없음 (하드 접합).
    nadir_falloff : 가장자리 가중 감쇠 강도 0~1. ``0`` 이면 균일,
        ``0.8`` 이면 최외곽 가중이 중심의 20%.
    band_count : 출력 밴드 수. RGB=3, 흑백/열화상=1.
    dtype : 출력 dtype. 열화상 16bit 원본이면 ``np.uint16``.
    max_out_pixels : 출력 상한.
    """

    def __init__(self,
                 feather_px: float = 120.0,
                 nadir_falloff: float = 0.6,
                 band_count: int = 3,
                 dtype=np.uint8,
                 max_out_pixels: int = MAX_OUT_PIXELS):
        self.feather_px = float(feather_px)
        self.nadir_falloff = float(np.clip(nadir_falloff, 0.0, 1.0))
        self.band_count = int(band_count)
        self.dtype = dtype
        self.max_out_pixels = int(max_out_pixels)


# ---------------------------------------------------------------------------
# 원본 이미지 공간의 가중치 맵
# ---------------------------------------------------------------------------
_WEIGHT_CACHE: dict[tuple, np.ndarray] = {}


def build_source_weight(width: int, height: int,
                        cx: float, cy: float,
                        cfg: MosaicConfig) -> np.ndarray:
    """(H, W) float32 가중치 맵. 카메라/설정 조합마다 캐시된다.

    feather 성분은 ``cv2.distanceTransform`` 으로 경계 거리를 구해
    ``feather_px`` 에서 포화시킨다. 연직 성분은 주점 기준 정규화 반경의
    이차 감쇠.
    """
    key = (width, height, round(cx, 3), round(cy, 3),
           cfg.feather_px, cfg.nadir_falloff)
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    # --- feather: 경계로부터의 거리 ---
    if cfg.feather_px > 0:
        mask = np.ones((height, width), dtype=np.uint8)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        w = np.clip(dist / cfg.feather_px, 0.0, 1.0).astype(np.float32)
    else:
        w = np.ones((height, width), dtype=np.float32)

    # --- 연직 근접도 ---
    if cfg.nadir_falloff > 0:
        ys, xs = np.mgrid[0:height, 0:width]
        r2 = ((xs - cx) ** 2 + (ys - cy) ** 2).astype(np.float32)
        r2_max = float(max(cx, width - cx) ** 2 + max(cy, height - cy) ** 2)
        w *= (1.0 - cfg.nadir_falloff * (r2 / r2_max)).astype(np.float32)

    # 0 가중은 분모 보호 로직과 충돌하므로 아주 작은 하한을 준다.
    w = np.maximum(w, 1e-4).astype(np.float32)

    if len(_WEIGHT_CACHE) > 8:
        _WEIGHT_CACHE.clear()
    _WEIGHT_CACHE[key] = w
    return w


# ---------------------------------------------------------------------------
# 창(window) 계산
# ---------------------------------------------------------------------------
def _frame_window(fh: FrameHomography,
                  x_min: float, y_max: float, gsd_m: float,
                  out_w: int, out_h: int) -> tuple[int, int, int, int] | None:
    """프레임 footprint 에 해당하는 출력 래스터 창 ``(u0, v0, w, h)``."""
    b = fh.footprint_bounds()
    if b is None:
        return None
    bx0, by0, bx1, by1 = b
    u0 = int(np.floor((bx0 - x_min) / gsd_m)) - 1
    u1 = int(np.ceil((bx1 - x_min) / gsd_m)) + 1
    v0 = int(np.floor((y_max - by1) / gsd_m)) - 1
    v1 = int(np.ceil((y_max - by0) / gsd_m)) + 1
    u0, v0 = max(u0, 0), max(v0, 0)
    u1, v1 = min(u1, out_w), min(v1, out_h)
    if u1 <= u0 or v1 <= v0:
        return None
    return u0, v0, u1 - u0, v1 - v0


# ---------------------------------------------------------------------------
# 단일 프레임 warp
# ---------------------------------------------------------------------------
def warp_frame(fh: FrameHomography,
               image: np.ndarray,
               x_min: float, y_max: float, gsd_m: float,
               out_w: int, out_h: int,
               cfg: MosaicConfig,
               undistort_maps=None):
    """프레임 하나를 자기 창 안에서 정사 warp.

    Returns
    -------
    ``(u0, v0, warped, weight)`` 또는 ``None``.
    ``warped`` 는 (h, w, band) — 원본 dtype 유지.
    ``weight`` 는 (h, w) float32, 창 밖/무효 픽셀은 0.
    """
    win = _frame_window(fh, x_min, y_max, gsd_m, out_w, out_h)
    if win is None:
        return None
    u0, v0, ww, wh = win

    if undistort_maps is not None:
        image = cv2.remap(image, undistort_maps[0], undistort_maps[1],
                          interpolation=cv2.INTER_LINEAR)

    # 창 좌상단을 기준으로 한 지상 원점.
    x_min_w = x_min + u0 * gsd_m
    y_max_w = y_max - v0 * gsd_m
    H = fh.ortho_pixel_matrix(x_min_w, y_max_w, gsd_m)

    warped = cv2.warpPerspective(
        image, H, (ww, wh),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    src_w = build_source_weight(fh.intr.width, fh.intr.height,
                                fh.intr.cx, fh.intr.cy, cfg)
    weight = cv2.warpPerspective(
        src_w, H, (ww, wh),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )

    # warpPerspective 는 소스 범위를 벗어난 곳을 0 으로 채우지만, 지평선 너머
    # (동차 분모의 부호가 뒤집힌 영역) 는 소스 안쪽 좌표로 접혀 들어와 유령
    # 픽셀을 만든다. 출력 픽셀의 동차 w 부호로 그 영역을 지운다.
    # w 는 (u, v) 의 아핀 함수라 meshgrid 없이 브로드캐스트로 충분하다.
    us = np.arange(ww, dtype=np.float32)
    vs = np.arange(wh, dtype=np.float32)[:, None]
    denom = H[2, 0] * us + H[2, 1] * vs + H[2, 2]
    weight[denom <= 1e-9] = 0.0

    if warped.ndim == 2:
        warped = warped[:, :, None]
    return u0, v0, warped, weight


# ---------------------------------------------------------------------------
# 단일 프레임 GeoTIFF
# ---------------------------------------------------------------------------
def orthorectify_frame(fh: FrameHomography,
                       image_path: str | Path,
                       output_path: str | Path,
                       gsd_m: float | None = None,
                       epsg: int = 5186,
                       cfg: MosaicConfig | None = None,
                       undistort: bool = True) -> bool:
    """사진 한 장 → 정사영상 GeoTIFF. ``simple_orthophoto`` 의 대체.

    Returns
    -------
    성공 여부. 실패 사유는 로그에 남는다 (호출자가 성공/실패 집계 가능).
    """
    import rasterio
    from rasterio.transform import from_origin

    cfg = cfg or MosaicConfig()
    b = fh.footprint_bounds()
    if b is None:
        logger.warning("정사영상 건너뜀 (footprint 계산 실패): %s", image_path)
        return False
    x_min, y_min, x_max, y_max = b

    if gsd_m is None:
        gsd_m = fh.gsd_at_principal_point()
    if not np.isfinite(gsd_m) or gsd_m <= 0:
        logger.warning("정사영상 건너뜀 (GSD 비정상: %s): %s", gsd_m, image_path)
        return False

    out_w = int(np.ceil((x_max - x_min) / gsd_m))
    out_h = int(np.ceil((y_max - y_min) / gsd_m))
    if out_w <= 0 or out_h <= 0 or out_w * out_h > cfg.max_out_pixels:
        logger.warning("정사영상 건너뜀 (출력 크기 %d×%d): %s",
                       out_w, out_h, image_path)
        return False

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.warning("정사영상 건너뜀 (이미지 읽기 실패): %s", image_path)
        return False
    if img.ndim == 3 and img.shape[2] >= 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)

    maps = fh.intr.undistort_maps() if undistort else None
    res = warp_frame(fh, img, x_min, y_max, gsd_m, out_w, out_h, cfg, maps)
    if res is None:
        logger.warning("정사영상 건너뜀 (warp 창 없음): %s", image_path)
        return False
    u0, v0, warped, weight = res

    bands = warped.shape[2]
    canvas = np.zeros((out_h, out_w, bands), dtype=warped.dtype)
    canvas[v0:v0 + warped.shape[0], u0:u0 + warped.shape[1]] = warped
    if weight is not None:
        m = np.zeros((out_h, out_w), dtype=bool)
        m[v0:v0 + weight.shape[0], u0:u0 + weight.shape[1]] = weight <= 0
        canvas[m] = 0

    transform = from_origin(x_min, y_max, gsd_m, gsd_m)
    with rasterio.open(output_path, "w", driver="GTiff",
                       height=out_h, width=out_w, count=bands,
                       dtype=canvas.dtype, crs=f"EPSG:{epsg}",
                       transform=transform, compress="lzw") as dst:
        for i in range(bands):
            dst.write(canvas[:, :, i], i + 1)

    logger.info("정사영상 저장: %s (%d×%d px, GSD=%.4f m, 평면경사 %.2f°)",
                output_path, out_w, out_h, gsd_m, fh.plane.slope_deg)
    return True


# ---------------------------------------------------------------------------
# 다중 프레임 모자이크
# ---------------------------------------------------------------------------
def mosaic_frames(frames: list[FrameHomography],
                  image_paths: list[str | Path],
                  output_path: str | Path,
                  gsd_m: float | None = None,
                  epsg: int = 5186,
                  cfg: MosaicConfig | None = None,
                  undistort: bool = True) -> dict:
    """여러 프레임을 하나의 정사 모자이크로 합성.

    프레임은 **한 번에 한 장씩** 읽어 창 안에 누산하고 즉시 버린다.
    전체 이미지를 메모리에 들고 있지 않으므로 수백 장도 처리 가능하다.
    누산기는 ``float32`` 이고 크기는 ``out_h × out_w × (bands+1)`` 이다
    (5000×5000, RGB 기준 약 400 MB).

    Returns
    -------
    통계 dict: 출력 크기, GSD, 성공/실패 장수, 미충전 픽셀 비율.
    """
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
            f"출력 {out_w}×{out_h} = {out_w*out_h/1e6:.0f} Mpx 가 상한 "
            f"{cfg.max_out_pixels/1e6:.0f} Mpx 초과. GSD 를 키우거나 "
            f"타일로 나눠 처리할 것 (현재 GSD={gsd_m:.4f} m)"
        )

    bands = cfg.band_count
    logger.info("모자이크 캔버스: %d×%d px, GSD=%.4f m, 범위 %.1f×%.1f m, "
                "누산기 %.0f MB",
                out_w, out_h, gsd_m, x_max - x_min, y_max - y_min,
                out_h * out_w * (bands + 1) * 4 / 1e6)

    acc = np.zeros((out_h, out_w, bands), dtype=np.float32)
    acc_w = np.zeros((out_h, out_w), dtype=np.float32)

    maps_cache: dict[tuple, object] = {}
    n_ok = n_fail = 0

    for fh, path in zip(frames, image_paths):
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
        sub = acc[v0:v0 + h, u0:u0 + w]
        sub += warped[:, :, :bands].astype(np.float32) * weight[:, :, None]
        acc_w[v0:v0 + h, u0:u0 + w] += weight
        n_ok += 1

    filled = acc_w > 1e-6
    fill_ratio = float(filled.mean())
    out = np.zeros((out_h, out_w, bands), dtype=cfg.dtype)
    info = np.iinfo(cfg.dtype) if np.issubdtype(cfg.dtype, np.integer) else None
    for i in range(bands):
        band = np.zeros((out_h, out_w), dtype=np.float32)
        band[filled] = acc[:, :, i][filled] / acc_w[filled]
        if info is not None:
            band = np.clip(band, info.min, info.max)
        out[:, :, i] = band.astype(cfg.dtype)

    transform = from_origin(x_min, y_max, gsd_m, gsd_m)
    with rasterio.open(output_path, "w", driver="GTiff",
                       height=out_h, width=out_w, count=bands,
                       dtype=out.dtype, crs=f"EPSG:{epsg}",
                       transform=transform, compress="lzw",
                       tiled=True, blockxsize=512, blockysize=512) as dst:
        for i in range(bands):
            dst.write(out[:, :, i], i + 1)

    logger.info("모자이크 저장: %s (%d장 합성, %d장 실패, 충전율 %.1f%%)",
                output_path, n_ok, n_fail, fill_ratio * 100)
    if fill_ratio < 0.5:
        logger.warning("충전율 %.1f%% — 촬영 경로에 공백이 있거나 프레임 간 "
                       "footprint 가 크게 떨어져 있다.", fill_ratio * 100)

    return {
        "output_path": str(output_path),
        "width": out_w, "height": out_h, "gsd_m": gsd_m, "epsg": epsg,
        "bounds": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
        "frames_ok": n_ok, "frames_failed": n_fail,
        "fill_ratio": fill_ratio,
        "plane": {"a": frames[0].plane.a, "b": frames[0].plane.b,
                  "c": frames[0].plane.c,
                  "slope_deg": frames[0].plane.slope_deg,
                  "rmse_m": frames[0].plane.inlier_rmse_m},
    }

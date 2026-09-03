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
    dsm : ``dsm.DSM`` 또는 ``None``. 주어지면 단일 평면 대신 **픽셀마다 자기
        높이로** 역투영한다. 이 현장처럼 지면과 패널 상면이 2.4 m 떨어진
        2층 구조에서는 평면 하나로 둘 다 맞출 수 없다 — 어느 높이를 골라도
        다른 층이 ``Δh·k`` 만큼 밀린다. DSM 은 그 항을 없앤다.
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
    offnadir_auto : ``True`` 면 ``max_offnadir_ratio`` 를 **커버리지에 맞춰
        자동 조정**한다. 이 값은 프레임 최대 off-nadir 에 상대적인 의미를
        갖는데, 초점거리가 보정되면 그 최대값이 변한다 (실측: f 보정으로
        k_max 가 0.479 → 0.552 로 커지자 같은 0.35 제한의 예외율이
        7.7% → 13.4% 로 뛰었다). 각 픽셀을 덮는 프레임들의 최소 k 를 모아
        **프레임 최대 k 의 일정 비율**로 잡으면 자동으로 따라간다.
    offnadir_frac : 자동 모드에서 쓸 비율 (프레임 최대 k 대비). 0.65 면
        H20T 기준 k ≈ 0.37 로, 원래 의도했던 0.35 와 비슷하다.
    offnadir_fallback : ``True`` 면 제한 때문에 아무 프레임도 덮지 못하는
        픽셀에 한해 제한을 풀어 채운다 (구멍 방지). ``False`` 면 그 부분을
        비운다 — 품질을 우선할 때.
    tile_memory_mb : 타일 하나의 메모리 예산 (MB). 출력이 아무리 커도
        메모리는 이 값으로 묶인다.
    prefetch_workers : 원본 이미지를 미리 읽는 스레드 수 (0 이면 끔).
        JPEG 디코딩이 모자이크 비용의 대부분이고 GIL 을 놓으므로, 합성과
        겹쳐 실행하면 시간이 줄어든다.
    prefetch_ahead : 몇 장 앞까지 미리 읽을지.
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
                 seam_panel_penalty: float = 2.0,
                 seam_cost_blur_m: float = 1.5,
                 exposure_compensate: bool = True,
                 glint_penalty: float = 0.7,
                 glint_threshold: int = 250,
                 seam_blend_px: float = 0.0,
                 dsm=None,
                 two_layer_builder=None,
                 max_offnadir_ratio: float = 0.35,
                 offnadir_auto: bool = True,
                 offnadir_frac: float = 0.65,
                 offnadir_fallback: bool = True,
                 offnadir_tolerance: float = 1.15,
                 offnadir_epsilon: float = 0.002,
                 tile_memory_mb: float = 256.0,
                 image_cache: int = 8,
                 prefetch_workers: int = 4,
                 prefetch_ahead: int = 4,
                 band_count: int = 3,
                 dtype=np.uint8,
                 max_out_pixels: int = MAX_OUT_PIXELS,
                 **_unsupported):
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
        self.seam_panel_penalty = float(max(seam_panel_penalty, 0.0))
        self.seam_cost_blur_m = float(max(seam_cost_blur_m, 0.0))
        self.exposure_compensate = bool(exposure_compensate)
        self.glint_penalty = float(np.clip(glint_penalty, 0.0, 1.0))
        self.glint_threshold = int(glint_threshold)
        self.seam_blend_px = float(max(seam_blend_px, 0.0))
        self.dsm = dsm
        self.two_layer_builder = two_layer_builder
        self.max_offnadir_ratio = float(max(max_offnadir_ratio, 0.0))
        self.offnadir_auto = bool(offnadir_auto)
        self.offnadir_frac = float(np.clip(offnadir_frac, 0.2, 1.0))
        self.offnadir_fallback = bool(offnadir_fallback)
        self.offnadir_tolerance = float(max(offnadir_tolerance, 1.0))
        self.offnadir_epsilon = float(max(offnadir_epsilon, 0.0))
        self.tile_memory_mb = float(max(tile_memory_mb, 32.0))
        self.image_cache = int(max(image_cache, 1))
        self.prefetch_workers = int(max(prefetch_workers, 0))
        self.prefetch_ahead = int(max(prefetch_ahead, 0))

        # ★ 버전 불일치 방어 (내부 경계).
        #   CLI→pipeline 은 이미 인자를 걸러내지만 pipeline→MosaicConfig
        #   에서도 같은 일이 났다: pipeline.py 는 새 버전인데 ortho.py 가
        #   이전 버전이라 `two_layer_builder` 에서 TypeError 로 죽었다.
        #   파일을 수동으로 옮기는 환경에서는 흔한 일이므로, 모르는 인자는
        #   경고만 하고 무시한다 — 새 기능만 빠진 채 정상 실행된다.
        if _unsupported:
            logger.warning(
                "ortho.py 가 지원하지 않는 MosaicConfig 인자 %d개를 무시합니다: "
                "%s — pipeline.py 와 ortho.py 버전이 다릅니다. 두 파일을 같은 "
                "배포본에서 가져오면 해당 기능이 켜집니다.",
                len(_unsupported), ", ".join(sorted(_unsupported)))
            for _k, _v in _unsupported.items():
                setattr(self, _k, _v)
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
    dsm_valid = None
    if cfg.dsm is not None:
        from .dsm import warp_frame_dsm
        warped, dsm_valid = warp_frame_dsm(fh, image, cfg.dsm,
                                           x_min, y_max, gsd_m,
                                           u0, v0, ww, wh)
    else:
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

    if dsm_valid is not None:
        weight[~dsm_valid] = 0.0
    else:
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


def _auto_offnadir_ratio(frames, bounds, cfg, coarse_px: int = 1200) -> float:
    """프레임 최대 off-nadir 의 일정 비율로 제한값을 잡는다.

    ★ 앞선 구현은 "예외율이 목표치가 되도록" 분위수로 정했는데, 실데이터에서
      **제한을 완전히 무력화**했다. 모자이크 가장자리 밴드가 전체의 5% 를
      넘으므로 95 분위수가 곧 프레임 최대값이 되어 ``k ≤ 0.5641`` =
      제한 없음, 예외율 0.0% 가 나왔다. 연직 제한의 이점이 통째로 사라졌다.

      예외율을 목표로 삼은 것이 잘못이었다. 예외 픽셀은 "더 나은 선택지가
      아예 없는 자리" 이므로 많아도 손해가 아니다. 정작 중요한 것은
      **선택지가 있는 자리에서 연직에 가까운 쪽을 쓰는 것**이다.

      그래서 규칙을 단순화한다: 제한값 = 프레임 최대 k × ``offnadir_frac``.
      초점거리가 보정돼 k_max 가 변해도 (실측 0.479 → 0.564) 같은 비율이
      유지되므로 자동으로 따라간다.
    """
    k_max = 0.0
    for fh in frames:
        k_max = max(k_max, float(np.hypot(fh.intr.width, fh.intr.height)
                                 / 2.0 / max(fh.intr.f_px, 1e-6)))
    if k_max <= 0:
        return cfg.max_offnadir_ratio
    ratio = float(np.clip(k_max * cfg.offnadir_frac, 0.15, k_max))
    logger.info("연직 제한 자동 결정: k ≤ %.3f (프레임 최대 %.3f 의 %.0f%%)",
                ratio, k_max, cfg.offnadir_frac * 100)
    return ratio


def _min_k_map(frames, bounds, ow: int, oh: int, g: float) -> np.ndarray:
    """각 출력 픽셀을 덮는 프레임들의 **최소 off-nadir 비 k**.

    이 값이 그 픽셀이 물리적으로 도달할 수 있는 최선이다. 내부는 작고
    (0.05~0.15), 가장자리는 클 수밖에 없다(0.3~0.47).
    """
    x_min, y_min, x_max, y_max = bounds
    best = np.full((oh, ow), np.inf, dtype=np.float32)
    for fh in frames:
        h = float(fh.camera_xyz[2]) - float(
            fh.plane.height_at(fh.camera_xyz[0], fh.camera_xyz[1]))
        if h <= 1e-6:
            continue
        b = fh.footprint_bounds()
        if b is None:
            continue
        u0 = max(int((b[0] - x_min) / g), 0)
        u1 = min(int(np.ceil((b[2] - x_min) / g)), ow)
        v0 = max(int((y_max - b[3]) / g), 0)
        v1 = min(int(np.ceil((y_max - b[1]) / g)), oh)
        if u1 <= u0 or v1 <= v0:
            continue
        xs = x_min + (np.arange(u0, u1) + 0.5) * g
        ys = y_max - (np.arange(v0, v1) + 0.5) * g
        X, Y = np.meshgrid(xs, ys)
        k = np.hypot(X - fh.camera_xyz[0], Y - fh.camera_xyz[1]) / h
        sub = best[v0:v1, u0:u1]
        np.minimum(sub, k.astype(np.float32), out=sub)
    return best


def _coarse_reference(frames, image_paths, order, bounds, cfg,
                      panel_mask=None,
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
        return {}, None, None, None, None

    # 픽셀별 허용 k = max(설정 상한, 도달 가능한 최소 k × 여유)
    k_allow = None
    X_all = Y_all = None
    if restrict and cfg.max_offnadir_ratio > 0 and cfg.offnadir_fallback:
        mk = _min_k_map(frames, bounds, ow, oh, g)
        # ★ ε 을 더한다. 최소 k 를 만드는 그 프레임은 k_here == min_k 이므로
        #   등호로 통과해야 하는데, _min_k_map 은 footprint 를 정수 픽셀로
        #   잘라 계산하고 본 루프는 전체 격자에서 계산해 격자 정렬이 미세하게
        #   다르다. 부동소수점에서 등호가 깨지면 **그 픽셀이 통째로 버려진다.**
        #   실측: tolerance 1.0 에서 충전율이 0.912 → 0.760 으로 15%p 손실.
        #   (1.15 에서는 여유가 오차를 덮어 문제가 보이지 않았다.)
        #   ε=0.002 의 기복변위 영향은 1.23 m × 0.002 = 2.5 mm 로 무시 가능.
        k_allow = np.maximum(cfg.max_offnadir_ratio,
                             mk * cfg.offnadir_tolerance
                             + cfg.offnadir_epsilon).astype(np.float32)
        k_allow[~np.isfinite(mk)] = np.inf
        xs_a = x_min + (np.arange(ow) + 0.5) * g
        ys_a = y_max - (np.arange(oh) + 0.5) * g
        X_all, Y_all = np.meshgrid(xs_a, ys_a)
        n_relaxed = float((k_allow > cfg.max_offnadir_ratio + 1e-6).mean())
        mk_med = float(np.median(mk[np.isfinite(mk)]))
        logger.info("픽셀별 연직 상한: %.1f%% 픽셀은 k≤%.2f 로 덮을 수 없어 "
                    "도달 가능한 최소 k 까지만 완화 (중앙값 %.2f, 최대 %.2f)",
                    n_relaxed * 100, cfg.max_offnadir_ratio, mk_med,
                    float(np.max(mk[np.isfinite(mk)])))

        # ★ '최소 k 중앙값' 은 **촬영 밀도의 직접 지표**다. 그 픽셀을 덮는
        #   프레임 중 가장 연직에 가까운 것이 얼마나 기울었는지를 뜻하므로,
        #   이 값이 크다는 것은 **어떤 지점도 바로 위에서 찍히지 않았다**는
        #   말이고, 파이프라인으로는 되돌릴 수 없다.
        #
        #   실측 대조:
        #     그린환경센터 380장/15,000㎡ → 중앙값 0.06, 패널 어긋남 0.055 m
        #     극동대      199장/15,100㎡ → 중앙값 0.21~0.65, 어긋남 0.55~0.78 m
        #   기복변위 = 패널높이(약 1.5 m) × k 로 실측과 자릿수가 맞는다
        #   (Wide: 1.5 × 0.65 = 0.98 m 예측 vs 0.78 m 실측).
        if mk_med > 0.15:
            logger.warning(
                "  ★ 최소 k 중앙값이 %.2f 입니다 — 대부분의 지점이 **연직으로 "
                "촬영된 적이 없습니다.** 패널 높이 1.5 m 기준 기복변위가 "
                "%.2f m 로 예상되며, 이는 파이프라인으로 줄일 수 없습니다. "
                "촬영 밀도를 높이거나(중복도 상향) 비행선 간격을 좁혀야 합니다. "
                "참고: 정상 사례는 중앙값 0.06 이었습니다.",
                mk_med, 1.5 * mk_med)

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
        h_cam = float(fh.camera_xyz[2]) - float(
            fh.plane.height_at(fh.camera_xyz[0], fh.camera_xyz[1]))
        # k_allow 를 쓰면 판정을 **지상 k 하나로 통일**한다. 이미지 공간
        # 하드 제한을 함께 걸면 그쪽이 먼저 프레임을 잘라내 픽셀별 완화가
        # 무력해진다 (실측: 충전율 39% 로 붕괴).
        _img_restrict = restrict and (k_allow is None)
        sw = build_source_weight(fh.intr.width, fh.intr.height,
                                 fh.intr.cx, fh.intr.cy, cfg,
                                 f_px=fh.intr.f_px, restrict=_img_restrict)
        sw_s = cv2.resize(sw, None, fx=0.25, fy=0.25,
                          interpolation=cv2.INTER_AREA)
        if _img_restrict and cfg.max_offnadir_ratio > 0:
            sw_s[sw_s <= 0] = 0.0
        wt = cv2.warpPerspective(
            sw_s, (S @ H).astype(np.float64), (ow, oh),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        wt[~valid] = 0.0
        if _img_restrict and cfg.max_offnadir_ratio > 0:
            valid = valid & (wt > 0)

        # ★ 픽셀별 k 상한. 단순 상수 상한은 가장자리에서 '아무도 못 덮음' →
        #   예외로 한 번에 최대 k 까지 열림 → 절벽이 된다. 단계로 나눠 봤더니
        #   단계마다 노출 이득과 시임 판정이 새로 계산돼 **오히려 나빠졌다**
        #   (바깥 0.427 → 0.480 m, 파손 16.9 → 19.8%).
        #
        #   대신 각 픽셀이 도달 가능한 **최소 k** 를 그 픽셀의 상한으로 쓴다.
        #   한 번의 패스로 끝나므로 이득·시임이 전역적으로 일관되고,
        #   내부는 그대로 작은 k, 가장자리는 필요한 만큼만 열린다.
        if k_allow is not None and h_cam > 0:
            k_here = np.hypot(X_all - fh.camera_xyz[0],
                              Y_all - fh.camera_xyz[1]) / h_cam
            valid = valid & (k_here <= k_allow)
        if cfg.glint_penalty > 0:
            wt[wg >= cfg.glint_threshold] *= (1.0 - cfg.glint_penalty)

        eff = wt
        ov = valid & filled
        if cfg.seam_optimize and cfg.seam_cost_weight > 0 and ov.sum() > 200:
            diff = np.zeros((oh, ow), dtype=np.float32)
            diff[ov] = np.abs(wg[ov] - canvas[ov])
            diff = cv2.blur(diff, (blur_c, blur_c))
            sc = float(np.percentile(diff[ov], 75)) + 1e-3
            cost = diff / sc
            # ★ 차이 영상만으로는 **높이를 모른다.**
            #   패널 상면은 기준면보다 약 1 m 높아 프레임마다 Δh·k 만큼 다른
            #   위치에 투영된다. 시임이 그 위를 지나면 그 크기의 단차가 그대로
            #   보인다. 실측(그린환경센터 RGB) 잘라낸 구간에서 패널 상단
            #   경계의 열 간 점프를 재니:
            #       중앙값 0 px, 90% 3 px, **99% 64 px(41 cm), 최대 155 px(99 cm)**
            #       3 px 초과 점프가 4.9% 의 열에서 발생
            #   99% 값이 패널 높이 1 m 와 k 의 곱과 자릿수가 맞는다.
            #   QC 중앙값(0.06 m)은 이 국소 단차를 못 잡고, p90(0.93 m)이 잡는다.
            #
            #   그래서 **패널 위를 지나는 시임에 직접 벌점**을 준다. 차이가
            #   우연히 작아도 패널이면 피하게 만든다.
            if panel_mask is not None and cfg.seam_panel_penalty > 0:
                cost = cost + cfg.seam_panel_penalty * panel_mask
            eff = wt * np.exp(-cfg.seam_cost_weight * cost)
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
    return gains, label, g, canvas, filled


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

    if cfg.offnadir_auto and cfg.max_offnadir_ratio > 0:
        cfg.max_offnadir_ratio = _auto_offnadir_ratio(frames, bounds, cfg)
        _WEIGHT_CACHE.clear()

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

    gains, label_map, ref_gsd, coarse_img, coarse_ok = _coarse_reference(
        frames, image_paths, order, bounds, cfg, restrict=True)

    # ★ 2패스 — 1패스 기준 영상에서 패널을 분할한 뒤 그 마스크로 시임을
    #   다시 배치한다. 시임이 패널 위를 지나면 Δh·k 만큼의 단차가 그대로
    #   보인다. 실측(그린환경센터 RGB, 잘라낸 구간)에서 패널 상단 경계의
    #   열 간 점프를 재니 중앙값 0 px 인데 **99% 가 64 px(41 cm), 최대
    #   155 px(99 cm)** 이고 4.9% 의 열에서 3 px 초과 점프가 났다.
    #   QC 중앙값(0.06 m)은 이 국소 단차를 못 잡고 p90(0.93 m)이 잡는다.
    #
    #   1패스 결과를 통째로 버리고 다시 만들므로 노출 이득·시임은 여전히
    #   **한 번만** 결정된다 (앞서 단계적 예외에서 겪은 이득 불일치 없음).
    if (cfg.seam_panel_penalty > 0 and cfg.seam_optimize
            and coarse_img is not None and coarse_ok is not None):
        try:
            from .two_layer import segment_panels
            pm = segment_panels(coarse_img, coarse_ok, ref_gsd)
            frac = float(pm[coarse_ok].mean()) if coarse_ok.any() else 0.0
            if 0.05 < frac < 0.90:
                pm_f = cv2.GaussianBlur(pm.astype(np.float32), (0, 0),
                                        max(0.5 / ref_gsd, 1.0))
                logger.info("시임 재배치: 패널 %.1f%% 영역에 벌점 %.1f 적용 "
                            "— 시임이 패널을 피해 잔디·그림자로 흐르게 합니다",
                            frac * 100, cfg.seam_panel_penalty)
                gains, label_map, ref_gsd, coarse_img, coarse_ok = \
                    _coarse_reference(frames, image_paths, order, bounds, cfg,
                                      panel_mask=pm_f, restrict=True)
            else:
                logger.info("시임 재배치 생략 — 패널 면적비 %.1f%% 가 타당 "
                            "범위를 벗어납니다", frac * 100)
        except Exception as exc:
            logger.warning("시임 재배치 실패 (계속 진행): %s", exc)

    # ★ 2층 높이맵은 '단일 평면 저해상도 정사영상' 이 있어야 만들 수 있는데,
    #   그것이 바로 라벨맵 패스의 부산물이다. 여기서 콜백으로 넘겨 준다.
    #   (점군이 아니라 영상 분할로 만들므로 점 밀도와 무관하다.)
    if cfg.two_layer_builder is not None and cfg.dsm is None:
        try:
            _tl = cfg.two_layer_builder(coarse_img, coarse_ok, bounds, ref_gsd)
        except Exception as exc:                       # pragma: no cover
            logger.warning("2층 높이맵 생성 실패 (단일 평면으로 진행): %s", exc)
            _tl = None
        if _tl is not None:
            cfg.dsm = _tl
            _WEIGHT_CACHE.clear()


    # ★ 예외 처리는 이제 필요 없다.
    #   픽셀별 k 상한(_min_k_map)이 저해상도 패스 안에서 이미 적용되므로,
    #   '아무도 못 덮는 픽셀' 자체가 생기지 않는다. 예전처럼 뒤에서 다시
    #   패스를 돌 이유가 없고, 그 덕에 노출 이득과 시임 판정이 전역적으로
    #   한 번만 계산된다 (단계별 재계산이 만들던 밝기 단차가 사라진다).
    n_fallback = 0.0
    if label_map is not None and cfg.max_offnadir_ratio > 0:
        n_fallback = float((label_map < 0).mean())
        if n_fallback > 0.001 and not cfg.offnadir_fallback:
            logger.info("연직 제한으로 비운 픽셀: %.1f%% "
                        "(--offnadir-fallback 를 켜면 도달 가능한 최소 k 로 "
                        "채웁니다)", n_fallback * 100)

    # 프레임별 footprint 를 출력 픽셀 창으로 미리 계산 (타일 교차 판정용).
    wins = {}
    for idx in range(len(frames)):
        w_ = _frame_window(frames[idx], x_min, y_max, gsd_m, out_w, out_h)
        if w_ is not None:
            wins[idx] = w_

    maps_cache: dict[tuple, object] = {}
    img_cache: dict[int, np.ndarray] = {}

    def _decode(idx):
        im = cv2.imread(str(image_paths[idx]), cv2.IMREAD_UNCHANGED)
        if im is not None:
            if im.ndim == 3 and im.shape[2] >= 3:
                im = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2RGB)
            elif im.ndim == 2 and bands == 3:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        return im

    # ★ 모자이크가 남은 최대 비용이다 (실측 21m15s 중 7m36s).
    #   비용의 대부분은 **JPEG 디코딩**이다 — 프레임이 걸치는 타일마다 다시
    #   디코딩하므로 380장 × 약 2타일 = 760회, 20 MP 이미지다.
    #   OpenCV 의 imread 는 GIL 을 놓으므로 스레드로 **미리 읽어두면**
    #   합성(warp/compositing)과 겹쳐 실행된다.
    #   전역 라벨맵이 승자를 이미 정해 두었으므로 프레임 처리 순서가
    #   결과에 영향을 주지 않는다 — 선읽기가 안전하다.
    _pool = None
    _futures: dict = {}
    if cfg.prefetch_workers > 0:
        from concurrent.futures import ThreadPoolExecutor
        _pool = ThreadPoolExecutor(max_workers=cfg.prefetch_workers)

    def prefetch(indices):
        if _pool is None:
            return
        for i in indices:
            if i not in img_cache and i not in _futures:
                _futures[i] = _pool.submit(_decode, i)

    def load(idx):
        if idx not in img_cache:
            fut = _futures.pop(idx, None)
            im = fut.result() if fut is not None else _decode(idx)
            img_cache[idx] = im
            while len(img_cache) > cfg.image_cache:
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

            tile_idx = [i for i in order
                        if wins.get(i) is not None
                        and not (wins[i][1] + wins[i][3] <= v0t
                                 or wins[i][1] >= v0t + th)]
            prefetch(tile_idx[:cfg.image_cache])

            for pos, idx in enumerate(tile_idx):
                # 앞으로 쓸 프레임을 미리 읽어 둔다 (디코딩과 합성을 겹침).
                prefetch(tile_idx[pos + 1: pos + 1 + cfg.prefetch_ahead])
                w_ = wins[idx]
                fu0, fv0, fw, fh_ = w_
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

    if _pool is not None:
        for f_ in _futures.values():
            f_.cancel()
        _pool.shutdown(wait=False)

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
        "dsm": cfg.dsm.stats() if cfg.dsm is not None else None,
        "offnadir_fallback_ratio": n_fallback,
        "offnadir_tolerance": cfg.offnadir_tolerance,
        "offnadir_epsilon": cfg.offnadir_epsilon,
        "bounds": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
        "frames_ok": len(n_ok), "frames_failed": len(frames) - len(n_ok),
        "outlier_frames": outliers,
        "fill_ratio": fill_ratio,
        "plane": {"a": frames[0].plane.a, "b": frames[0].plane.b,
                  "c": frames[0].plane.c,
                  "slope_deg": frames[0].plane.slope_deg,
                  "rmse_m": frames[0].plane.inlier_rmse_m},
    }

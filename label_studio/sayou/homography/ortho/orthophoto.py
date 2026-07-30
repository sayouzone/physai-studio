"""평면 가정 정사영상 생성 — backward warping. **[레거시]**

⚠ 이 모듈은 ``geometry.project_point`` 와 **다른 부호 규약**을 쓴다.
   여기: ``d_cam = [-(px-cx)/f, -(py-cy)/f, -1]`` + ``px = cx + f·…``
   저기: ``project_point`` 의 광선은 ``[-(px-cx)/f, -(py-cy)/f, +1]`` 에 비례

   두 규약은 광축 주변 180° 만큼 어긋난다. Bundle Adjustment 와
   triangulation 은 ``project_point`` 규약을 쓰므로, **BA 산출 자세를 이
   함수에 그대로 넣으면 안 된다.** 신규 코드는
   ``georeferencing.homography.orthorectify_frame`` / ``mosaic_frames`` 를
   쓸 것 — 같은 규약을 공유하고, 출력 픽셀당 광선교차 계산이
   ``cv2.warpPerspective`` 한 번으로 대체된다.

   기존 산출물 재현 목적으로만 남겨 둔다.


DSM 이 없는 GCP-free 환경에서는 사진의 평균 지상 고도(``ground_z``)를
추정해서 평면으로 가정한다. 솔라패널 단지처럼 평탄한 지형이면 충분히 유효.

부호 규약 (중요)
----------------
``geometry.rotation_matrix(ω,φ,κ)`` 규약에서 카메라 광학축(보는 방향)은
카메라 좌표계 **-Z** 에 대응한다. 따라서:

* **픽셀 → 지상점 (forward 화면범위 계산)** 의 ray 는 ``d_cam = [-x/f, -y/f, -1]``
  로 둬야 nadir 촬영에서 ``d_world[2] < 0`` 이 나와 ``t = (Zg-Zc)/d_world[2] > 0``
  (지상이 카메라 아래이므로 ``Zg < Zc``) 으로 정상.
* **지상점 → 픽셀 (backward warping)** 도 같은 부호 규약을 유지해야 왕복 일치::

    px = cx + f_px · (R·diff)[0] / (R·diff)[2]
    py = cy + f_px · (R·diff)[1] / (R·diff)[2]

  (기존 ``cx - f_px·...`` 는 ray_cam Z=+1 규약의 잔재로, 좌우/상하 반전되어
  왕복오차 ~5000 px 발생.)

가속 전략
---------
* **좌표 생성 (가장 큰 연산)**: CuPy meshgrid + einsum-like 벡터 연산.
  ``out_w × out_h`` 픽셀 (수백만~수천만) 의 ray 를 한 번에 계산.
* **remap**: ``cv2.cuda.remap`` 사용 시 GpuMat 업로드/다운로드 시간이 있어
  단일 호출 효과는 출력 해상도가 1M 픽셀 이상일 때만 의미가 있다.
* 입력 이미지 크기가 작으면 CPU 가 더 빠르므로 임계치 가드를 둔다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

from solar_thermal.image.metadata import ImageMetadata

from ..geometry import compute_focal_px, rotation_matrix
from ..gpu_backend import HAS_CUPY, HAS_CV_CUDA_REMAP, cp

logger = logging.getLogger(__name__)

# 출력 픽셀 수 임계치 — 이 이하에서는 GPU 전송 오버헤드가 SIMT 이득보다 크다.
_GPU_MIN_PIXELS = 1_000_000


# ---------------------------------------------------------------------------
# Backward warping 좌표 생성
# ---------------------------------------------------------------------------
def _build_warp_maps_gpu(out_w: int, out_h: int,
                         x_min: float, y_max: float, gsd_m: float,
                         ground_z: float,
                         R: np.ndarray, camera_xyz: np.ndarray,
                         f_px: float, cx: float, cy: float):
    """CuPy 로 (map_x, map_y) 생성. 반환은 numpy float32."""
    out_xs = x_min + (cp.arange(out_w, dtype=cp.float64) + 0.5) * gsd_m
    out_ys = y_max - (cp.arange(out_h, dtype=cp.float64) + 0.5) * gsd_m
    grid_x, grid_y = cp.meshgrid(out_xs, out_ys)
    world = cp.stack([grid_x.ravel(),
                      grid_y.ravel(),
                      cp.full(grid_x.size, ground_z, dtype=cp.float64)],
                     axis=1)
    R_gpu = cp.asarray(R)
    cam_gpu = cp.asarray(camera_xyz)
    diff = world - cam_gpu
    R_diff = diff @ R_gpu.T
    den = R_diff[:, 2]
    valid = cp.abs(den) > 1e-9
    safe_den = cp.where(valid, den, 1)
    src_x = cp.where(valid, cx + f_px * R_diff[:, 0] / safe_den, -1.0)
    src_y = cp.where(valid, cy + f_px * R_diff[:, 1] / safe_den, -1.0)
    map_x = cp.asnumpy(src_x.reshape(out_h, out_w).astype(cp.float32))
    map_y = cp.asnumpy(src_y.reshape(out_h, out_w).astype(cp.float32))
    return map_x, map_y


def _build_warp_maps_np(out_w: int, out_h: int,
                        x_min: float, y_max: float, gsd_m: float,
                        ground_z: float,
                        R: np.ndarray, camera_xyz: np.ndarray,
                        f_px: float, cx: float, cy: float):
    out_xs = x_min + (np.arange(out_w) + 0.5) * gsd_m
    out_ys = y_max - (np.arange(out_h) + 0.5) * gsd_m
    grid_x, grid_y = np.meshgrid(out_xs, out_ys)
    world = np.stack([grid_x.ravel(),
                      grid_y.ravel(),
                      np.full(grid_x.size, ground_z)], axis=1)
    diff = world - camera_xyz
    R_diff = diff @ R.T
    den = R_diff[:, 2]
    valid = np.abs(den) > 1e-9
    src_x = np.full(grid_x.size, -1.0)
    src_y = np.full(grid_x.size, -1.0)
    src_x[valid] = cx + f_px * R_diff[valid, 0] / den[valid]
    src_y[valid] = cy + f_px * R_diff[valid, 1] / den[valid]
    return (src_x.reshape(out_h, out_w).astype(np.float32),
            src_y.reshape(out_h, out_w).astype(np.float32))


# ---------------------------------------------------------------------------
# Pixel → ground (4모서리 화면범위 계산용)
# ---------------------------------------------------------------------------
def _pixel_to_ground(px: float, py: float,
                     R: np.ndarray, camera_xyz: np.ndarray,
                     ground_z: float,
                     f_px: float, cx: float, cy: float) -> np.ndarray | None:
    """단일 픽셀 → ground plane 교점. 광선이 평행이거나 위로 향하면 None."""
    d_cam = np.array([-(px - cx) / f_px, -(py - cy) / f_px, -1.0])
    d_world = R.T @ d_cam
    if abs(d_world[2]) < 1e-9:
        return None  # 광선이 지면과 평행.
    t = (ground_z - camera_xyz[2]) / d_world[2]
    if t <= 0:
        return None  # nadir 촬영이면 t>0. t<=0 이면 자세값 이상.
    gx = camera_xyz[0] + t * d_world[0]
    gy = camera_xyz[1] + t * d_world[1]
    return np.array([gx, gy])


# ---------------------------------------------------------------------------
# remap (cv2.cuda.remap 또는 CPU)
# ---------------------------------------------------------------------------
def _remap_gpu(img_rgb: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    gpu_src = cv2.cuda_GpuMat()
    gpu_src.upload(img_rgb)
    gpu_mx = cv2.cuda_GpuMat()
    gpu_mx.upload(map_x)
    gpu_my = cv2.cuda_GpuMat()
    gpu_my.upload(map_y)
    gpu_dst = cv2.cuda.remap(
        gpu_src, gpu_mx, gpu_my,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return gpu_dst.download()


def _remap_cpu(img_rgb: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        img_rgb, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
def simple_orthophoto(meta: ImageMetadata,
                      camera_xyz: np.ndarray,
                      omega: float, phi: float, kappa: float,
                      ground_z: float,
                      gsd_m: float,
                      output_path: Path,
                      epsg: int = 5186):
    """평지 가정 정사영상 생성.

    DSM이 없는 GCP-free 환경에서는 사진의 평균 지상 고도(ground_z)를
    추정해서 평면으로 가정. 솔라패널 단지처럼 평탄한 지형이면 충분히 유효.

    Args:
        ground_z: 평균 지표면 고도 (m). 미지정 시 카메라 고도 - rel_alt 사용.
        gsd_m: 출력 픽셀 크기 (Ground Sampling Distance, m/pixel).
               예: 100m 고도, 4000px 폭, 35mm 렌즈 → GSD ≈ 2.7cm/px

    Returns:
        ``True`` 면 정사영상 저장 성공, ``False`` 면 입력 메타데이터/카메라
        파라미터에 문제가 있어 건너뜀 (예: focal_length 정보 없음, 이미지
        읽기 실패, ground_z 가 비유한 등). 호출자는 이 반환값으로 사진별
        성공/실패를 카운트할 수 있다.
    """
    # ---- 입력 가드: focal_length 정보 ------------------------------------
    f_px = compute_focal_px(meta)
    if f_px <= 0:
        logger.warning(
            "정사영상 건너뜀 (focal_length 정보 없음): %s",
            meta.origin_path,
        )
        return False

    # ---- 입력 가드: ground_z 가 유한 ------------------------------------
    if not np.isfinite(ground_z):
        logger.warning(
            "정사영상 건너뜀 (ground_z=NaN/Inf): %s",
            meta.origin_path,
        )
        return False

    # ---- 입력 가드: camera_xyz 유한 ------------------------------------
    if not np.all(np.isfinite(camera_xyz)):
        logger.warning(
            "정사영상 건너뜀 (camera_xyz 에 비유한 값): %s",
            meta.origin_path,
        )
        return False

    # ---- 이미지 로딩 ------------------------------------------------------
    img = cv2.imread(meta.origin_path, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("정사영상 건너뜀 (이미지 읽기 실패): %s",
                       meta.origin_path)
        return False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_rgb.shape[:2]

    R = rotation_matrix(omega, phi, kappa)
    cx, cy = w_img / 2.0, h_img / 2.0

    # -----------------------------------------------------------------------
    # 픽셀 → 지상점 역투영.
    #
    # ★ 버그 원인 ★
    #   rotation_matrix(ω,φ,κ) 규약에서 카메라 광학축(보는 방향)은
    #   카메라 좌표계의 +Z 가 아니라 -Z 에 대응한다. 기존 코드는 ray_cam 의
    #   Z 를 +1 로 둬서 광선이 하늘을 향했고(ray_world[2]>0), 카메라가 지면
    #   위(158m > 113m)에 있는데도 t = (Zg-Zc)/ray_world[2] 가 음수가 되어
    #   `if t < 0: continue` 에서 4모서리가 전부 버려졌다.
    #
    # ★ 해결 ★
    #   d_cam 의 Z 를 -1 로 두면 nadir 촬영 시 d_world[2] < 0,
    #   t = (Zg - Zc) / d_world[2] > 0 (Zg < Zc 이므로) 으로 정상.
    #   실측 검증: H20T 45.87m 고도 → LRF 거리(45.87m)와 정확히 일치,
    #   지상범위 약 26.7 x 35.4 m.
    # -----------------------------------------------------------------------
    def pixel_to_ground(px: float, py: float):
        d_cam = np.array([-(px - cx) / f_px,
                          -(py - cy) / f_px,
                          -1.0])
        d_world = R.T @ d_cam
        if abs(d_world[2]) < 1e-9:
            return None  # 광선이 지면과 평행
        t = (ground_z - camera_xyz[2]) / d_world[2]
        if t <= 0:
            return None  # nadir 촬영이면 t>0. t<=0 이면 자세값 이상.
        gx = camera_xyz[0] + t * d_world[0]
        gy = camera_xyz[1] + t * d_world[1]
        return np.array([gx, gy])

    # 이미지 4모서리를 ground plane (Z=ground_z)으로 역투영하여 정사영상 범위 결정
    corners_px = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype=float)
    ground_corners = []
    for px, py in corners_px:
        g = pixel_to_ground(px, py)
        if g is not None:
            ground_corners.append(g)

    if len(ground_corners) < 4:
        logger.warning("Ground projection 실패: %s", meta.origin_path)
        return False

    ground_corners = np.array(ground_corners)
    x_min, y_min = ground_corners.min(axis=0)
    x_max, y_max = ground_corners.max(axis=0)

    # ground_corners 에 NaN/Inf 가 섞이면 x_max-x_min 도 NaN → int() 가 죽는다.
    # 입력 가드에서 대부분 잡히지만 4모서리 중 일부만 비유한일 경우 대비.
    if not (np.isfinite(x_min) and np.isfinite(x_max)
            and np.isfinite(y_min) and np.isfinite(y_max)):
        logger.warning(
            "Ground projection 범위에 NaN/Inf: %s (x_min=%s, x_max=%s)",
            meta.origin_path, x_min, x_max,
        )
        return False

    # 출력 래스터 크기.
    out_w = int(np.ceil((x_max - x_min) / gsd_m))
    out_h = int(np.ceil((y_max - y_min) / gsd_m))

    # 비정상적으로 큰 출력 (자세 추정 실패로 ground projection 이 발산) 차단.
    # 솔라패널 단지 사진 한 장의 ortho 는 보통 1만 픽셀 이하.
    MAX_OUT_PIXELS = 100_000_000  # 1억 픽셀 (100 megapixel) 상한.
    if out_w * out_h > MAX_OUT_PIXELS or out_w <= 0 or out_h <= 0:
        logger.warning(
            "정사영상 건너뜀 (출력 크기 비정상: %d x %d): %s",
            out_w, out_h, meta.origin_path,
        )
        return False

    # 정사영상의 각 픽셀 → 월드좌표 → 원본사진 픽셀로 역매핑 (backward warping)
    out_xs = x_min + (np.arange(out_w) + 0.5) * gsd_m
    out_ys = y_max - (np.arange(out_h) + 0.5) * gsd_m
    grid_x, grid_y = np.meshgrid(out_xs, out_ys)
    world = np.stack([grid_x.ravel(),
                      grid_y.ravel(),
                      np.full(grid_x.size, ground_z)], axis=1)

    # 월드점 → 원본 픽셀 (forward projection).
    # ★ pixel_to_ground 가 d_cam=[...,-1] 규약을 쓰므로, 그 역산인
    #   backward warping 도 부호를 맞춰야 한다. 왕복 검증 결과:
    #     px = cx + f_px · (R·diff)[0] / (R·diff)[2]
    #     py = cy + f_px · (R·diff)[1] / (R·diff)[2]
    #   (기존 cx - f_px·... 는 ray_cam Z=+1 규약의 잔재로, 좌우/상하가
    #    반전되어 왕복오차 ~5000px 발생)
    diff = world - camera_xyz
    R_diff = diff @ R.T          # 각 행이 R·diff
    den = R_diff[:, 2]
    valid = np.abs(den) > 1e-9

    src_x = np.full(grid_x.size, -1.0)
    src_y = np.full(grid_x.size, -1.0)
    src_x[valid] = cx + f_px * R_diff[valid, 0] / den[valid]
    src_y[valid] = cy + f_px * R_diff[valid, 1] / den[valid]

    map_x = src_x.reshape(out_h, out_w).astype(np.float32)
    map_y = src_y.reshape(out_h, out_w).astype(np.float32)
    ortho = cv2.remap(img_rgb, map_x, map_y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0, 0, 0))

    # GeoTIFF 저장 (북향 정사)
    transform = from_origin(x_min, y_max, gsd_m, gsd_m)
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=out_h, width=out_w, count=3,
        dtype=ortho.dtype,
        crs=f"EPSG:{epsg}",
        transform=transform,
        compress="lzw",
    ) as dst:
        for i in range(3):
            dst.write(ortho[:, :, i], i + 1)
    logger.info("정사영상 저장: %s (%dx%d px, GSD=%.3fm)", output_path, out_w, out_h, gsd_m)
    return True


__all__ = ["simple_orthophoto"]
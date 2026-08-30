"""겹침 시차(disparity)로 기준 평면 높이를 데이터에서 직접 추정.

왜 필요한가
-----------
``--panel-top`` 값을 손으로 맞추는 것은 추측이다. 게다가 ``estimate_ground_z``
가 쓰는 LRF (``LRFTargetAbsAlt``) 는 "레이저가 맞은 지점"의 고도이지 지면이
아니다. 태양광 단지 위를 nadir 로 날면 레이저는 **패널 상면에 맞는 경우가
많다**. 즉 기준면이 이미 패널 상면 근처인데 거기에 ``+1.8`` 을 더하면 오히려
패널 위로 1.8 m 떠버린다. 이 모듈은 그 추측을 없앤다.

원리
----
높이 ``H`` 인 실제 지상점을 고도 ``z0`` 평면으로 정사보정하면, 프레임 ``i``
에서 그 점은 카메라 연직점 바깥으로::

    d_i = (H − z0) · u_i,     u_i = (P_xy − C_i,xy) / (C_i,z − z0)

만큼 밀린다 (``u_i`` 는 그 지점의 off-nadir 벡터). 따라서 두 프레임 사이의
겹침에서 관측되는 **시차**는::

    d_ij = d_i − d_j = (H − z0) · (u_i − u_j)

``u_i, u_j`` 는 카메라 위치에서 계산되고, ``d_ij`` 는 위상상관으로 측정된다.
남은 미지수는 스칼라 ``(H − z0)`` 하나뿐이므로 겹침 쌍들에 대해 최소제곱으로
풀면 된다. 그 값을 ``z0`` 에 더하면 **패널 상면(정확히는 겹침 영역에서 시차를
지배하는 표면)의 실제 고도**가 나온다.

이 추정은 특징점 매칭이나 BA 를 요구하지 않는다 — 이미 만든 호모그래피로
두 프레임을 같은 격자에 얹고 위상상관 한 번이면 끝난다.

한계
----
* 겹침 영역에 시차를 잴 만한 텍스처가 있어야 한다. 균질한 잔디만 있는 쌍은
  응답이 낮아 자동으로 버려진다.
* 지면과 패널이 섞인 영역에서는 **면적을 더 많이 차지하는 쪽**의 높이로
  수렴한다. 태양광 단지 중심부라면 패널 상면이다. 단지 외곽(지면 위주)
  프레임 쌍이 섞이면 값이 낮게 끌리므로, ``robust=True`` 가 중앙값으로
  이상치를 눌러 준다.
* ``u_i − u_j`` 가 0 에 가까운 쌍(거의 같은 위치에서 찍은 두 장)은 시차가
  안 생겨 정보가 없다 — ``min_baseline_ratio`` 로 걸러낸다.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .homography import GroundPlane

logger = logging.getLogger(__name__)

__all__ = ["estimate_plane_height_from_overlaps", "apply_calibration",
           "calibrate_plane", "PlaneCalibration"]


class PlaneCalibration:
    """평면 높이 보정 결과."""

    def __init__(self, delta_z_m: float, n_pairs: int, n_used: int,
                 residual_m: float, samples: list[float]):
        self.delta_z_m = delta_z_m      # 현재 평면에 더해야 할 값 (m)
        self.n_pairs = n_pairs
        self.n_used = n_used
        self.residual_m = residual_m
        self.samples = samples

    def __repr__(self) -> str:
        return (f"PlaneCalibration(Δz={self.delta_z_m:+.3f} m, "
                f"{self.n_used}/{self.n_pairs} pairs, "
                f"residual={self.residual_m:.3f} m)")


def _overlap_window(fh_i, fh_j, margin: float = 0.75):
    """두 프레임 footprint 의 교집합 중앙부 (x_min, y_max, w_m, h_m). 없으면 None."""
    bi, bj = fh_i.footprint_bounds(), fh_j.footprint_bounds()
    if bi is None or bj is None:
        return None
    x0 = max(bi[0], bj[0]); y0 = max(bi[1], bj[1])
    x1 = min(bi[2], bj[2]); y1 = min(bi[3], bj[3])
    if x1 - x0 < 3.0 or y1 - y0 < 3.0:
        return None
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    w = (x1 - x0) * margin; h = (y1 - y0) * margin
    return cx - w / 2, cy + h / 2, w, h


def _warp_to_window(fh, image, x_min, y_max, gsd, nx, ny):
    """프레임을 지정 지상격자로 warp (그레이스케일 float32)."""
    M = fh.ortho_pixel_matrix(x_min, y_max, gsd)
    out = cv2.warpPerspective(
        image, M, (nx, ny),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.astype(np.float32)


def estimate_plane_height_from_overlaps(
    frames: list,
    image_paths: list,
    plane: GroundPlane,
    intrinsics: list | None = None,
    *,
    max_pairs: int = 24,
    window_px: int = 384,
    min_response: float = 0.05,
    min_baseline_ratio: float = 0.03,
    robust: bool = True,
    max_shift_m: float = 3.0,
    seed: int = 0,
) -> PlaneCalibration | None:
    """겹침 시차로 ``plane`` 에 더해야 할 높이 보정량을 추정.

    Parameters
    ----------
    frames : ``FrameHomography`` 리스트 (현재 ``plane`` 으로 만든 것).
    image_paths : 같은 순서의 원본 이미지 경로.
    plane : 현재 기준 평면.
    max_pairs : 사용할 겹침 쌍 수 상한. 24 쌍이면 보통 충분하고 수 초면 끝난다.
    window_px : 시차 측정 창 한 변 (출력 픽셀). 클수록 안정적이지만 느리다.
    min_response : 위상상관 응답 하한. 텍스처 없는 쌍 제거.
    min_baseline_ratio : ``|u_i − u_j|`` 하한. 너무 작으면 시차가 안 생긴다.
    robust : 쌍별 추정치의 중앙값 사용 (이상치에 강함).
    max_shift_m : 1회 보정량의 절대 상한 (m). 기준면은 지면과 구조물 상단
        사이에서만 움직일 수 있다. 크게 잡으면 겹침의 시차 외 불일치를
        평면 높이로 흡수해 버린다 (실측 −8.2 m 사례).

    Returns
    -------
    ``PlaneCalibration`` 또는 ``None`` (유효 쌍 부족).
    ``delta_z_m`` 을 현재 평면 ``c`` 에 더하면 보정된 평면이 된다.
    """
    n = len(frames)
    if n < 2:
        return None

    # --- 겹치는 쌍 후보 (카메라 수평거리 기준 근접쌍) ---
    C = np.array([f.camera_xyz for f in frames])
    rng = np.random.default_rng(seed)
    cand = []
    for i in range(n):
        d = np.hypot(C[:, 0] - C[i, 0], C[:, 1] - C[i, 1])
        order = np.argsort(d)
        for j in order[1:4]:
            if j > i:
                cand.append((i, int(j)))
    if not cand:
        return None
    rng.shuffle(cand)
    cand = cand[:max_pairs]

    cache: dict[int, np.ndarray] = {}

    def gray(idx):
        if idx not in cache:
            img = cv2.imread(str(image_paths[idx]), cv2.IMREAD_GRAYSCALE)
            cache[idx] = img
            if len(cache) > 8:
                cache.pop(next(iter(cache)))
        return cache[idx]

    samples: list[float] = []
    n_used = 0
    for i, j in cand:
        win = _overlap_window(frames[i], frames[j])
        if win is None:
            continue
        x_min, y_max, w_m, h_m = win
        gsd = max(w_m, h_m) / window_px
        nx, ny = int(w_m / gsd), int(h_m / gsd)
        if nx < 64 or ny < 64:
            continue

        gi, gj = gray(i), gray(j)
        if gi is None or gj is None:
            continue
        wi = _warp_to_window(frames[i], gi, x_min, y_max, gsd, nx, ny)
        wj = _warp_to_window(frames[j], gj, x_min, y_max, gsd, nx, ny)
        valid = (wi > 0) & (wj > 0)
        if valid.mean() < 0.5:
            continue

        wi = np.where(valid, wi, 0.0); wj = np.where(valid, wj, 0.0)
        wi -= wi[valid].mean(); wj -= wj[valid].mean()
        hann = cv2.createHanningWindow((nx, ny), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(wi * hann, wj * hann)
        if resp < min_response:
            continue

        # 관측 시차 (m). warp 격자는 x 증가 → 동, y 증가 → 남.
        # ``cv2.phaseCorrelate(a, b)`` 는 b 를 a 에 맞추는 이동을 주므로
        # (i → j) 시차로 쓰려면 부호를 뒤집는다. 이 부호는 합성 검증으로
        # 확정했다 (진짜 표면고도 2.4 m 를 세 개의 서로 다른 시작 평면에서
        # 모두 2.4 m 로 복원).
        d_obs = -np.array([dx * gsd, -dy * gsd])

        # 기하학적 off-nadir 차이 벡터.
        P = np.array([x_min + w_m / 2, y_max - h_m / 2])
        z_p = plane.height_at(P[0], P[1])
        ui = (P - C[i, :2]) / max(C[i, 2] - z_p, 1e-6)
        uj = (P - C[j, :2]) / max(C[j, 2] - z_p, 1e-6)
        du = ui - uj
        if np.linalg.norm(du) < min_baseline_ratio:
            continue

        # d_obs = (H − z0) · du  →  스칼라 최소제곱 투영.
        dh = float(np.dot(d_obs, du) / np.dot(du, du))
        samples.append(dh)
        n_used += 1

    if n_used < 3:
        logger.warning("평면 높이 자동보정 실패: 유효 겹침 쌍 %d개 (최소 3개). "
                       "겹침 영역에 텍스처가 부족하거나 프레임 간격이 너무 좁음.",
                       n_used)
        return None

    arr = np.array(samples)
    dz = float(np.median(arr)) if robust else float(arr.mean())
    resid = float(np.median(np.abs(arr - dz)))

    # ---- 물리적 타당성 가드 -------------------------------------------
    # ★ 실측 데이터에서 이 추정기가 평면을 8.4 m 아래(지하) 로 밀어버린
    #   사례가 있었다. 겹침에 시차 외의 불일치(자세 오차, 정반사로 인한
    #   밝기 급변)가 섞이면 위상상관이 엉뚱한 봉우리를 잡고, 그 값이
    #   그대로 평면 높이로 흡수된다.
    #
    #   기준면은 물리적으로 '지면 근처 ~ 구조물 상단' 범위를 벗어날 수
    #   없다. 카메라-평면 거리의 20% 를 넘는 보정은 추정 실패로 본다.
    cam_h = float(np.median([f.camera_xyz[2] for f in frames])) - float(
        plane.height_at(*np.mean([f.camera_xyz[:2] for f in frames], axis=0)))
    # ★ 이전 상한 max(0.2·cam_h, 3.0) 은 44.9 m 고도에서 8.9 m 를 허용했다.
    #   실데이터에서 보정이 −8.2 m 로 그 안에 들어와 통과했고, 기준면이 지면
    #   보다 6.35 m 아래로 내려갔다. 그 상태에서는 패널이 기준면보다 8.75 m
    #   위에 있어 기복변위가 시임에서 1.75 m 에 달한다 — BA 를 고쳐 얻은
    #   이득을 그대로 까먹는다. 기준면 높이는 지면과 구조물 상단 사이
    #   몇 m 안에서만 움직일 수 있으므로 상한을 절대값으로 묶는다.
    limit = max_shift_m
    if abs(dz) > limit:
        logger.warning(
            "평면 높이 자동보정 기각: Δz=%+.2f m 가 타당 범위 ±%.1f m 를 "
            "벗어났습니다 (카메라-평면 %.1f m). 겹침에 시차 외 불일치가 "
            "많다는 뜻이므로 보정을 적용하지 않습니다.", dz, limit, cam_h)
        return None
    if resid > 0.5 * max(abs(dz), 0.3):
        logger.warning(
            "평면 높이 자동보정 기각: 쌍별 산포 ±%.2f m 가 추정값 %+.2f m "
            "대비 너무 큽니다 — 신뢰할 수 없는 추정입니다.", resid, dz)
        return None

    logger.info("평면 높이 자동보정: Δz = %+.3f m (쌍 %d/%d, 산포 ±%.3f m)",
                dz, n_used, len(cand), resid)
    if resid > 0.5:
        logger.warning("쌍별 추정 산포가 %.2f m 로 큽니다 — 겹침 영역에 지면과 "
                       "패널이 섞여 있거나 자세 오차가 큽니다. 값을 그대로 "
                       "쓰기 전에 결과를 눈으로 확인하세요.", resid)
    return PlaneCalibration(dz, len(cand), n_used, resid, samples)


def apply_calibration(plane: GroundPlane, calib: PlaneCalibration) -> GroundPlane:
    """보정량을 적용한 새 평면."""
    return GroundPlane(plane.a, plane.b, plane.c + calib.delta_z_m,
                       origin_xy=plane.origin_xy,
                       inlier_rmse_m=plane.inlier_rmse_m,
                       n_inliers=plane.n_inliers)


def calibrate_plane(frames_builder,
                    image_paths: list,
                    plane: GroundPlane,
                    *,
                    max_iter: int = 3,
                    tol_m: float = 0.05,
                    total_limit_m: float = 3.0,
                    **kwargs) -> tuple[GroundPlane, PlaneCalibration | None]:
    """반복 보정 — 평면을 옮기면 ``u_i`` 도 조금 바뀌므로 2~3 회면 수렴한다.

    Parameters
    ----------
    frames_builder : ``plane -> list[FrameHomography]`` 콜러블. 평면이 바뀔
        때마다 호모그래피를 다시 만들어야 하므로 함수로 받는다.
    plane : 시작 평면.
    max_iter : 최대 반복. 보통 2 회면 ``tol_m`` 안에 든다.
    tol_m : 보정량이 이 값보다 작아지면 종료.

    Returns
    -------
    (보정된 평면, 마지막 ``PlaneCalibration``). 추정 실패 시 원래 평면과
    ``None`` 을 돌려주므로 호출자는 그대로 진행하면 된다.
    """
    cur = plane
    last: PlaneCalibration | None = None
    c0 = plane.c
    for it in range(max_iter):
        frames = frames_builder(cur)
        cal = estimate_plane_height_from_overlaps(frames, image_paths, cur, **kwargs)
        if cal is None:
            break
        last = cal
        nxt = apply_calibration(cur, cal)
        # 누적 이동도 묶는다 — 반복이 조금씩 같은 방향으로 밀 수 있다.
        if abs(nxt.c - c0) > total_limit_m:
            logger.warning("평면 보정 누적 %+.2f m 가 상한 ±%.1f m 를 넘어 "
                           "중단합니다 (현재 %.2f m 유지).",
                           nxt.c - c0, total_limit_m, cur.c)
            break
        cur = nxt
        if abs(cal.delta_z_m) < tol_m:
            logger.info("평면 보정 수렴 (%d회): 최종 기준면 표고 %.2f m", it + 1, cur.c)
            break
    else:
        if last is not None:
            logger.warning("평면 보정이 %d회 안에 수렴하지 않음 (마지막 Δz=%+.3f m)",
                           max_iter, last.delta_z_m)
    return cur, last

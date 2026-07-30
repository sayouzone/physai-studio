"""DLT(Direct Linear Transform) 다중시점 삼각측량.

원리
----
각 관측 (u, v) 와 카메라 P 에 대해 ``[u·w, v·w, w]ᵀ = P · [X,Y,Z,1]ᵀ`` 에서
u, v 를 소거하면 ``X = [X,Y,Z,1]ᵀ`` 에 대한 2 개의 동차 선형식이 나온다::

    u · (P[2]·X) - (P[0]·X) = 0
    v · (P[2]·X) - (P[1]·X) = 0

N 개 시점이면 ``2N × 4`` 행렬 A 가 되고, ``A · X = 0`` 의 최소제곱해는
``AᵀA`` 의 최소 특이값에 대응하는 우특이벡터 (SVD 마지막 행).

GPU 가속 전략
-------------
SVD 는 length-N 마다 행렬 크기가 달라 단순 ``(B, 2N, 4)`` 배치가 안 된다.
→ **동일 length 끼리 그룹핑** 후 그룹별로 ``cp.linalg.svd`` 배치 호출.
드론 사진의 track 길이 분포는 보통 2~6 에 집중돼 있어 그룹 수가 적다.

삼각측량 후 두 가지 품질 필터:
1. **시선각 (parallax)** — 가장 멀리 떨어진 두 카메라에서 점을 바라본
   방향벡터 사이 각도가 ``min_triangulation_angle_deg`` 미만이면 폐기
   (깊이가 불안정). nadir 드론은 베이스라인이 짧아 2도 정도로 완화.
2. **재투영 오차** — 복원된 X 를 다시 모든 카메라로 투영한 오차의 평균이
   ``max_reproj_err_px`` 초과면 폐기.
"""

from __future__ import annotations

import logging

import numpy as np

from ..geometry import camera_projection_matrix
from ..gpu_backend import HAS_CUPY, cp, free_gpu_memory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CPU helpers
# ---------------------------------------------------------------------------
def _triangulate_dlt_cpu(proj_mats: list[np.ndarray],
                         points_2d: list[np.ndarray]) -> np.ndarray:
    """단일 track CPU DLT."""
    A = []
    for P, (u, v) in zip(proj_mats, points_2d):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A)
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return np.array([np.nan, np.nan, np.nan])
    return X[:3] / X[3]

def triangulate_dlt(proj_mats: list[np.ndarray],
                    points_2d: list[np.ndarray]) -> np.ndarray:
    """DLT(Direct Linear Transform) 다중시점 삼각측량.

    각 관측 (u, v) 와 카메라 P 에 대해, [u·w, v·w, w]ᵀ = P·[X,Y,Z,1]ᵀ 에서
    u, v 를 소거하면 X=[X,Y,Z,1]ᵀ 에 대한 2개의 동차 선형식을 얻는다:
        u·(P[2]·X) - (P[0]·X) = 0
        v·(P[2]·X) - (P[1]·X) = 0
    N개 시점이면 2N x 4 행렬 A 가 되고, A·X = 0 의 최소제곱해는
    AᵀA 의 최소 특이값에 대응하는 우특이벡터 (SVD 마지막 행).

    Parameters
    ----------
    proj_mats : 각 관측에 대응하는 3x4 카메라 행렬 P 리스트
    points_2d : 각 관측의 (u, v) 픽셀좌표 리스트

    Returns
    -------
    point_3d : (3,) 복원된 지상점 (X, Y, Z)
    """
    A = []
    for P, (u, v) in zip(proj_mats, points_2d):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A)                       # (2N, 4)
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]                              # 최소 특이값의 우특이벡터
    if abs(X[3]) < 1e-12:
        return np.array([np.nan, np.nan, np.nan])
    return X[:3] / X[3]                     # 동차 → 3D


def _compute_proj_and_centers(cameras: np.ndarray,
                              intrinsics: list[tuple[float, float, float]]):
    """카메라별 P 행렬 + 중심 좌표를 numpy 배열로 일괄 계산."""
    n_cam = len(cameras)
    Ps = np.empty((n_cam, 3, 4), dtype=np.float64)
    centers = np.empty((n_cam, 3), dtype=np.float64)
    for ci in range(n_cam):
        Xc, Yc, Zc, om, ph, ka = cameras[ci]
        f_px, cx, cy = intrinsics[ci]
        C = np.array([Xc, Yc, Zc])
        Ps[ci] = camera_projection_matrix(C, om, ph, ka, f_px, cx, cy)
        centers[ci] = C
    return Ps, centers


# ---------------------------------------------------------------------------
# GPU 배치 SVD
# ---------------------------------------------------------------------------
def _triangulate_batched_gpu(tracks: list,
                             cameras: np.ndarray,
                             intrinsics: list[tuple[float, float, float]],
                             max_reproj_err_px: float,
                             min_triangulation_angle_deg: float):
    Ps_np, centers_np = _compute_proj_and_centers(cameras, intrinsics)
    Ps = cp.asarray(Ps_np)
    centers = cp.asarray(centers_np)

    # length 별 그룹핑.
    by_len: dict[int, list[int]] = {}
    for t_idx, track in enumerate(tracks):
        by_len.setdefault(len(track), []).append(t_idx)

    observations: list = []
    initial_points: list = []
    track_point_map: list = []
    dropped_angle = dropped_reproj = dropped_degenerate = 0

    for length, t_indices in by_len.items():
        B = len(t_indices)
        cam_idx_arr = np.empty((B, length), dtype=np.int64)
        pts_arr = np.empty((B, length, 2), dtype=np.float64)
        for b, t_idx in enumerate(t_indices):
            for k, obs in enumerate(tracks[t_idx]):
                cam_idx_arr[b, k] = obs[0]
                pts_arr[b, k, 0] = obs[2]
                pts_arr[b, k, 1] = obs[3]

        cam_idx_gpu = cp.asarray(cam_idx_arr)
        pts_gpu = cp.asarray(pts_arr)
        Ps_batch = Ps[cam_idx_gpu]           # (B, L, 3, 4)
        centers_batch = centers[cam_idx_gpu]  # (B, L, 3)

        # A 행렬 구성 (B, 2L, 4).
        u = pts_gpu[..., 0:1]   # (B, L, 1)
        v = pts_gpu[..., 1:2]
        P0 = Ps_batch[..., 0, :]  # (B, L, 4)
        P1 = Ps_batch[..., 1, :]
        P2 = Ps_batch[..., 2, :]
        row_u = u * P2 - P0
        row_v = v * P2 - P1
        A = cp.empty((B, 2 * length, 4), dtype=cp.float64)
        A[:, 0::2, :] = row_u
        A[:, 1::2, :] = row_v

        # 배치 SVD — CuPy 는 (B, M, N) 입력 지원.
        _, _, Vt = cp.linalg.svd(A, full_matrices=False)
        X_h = Vt[:, -1, :]              # (B, 4)
        w = X_h[:, 3:4]
        degenerate = (cp.abs(w[:, 0]) < 1e-12)
        # w == 0 으로 NaN/Inf 가 안 나오게 안전한 분모로 대체.
        safe_w = cp.where(w == 0, 1, w)
        X = cp.where(degenerate[:, None], cp.nan, X_h[:, :3] / safe_w)

        # 시선각 (parallax) — 배치 벡터화.
        rays = X[:, None, :] - centers_batch      # (B, L, 3)
        norms = cp.linalg.norm(rays, axis=2, keepdims=True) + 1e-12
        rays_n = rays / norms
        cos_mat = cp.einsum("bik,bjk->bij", rays_n, rays_n)
        cos_mat = cp.clip(cos_mat, -1.0, 1.0)
        max_angle = cp.max(cp.arccos(cos_mat) * (180.0 / cp.pi),
                           axis=(1, 2))  # (B,)

        # 재투영 오차.
        X_full = cp.concatenate([X, cp.ones((B, 1), dtype=X.dtype)], axis=1)
        proj = cp.einsum("blij,bj->bli", Ps_batch, X_full)
        proj_uv = proj[..., :2] / (proj[..., 2:3] + 1e-12)  # (B, L, 2)
        reproj_err = cp.linalg.norm(proj_uv - pts_gpu, axis=2)  # (B, L)
        mean_err = cp.mean(reproj_err, axis=1)  # (B,)

        # CPU 로 마스크 한 번에 전송.
        bad_deg = cp.asnumpy(degenerate | cp.any(cp.isnan(X), axis=1))
        bad_ang = cp.asnumpy(max_angle < min_triangulation_angle_deg)
        bad_rep = cp.asnumpy(mean_err > max_reproj_err_px)
        X_cpu = cp.asnumpy(X)

        for b, t_idx in enumerate(t_indices):
            if bad_deg[b]:
                dropped_degenerate += 1
                continue
            if bad_ang[b]:
                dropped_angle += 1
                continue
            if bad_rep[b]:
                dropped_reproj += 1
                continue
            point_idx = len(initial_points)
            initial_points.append(X_cpu[b])
            track_point_map.append(t_idx)
            for k in range(length):
                observations.append(
                    (int(cam_idx_arr[b, k]), point_idx, pts_arr[b, k].copy())
                )

    initial_points_arr = (np.array(initial_points)
                          if initial_points else np.zeros((0, 3)))
    logger.info(
        "삼각측량 [GPU]: %d개 3D점 복원, %d개 관측 "
        "(제거: 시선각 %d, 재투영 %d, 퇴화 %d)",
        len(initial_points_arr), len(observations),
        dropped_angle, dropped_reproj, dropped_degenerate,
    )
    if len(initial_points_arr) > 0:
        zs = initial_points_arr[:, 2]
        logger.info("  복원점 Z범위: %.1f ~ %.1f m (중앙값 %.1f)",
                    zs.min(), zs.max(), np.median(zs))
    free_gpu_memory()
    return observations, initial_points_arr, track_point_map


# ---------------------------------------------------------------------------
# CPU 단일 track 순회 (fallback)
# ---------------------------------------------------------------------------
def _triangulate_cpu(tracks: list,
                     cameras: np.ndarray,
                     intrinsics: list[tuple[float, float, float]],
                     max_reproj_err_px: float,
                     min_triangulation_angle_deg: float):
    Ps_np, centers_np = _compute_proj_and_centers(cameras, intrinsics)

    observations: list = []
    initial_points: list = []
    track_point_map: list = []
    dropped_angle = dropped_reproj = dropped_degenerate = 0

    for t_idx, track in enumerate(tracks):
        cam_indices = [obs[0] for obs in track]
        pts2d = [np.array([obs[2], obs[3]]) for obs in track]
        Ps = [Ps_np[ci] for ci in cam_indices]

        X = _triangulate_dlt_cpu(Ps, pts2d)
        if not np.all(np.isfinite(X)):
            dropped_degenerate += 1
            continue

        rays = [(X - centers_np[ci]) for ci in cam_indices]
        rays = [r / (np.linalg.norm(r) + 1e-12) for r in rays]
        max_angle = 0.0
        for a in range(len(rays)):
            for b in range(a + 1, len(rays)):
                cos_ang = np.clip(rays[a] @ rays[b], -1.0, 1.0)
                max_angle = max(max_angle, np.degrees(np.arccos(cos_ang)))
        if max_angle < min_triangulation_angle_deg:
            dropped_angle += 1
            continue

        Xh = np.append(X, 1.0)
        reproj_errs = []
        for P, uv in zip(Ps, pts2d):
            uvw = P @ Xh
            if abs(uvw[2]) < 1e-12:
                reproj_errs.append(1e9)
                continue
            uv_pred = uvw[:2] / uvw[2]
            reproj_errs.append(np.linalg.norm(uv_pred - uv))
        if np.mean(reproj_errs) > max_reproj_err_px:
            dropped_reproj += 1
            continue

        point_idx = len(initial_points)
        initial_points.append(X)
        track_point_map.append(t_idx)
        for ci, uv in zip(cam_indices, pts2d):
            observations.append((ci, point_idx, uv))

    initial_points_arr = (np.array(initial_points)
                          if initial_points else np.zeros((0, 3)))
    logger.info(
        "삼각측량 [CPU]: %d개 3D점 복원, %d개 관측 "
        "(제거: 시선각 %d, 재투영 %d, 퇴화 %d)",
        len(initial_points_arr), len(observations),
        dropped_angle, dropped_reproj, dropped_degenerate,
    )
    if len(initial_points_arr) > 0:
        zs = initial_points_arr[:, 2]
        logger.info("  복원점 Z범위: %.1f ~ %.1f m (중앙값 %.1f)",
                    zs.min(), zs.max(), np.median(zs))
    return observations, initial_points_arr, track_point_map


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------
def triangulate_tracks(tracks: list,
                       cameras: np.ndarray,
                       intrinsics: list[tuple[float, float, float]],
                       max_reproj_err_px: float = 3.0,
                       min_triangulation_angle_deg: float = 2.0):
    """track 들을 삼각측량해서 BA 입력(observations, initial_points) 생성.

    Parameters
    ----------
    tracks : build_tracks 의 출력
    cameras : (n_cam, 6) 초기 외부표정 [Xc, Yc, Zc, ω, φ, κ]
              (RTK 좌표 + 짐벌 자세에서 만든 초기값)
    intrinsics : 카메라별 (f_px, cx, cy) 리스트
    max_reproj_err_px : 삼각측량 후 재투영 오차가 이 값을 넘는 track 은 제거
    min_triangulation_angle_deg : 시선각(parallax)이 너무 작으면 깊이가
        불안정하므로 제거. nadir 드론 사진은 베이스라인이 짧아 각이
        작은 편이라 2도 정도로 완화.

    Returns
    -------
    observations : list[(cam_idx, point_idx, np.array([u, v]))]
    initial_points : (n_pts, 3) 삼각측량된 3D 점들
    track_point_map : list[int] — initial_points[k] 가 몇 번째 track 인지
    """
    # 카메라별 P 행렬 미리 계산
    proj_cache = {}
    centers = {}
    for cam_idx in range(len(cameras)):
        Xc, Yc, Zc, om, ph, ka = cameras[cam_idx]
        f_px, cx, cy = intrinsics[cam_idx]
        C = np.array([Xc, Yc, Zc])
        proj_cache[cam_idx] = camera_projection_matrix(C, om, ph, ka, f_px, cx, cy)
        centers[cam_idx] = C

    observations = []
    initial_points = []
    track_point_map = []
    dropped_angle = dropped_reproj = dropped_degenerate = 0

    for t_idx, track in enumerate(tracks):
        cam_indices = [obs[0] for obs in track]
        pts2d = [np.array([obs[2], obs[3]]) for obs in track]
        Ps = [proj_cache[ci] for ci in cam_indices]

        # --- DLT 삼각측량 ---
        X = triangulate_dlt(Ps, pts2d)
        if not np.all(np.isfinite(X)):
            dropped_degenerate += 1
            continue

        # --- 시선각(parallax) 체크 ---
        # 가장 멀리 떨어진 두 카메라에서 점을 바라본 방향벡터 사이 각도.
        rays = [(X - centers[ci]) for ci in cam_indices]
        rays = [r / (np.linalg.norm(r) + 1e-12) for r in rays]
        max_angle = 0.0
        for a in range(len(rays)):
            for b in range(a + 1, len(rays)):
                cos_ang = np.clip(rays[a] @ rays[b], -1.0, 1.0)
                max_angle = max(max_angle, np.degrees(np.arccos(cos_ang)))
        if max_angle < min_triangulation_angle_deg:
            dropped_angle += 1
            continue

        # --- 재투영 오차 체크 ---
        Xh = np.append(X, 1.0)
        reproj_errs = []
        for P, uv in zip(Ps, pts2d):
            uvw = P @ Xh
            if abs(uvw[2]) < 1e-12:
                reproj_errs.append(1e9)
                continue
            uv_pred = uvw[:2] / uvw[2]
            reproj_errs.append(np.linalg.norm(uv_pred - uv))
        if np.mean(reproj_errs) > max_reproj_err_px:
            dropped_reproj += 1
            continue

        # --- 통과: observations 에 추가 ---
        point_idx = len(initial_points)
        initial_points.append(X)
        track_point_map.append(t_idx)
        for ci, uv in zip(cam_indices, pts2d):
            observations.append((ci, point_idx, uv))

    initial_points = (np.array(initial_points)
                      if initial_points else np.zeros((0, 3)))
    logger.info(
        "삼각측량: %d개 3D점 복원, %d개 관측 (제거: 시선각 %d, 재투영 %d, 퇴화 %d)",
        len(initial_points), len(observations),
        dropped_angle, dropped_reproj, dropped_degenerate,
    )
    if len(initial_points) > 0:
        zs = initial_points[:, 2]
        logger.info("  복원점 Z범위: %.1f ~ %.1f m (중앙값 %.1f)",
                    zs.min(), zs.max(), np.median(zs))
    return observations, initial_points, track_point_map


__all__ = ["triangulate_tracks"]
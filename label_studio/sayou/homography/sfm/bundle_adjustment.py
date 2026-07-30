"""RTK 제약 Bundle Adjustment.

핵심 아이디어
-------------
GCP 가 없으므로 절대 좌표계 기준점은 RTK 측정값. 하지만 RTK 도 cm 급 오차가
있으므로 hard constraint 가 아닌 **soft constraint (가중치 = 1/σ²)** 로 잔차에
추가::

    residual = [reprojection_errors, sqrt(w) * (camera_pos - rtk_prior)]

* reprojection error 가 일관된 internal geometry 를 보장
* RTK prior 가 절대 georeferencing 을 보장
* 둘이 가중 평균되어 outlier 에 강건한 해를 찾음

★ 이 버전에서 고친 것
---------------------
원본은 ``_build_residuals_np`` / ``_build_residuals_gpu`` /
``_build_jacobian_sparsity`` 를 **정의만 하고 한 번도 호출하지 않았다**.
``rtk_constrained_bundle_adjustment`` 안에 중첩 정의된 Python 루프
``residuals`` 가 실제로 쓰였다. 즉 docstring 이 설명하는 가속은 전부
미적용 상태였다. 이 파일은 세 함수를 실제 경로에 연결한다.

1. **관측 평탄화 + 벡터화 residual** — ``residuals(x)`` 호출당 Python 루프
   M 회 → 단일 ``einsum`` 한 번.
2. **희소 jacobian** — ``least_squares(jac_sparsity=...)``. 이게 없으면
   ``trf`` 가 finite-difference 로 **전체 파라미터 수 P 만큼** residual 을
   재평가한다 (P = 6·n_cam + 3·n_pts, 보통 수만~수십만). sparsity 를 주면
   그래프 컬러링으로 수십 회로 줄어든다. **GPU 없이 얻는 가장 큰 이득.**
3. **CuPy backend** — 관측 M ≥ 5000 일 때만. 적으면 PCIe 전송이 더 비싸다.
4. **2단계 robust 스케줄** — 원본의 ``loss="huber", f_scale=2.0`` 단일 호출은
   초기 재투영 오차가 몇 px 를 넘으면 전 관측이 이상점 취급되어 최적화가
   시작조차 못 한다 (실측: 84초 소요, 카메라 이동량 0.000 m). 1단계는 잔차
   분포에 맞춘 느슨한 ``soft_l1``, 2단계에서 ``huber 2px`` 로 조인다.

★ 좌표 중심화 (추가)
--------------------
EPSG:5186 절대좌표는 (200000, 400000) 근방이다. ``diff = P_o − C_o`` 에서
두 항 모두 10⁵ 스케일인데 차이는 10¹ 스케일 → 상쇄로 유효자릿수 약 4 자리
손실. 또 ``least_squares`` 의 finite-difference 스텝은 파라미터 크기에
비례하므로 10⁵ 파라미터에 대한 상대 스텝이 지나치게 커진다.
→ 블록 중심을 빼고 최적화한 뒤 되돌린다. 결과는 동일하고 수치만 좋아진다.

★ RMSE 계산 (수정)
------------------
원본의 ``rmse_px = sqrt(2·result.cost / len(observations))`` 는
(a) ``cost`` 에 RTK prior 잔차가 섞여 있고 (b) huber 감쇠가 적용된 값이라
재투영 RMSE 가 아니다. 해에서 재투영 잔차만 다시 계산한다.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csr_matrix

from ..geometry import rotation_matrices_batch_cp, rotation_matrices_batch_np
from ..gpu_backend import HAS_CUPY, cp, free_gpu_memory
from ..utils import fmt_elapsed

logger = logging.getLogger(__name__)

# GPU 전환 임계치 (관측 수). 이 이하에서는 numpy 벡터화가 빠르다.
_GPU_MIN_OBS = 5000


# ---------------------------------------------------------------------------
# Residual 콜백 빌더
# ---------------------------------------------------------------------------
def _build_residuals_gpu(n_cam: int, n_pts: int,
                         obs_cam_idx: np.ndarray, obs_pt_idx: np.ndarray,
                         obs_uv: np.ndarray,
                         rtk_priors: np.ndarray, rtk_w_sqrt: np.ndarray,
                         f_px: float, cx: float, cy: float):
    """CuPy 벡터화 residual. 입력은 numpy, 내부에서 GPU 로 이전."""
    obs_cam_idx_g = cp.asarray(obs_cam_idx)
    obs_pt_idx_g = cp.asarray(obs_pt_idx)
    obs_uv_g = cp.asarray(obs_uv)
    rtk_priors_g = cp.asarray(rtk_priors)
    rtk_w_sqrt_g = cp.asarray(rtk_w_sqrt)

    def residuals(x):
        x_g = cp.asarray(x)
        cams = x_g[: n_cam * 6].reshape(n_cam, 6)
        pts = x_g[n_cam * 6:].reshape(n_pts, 3)

        R = rotation_matrices_batch_cp(cams[:, 3:6], cp)
        C = cams[:, :3]

        R_o = R[obs_cam_idx_g]
        C_o = C[obs_cam_idx_g]
        P_o = pts[obs_pt_idx_g]
        diff = P_o - C_o
        Rdiff = cp.einsum("mij,mj->mi", R_o, diff)
        den = Rdiff[:, 2]
        den_safe = cp.where(cp.abs(den) < 1e-9, 1e-9, den)
        x_pred = cx - f_px * Rdiff[:, 0] / den_safe
        y_pred = cy - f_px * Rdiff[:, 1] / den_safe
        reproj = cp.stack([x_pred - obs_uv_g[:, 0],
                           y_pred - obs_uv_g[:, 1]], axis=1)

        rtk_res = rtk_w_sqrt_g * (C - rtk_priors_g)

        return cp.asnumpy(cp.concatenate([reproj.ravel(), rtk_res.ravel()]))

    return residuals


def _build_residuals_np(n_cam: int, n_pts: int,
                        obs_cam_idx: np.ndarray, obs_pt_idx: np.ndarray,
                        obs_uv: np.ndarray,
                        rtk_priors: np.ndarray, rtk_w_sqrt: np.ndarray,
                        f_px: float, cx: float, cy: float):
    """numpy 벡터화 residual (CPU). 원본 Python 루프 대비 수십~수백 배 빠름."""

    def residuals(x):
        cams = x[: n_cam * 6].reshape(n_cam, 6)
        pts = x[n_cam * 6:].reshape(n_pts, 3)

        R = rotation_matrices_batch_np(cams[:, 3:6])
        C = cams[:, :3]

        R_o = R[obs_cam_idx]
        C_o = C[obs_cam_idx]
        P_o = pts[obs_pt_idx]
        diff = P_o - C_o
        Rdiff = np.einsum("mij,mj->mi", R_o, diff)
        den = Rdiff[:, 2]
        den_safe = np.where(np.abs(den) < 1e-9, 1e-9, den)
        x_pred = cx - f_px * Rdiff[:, 0] / den_safe
        y_pred = cy - f_px * Rdiff[:, 1] / den_safe
        reproj = np.stack([x_pred - obs_uv[:, 0],
                           y_pred - obs_uv[:, 1]], axis=1)

        rtk_res = rtk_w_sqrt * (C - rtk_priors)

        return np.concatenate([reproj.ravel(), rtk_res.ravel()])

    return residuals


# ---------------------------------------------------------------------------
# Jacobian sparsity
# ---------------------------------------------------------------------------
def _build_jacobian_sparsity(n_cam: int, n_pts: int,
                             obs_cam_idx: np.ndarray,
                             obs_pt_idx: np.ndarray) -> csr_matrix:
    """trf 알고리즘에 넘길 binary sparsity 행렬.

    각 reprojection residual (2개) 은 자기 카메라 6개 + 자기 3D 점 3개에만
    의존. 각 RTK residual (3개) 은 자기 카메라 위치 1개에만 의존. → 극도로 희소.

    원본은 ``lil_matrix`` 에 fancy indexing 을 18번 반복했는데, 관측 수십만
    규모에서 이 구성 자체가 수십 초 걸린다. COO 트리플렛을 한 번에 만들어
    ``tocsr()`` 하면 벡터 연산 몇 번으로 끝난다.
    """
    M = int(obs_cam_idx.shape[0])
    n_params = n_cam * 6 + n_pts * 3
    n_res = 2 * M + 3 * n_cam
    pt_offset = n_cam * 6

    rows_even = np.arange(M, dtype=np.int64) * 2

    # --- reprojection: 카메라 6개 파라미터 ---
    cam_cols = obs_cam_idx[:, None] * 6 + np.arange(6)[None, :]      # (M, 6)
    # --- reprojection: 3D 점 3개 파라미터 ---
    pt_cols = pt_offset + obs_pt_idx[:, None] * 3 + np.arange(3)[None, :]  # (M, 3)
    cols_per_obs = np.concatenate([cam_cols, pt_cols], axis=1)       # (M, 9)

    k = cols_per_obs.shape[1]
    rows = np.repeat(rows_even, k)
    cols = cols_per_obs.ravel()
    # residual 이 (x, y) 두 개이므로 각 행의 짝을 함께 채운다.
    rows = np.concatenate([rows, rows + 1])
    cols = np.concatenate([cols, cols])

    # --- RTK prior: 자기 카메라 위치 3개 ---
    cam_ids = np.arange(n_cam, dtype=np.int64)
    rtk_rows = 2 * M + (cam_ids[:, None] * 3 + np.arange(3)[None, :])   # (n_cam, 3)
    rtk_cols = cam_ids[:, None] * 6 + np.arange(3)[None, :]
    rows = np.concatenate([rows, rtk_rows.ravel()])
    cols = np.concatenate([cols, rtk_cols.ravel()])

    data = np.ones(rows.shape[0], dtype=np.uint8)
    return coo_matrix((data, (rows, cols)), shape=(n_res, n_params)).tocsr()


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------
def rtk_constrained_bundle_adjustment(
    initial_cameras: np.ndarray,        # (n_cam, 6) [Xc,Yc,Zc, ω,φ,κ]
    initial_points: np.ndarray,         # (n_pts, 3)
    observations: list[tuple[int, int, np.ndarray]],
    rtk_priors: np.ndarray,             # (n_cam, 3) RTK 측정 카메라 위치
    rtk_weights: np.ndarray,            # (n_cam, 3) 1/σ² 가중치
    f_px: float, cx: float, cy: float,
    max_nfev: int = 300,
    verbose: int = 0,
):
    """RTK 좌표를 카메라 위치의 사전확률(prior)로 묶는 번들 조정.

    Returns
    -------
    cams_opt : (n_cam, 6) 최적화된 외부표정 (절대 투영좌표계).
    pts_opt : (n_pts, 3) 최적화된 3D 점.
    rmse_px : 재투영 RMSE (픽셀). RTK 잔차와 huber 감쇠를 제외한 순수 값.
    """
    initial_cameras = np.asarray(initial_cameras, dtype=np.float64)
    initial_points = np.asarray(initial_points, dtype=np.float64)
    rtk_priors = np.asarray(rtk_priors, dtype=np.float64)
    rtk_weights = np.asarray(rtk_weights, dtype=np.float64)

    n_cam = len(initial_cameras)
    n_pts = len(initial_points)
    if not observations:
        raise ValueError("observations 가 비어 있음")

    # ---- 관측 평탄화 ----------------------------------------------------
    obs_cam_idx = np.fromiter((o[0] for o in observations),
                              dtype=np.int64, count=len(observations))
    obs_pt_idx = np.fromiter((o[1] for o in observations),
                             dtype=np.int64, count=len(observations))
    obs_uv = np.asarray([o[2] for o in observations], dtype=np.float64)
    M = len(obs_cam_idx)

    if obs_cam_idx.max(initial=-1) >= n_cam or obs_pt_idx.max(initial=-1) >= n_pts:
        raise ValueError("observation 인덱스가 카메라/점 배열 범위를 벗어남")

    # ---- 좌표 중심화 (수치 안정) ----------------------------------------
    # 회전각(3열~5열)은 건드리지 않는다. 위치 3열만 이동.
    origin = initial_cameras[:, :3].mean(axis=0)
    cams0 = initial_cameras.copy()
    cams0[:, :3] -= origin
    pts0 = initial_points - origin
    priors0 = rtk_priors - origin

    rtk_w_sqrt = np.sqrt(rtk_weights)

    # ---- residual 콜백 선택 ---------------------------------------------
    use_gpu = HAS_CUPY and M >= _GPU_MIN_OBS
    build = _build_residuals_gpu if use_gpu else _build_residuals_np
    residuals = build(n_cam, n_pts, obs_cam_idx, obs_pt_idx, obs_uv,
                      priors0, rtk_w_sqrt, f_px, cx, cy)

    # ---- 희소 jacobian ---------------------------------------------------
    t0 = time.perf_counter()
    J_sparsity = _build_jacobian_sparsity(n_cam, n_pts, obs_cam_idx, obs_pt_idx)
    n_params = n_cam * 6 + n_pts * 3
    density = J_sparsity.nnz / (J_sparsity.shape[0] * J_sparsity.shape[1])
    logger.info("BA 준비: 카메라 %d, 점 %d, 관측 %d, 파라미터 %d, "
                "jacobian 밀도 %.2e (%s, %s)",
                n_cam, n_pts, M, n_params, density,
                "CuPy" if use_gpu else "numpy",
                fmt_elapsed(time.perf_counter() - t0))

    x0 = np.concatenate([cams0.ravel(), pts0.ravel()])

    # ---- 초기 상태 진단 --------------------------------------------------
    # 자세 초기값이 틀리면 (예: 광축이 하늘을 향하면) 여기서 바로 드러난다.
    r0 = residuals(x0)
    reproj0 = r0[: 2 * M].reshape(M, 2)
    rmse0 = float(np.sqrt(np.mean(np.sum(reproj0 ** 2, axis=1))))
    logger.info("BA 시작 전 재투영 RMSE: %.1f px", rmse0)
    if rmse0 > 500:
        logger.warning(
            "초기 재투영 오차가 %.0f px 로 비정상적으로 큽니다. 짐벌 각 → "
            "(ω, φ, κ) 변환 규약을 확인하세요 — 광축이 반대를 향하면 이 값이 "
            "수천 px 가 되고 BA 가 엉뚱한 해로 수렴합니다.", rmse0,
        )

    # ---- 2단계 스케줄 ---------------------------------------------------
    # ★ 원본의 loss="huber", f_scale=2.0 단일 호출은 초기 오차가 몇 px 를
    #   넘는 순간 무너진다. huber 는 |r| > f_scale 인 잔차의 기울기를 눌러
    #   버리므로, 초기 재투영 오차가 185 px 이면 **모든 관측이 이상점으로
    #   취급되어** 기울기가 소멸하고 최적화가 첫 몇 스텝에서 멈춘다.
    #   (실측: 원본은 84초를 쓰고도 카메라가 0.000 m 움직였다.)
    #
    #   삼각측량 초기값은 보통 수십~수백 px 오차를 갖는 것이 정상이므로,
    #   1단계는 잔차 분포에 맞춘 느슨한 f_scale 로 해를 끌어오고,
    #   2단계에서 huber 2 px 로 조여 진짜 오매칭만 걷어낸다.
    stage1_scale = max(4.0 * float(np.median(np.sqrt(
        np.sum(reproj0 ** 2, axis=1)))), 8.0)

    t0 = time.perf_counter()
    common = dict(jac_sparsity=J_sparsity, method="trf",
                  x_scale="jac", verbose=verbose)

    result = least_squares(residuals, x0, loss="soft_l1",
                           f_scale=stage1_scale,
                           max_nfev=max_nfev, **common)
    logger.info("- BA 1단계 (soft_l1, f_scale=%.1f px): %s (nfev=%d)",
                stage1_scale, fmt_elapsed(time.perf_counter() - t0), result.nfev)

    t0 = time.perf_counter()
    result = least_squares(residuals, result.x, loss="huber",
                           f_scale=2.0,
                           max_nfev=max_nfev, **common)
    logger.info("- BA 2단계 (huber, f_scale=2.0 px): %s (nfev=%d, status=%d)",
                fmt_elapsed(time.perf_counter() - t0),
                result.nfev, result.status)

    cams_opt = result.x[: n_cam * 6].reshape(n_cam, 6).copy()
    pts_opt = result.x[n_cam * 6:].reshape(n_pts, 3).copy()

    # ---- 순수 재투영 RMSE (RTK 잔차 / huber 감쇠 제외) -------------------
    r = residuals(result.x)
    reproj = r[: 2 * M].reshape(M, 2)
    per_obs = np.sqrt(np.sum(reproj ** 2, axis=1))
    rmse_px = float(np.sqrt(np.mean(per_obs ** 2)))

    # ---- 중심화 해제 -----------------------------------------------------
    cams_opt[:, :3] += origin
    pts_opt += origin

    rtk_shift = np.linalg.norm(cams_opt[:, :3] - rtk_priors, axis=1)
    logger.info("BA 완료. 재투영 RMSE %.2f px (중앙값 %.2f, 95%% %.2f) | "
                "RTK prior 대비 카메라 이동: 중앙값 %.3f m, 최대 %.3f m",
                rmse_px, float(np.median(per_obs)),
                float(np.percentile(per_obs, 95)),
                float(np.median(rtk_shift)), float(rtk_shift.max()))

    if use_gpu:
        free_gpu_memory()
    return cams_opt, pts_opt, rmse_px


__all__ = ["rtk_constrained_bundle_adjustment"]

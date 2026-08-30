"""RTK 고정 + 자세 prior Bundle Adjustment.

실측이 드러낸 발산 (380장, 그린환경센터)
----------------------------------------
```
누적 자세 변화: 중앙값 10.728°, 최대 20.758°
RTK prior 대비 카메라 이동: 중앙값 1.823 m, 최대 8.529 m
```

* RTK σ_xy 는 **1.1 cm** 인데 카메라가 1.823 m 움직였습니다 — **166σ**.
* 짐벌 자세는 ±1° 수준인데 **10.7°** 가 변했습니다.
* 초기 재투영 오차 110.7 px = 0.94° 이므로, BA 가 고쳐야 할 양은 1° 미만입니다.
  10.7° 는 고친 것이 아니라 **달아난 것**입니다.

왜 달아나나
-----------
잔차 개수가 압도적으로 비대칭입니다::

    영상 잔차 1,799,364개  vs  RTK 잔차 1,140개   (1578 : 1)

게다가 2장짜리 track 이 55% (147,680/269,615) 라 블록을 붙잡는 힘이 약합니다.
그 결과 **블록 전체가 회전·변형**하면서 RTK 를 크게 위반해도, 영상 잔차가
조금만 줄면 최적화는 그쪽으로 갑니다. 자유도가 남아 있으면 반드시 그렇게 됩니다.

연쇄 피해도 확인됐습니다: 점군 Z 가 72.9~212.3 m 로 퍼져 평면 RANSAC inlier
비율이 23.7% 로 실패했고(→ LRF 폴백), DSM 은 점 밀도 부족으로 폐기됐습니다.

이 모듈의 처방
--------------
1. **카메라 위치 고정** (``fix_positions=True``, 기본).
   RTK Fixed 의 1.1 cm 는 BA 가 영상만으로 도달할 수 있는 정확도보다 훨씬
   좋습니다. **추정할 이유가 없습니다.** 위치를 파라미터에서 빼면
   RTK 잔차 자체가 사라져 가중치 균형 문제가 없어지고, 블록의 평행이동·
   축척 자유도가 원천적으로 제거됩니다. 파라미터도 20% 이상 줄어듭니다.

2. **자세 prior** (``attitude_sigma_deg``, 기본 2°).
   짐벌 자세는 절대 정확도가 ±1~2° 이지 20° 가 아닙니다. 초기값에서 이만큼
   벗어나면 벌점을 줍니다. 이것이 관측이 적은 가장자리 프레임이 소수의
   오매칭에 끌려가는 것을 막습니다.

3. **prior 잔차를 robust loss 에서 보호**.
   ``scipy.optimize.least_squares`` 의 ``loss`` 는 잔차 벡터 전체에
   적용됩니다. 느슨한 단계의 ``f_scale`` 은 수백 px 이므로, 그대로 두면
   자세 prior 위반도 "작은 값" 으로 취급돼 무력해집니다. prior 잔차를
   ``f_scale`` 기준으로 미리 스케일해 항상 유효하게 만듭니다.

위치를 정말 풀어야 한다면 ``fix_positions=False`` 로 되돌릴 수 있지만,
그때는 ``rtk_weights`` 가 실제로 지켜지는지 로그의 "카메라 이동" 을 반드시
확인해야 합니다.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csr_matrix

from ..geometry import rotation_matrices_batch_np
from ..utils import fmt_elapsed

logger = logging.getLogger(__name__)

__all__ = ["rtk_fixed_bundle_adjustment"]


def _sparsity(n_cam: int, n_pts: int, obs_cam: np.ndarray, obs_pt: np.ndarray,
              n_cam_par: int) -> csr_matrix:
    """(reprojection + 자세 prior) 잔차의 희소 패턴."""
    M = int(obs_cam.shape[0])
    n_par = n_cam * n_cam_par + n_pts * 3
    n_res = 2 * M + 3 * n_cam
    pt_off = n_cam * n_cam_par

    rows_e = np.arange(M, dtype=np.int64) * 2
    cam_cols = obs_cam[:, None] * n_cam_par + np.arange(n_cam_par)[None, :]
    pt_cols = pt_off + obs_pt[:, None] * 3 + np.arange(3)[None, :]
    cols = np.concatenate([cam_cols, pt_cols], axis=1)
    k = cols.shape[1]
    r = np.repeat(rows_e, k)
    c = cols.ravel()
    rows = np.concatenate([r, r + 1])
    cols_all = np.concatenate([c, c])

    # 자세 prior: 자기 카메라의 각 3개.
    ids = np.arange(n_cam, dtype=np.int64)
    a_rows = 2 * M + (ids[:, None] * 3 + np.arange(3)[None, :])
    ang_off = n_cam_par - 3          # 위치 고정이면 0, 아니면 3
    a_cols = ids[:, None] * n_cam_par + ang_off + np.arange(3)[None, :]
    rows = np.concatenate([rows, a_rows.ravel()])
    cols_all = np.concatenate([cols_all, a_cols.ravel()])

    data = np.ones(rows.shape[0], dtype=np.uint8)
    return coo_matrix((data, (rows, cols_all)), shape=(n_res, n_par)).tocsr()


def rtk_fixed_bundle_adjustment(initial_cameras: np.ndarray,
                                initial_points: np.ndarray,
                                observations: list,
                                f_px: float, cx: float, cy: float,
                                *,
                                fix_positions: bool = True,
                                attitude_sigma_deg: float = 1.0,
                                attitude_sigma_weak_deg: float | None = None,
                                weak_obs_ratio: float = 0.30,
                                max_nfev: int = 200,
                                verbose: int = 0):
    """카메라 위치를 RTK 에 고정하고 자세 + 점만 최적화.

    Returns ``(cams_opt, pts_opt, rmse_px)`` — 기존 API 와 동일한 형태.
    """
    cams0 = np.asarray(initial_cameras, dtype=np.float64).copy()
    pts0 = np.asarray(initial_points, dtype=np.float64).copy()
    n_cam, n_pts = len(cams0), len(pts0)
    if not observations:
        raise ValueError("observations 가 비어 있음")

    obs_cam = np.fromiter((o[0] for o in observations), dtype=np.int64,
                          count=len(observations))
    obs_pt = np.fromiter((o[1] for o in observations), dtype=np.int64,
                         count=len(observations))
    obs_uv = np.asarray([o[2] for o in observations], dtype=np.float64)
    M = len(obs_cam)

    origin = cams0[:, :3].mean(axis=0)
    C_fixed = cams0[:, :3] - origin
    P0 = pts0 - origin
    ang0 = cams0[:, 3:6].copy()

    n_cam_par = 3 if fix_positions else 6
    # ★ 관측수에 따라 자세 prior 를 차등 적용 (실측 근거)
    #   프레임당 관측 중앙값 820 인데 하위 10% 는 92 개 (11%) 뿐이고
    #   tie point 가 0 인 프레임도 2 장 있었다. 이런 가장자리 프레임은
    #   자세가 사실상 prior 로만 결정되는데, σ=1° 는 지상 0.78 m 에
    #   해당한다. 실측 바깥 어긋남 0.475 m 와 같은 자릿수다.
    #
    #   관측이 충분한 프레임은 영상이 자세를 잡아 주므로 prior 를 느슨히
    #   둬도 되고, 관측이 적은 프레임은 짐벌 값을 더 믿어야 한다.
    #   → 관측수가 중앙값의 weak_obs_ratio 미만인 프레임에 더 작은 σ 적용.
    obs_cnt = np.bincount(obs_cam, minlength=n_cam).astype(float)
    med_obs = max(float(np.median(obs_cnt[obs_cnt > 0])), 1.0)
    sig_vec = np.full(n_cam, np.radians(max(attitude_sigma_deg, 1e-3)))
    # ★ 차등 prior 는 실험 결과 **역효과**였다. 관측이 적은 프레임의 σ 를
    #   0.3° 로 조이니 자세오차가 0.366° → 0.748° 로 나빠졌다. 짐벌 초기값
    #   자체에 0.9° 오차가 있어서, 거기에 단단히 묶으면 소수의 영상 관측이
    #   해 주던 보정마저 막힌다. 기본값은 차등 없음(동일 σ)으로 둔다.
    if attitude_sigma_weak_deg is None:
        attitude_sigma_weak_deg = attitude_sigma_deg
    weak = obs_cnt < weak_obs_ratio * med_obs
    if weak.any() and abs(attitude_sigma_weak_deg - attitude_sigma_deg) > 1e-9:
        sig_vec[weak] = np.radians(max(attitude_sigma_weak_deg, 1e-3))
        logger.info("  자세 prior 차등: 관측 부족 %d장에 σ=%.2f° "
                    "(나머지 %.2f°) — 가장자리 프레임이 소수 오매칭에 "
                    "끌려가는 것을 막습니다", int(weak.sum()),
                    attitude_sigma_weak_deg, attitude_sigma_deg)
    sig_att = sig_vec[:, None]

    def unpack(x):
        cp = x[:n_cam * n_cam_par].reshape(n_cam, n_cam_par)
        pts = x[n_cam * n_cam_par:].reshape(n_pts, 3)
        if fix_positions:
            return C_fixed, cp, pts
        return cp[:, :3], cp[:, 3:6], pts

    # prior 잔차 스케일 — robust loss 의 f_scale 에 묻히지 않도록 보정한다.
    prior_gain = [1.0]

    def residuals(x):
        C, ang, pts = unpack(x)
        R = rotation_matrices_batch_np(ang)
        diff = pts[obs_pt] - C[obs_cam]
        Rd = np.einsum("mij,mj->mi", R[obs_cam], diff)
        den = Rd[:, 2]
        den_s = np.where(np.abs(den) < 1e-9, 1e-9, den)
        xp = cx - f_px * Rd[:, 0] / den_s
        yp = cy - f_px * Rd[:, 1] / den_s
        reproj = np.stack([xp - obs_uv[:, 0], yp - obs_uv[:, 1]], axis=1)
        # ★ 각도 wrap 필수. nadir 정북은 ω ≈ −180° 라 prior 와 값이 ±180°
        #   경계를 넘나든다. wrap 없이 빼면 0.2° 차이가 359.8° 로 계산돼
        #   prior 가 프레임을 정반대로 밀어낸다.
        d = (ang - ang0 + np.pi) % (2.0 * np.pi) - np.pi
        att = d / sig_att * prior_gain[0]
        return np.concatenate([reproj.ravel(), att.ravel()])

    x0 = np.concatenate([
        (ang0 if fix_positions
         else np.hstack([C_fixed, ang0])).ravel(),
        P0.ravel()])

    J = _sparsity(n_cam, n_pts, obs_cam, obs_pt, n_cam_par)
    r0 = residuals(x0)
    rep0 = r0[:2 * M].reshape(M, 2)
    rmse0 = float(np.sqrt(np.mean(np.sum(rep0 ** 2, axis=1))))
    logger.info("BA(RTK고정): 카메라 %d(자세만), 점 %d, 관측 %d, 파라미터 %d | "
                "시작 재투영 RMSE %.1f px", n_cam, n_pts, M,
                n_cam * n_cam_par + n_pts * 3, rmse0)

    med = float(np.median(np.sqrt(np.sum(rep0 ** 2, axis=1))))
    stage1 = max(4.0 * med, 8.0)
    common = dict(jac_sparsity=J, method="trf", x_scale="jac", verbose=verbose)

    # ★ prior 잔차가 robust loss 의 선형 구간에 묻히지 않도록, f_scale 에
    #   비례해 키운다. 이렇게 하면 "자세를 σ 만큼 어기는 비용" 이 "관측 하나를
    #   f_scale 만큼 어기는 비용" 과 같은 척도가 된다.
    t0 = time.perf_counter()
    prior_gain[0] = stage1
    res = least_squares(residuals, x0, loss="soft_l1", f_scale=stage1,
                        max_nfev=max_nfev, **common)
    logger.info("- BA 1단계 (soft_l1, f_scale=%.1f px): %s (nfev=%d)",
                stage1, fmt_elapsed(time.perf_counter() - t0), res.nfev)

    t0 = time.perf_counter()
    prior_gain[0] = 2.0
    res = least_squares(residuals, res.x, loss="huber", f_scale=2.0,
                        max_nfev=max_nfev, **common)
    logger.info("- BA 2단계 (huber, f_scale=2.0 px): %s (nfev=%d, status=%d)",
                fmt_elapsed(time.perf_counter() - t0), res.nfev, res.status)

    C, ang, pts = unpack(res.x)
    cams_opt = np.hstack([C + origin, ang])
    pts_opt = pts + origin

    r = residuals(res.x)
    per = np.sqrt(np.sum(r[:2 * M].reshape(M, 2) ** 2, axis=1))
    rmse = float(np.sqrt(np.mean(per ** 2)))

    d_att = np.degrees(np.linalg.norm(
        (ang - ang0 + np.pi) % (2.0 * np.pi) - np.pi, axis=1))
    logger.info("BA 완료. 재투영 RMSE %.2f px (중앙값 %.2f, 95%% %.2f) | "
                "자세 변화 중앙값 %.3f°, 최대 %.3f° | 카메라 위치 %s",
                rmse, float(np.median(per)), float(np.percentile(per, 95)),
                float(np.median(d_att)), float(d_att.max()),
                "RTK 고정" if fix_positions else "자유")
    if d_att.max() > 3.0 * attitude_sigma_deg:
        logger.warning("  자세가 prior σ(%.1f°) 의 3배를 넘게 움직인 프레임이 "
                       "있습니다 (최대 %.2f°). 매칭 이상치를 의심하세요.",
                       attitude_sigma_deg, float(d_att.max()))
    return cams_opt, pts_opt, rmse

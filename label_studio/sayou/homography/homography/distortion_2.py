"""방사 왜곡 자기보정 — 재투영 잔차에서 k1, k2 를 직접 추정한다.

왜 필요한가 (실측 근거)
-----------------------
BA 가 정상화되고 초점거리도 수렴한 뒤에도 **재투영 잔차가 바닥을 치지
않았습니다.**

```
중간 단계 후: RMSE 5.21 px, 중앙값 2.66 px, 95% 10.15 px
정밀 게이트 3 px → 268,059 → 133,011 (재투영으로 89,740개 제거)
```

RMSE(꼬리)는 9.43 → 5.21 로 줄었는데 **중앙값은 2.69 → 2.66 으로 거의 그대로**
였습니다. 이상치를 걷어내도 남는 바닥이 있다는 뜻입니다.

그 바닥의 정체는 **미보정 렌즈 왜곡**입니다. 이 데이터의 XMP 에는
``DewarpData`` 가 없고, plumb-line 으로 잰 ``k1 ≈ −0.017`` 을 넣어
계산하면 (``f_px = 7045``):

| 이미지 반경 | 왜곡량 |
|---|---|
| 1000 px | 0.34 px |
| 2000 px | 2.74 px |
| 3240 px (모서리) | 11.65 px |

**잔차 중앙값 2.66 px 와 95% 10.15 px 가 각각 중간반경·모서리 왜곡량과
정확히 맞습니다.** BA 가 더 내려갈 수 없었던 게 아니라 모델에 그 항이
없었던 것입니다.

왜 이제야 보이나
----------------
앞서 plumb-line 으로 k1 을 쟀을 때는 "모서리에서 지상 0.08 m 밖에 안 되니
무시해도 된다" 고 판단했습니다. **지상 거리로는 작지만 픽셀로는 12 px 이고,
그게 3 px 게이트를 통과하지 못하게 만드는 크기**라는 점을 놓쳤습니다.
왜곡의 진짜 피해는 정사영상 위치오차가 아니라 **BA 가 쓸 수 있는 점의 수**
였습니다.

추정 방법
---------
BA 해에서 각 관측의 재투영 잔차를 그 관측의 **반경 방향 성분**으로 투영하면,
왜곡은 반경의 홀수 멱급수로 나타납니다:

    Δr(r) = k1·r³/f² + k2·r⁵/f⁴

관측이 수십만 개이므로 이 1~2개 계수는 매우 안정적으로 풀립니다. 접선
왜곡은 이 렌즈에서 무시할 수준이라 넣지 않습니다 (필요하면 확장).

추정한 계수로 **관측 좌표를 보정**한 뒤 삼각측량·BA 를 다시 돌리면 잔차
바닥이 사라지고, 같은 게이트로 훨씬 많은 점이 살아남습니다.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["diagnose_residual_vs_radius", "estimate_radial_distortion",
           "undistort_observations"]


def estimate_radial_distortion(observations,
                               cams: np.ndarray,
                               points: np.ndarray,
                               f_px: float, cx: float, cy: float,
                               *,
                               use_k2: bool = True,
                               max_abs_resid_px: float = 40.0,
                               min_obs: int = 5000,
                               clip_k1: float = 0.30
                               ) -> tuple[float, float, dict] | None:
    """재투영 잔차의 반경 성분에서 ``(k1, k2)`` 추정.

    Returns ``(k1, k2, 진단 dict)`` 또는 ``None``.

    모델은 ``r_observed = r_ideal · (1 + k1·(r/f)² + k2·(r/f)⁴)`` 이고,
    잔차 ``Δr = r_obs − r_pred`` 를 ``r`` 에 대해 회귀한다.
    """
    from ..geometry import rotation_matrix

    if len(observations) < min_obs:
        return None

    obs_cam = np.fromiter((o[0] for o in observations), dtype=np.int64,
                          count=len(observations))
    obs_pt = np.fromiter((o[1] for o in observations), dtype=np.int64,
                         count=len(observations))
    obs_uv = np.asarray([o[2] for o in observations], dtype=np.float64)

    # 예측 픽셀 계산 (공선조건, project_point 와 동일 부호 규약).
    R_all = np.stack([rotation_matrix(*c[3:6]) for c in cams])
    diff = points[obs_pt] - cams[obs_cam, :3]
    Rd = np.einsum("mij,mj->mi", R_all[obs_cam], diff)
    den = Rd[:, 2]
    ok = np.abs(den) > 1e-9
    if ok.sum() < min_obs:
        return None
    den = np.where(ok, den, 1.0)
    u_pred = cx - f_px * Rd[:, 0] / den
    v_pred = cy - f_px * Rd[:, 1] / den

    du = obs_uv[:, 0] - u_pred
    dv = obs_uv[:, 1] - v_pred
    resid = np.hypot(du, dv)

    # 예측점의 반경과 반경 단위벡터.
    px = u_pred - cx; py = v_pred - cy
    r = np.hypot(px, py)
    good = ok & (r > 50.0) & (resid < max_abs_resid_px)
    if good.sum() < min_obs:
        return None

    ur = px[good] / r[good]
    vr = py[good] / r[good]
    d_rad = du[good] * ur + dv[good] * vr        # 잔차의 반경 성분
    rr = r[good]

    x = rr / f_px
    if use_k2:
        A = np.stack([rr * x ** 2, rr * x ** 4], axis=1)
    else:
        A = (rr * x ** 2)[:, None]

    # IRLS 로 이상치 억제.
    w = np.ones(len(rr))
    coef = np.zeros(A.shape[1])
    for _ in range(4):
        Aw = A * w[:, None]
        coef, *_ = np.linalg.lstsq(Aw, d_rad * w, rcond=None)
        res = d_rad - A @ coef
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        w = np.minimum(1.0, 2.0 * s / np.maximum(np.abs(res), 1e-9))

    k1 = float(coef[0])
    k2 = float(coef[1]) if use_k2 else 0.0
    if not np.isfinite(k1) or abs(k1) > clip_k1:
        logger.warning("방사 왜곡 추정 기각: k1=%.4f 가 타당 범위 ±%.2f 밖",
                       k1, clip_k1)
        return None

    before = float(np.median(np.abs(d_rad)))
    after = float(np.median(np.abs(d_rad - A @ coef)))
    r_corner = float(np.percentile(rr, 99))
    info = {
        "k1": k1, "k2": k2,
        "n_obs": int(good.sum()),
        "radial_resid_median_before_px": before,
        "radial_resid_median_after_px": after,
        "corner_distortion_px": float(
            abs(k1 * r_corner ** 3 / f_px ** 2
                + k2 * r_corner ** 5 / f_px ** 4)),
    }
    logger.info("방사 왜곡 추정: k1=%+.5f, k2=%+.5f (관측 %d개) — "
                "반경 잔차 중앙값 %.2f → %.2f px, 모서리 왜곡 %.1f px",
                k1, k2, info["n_obs"], before, after,
                info["corner_distortion_px"])
    if after > before * 0.8:
        logger.warning("  왜곡 모델로 설명되는 부분이 적습니다 "
                       "(%.2f → %.2f px). 잔차의 주원인이 왜곡이 아닐 수 "
                       "있으니 적용 결과를 확인하세요.", before, after)
    return k1, k2, info


def undistort_observations(observations, k1: float, k2: float,
                           f_px: float, cx: float, cy: float):
    """관측 픽셀 좌표에서 방사 왜곡을 제거한 새 관측 리스트.

    ``r_ideal = r_obs / (1 + k1·(r/f)² + k2·(r/f)⁴)`` 를 한 번의 고정점
    반복으로 푼다 (이 크기의 왜곡에서는 2회면 수렴).
    """
    out = []
    for ci, pi, uv in observations:
        dx = uv[0] - cx; dy = uv[1] - cy
        r = np.hypot(dx, dy)
        if r < 1e-6:
            out.append((ci, pi, uv))
            continue
        ri = r
        for _ in range(2):
            x = ri / f_px
            ri = r / (1.0 + k1 * x * x + k2 * x ** 4)
        s = ri / r
        out.append((ci, pi, np.array([cx + dx * s, cy + dy * s])))
    return out


def diagnose_residual_vs_radius(observations, cams, points,
                                f_px: float, cx: float, cy: float,
                                *, n_bins: int = 10, max_obs: int = 400_000):
    """재투영 잔차를 **이미지 반경 구간별**로 나눠 보고한다.

    이 한 줄짜리 진단이 "잔차 바닥이 렌즈 왜곡인가" 를 결정한다:

    * 왜곡이면 잔차가 반경의 **3승**으로 자란다 (중심 ≈ 0, 모서리에서 최대).
    * 오매칭·자세오차면 반경과 **무관**하게 고르게 퍼진다.

    원격으로 추측할 필요 없이 다음 실행 로그에서 바로 갈립니다.
    """
    from ..geometry import rotation_matrix

    if not observations:
        return None
    n = len(observations)
    step = max(1, n // max_obs)
    obs = observations[::step]

    obs_cam = np.fromiter((o[0] for o in obs), dtype=np.int64, count=len(obs))
    obs_pt = np.fromiter((o[1] for o in obs), dtype=np.int64, count=len(obs))
    obs_uv = np.asarray([o[2] for o in obs], dtype=np.float64)

    R_all = np.stack([rotation_matrix(*c[3:6]) for c in cams])
    diff = points[obs_pt] - cams[obs_cam, :3]
    Rd = np.einsum("mij,mj->mi", R_all[obs_cam], diff)
    den = Rd[:, 2]
    ok = np.abs(den) > 1e-9
    den = np.where(ok, den, 1.0)
    u = cx - f_px * Rd[:, 0] / den
    v = cy - f_px * Rd[:, 1] / den
    resid = np.hypot(obs_uv[:, 0] - u, obs_uv[:, 1] - v)
    r = np.hypot(u - cx, v - cy)
    m = ok & np.isfinite(resid) & (resid < 200.0)
    if m.sum() < 1000:
        return None
    r, resid = r[m], resid[m]

    edges = np.linspace(0.0, np.percentile(r, 99.5), n_bins + 1)
    rows, meds, ctrs = [], [], []
    for i in range(n_bins):
        sel = (r >= edges[i]) & (r < edges[i + 1])
        if sel.sum() < 200:
            continue
        c_ = 0.5 * (edges[i] + edges[i + 1])
        md = float(np.median(resid[sel]))
        ctrs.append(c_); meds.append(md)
        rows.append(f"    r {edges[i]:6.0f}~{edges[i+1]:6.0f} px : "
                    f"중앙값 {md:6.2f} px  (관측 {int(sel.sum()):,})")
    if len(meds) < 3:
        return None

    ctrs = np.array(ctrs); meds = np.array(meds)
    # ★ 절편(반경 무관 바닥) 을 반드시 함께 적합해야 한다.
    #   잔차 = '오매칭·잡음이 만드는 바닥' + '왜곡이 만드는 r³ 성분' 의 합인데,
    #   절편 없이 r³ 만 맞추면 바닥까지 r³ 로 설명하려다 적합이 무너진다.
    #   실측에서 절편 없는 적합이 R²=0.14 로 "왜곡 아님" 이라 오판했고,
    #   절편을 넣자 R²=0.985, 바닥 1.72 px + k1=+0.0107 로 깨끗하게 갈렸다.
    A = np.stack([np.ones_like(ctrs), ctrs ** 3 / f_px ** 2], axis=1)
    coef = np.linalg.lstsq(A, meds, rcond=None)[0]
    floor_px = float(coef[0]); k1_imp = float(coef[1])
    pred = A @ coef
    ss_res = float(np.sum((meds - pred) ** 2))
    ss_tot = float(np.sum((meds - meds.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    flat = meds.max() / max(meds.min(), 1e-6)
    r_max = float(ctrs.max())
    radial_at_max = abs(k1_imp) * r_max ** 3 / f_px ** 2

    logger.info("재투영 잔차의 반경 의존성 (왜곡 판별):")
    for line in rows:
        logger.info(line)
    logger.info("    → 바닥 %.2f px + 반경성분 (최대 반경에서 %.2f px), "
                "적합 R²=%.3f, 함의 k1=%+.5f, 최대/최소 비 %.1f배",
                floor_px, radial_at_max, r2, k1_imp, flat)
    distorted = (r2 > 0.7 and radial_at_max > max(0.5 * floor_px, 0.8))
    if distorted:
        logger.warning("    ★ 반경 성분이 뚜렷합니다 — **미보정 렌즈 왜곡**이 "
                       "확인됩니다. 바닥(%.2f px)은 오매칭·잡음이고, 그 위에 "
                       "얹힌 %.2f px 가 왜곡입니다. 왜곡을 펴면 같은 게이트로 "
                       "훨씬 많은 점이 살아남습니다.", floor_px, radial_at_max)
    else:
        logger.info("    잔차가 반경과 거의 무관합니다 — 왜곡이 아니라 "
                    "오매칭/자세오차가 주원인입니다.")
    return {"radii": ctrs.tolist(), "median_px": meds.tolist(),
            "max_over_min": flat, "fit_r2": r2, "implied_k1": k1_imp,
            "floor_px": floor_px, "radial_at_max_px": radial_at_max,
            "distortion_detected": bool(distorted)}

"""RTK 유도 tie point 매칭 — 반복 격자에서 track 기아를 푸는 모듈.

실측 진단 (380 장, 그린환경센터)
--------------------------------
``summary.json``:

* 평면 적합 inlier **147 개 / 이미지 380 장** = 이미지당 0.4 개.
  고중복 380 장이면 수천~수만 개가 나와야 정상이다. **BA 점군이 사실상 비어
  있다.**
* 재투영 RMSE 0.58 px 는 관측이 극소수라 낮게 나온 값이지 자세가 잘 풀렸다는
  뜻이 아니다.
* 모자이크 정렬 보정이 60/380 장에만 적용됐고 **최대 0.999 m 로 상한(1.0 m)에
  정확히 잘렸다** — 실제 프레임 어긋남은 1 m 를 넘는다.

오차 예산으로 역산하면 1 m 를 만들 수 있는 것은 **자세오차 약 1.3°** 뿐이다
(RTK σ 1 cm → 0.01 m, 실측 렌즈왜곡 k1≈−0.017 → 모서리 0.08 m). 짐벌 보고값은
0.1° 단위지만 광축 절대 자세의 정확도는 그보다 훨씬 나쁘고, **그걸 잡는 것이
BA 의 역할인데 track 이 없어서 못 잡고 있다.**

왜 track 이 없나
----------------
태양광 패널은 셀 격자(실측 주기 0.23 m)와 모듈 행(2.72 m)이 반복된다. SIFT
descriptor 는 이웃한 셀끼리 거의 구별되지 않으므로 **Lowe ratio test 가
"1등과 2등이 비슷하다"는 이유로 올바른 매칭까지 전부 기각한다.** 반복 텍스처의
전형적인 실패이고, ratio 를 완화하면 이번엔 오매칭이 쏟아진다.

해법: RTK 로 탐색을 좁힌다
--------------------------
RTK 가 cm 급 위치를, 짐벌이 대략의 자세를 준다. 두 정보로 지상 평면을 거쳐
**이미지 i 의 점이 이미지 j 의 어디에 떨어져야 하는지 예측**할 수 있다
(``predict_correspondence``). 자세오차 1.3° 여도 예측 오차는 1 m 수준,
즉 픽셀로 150 px 정도다.

그러면 매칭을 **예측 위치 반경 안으로 제한**할 수 있다. 반경 안에는 보통
같은 셀 하나뿐이므로 모호성이 사라지고, ratio test 를 통과 못 하던 매칭이
살아난다. 이렇게 얻은 track 으로 BA 가 비로소 자세를 잡는다.

이 전략은 사용자가 앞서 독립적으로 검증한 결과와 일치한다 — 매칭 필터를
``max_dist 0.8 m`` 로 조인 실행이 프레임 간 잔차 RMS 를 1.197 → 0.334 m 로,
네트워크 조정 후 0.113 m 로 낮췄다.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "predict_correspondence",
    "guided_match_pair",
    "guided_build_tie_points",
    "guided_match_pair_fast",
    "guided_rematch",
    "merge_matches",
    "fit_points_plane",
]


def predict_correspondence(fh_i, fh_j, pts_i: np.ndarray) -> np.ndarray | None:
    """이미지 i 의 픽셀들이 이미지 j 에서 있어야 할 위치 (평면 경유).

    ``pts_i`` (N, 2) → ``pts_j_pred`` (N, 2). 두 프레임이 같은 지상 평면을
    공유하므로 합성 호모그래피 ``H_j ∘ H_i⁻¹`` 로 닫힌다.
    """
    g = fh_i.pixel_to_ground(np.asarray(pts_i, dtype=np.float64))
    if g is None or len(g) == 0:
        return None
    out = fh_j.ground_to_pixel(g[:, :2] if g.shape[1] > 2 else g)
    return out


def guided_match_pair(kp_i, desc_i, kp_j, desc_j,
                      fh_i, fh_j,
                      *,
                      search_radius_px: float = 200.0,
                      ratio: float = 0.85,
                      min_matches: int = 25,
                      ransac_thresh_px: float = 3.0):
    """RTK 예측으로 탐색을 좁힌 뒤 매칭.

    일반 매칭과 다른 점:

    1. 이미지 i 의 각 keypoint 에 대해 **예측 위치 반경 안의 후보만** 본다.
       반복 격자에서도 반경 안에는 보통 정답 하나뿐이라 모호성이 사라진다.
    2. 그래서 ``ratio`` 를 0.75 보다 **완화**해도 안전하다 (기본 0.85).
       완화 없이는 반복 텍스처에서 정답까지 기각된다.
    3. 마지막에 평행이동+회전(부분 아핀) RANSAC 으로 잔여 이상치를 제거한다.
       ``findFundamentalMat`` 은 거의 평면인 장면에서 퇴화하므로 쓰지 않는다.

    Returns ``(pts_i, pts_j, idx_i, idx_j)`` 또는 ``None``.
    """
    if desc_i is None or desc_j is None:
        return None
    if len(kp_i) < min_matches or len(kp_j) < min_matches:
        return None

    P_i = np.array([k.pt for k in kp_i], dtype=np.float64)
    P_j = np.array([k.pt for k in kp_j], dtype=np.float64)

    pred = predict_correspondence(fh_i, fh_j, P_i)
    if pred is None or not np.all(np.isfinite(pred)):
        return None

    # j 의 keypoint 를 격자에 넣어 반경 질의를 빠르게.
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(P_j)
        neigh = tree.query_ball_point(pred, r=search_radius_px)
    except ImportError:                                   # pragma: no cover
        neigh = [np.nonzero(np.hypot(P_j[:, 0] - p[0],
                                     P_j[:, 1] - p[1]) <= search_radius_px)[0]
                 for p in pred]

    di = np.asarray(desc_i, dtype=np.float32)
    dj = np.asarray(desc_j, dtype=np.float32)

    mi, mj = [], []
    for i, cand in enumerate(neigh):
        if len(cand) == 0:
            continue
        cand = np.asarray(cand, dtype=np.int64)
        d = np.linalg.norm(dj[cand] - di[i], axis=1)
        if len(d) == 1:
            # 후보가 하나뿐 — 모호성 자체가 없다. 거리 상한만 확인.
            if d[0] < 300.0:
                mi.append(i); mj.append(int(cand[0]))
            continue
        order = np.argsort(d)
        if d[order[0]] < ratio * d[order[1]]:
            mi.append(i); mj.append(int(cand[order[0]]))

    if len(mi) < min_matches:
        return None

    pts_i = P_i[mi].astype(np.float32)
    pts_j = P_j[mj].astype(np.float32)

    M, inl = cv2.estimateAffinePartial2D(
        pts_i, pts_j, method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh_px, maxIters=4000)
    if M is None or inl is None:
        return None
    inl = inl.ravel().astype(bool)
    if inl.sum() < min_matches:
        return None

    idx_i = np.array(mi, dtype=np.int64)[inl]
    idx_j = np.array(mj, dtype=np.int64)[inl]
    return pts_i[inl], pts_j[inl], idx_i, idx_j


def guided_build_tie_points(metas, pairs, frames, features,
                            *,
                            search_radius_px: float = 200.0,
                            ratio: float = 0.85,
                            min_matches: int = 25):
    """``features.pairs.build_tie_points`` 의 RTK 유도 버전.

    Parameters
    ----------
    frames : 프레임별 ``FrameHomography`` (RTK+짐벌+평면으로 만든 초기값).
        예측에만 쓰이므로 정확할 필요는 없고 1 m 수준이면 충분하다.
    features : ``{idx: (keypoints, descriptors, shape)}``.

    Returns
    -------
    ``matches`` dict — ``build_tracks`` 가 그대로 소비하는 형식.
    """
    matches = {}
    n_try = n_ok = 0
    counts = []
    for i, j in pairs:
        if i not in features or j not in features:
            continue
        kp_i, desc_i, _ = features[i]
        kp_j, desc_j, _ = features[j]
        n_try += 1
        res = guided_match_pair(kp_i, desc_i, kp_j, desc_j,
                                frames[i], frames[j],
                                search_radius_px=search_radius_px,
                                ratio=ratio, min_matches=min_matches)
        if res is None:
            continue
        matches[(i, j)] = res
        counts.append(len(res[0]))
        n_ok += 1

    if counts:
        logger.info("RTK 유도 매칭: %d/%d 쌍 성공, 쌍당 inlier 중앙값 %d개 "
                    "(총 %d개 대응)", n_ok, n_try, int(np.median(counts)),
                    int(np.sum(counts)))
    else:
        logger.warning("RTK 유도 매칭: 성공한 쌍이 없습니다 — 예측 반경(%.0f px)"
                       "이 너무 작거나 초기 자세가 크게 틀렸을 수 있습니다.",
                       search_radius_px)
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# 2차 안내 매칭 — 번들조정을 마친 자세로 예측
#
# ★ 1차(짐벌 자세) 예측으로는 안내 매칭이 성립하지 않았다. 예측 차이 중앙값이
#   EWP 열화상 35 px, Site-2-29719 열화상 41 px 였고, 예측 평면을 확정 기준면으로
#   바꿔도 35 → 35, 41 → 42 px 로 움직이지 않았다 — 지형이 아니라 **짐벌 자세 오차**
#   (두 프레임 사이 약 2°)와 옥상 시차가 지배했다. 반경을 줄무늬 주기의 절반
#   (64~85 px)보다 작게 잡을 수 없으니 반복 패널을 가를 수 없었다.
#   번들조정 뒤에는 재투영 RMSE 가 0.7 px 수준이라 자세 몫이 수 px 로 떨어진다.
#   그 자세로 예측하면 반경을 주기의 30% 정도로 좁혀도 맞는 짝이 반경 안에 든다.
# ─────────────────────────────────────────────────────────────────────────────

def guided_match_pair_fast(kp_i, desc_i, kp_j, desc_j, fh_i, fh_j, *,
                           radius_px: float, shape_j=None,
                           ratio: float = 0.85, min_matches: int = 20,
                           ransac_thresh_px: float = 4.0, k_max: int = 64,
                           chunk: int = 2048, pred=None):
    """``guided_match_pair`` 와 같은 판정을 배열 연산으로 (특징점마다 파이썬 반복 없음).

    반경 안 후보를 ``cKDTree.query(k=K, distance_upper_bound=r)`` 로 한꺼번에 뽑고,
    서술자 거리를 (N, K) 배열로 계산해 1등 · 2등 비율 검사를 한다. 같은 j 에 여러 i 가
    붙으면 거리가 가장 짧은 것만 남긴다.
    """
    if desc_i is None or desc_j is None or len(kp_i) < min_matches or len(kp_j) < min_matches:
        return None
    from scipy.spatial import cKDTree
    P_i = np.array([k.pt for k in kp_i], dtype=np.float64)
    P_j = np.array([k.pt for k in kp_j], dtype=np.float64)
    if pred is None:                                    # 미리 계산한 예측이 없으면 평면 하나로
        pred = predict_correspondence(fh_i, fh_j, P_i)
    if pred is None:
        return None
    pred = np.asarray(pred, dtype=np.float64)
    ok = np.all(np.isfinite(pred), axis=1)
    if shape_j is not None:
        h, w = shape_j[:2]
        ok &= (pred[:, 0] > -radius_px) & (pred[:, 0] < w + radius_px) & \
              (pred[:, 1] > -radius_px) & (pred[:, 1] < h + radius_px)
        area = float(w * h)
    else:
        area = float(np.ptp(P_j[:, 0]) * np.ptp(P_j[:, 1]) + 1.0)
    ii = np.nonzero(ok)[0]
    if len(ii) < min_matches:
        return None
    tree = cKDTree(P_j)
    # ★ 짝은 반경 r 안에서 고르되, 비율 검사의 2등은 2r 안에서 찾는다. 합성 반복 패널
    #   (옥상 포함)에서 r 안 후보가 하나뿐이면 받아들이던 판정이 **한 칸 밀린 짝** 8,212개
    #   (2.9%)를 만들었다. 진짜 짝이 r 바로 밖에 있고 옆 셀만 r 안에 드는 경우다. 2r 안의
    #   진짜 짝이 2등이 되면 비율 검사에서 모호함으로 걸러진다.
    r2 = 2.0 * radius_px
    # 후보 수를 먼저 세고(C 구현, 빠름) K 를 그 99% 에 맞춘다. 넉넉한 고정 K 는
    # (N, K, 128) 배열을 키워 오히려 느렸다 (합성 시험: K=64 에서 기존 반복보다 2배 느림).
    cnt = np.asarray(tree.query_ball_point(pred[ii], r=r2, return_length=True))
    if cnt.size == 0 or cnt.max() == 0:
        return None
    K = int(np.clip(np.percentile(cnt, 99) + 1, 2, k_max))
    K = min(K, len(P_j))
    ii = ii[cnt > 0]
    di = np.asarray(desc_i, dtype=np.float32)
    dj = np.vstack([np.asarray(desc_j, dtype=np.float32), np.zeros((1, di.shape[1]), np.float32)])
    bi, bj, bd = [], [], []
    for s in range(0, len(ii), chunk):
        sel = ii[s:s + chunk]
        pdist, nb = tree.query(pred[sel], k=K, distance_upper_bound=r2)
        nb = np.asarray(nb).reshape(len(sel), -1); pdist = np.asarray(pdist).reshape(len(sel), -1)
        valid = nb < len(P_j)
        d = np.linalg.norm(dj[nb] - di[sel][:, None, :], axis=2)
        d[~valid] = np.inf
        inner = valid & (pdist <= radius_px)             # 짝 후보는 r 안
        d_in = np.where(inner, d, np.inf)
        b0 = np.argmin(d_in, axis=1)
        d0 = d_in[np.arange(len(sel)), b0]
        j0 = nb[np.arange(len(sel)), b0]
        d_rest = d.copy(); d_rest[np.arange(len(sel)), b0] = np.inf   # 2등은 2r 안 전체에서
        d1 = d_rest.min(axis=1)
        nv = valid.sum(axis=1)
        acc = np.isfinite(d0) & (((nv == 1) & (d0 < 300.0)) | ((nv >= 2) & (d0 < ratio * d1)))
        bi.append(sel[acc]); bj.append(j0[acc]); bd.append(d0[acc])
    if not bi:
        return None
    bi, bj, bd = np.concatenate(bi), np.concatenate(bj), np.concatenate(bd)
    if len(bi) < min_matches:
        return None
    # 같은 j 에 여러 i 가 붙으면 가장 가까운 하나만
    o = np.lexsort((bd, bj))
    bi, bj = bi[o], bj[o]
    first = np.r_[True, bj[1:] != bj[:-1]]
    bi, bj = bi[first], bj[first]
    if len(bi) < min_matches:
        return None
    pts_i = P_i[bi].astype(np.float32); pts_j = P_j[bj].astype(np.float32)
    M, inl = cv2.estimateAffinePartial2D(pts_i, pts_j, method=cv2.RANSAC,
                                         ransacReprojThreshold=ransac_thresh_px, maxIters=4000)
    if M is None or inl is None:
        return None
    inl = inl.ravel().astype(bool)
    if inl.sum() < min_matches:
        return None
    return pts_i[inl], pts_j[inl], bi[inl].astype(np.int64), bj[inl].astype(np.int64)


def pointwise_prediction(P_i, i, j, pl, sup, fh_fn, *, sup_r_px, bin_m=0.25, drop_unsupported=False):
    """특징점마다 가까운 번들조정 관측점의 높이로 예측한다.

    ★ 쌍마다 평면 하나로 예측하면, 한 장에 옥상과 땅이 함께 드는 RGB(합성: 프레임 88 × 66 m)
      에서 옥상 점이 시차(약 50 px)만큼 엉뚱하게 예측돼 반경 안의 옆 셀에 붙었다 — 틀린 대응
      1,578개(4.74%). '여러 층 쌍' 판정은 옥상 점이 쌍의 5% 를 넘어야 걸려 대부분 놓쳤다.
      번들조정 점군에는 옥상 점의 실제 높이가 있으니, 영상 i 에서 각 특징점 둘레 sup_r_px 안의
      관측점 3개의 높이 중앙값만큼 평면을 올리거나 내려 예측한다. 높이는 bin_m 칸으로 묶어
      칸마다 검증된 호모그래피 코드(build_frame_homography · predict_correspondence)를 쓴다.

    관측점이 둘레에 없는 특징점은 평면 pl 로 예측한다(tie point 없던 프레임 포함).
    drop_unsupported 면 그런 점을 버린다(NaN) — 여러 층 쌍에서 쓴다.
    Returns ``(pred (N,2), n_supported)``.
    """
    from .homography import GroundPlane
    from scipy.spatial import cKDTree
    P_i = np.asarray(P_i, dtype=np.float64)
    pred = np.full((len(P_i), 2), np.nan)
    s = sup.get(i)
    off = np.zeros(len(P_i))
    has = np.zeros(len(P_i), bool)
    if s is not None and len(s[1]) >= 3:
        tree, uv, xyz = s
        d, nb = tree.query(P_i, k=3, distance_upper_bound=sup_r_px)
        d = np.atleast_2d(d); nb = np.atleast_2d(nb)
        valid = nb < len(uv)
        cnt = valid.sum(axis=1)
        has = cnt >= 1
        if has.any():
            z = np.where(valid, xyz[np.minimum(nb, len(xyz) - 1), 2], np.nan)
            xs = np.where(valid, xyz[np.minimum(nb, len(xyz) - 1), 0], np.nan)
            ys = np.where(valid, xyz[np.minimum(nb, len(xyz) - 1), 1], np.nan)
            zm = np.nanmedian(z[has], axis=1)
            base = pl.height_at(np.nanmedian(xs[has], axis=1), np.nanmedian(ys[has], axis=1))
            off[has] = zm - base
    q = np.round(off / bin_m) * bin_m
    if drop_unsupported:
        groups = np.unique(q[has])
    else:
        groups = np.unique(q)
    for g in groups:
        m = (q == g) & (has if drop_unsupported else True)
        if not m.any():
            continue
        pl_g = pl if abs(g) < 1e-9 else GroundPlane(a=pl.a, b=pl.b, c=pl.c + float(g), origin_xy=pl.origin_xy)
        fi, fj = fh_fn(i, pl_g), fh_fn(j, pl_g)
        if fi is None or fj is None:
            continue
        r = predict_correspondence(fi, fj, P_i[m])
        if r is not None:
            pred[m] = r
    return pred, int(has.sum())


def fit_points_plane(points):
    """번들조정 점군에 이상치를 깎아 가며 평면을 적합 — 대응점이 실제로 놓인 면."""
    from .homography import GroundPlane
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or len(P) < 20:
        return None
    x0, y0 = float(np.median(P[:, 0])), float(np.median(P[:, 1]))
    A = np.c_[P[:, 0] - x0, P[:, 1] - y0, np.ones(len(P))]
    keep = np.ones(len(P), bool); coef = None
    for _ in range(4):
        if keep.sum() < 10:
            break
        coef, *_ = np.linalg.lstsq(A[keep], P[keep, 2], rcond=None)
        r = P[:, 2] - A @ coef
        mad = float(np.median(np.abs(r[keep] - np.median(r[keep])))) * 1.4826 + 1e-6
        keep = np.abs(r) < max(2.5 * mad, 0.3)
    if coef is None:
        return None
    return GroundPlane(a=float(coef[0]), b=float(coef[1]), c=float(coef[2]), origin_xy=(x0, y0))


def merge_matches(base, extra):
    """쌍마다 1차 대응에 2차 대응을 더한다. 같은 i 특징점이 다른 짝이면 2차를 따른다."""
    out = dict(base)
    added = new_pairs = 0
    for key, val in extra.items():
        if key not in out:
            out[key] = val; new_pairs += 1; added += len(val[0]); continue
        pi0, pj0, ii0, jj0 = (np.asarray(v) for v in out[key])
        pi1, pj1, ii1, jj1 = (np.asarray(v) for v in val)
        by_i = {int(a): (pi0[k], pj0[k], int(jj0[k])) for k, a in enumerate(ii0)}
        before = len(by_i)
        for k, a in enumerate(ii1):
            by_i[int(a)] = (pi1[k], pj1[k], int(jj1[k]))
        # 같은 j 에 두 i 가 붙었으면 하나만
        seen_j, rows = set(), []
        for a, (p, q, b) in by_i.items():
            if b in seen_j: continue
            seen_j.add(b); rows.append((p, q, a, b))
        added += max(len(rows) - before, 0)
        out[key] = (np.array([r[0] for r in rows], np.float32), np.array([r[1] for r in rows], np.float32),
                    np.array([r[2] for r in rows], np.int64), np.array([r[3] for r in rows], np.int64))
    return out, added, new_pairs


def guided_rematch(pairs, features, base_matches, cams, intrinsics, observations, points, *,
                   global_plane, rot_fn, weak_frames=(), fixed_radius_px: float = 0.0,
                   factor: float = 3.0, r_min: float = 2.0, r_max: float = 15.0,
                   err_max_px: float = 6.0,
                   ratio: float = 0.85, min_matches: int = 20, min_local_points: int = 30,
                   multilevel_dz_m: float = 3.0, multilevel_frac: float = 0.05,
                   multilevel: str = "skip", pointwise: bool = True, sup_frac: float = 0.02,
                   reference_plane=None, ba_intrinsics=None):
    """번들조정을 마친 자세로 모든 인접쌍을 다시 매칭한다.

    쌍마다
      1) 예측 평면 — 두 영상에 함께 보이는 번들조정 점(30개 이상)으로 국소 평면을 적합.
         모자라면 전체 점군 평면. ★ 합성 시험: 예측 평면 높이가 1.5 m 틀리면 RGB(f 2950)
         에서 예측이 11 px 밀려 반경 4~8 px 로는 매칭이 무너졌다.
      2) 반경 — 그 쌍의 1차 대응이 예측 위치에서 떨어진 거리의 중앙값 × factor
         (r_min ~ r_max). 1차 대응이 없는 쌍은 전체 중앙값 × factor, 자세가 다듬어지지
         않은 프레임(weak)이 끼면 1.5배. ★ 합성 반복 패널(열화상 셀 4.2 px, RGB 12.1 px):
         반경 30 px 은 맞음 298 · 틀림 11 로 지금 방식(278 · 0)과 비슷했고, 반경을 예측
         오차에 맞춰 2.5~8 px 로 좁히자 열화상 3,625 · 0, RGB 9,331 · 0 이 됐다.
    """
    from collections import defaultdict
    from scipy.spatial import cKDTree
    from .homography import build_frame_homography, GroundPlane
    cam_pts = defaultdict(set)
    cam_obs = defaultdict(list)
    for o in observations:
        cam_pts[int(o[0])].add(int(o[1]))
        cam_obs[int(o[0])].append((int(o[1]), float(o[2][0]), float(o[2][1])))
    P = np.asarray(points, dtype=float)
    weak = set(int(w) for w in weak_frames)
    # 영상마다 관측점의 영상 좌표 → 3차원 점 (특징점별 높이 예측용)
    sup = {}
    if pointwise:
        for ci, lst in cam_obs.items():
            a = np.asarray(lst, dtype=float)
            ok = (a[:, 0] >= 0) & (a[:, 0] < len(P))
            a = a[ok]
            if len(a) >= 3:
                uv = a[:, 1:3]; xyz = P[a[:, 0].astype(int)]
                sup[ci] = (cKDTree(uv), uv, xyz)
    # ★ 옥상이 든 영상 — 관측점 중 기준면에서 multilevel_dz_m 넘게 벗어난 점이 2% 를 넘으면.
    #   쌍 단위 판정(점의 5%)은 RGB 처럼 한 장에 옥상이 작게 드는 경우를 놓쳤다(합성: 1쌍만
    #   걸림, 틀린 대응 0.62%). 그런 영상이 낀 쌍에서는 높이를 모르는 특징점을 버린다.
    ml_frame = set()
    if pointwise and global_plane is not None:
        for ci, (_t, _uv, xyz) in sup.items():
            dev = np.abs(xyz[:, 2] - global_plane.height_at(xyz[:, 0], xyz[:, 1]))
            if float(np.mean(dev > multilevel_dz_m)) > 0.02:
                ml_frame.add(ci)

    def fh(i, pl):
        try:
            return build_frame_homography(np.asarray(cams[i][:3], float), rot_fn(*cams[i][3:6]),
                                          intrinsics[i], pl)
        except Exception:
            return None

    # ★ 모델 일치 진단 — 갈평 줌은 번들조정이 잘 묶은 쌍(strong)조차 예측이 723 px 틀렸다.
    #   같은 자세로 번들조정은 재투영 1.2 px 이니, 2차 매칭이 번들조정과 다른 카메라 모델을 쓰는
    #   것이다. 번들조정은 초점 · 주점을 전체 중앙값 하나(f_rep, cx_rep, cy_rep)로 쓰고, 여기서는
    #   프레임마다의 값(intrinsics)을 쓴다. 잘 묶인 프레임에서 번들조정 3차원 점을 그 높이의 평면으로
    #   투영해 관측과 비교하고, 두 내부 파라미터 중 일치가 좋은 쪽으로 예측한다.
    model_check = {}
    if ba_intrinsics is not None and sup:
        from dataclasses import replace as _rep
        f_b, cx_b, cy_b = (float(v) for v in ba_intrinsics)
        intr_ba = [(_rep(k, f_px=f_b, cx=cx_b, cy=cy_b) if k is not None else None) for k in intrinsics]
        def _consistency(intr_list):
            vals = []
            frames = sorted(sup, key=lambda c: -len(sup[c][1]))[:40]
            for ci in frames:
                if intr_list[ci] is None:
                    continue
                _t, uv, xyz = sup[ci]
                sel = np.arange(len(uv))[:400]
                for zb in np.unique(np.round(xyz[sel, 2] / 0.25) * 0.25):
                    m = sel[np.abs(np.round(xyz[sel, 2] / 0.25) * 0.25 - zb) < 1e-6]
                    try:
                        fhz = build_frame_homography(np.asarray(cams[ci][:3], float), rot_fn(*cams[ci][3:6]),
                                                     intr_list[ci], GroundPlane(0.0, 0.0, float(zb), origin_xy=(0.0, 0.0)))
                    except Exception:
                        fhz = None
                    if fhz is None:
                        continue
                    q = fhz.ground_to_pixel(xyz[m, :2])
                    d = np.hypot(*(np.asarray(q) - uv[m]).T)
                    vals.extend(d[np.isfinite(d)].tolist())
            return float(np.median(vals)) if vals else None
        c_frame, c_ba = _consistency(intrinsics), _consistency(intr_ba)
        model_check = {"frame_intrinsics_px": None if c_frame is None else round(c_frame, 2),
                       "ba_intrinsics_px": None if c_ba is None else round(c_ba, 2)}
        if c_ba is not None and (c_frame is None or c_ba < c_frame):
            intrinsics = intr_ba
            model_check["used"] = "ba"
        else:
            model_check["used"] = "frame"
    plan, errs, n_local = [], [], 0
    mode_votes = {}
    n_obs_ref = 0
    cam_uv = defaultdict(dict)                      # 영상 → {3차원 점 번호: 관측 영상 좌표}
    for o in observations:
        cam_uv[int(o[0])][int(o[1])] = (float(o[2][0]), float(o[2][1]))
    _obs_n = defaultdict(int)
    for o in observations:
        _obs_n[int(o[0])] += 1
    _med_obs = float(np.median([v for v in _obs_n.values()])) if _obs_n else 0.0
    def _cls(a, b):
        na, nb = _obs_n.get(a, 0), _obs_n.get(b, 0)
        if na == 0 or nb == 0: return "zero"
        if min(na, nb) < 0.3 * _med_obs: return "weak"
        return "strong"
    diag = defaultdict(lambda: {"raw": [], "corr": []})
    for i, j in pairs:
        if i not in features or j not in features or intrinsics[i] is None or intrinsics[j] is None:
            continue
        common = cam_pts.get(i, set()) & cam_pts.get(j, set())
        pl = None
        mixed = False
        if len(common) >= min_local_points:
            Q = P[sorted(common)]
            pl = fit_points_plane(Q)
            if pl is not None:
                n_local += 1
                # ★ 옥상과 땅이 한 쌍에 섞이면 국소 평면이 두 층 사이에 놓여 양쪽 점의 예측이
                #   틀어진다 (합성: 옥상 있으면 틀린 짝 0.76%, 없으면 0.00%). 점들의 평면 잔차가
                #   크게 흩어지는 쌍을 '여러 층 쌍'으로 본다.
                rr = Q[:, 2] - pl.height_at(Q[:, 0], Q[:, 1])
                # ★ 중앙값 기반(MAD)은 옥상이 쌍의 일부만 차지하면 작게 나와 못 잡았다(합성: MAD
                #   판정으로 17쌍을 건너뛰어도 틀린 짝 869 → 869). 그런 쌍에서 옥상 점은 땅 평면으로
                #   예측돼 시차만큼 엉뚱한 옥상 셀에 붙고, 위치가 땅 예측과 맞아 RANSAC 도 통과한다.
                #   평면에서 크게 벗어난 점의 **비율**로 판정한다.
                mixed = float(np.mean(np.abs(rr) > multilevel_dz_m)) > multilevel_frac
        pl = pl or global_plane
        fi, fj = fh(i, pl), fh(j, pl)
        if fi is None or fj is None:
            continue
        sh_i = features[i][2] if len(features[i]) > 2 else None
        sup_r = max(4.0, sup_frac * float(max(sh_i[:2]) if sh_i is not None else 1000.0))
        # ★ 쌍마다 예측 방식을 1차 대응으로 직접 재서 고른다.
        #   갈평저수지 RGB(줌, 비행선 교차 2%)는 번들조정 점의 깊이가 불안정해 국소 평면 ·
        #   특징점별 높이 예측이 1,094 px 틀렸고(314쌍 건너뜀, 113장 '여러 층' 오판), 확정
        #   기준면(0.023°)은 거의 완벽했다. Site-2 는 반대로 옥상 때문에 특징점별 높이가
        #   필요했다. 어느 쪽인지 부지마다 미리 알 수 없으니 재서 고른다.
        err, mode = None, ("point" if pointwise else "plane")
        # ★ 예측 오차는 **번들조정을 통과한 관측**(두 영상에 함께 보이는 3차원 점의 실제 위치)으로
        #   잰다. 갈평 줌은 모델 일치가 1.9 px 인데 1차 대응 기준 오차가 strong 쌍에서도 723 px 였다
        #   — 예측이 아니라 **1차 대응이 틀렸다**(반복 패널 + 평면 장면의 F 퇴화로 RANSAC 이 틀린
        #   대응 묶음을 받아들임). 함께 보이는 점이 10개 미만인 쌍만 1차 대응을 쓴다.
        bm = None
        _ci, _cj = cam_uv.get(i), cam_uv.get(j)
        if _ci and _cj:
            _com = [pid for pid in _ci if pid in _cj]
            if len(_com) >= 10:
                bm = (np.array([_ci[pid] for pid in _com]), np.array([_cj[pid] for pid in _com]))
                n_obs_ref += 1
        if bm is None:
            bm = base_matches.get((i, j))
        if bm is not None and len(bm[0]) >= 10:
            src = np.asarray(bm[0], float); dst = np.asarray(bm[1], float)
            cand = {}
            def _e(pred):
                if pred is None: return None
                d = np.hypot(*(dst - pred).T); d = d[np.isfinite(d)]
                return float(np.median(d)) if d.size else None
            if pointwise:
                cand["point"] = _e(pointwise_prediction(src, i, j, pl, sup, fh, sup_r_px=sup_r)[0])
            cand["plane"] = _e(predict_correspondence(fi, fj, src))
            if reference_plane is not None:
                ri, rj = fh(i, reference_plane), fh(j, reference_plane)
                if ri is not None and rj is not None:
                    cand["ref"] = _e(predict_correspondence(ri, rj, src))
            cand = {k: v for k, v in cand.items() if v is not None}
            A = None
            if cand:
                mode = min(cand, key=cand.get); err = cand[mode]
                mode_votes[mode] = mode_votes.get(mode, 0) + 1
                # ★ 1차 대응으로 예측을 보정한다 — 갈평저수지 RGB(줌)는 어떤 평면으로도 예측 오차
                #   중앙값이 1,094 px 였다. 자세를 고치지 못한 프레임(tie point 없음 81장 · 관측 적음
                #   120장)이 낀 쌍이 다수라서다. 1차 대응이 정답 위치를 알려 주므로, 예측 → 실제의
                #   닮음 변환을 RANSAC 으로 맞춰 모든 특징점 예측에 적용한다.
                if mode == "point":
                    pp = pointwise_prediction(src, i, j, pl, sup, fh, sup_r_px=sup_r)[0]
                elif mode == "ref":
                    pp = predict_correspondence(fh(i, reference_plane), fh(j, reference_plane), src)
                else:
                    pp = predict_correspondence(fi, fj, src)
                if pp is not None and len(src) >= 10:
                    okp = np.all(np.isfinite(pp), axis=1)
                    if okp.sum() >= 10:
                        M, inl = cv2.estimateAffinePartial2D(pp[okp].astype(np.float32), dst[okp].astype(np.float32),
                                                             method=cv2.RANSAC, ransacReprojThreshold=3.0)
                        if M is not None and inl is not None and inl.sum() >= 10:
                            q = (M[:, :2] @ pp[okp].T).T + M[:, 2]
                            ec = float(np.median(np.hypot(*(dst[okp] - q).T)))
                            diag[_cls(i, j)]["corr"].append(ec)
                            if ec < err:
                                A = M; err = ec
                diag[_cls(i, j)]["raw"].append(cand[mode])
                errs.append(err)
        plan.append([i, j, fi, fj, err, mixed, pl, sup_r, mode if err is not None else None,
                     A if err is not None else None])

    g_err = float(np.median(errs)) if errs else 5.0
    out, counts, radii = {}, [], []
    n_mixed = n_skip = 0
    n_sup = n_kp = 0
    n_bad_pred = 0
    site_mode = max(mode_votes, key=mode_votes.get) if mode_votes else ("point" if pointwise else "plane")
    n_corr = 0
    for i, j, fi, fj, err, mixed, pl, sup_r, mode, A in plan:
        mode = mode or site_mode                      # 1차 대응 없는 쌍은 부지에서 가장 많이 이긴 방식
        if mode == "ref" and reference_plane is not None:
            _ri, _rj = fh(i, reference_plane), fh(j, reference_plane)
            if _ri is not None and _rj is not None:
                fi, fj = _ri, _rj
        rt, th_scale = ratio, 0.5
        if mixed:
            n_mixed += 1
            if multilevel == "skip" and not pointwise:
                n_skip += 1
                continue
            if not pointwise:
                rt, th_scale = min(ratio, 0.7), 0.3                 # 엄격
        # ★ 합성 RGB 반복 패널: 틀린 2차 대응 1,459개 중 1,452개가 참 위치에서 19~23 px —
        #   같은 무늬가 되풀이되는 간격(셀 두 칸 0.46 m ≈ 20.8 px)의 닮은 셀이었다. 예측이 나쁜
        #   쌍의 반경이 '오차 × 3' 으로 40 px 까지 커져 닮은 셀이 반경 안에 들어온 것이다.
        #   예측이 그만큼 틀린 쌍에서는 반경을 키워도 반복을 가를 수 없으니 건너뛴다.
        if fixed_radius_px <= 0 and err is not None and err > err_max_px:
            n_bad_pred += 1
            continue
        if fixed_radius_px > 0:
            r = float(fixed_radius_px)
        elif err is not None:
            r = float(np.clip(factor * err, r_min, r_max))
        else:
            r = float(np.clip(factor * g_err * (1.5 if (i in weak or j in weak) else 1.0), r_min, r_max))
        radii.append(r)
        kp_i, d_i, _ = features[i]; kp_j, d_j, sh_j = features[j]
        pred = None
        if pointwise and mode == "point":
            Pk = np.array([k.pt for k in kp_i], dtype=np.float64)
            # 여러 층 쌍에서는 높이를 모르는 점을 버린다 — 땅 평면으로 예측하면 옥상 점이 옆 셀에 붙는다
            pred, ns = pointwise_prediction(Pk, i, j, pl, sup, fh, sup_r_px=sup_r,
                                            drop_unsupported=mixed or (i in ml_frame) or (j in ml_frame))
            n_sup += ns; n_kp += len(Pk)
        if A is not None:                                  # 1차 대응으로 맞춘 보정을 예측에 적용
            Pk = np.array([k.pt for k in kp_i], dtype=np.float64)
            if pred is None:
                pred = predict_correspondence(fi, fj, Pk)
            if pred is not None:
                pred = (A[:, :2] @ np.asarray(pred, float).T).T + A[:, 2]
                n_corr += 1
        res = guided_match_pair_fast(kp_i, d_i, kp_j, d_j, fi, fj, radius_px=r, shape_j=sh_j,
                                     ratio=rt, min_matches=min_matches,
                                     ransac_thresh_px=max(1.5, th_scale * r),   # 셀 한 칸보다 작게
                                     pred=pred)
        if res is not None:
            out[(i, j)] = res; counts.append(len(res[0]))
    info = dict(pairs_tried=len(plan), pairs_ok=len(out),
                inliers_median=int(np.median(counts)) if counts else 0,
                inliers_total=int(np.sum(counts)) if counts else 0,
                pred_err_median_px=round(g_err, 2),
                radius_median_px=round(float(np.median(radii)), 2) if radii else None,
                local_plane_pairs=n_local, multilevel_pairs=n_mixed, multilevel_skipped=n_skip,
                pointwise=bool(pointwise), pairs_skipped_bad_prediction=n_bad_pred,
                multilevel_frames=len(ml_frame) if pointwise else None,
                prediction_mode_votes=dict(mode_votes), prediction_mode_site=site_mode,
                pairs_corrected_by_base_matches=n_corr,
                model_consistency=model_check or None,
                pairs_error_from_ba_observations=n_obs_ref,
                pred_err_by_pair_class={k: {"pairs": len(v["raw"]),
                                            "raw_median_px": round(float(np.median(v["raw"])), 1) if v["raw"] else None,
                                            "corrected_median_px": round(float(np.median(v["corr"])), 1) if v["corr"] else None}
                                        for k, v in diag.items()},
                keypoints_height_supported=round(n_sup / max(n_kp, 1), 3) if pointwise else None)
    return out, info

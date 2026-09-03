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

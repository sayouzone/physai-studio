"""'거대한 평면(가상 지표면)' 추정 — 파이프라인 7단계의 입력.

DSM 을 만들지 않는 대신 대상 지역을 평면 하나로 근사한다. 그 평면을 얼마나
잘 잡느냐가 정사영상 정확도를 지배한다.

오차의 크기 (직관)
------------------
평면에서 ``Δh`` 만큼 떨어진 실제 지형점은, 연직에서 ``θ`` 만큼 기울어진
시선으로 보면 지상에서 ``Δh · tan θ`` 만큼 밀린다. 화각 반각이 20° 인
사진의 가장자리에서 ``Δh = 1 m`` 면 밀림은 약 0.36 m. 즉:

* 패널 상면과 지면을 혼동 (Δh ≈ 1.5 m) → 가장자리에서 0.5 m 급 오차
* 경사 3% 인 부지를 수평평면으로 근사, 촬영폭 120 m → 양끝 ±1.8 m 편차
  → 가장자리 오차 0.6 m 급

두 번째 항목 때문에 이 모듈은 **경사평면** (``Z = a·x + b·y + c``) 을 기본으로
지원한다. 호모그래피는 임의 평면에 대해 정확하므로 경사를 넣어도 비용이 0 이다.
수평평면만 쓰는 것은 근거 없는 제약이다.

추정 우선순위
-------------
1. **BA 3D 점군** — 가장 신뢰도 높음. RANSAC 평면적합으로 이상점 제거.
   태양광 단지는 패널 상면이 점군의 대부분을 차지하므로, 적합된 평면은
   자연히 '패널 상면 평면' 이 된다. 검출 결과를 패널 위치로 쓸 거라면
   이게 오히려 **원하는 기준면**이다 (지면이 아니라).
2. **LRF** — H20T 는 촬영 시점 조준점의 실측 절대고도를 준다.
   ``altitude − relative_height`` 보다 훨씬 정확.
3. **메타데이터 폴백** — ``절대고도 − 상대고도``. 이륙지점 기준이라
   지형 기복이 있으면 편향된다.

⚠ 절대고도 하드코딩 금지
------------------------
"이 사이트의 패널 상면은 114.87 m" 같은 절대값은 다른 사이트에서 즉시
깨진다 (지형 표고 종속). 이식 가능한 값은 **지면 대비 상대 높이**
(예: 패널 상면 1.48 m) 이므로, 폴백 경로에서는 ``panel_top_offset_m`` 으로
받는다.
"""

from __future__ import annotations

import logging

import numpy as np

from .homography import GroundPlane

logger = logging.getLogger(__name__)

__all__ = [
    "fit_plane_ransac",
    "plane_from_lrf",
    "plane_from_metadata",
    "estimate_ground_plane",
]


# ---------------------------------------------------------------------------
# RANSAC 평면 적합
# ---------------------------------------------------------------------------
def fit_plane_ransac(points_xyz: np.ndarray,
                     *,
                     allow_tilt: bool = True,
                     max_tilt_deg: float = 15.0,
                     inlier_threshold_m: float = 0.5,
                     n_iterations: int = 500,
                     min_inlier_ratio: float = 0.3,
                     surface: str = "upper",
                     surface_gap_m: float = 0.6,
                     seed: int = 0) -> GroundPlane | None:
    """3D 점군 → ``Z = a·x + b·y + c`` 평면 (RANSAC + 최소제곱 재적합).

    수직 평면은 이 매개화로 표현할 수 없다 (``Z`` 를 ``x, y`` 의 함수로
    두므로). 지표면 적합이라 문제되지 않으며, 오히려 벽면 같은 병적인 해를
    구조적으로 배제해 준다.

    Parameters
    ----------
    points_xyz : (N, 3) 투영좌표계 점군. BA 산출 ``points_opt`` 를 그대로.
    allow_tilt : ``False`` 면 경사를 0 으로 강제하고 로버스트 중앙값 높이만
        추정 (지형이 정말 평탄하고 점군이 얇을 때).
    max_tilt_deg : 이보다 급한 경사가 나오면 적합 실패로 간주. 자세 오차나
        이상점 군집이 만든 가짜 경사를 막는 안전장치.
    inlier_threshold_m : 평면으로부터 이 거리 안이면 inlier. 태양광 단지는
        패널 두께 + 지지구조 때문에 0.3~0.8 m 가 적당.
    min_inlier_ratio : inlier 비율이 이보다 낮으면 ``None``.

    Returns
    -------
    ``GroundPlane`` 또는 ``None`` (점 부족 / 적합 실패).
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 3:
        logger.warning("평면 적합 실패: 유효 점 %d개 (최소 3개)", len(pts))
        return None

    if not allow_tilt:
        z = float(np.median(pts[:, 2]))
        resid = pts[:, 2] - z
        keep = np.abs(resid) <= inlier_threshold_m
        rmse = float(np.sqrt(np.mean(resid[keep] ** 2))) if keep.any() else float("nan")
        ctr = pts[:, :2].mean(axis=0)
        return GroundPlane(0.0, 0.0, z, origin_xy=(float(ctr[0]), float(ctr[1])),
                           inlier_rmse_m=rmse, n_inliers=int(keep.sum()))

    # 투영좌표는 수십만 m 라서 그대로 최소제곱하면 조건수가 나빠진다.
    # 중심화 후 적합하고 마지막에 원점 기준으로 되돌린다.
    ctr = pts.mean(axis=0)
    P = pts - ctr
    rng = np.random.default_rng(seed)
    n = len(P)

    best_inliers = None
    best_count = 0

    if n >= 3:
        n_iter = min(n_iterations, max(1, n * (n - 1) * (n - 2) // 6))
        for _ in range(n_iter):
            idx = rng.choice(n, size=3, replace=False)
            sample = P[idx]
            A = np.column_stack([sample[:, 0], sample[:, 1], np.ones(3)])
            try:
                coef = np.linalg.solve(A, sample[:, 2])
            except np.linalg.LinAlgError:
                continue  # 3점이 수직선상 → 특이.
            resid = P[:, 2] - (coef[0] * P[:, 0] + coef[1] * P[:, 1] + coef[2])
            inliers = np.abs(resid) <= inlier_threshold_m
            cnt = int(inliers.sum())
            if cnt > best_count:
                best_count, best_inliers = cnt, inliers

    if best_inliers is None or best_count < 3:
        logger.warning("평면 RANSAC 실패: 최대 inlier %d개", best_count)
        return None

    ratio = best_count / n
    if ratio < min_inlier_ratio:
        logger.warning("평면 RANSAC inlier 비율 부족: %.1f%% < %.1f%% "
                       "(지형이 평면과 거리가 멀거나 자세 오차가 큼)",
                       ratio * 100, min_inlier_ratio * 100)
        return None

    # inlier 전체로 최소제곱 재적합 — RANSAC 의 3점 해보다 훨씬 정밀.
    Q = P[best_inliers]
    A = np.column_stack([Q[:, 0], Q[:, 1], np.ones(len(Q))])
    coef, *_ = np.linalg.lstsq(A, Q[:, 2], rcond=None)
    a, b, c_ctr = float(coef[0]), float(coef[1]), float(coef[2])

    tilt = float(np.degrees(np.arctan(np.hypot(a, b))))
    if tilt > max_tilt_deg:
        logger.warning("평면 경사 %.2f° > %.2f° — 수평평면으로 강등 "
                       "(자세 오차 또는 이상점 군집 의심)", tilt, max_tilt_deg)
        return fit_plane_ransac(pts, allow_tilt=False,
                                inlier_threshold_m=inlier_threshold_m, seed=seed)

    resid = Q[:, 2] - (a * Q[:, 0] + b * Q[:, 1] + c_ctr)
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    # 중심화 해제 없이 origin 을 그대로 싣는다 — GroundPlane 이 원점을 알고
    # 있으므로 c 는 '중심점에서의 표고' 라는 읽을 수 있는 값으로 남는다.
    c = c_ctr + ctr[2]

    # ---- 상면(패널 상면) 으로 이동 -------------------------------------
    # ★ 태양광 단지의 점군은 '지면' 과 '패널 상면' 두 층으로 갈린다. RANSAC
    #   은 그중 점이 많은 쪽 하나만 잡는데, 지면(잔디/자갈) 텍스처가 특징점을
    #   더 많이 내면 **지면 평면**이 선택된다. 그러면 패널이 프레임마다
    #   Δh·tanθ 만큼 밀려(relief displacement) 모자이크 시임에서 뚝 끊긴다
    #   (H20T 45.9 m 고도, 패널 1.8 m 기준 최대 174 cm = 패널 한 장 폭).
    #
    #   검사 대상이 패널이므로 기준면은 **패널 상면**이어야 한다. 전체 점군의
    #   잔차 분포에서 적합 평면보다 위쪽에 유의한 층이 또 있으면 그 층으로
    #   c 를 올린다. 경사(a, b) 는 두 층이 평행하다고 보고 그대로 쓴다.
    if surface == "upper":
        all_resid = P[:, 2] - (a * P[:, 0] + b * P[:, 1] + c_ctr)
        c_shift = _find_upper_layer(all_resid, inlier_threshold_m,
                                    surface_gap_m)
        if c_shift > 0:
            c += c_shift
            upper_n = int(np.sum(np.abs(all_resid - c_shift)
                                 <= inlier_threshold_m))
            logger.info("상면 층 검출: 적합 평면보다 %.2f m 위에 점 %d개 — "
                        "기준면을 그쪽으로 올림 (태양광 패널 상면으로 추정)",
                        c_shift, upper_n)
            best_count = upper_n
            rmse = float(np.sqrt(np.mean(
                (all_resid[np.abs(all_resid - c_shift) <= inlier_threshold_m]
                 - c_shift) ** 2))) if upper_n else rmse

    logger.info("지상평면 적합: 경사 %.3f°, 중심표고 %.2f m, RMSE %.3f m, "
                "inlier %d/%d (%.1f%%)",
                tilt, c, rmse, best_count, n, ratio * 100)
    return GroundPlane(a=a, b=b, c=c,
                       origin_xy=(float(ctr[0]), float(ctr[1])),
                       inlier_rmse_m=rmse, n_inliers=best_count)


def _find_upper_layer(residuals: np.ndarray,
                      bin_m: float,
                      min_gap_m: float,
                      min_fraction: float = 0.10) -> float:
    """적합 평면 위쪽에 있는 두 번째 층까지의 거리 (m). 없으면 0.

    잔차 히스토그램에서 최빈 봉우리(=적합된 층)보다 ``min_gap_m`` 이상 위에
    있으면서 전체의 ``min_fraction`` 이상을 차지하는 봉우리를 찾는다.
    태양광 단지에서 이 봉우리가 패널 상면이다.

    ``min_fraction`` 이 문턱 역할을 한다 — 잡음이나 소수의 이상점(전신주,
    새) 때문에 기준면이 엉뚱하게 올라가지 않도록.
    """
    r = residuals[np.isfinite(residuals)]
    if r.size < 20:
        return 0.0
    lo, hi = float(np.percentile(r, 1)), float(np.percentile(r, 99))
    if hi - lo < min_gap_m:
        return 0.0                      # 단일 층 — 이동할 이유 없음.
    bins = max(int(np.ceil((hi - lo) / max(bin_m, 1e-3))), 4)
    hist, edges = np.histogram(r, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])

    base_i = int(np.argmax(hist))
    base_c = centers[base_i]
    upper = (centers > base_c + min_gap_m)
    if not upper.any():
        return 0.0
    cand_i = np.nonzero(upper)[0][int(np.argmax(hist[upper]))]
    if hist[cand_i] < min_fraction * r.size:
        return 0.0                      # 위쪽 층이 너무 얇다 — 이상점일 것.
    return float(centers[cand_i] - base_c)


# ---------------------------------------------------------------------------
# 메타데이터 기반 폴백
# ---------------------------------------------------------------------------
def plane_from_lrf(metas: list, camera_xyz: np.ndarray) -> GroundPlane | None:
    """LRF 조준점 절대고도들로 평면 적합.

    H20T 의 ``LRFTargetAbsAlt`` 는 촬영 시점 화면 중앙 지점의 실측 고도다.
    프레임마다 한 점씩 생기므로, 프레임이 여러 장이면 경사까지 잡을 수 있다.
    LRF 조준점의 수평 위치는 (nadir 근처라면) 카메라 연직 아래로 근사한다.
    """
    xs, ys, zs = [], [], []
    C = np.asarray(camera_xyz).reshape(-1, 3)
    for i, meta in enumerate(metas):
        z = getattr(meta, "lrf_target_abs_alt", None)
        if z is None or not np.isfinite(z) or z == 0:
            continue
        xs.append(C[i, 0])
        ys.append(C[i, 1])
        zs.append(float(z))

    if len(zs) < 3:
        if not zs:
            return None
        z = float(np.median(zs))
        logger.info("LRF 기준 평면 (수평, %d점): Z=%.2f m", len(zs), z)
        return GroundPlane.horizontal(z)

    pts = np.column_stack([xs, ys, zs])
    plane = fit_plane_ransac(pts, inlier_threshold_m=1.0, min_inlier_ratio=0.5)
    if plane is not None:
        logger.info("LRF 기준 평면: %d점, 경사 %.3f°", len(zs), plane.slope_deg)
    return plane


def plane_from_metadata(metas: list,
                        panel_top_offset_m: float = 0.0) -> GroundPlane | None:
    """``절대고도 − 상대고도`` 폴백.

    Parameters
    ----------
    panel_top_offset_m : 지면 대비 기준면 높이 (m). 태양광 패널 상면을
        기준으로 삼으려면 지면에서 패널 상면까지의 높이를 넣는다.
        **절대고도가 아니라 상대 높이** — 이 값만이 사이트 간 이식 가능하다.
    """
    zs = []
    for meta in metas:
        alt = getattr(getattr(meta, "gps", None), "altitude", None)
        rel = getattr(meta, "relative_height", None)
        if alt is None or rel is None:
            continue
        if not (np.isfinite(alt) and np.isfinite(rel)):
            continue
        zs.append(float(alt) - float(rel))
    if not zs:
        return None
    z = float(np.median(zs)) + panel_top_offset_m
    spread = float(np.percentile(zs, 90) - np.percentile(zs, 10)) if len(zs) > 4 else 0.0
    logger.info("메타데이터 폴백 평면: Z=%.2f m (지면 %.2f + 오프셋 %.2f), "
                "이륙고도 산포 %.2f m", z, np.median(zs), panel_top_offset_m, spread)
    if spread > 3.0:
        logger.warning("이륙지점 기준 고도 산포가 %.1f m — 지형 기복이 크므로 "
                       "이 폴백 평면의 정확도는 낮다. BA 점군 사용을 권장.", spread)
    return GroundPlane.horizontal(z)


def estimate_ground_plane(*,
                          points_xyz: np.ndarray | None = None,
                          metas: list | None = None,
                          camera_xyz: np.ndarray | None = None,
                          panel_top_offset_m: float = 0.0,
                          allow_tilt: bool = True,
                          inlier_threshold_m: float = 0.5,
                          surface: str = "upper") -> GroundPlane:
    """우선순위에 따라 지상평면을 결정. 모든 경로가 실패하면 ``ValueError``.

    ``surface="upper"`` (기본) 는 점군에 두 층이 있으면 위층(태양광 패널
    상면) 을 기준면으로 삼는다. 지면 기준이 필요하면 ``"dominant"``.
    """
    if points_xyz is not None and len(points_xyz) >= 3:
        plane = fit_plane_ransac(points_xyz, allow_tilt=allow_tilt,
                                 inlier_threshold_m=inlier_threshold_m,
                                 surface=surface)
        if plane is not None:
            return plane
        logger.warning("BA 점군 평면 적합 실패 — LRF/메타데이터로 폴백")

    if metas is not None and camera_xyz is not None:
        plane = plane_from_lrf(metas, camera_xyz)
        if plane is not None:
            # LRF 는 조준점(=지면) 실측이므로 패널 상면 기준이 필요하면
            # 오프셋을 더해야 한다. BA 점군 경로와 기준면을 맞춘다.
            if panel_top_offset_m:
                plane = GroundPlane(plane.a, plane.b,
                                    plane.c + panel_top_offset_m,
                                    origin_xy=plane.origin_xy,
                                    inlier_rmse_m=plane.inlier_rmse_m,
                                    n_inliers=plane.n_inliers)
                logger.info("LRF 평면에 패널 상면 오프셋 +%.2f m 적용",
                            panel_top_offset_m)
            return plane

    if metas is not None:
        plane = plane_from_metadata(metas, panel_top_offset_m=panel_top_offset_m)
        if plane is not None:
            return plane

    raise ValueError(
        "지상평면을 추정할 수 없음 — BA 점군, LRF, 고도 메타데이터가 모두 없음"
    )

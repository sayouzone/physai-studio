"""BA 점군 → 성긴 DSM, 그리고 DSM 기반 정사보정.

왜 이제야 DSM 인가
------------------
이 파이프라인은 처음부터 "DSM 없이 평면 하나로" 를 목표로 했습니다. 점군이
147 개뿐일 때는 그것이 유일한 선택이었습니다. 그러나 2단계 BA 수정 이후
점군이 **231,713 개**가 됐고, 상황이 달라졌습니다.

단일 평면이 남긴 오차가 이제 지배적입니다. 이 현장은 높이가 뚜렷이 다른
두 층으로 되어 있습니다:

* 지면 ≈ 113.4 m (절대고도 − 상대고도)
* 패널 상면 ≈ 115.8 m (지면 + 가대 높이 약 2.4 m)

**평면 하나로는 둘 다 맞출 수 없습니다.** 평면을 어디에 두든 다른 층이
``Δh · k`` 만큼 밀립니다 (``k`` = off-nadir 비). 실측으로도 이 한계가 드러났습니다:

* 기준면을 113.67 로 두면 (mosaic_6) 지면이 밀리고,
* 105.45 로 두면 (mosaic_5) 패널이 밀립니다.

어느 쪽도 정답이 아니며, 두 결과의 국소 어긋남 지표가 서로 엎치락뒤치락한
것이 그 증거입니다 (안쪽 0.071 vs 0.091 m, 바깥 0.698 vs 0.417 m).

DSM 을 쓰면 이 오차 항이 **사라집니다.** 각 출력 픽셀이 자기 자리의 실제
높이로 역투영되므로 지면은 지면대로, 패널은 패널대로 제자리에 놓입니다.

성긴 격자로 충분한 이유
----------------------
필요한 것은 "층 구분" 이지 정밀 표고모델이 아닙니다. 셀 크기 0.5~1.0 m 면
패널 행(폭 약 1.3 m, 간격 2.72 m)을 분리하기에 충분하고, 231,713 점이면
1 m 격자 기준 셀당 평균 수십 점이 들어갑니다.

구멍(점이 없는 셀)은 주변에서 메우고, 마지막에 약하게 다듬습니다. 다듬기를
과하게 하면 층 경계가 뭉개져 DSM 의 이점이 사라지므로 기본값은 보수적입니다.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["DSM", "build_dsm", "warp_frame_dsm"]


class DSM:
    """성긴 표고 격자. ``height[i, j]`` 는 셀 중심의 표고 (m)."""

    def __init__(self, height: np.ndarray, x_min: float, y_max: float,
                 cell_m: float, n_points: int, coverage: float):
        self.height = height
        self.x_min = float(x_min)
        self.y_max = float(y_max)
        self.cell_m = float(cell_m)
        self.n_points = int(n_points)
        self.coverage = float(coverage)

    @property
    def shape(self):
        return self.height.shape

    def sample(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """투영좌표 (X, Y) 에서의 표고를 이중선형 보간으로."""
        u = (X - self.x_min) / self.cell_m - 0.5
        v = (self.y_max - Y) / self.cell_m - 0.5
        return cv2.remap(self.height,
                         u.astype(np.float32), v.astype(np.float32),
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def stats(self) -> dict:
        h = self.height[np.isfinite(self.height)]
        return {
            "cell_m": self.cell_m,
            "shape": list(self.shape),
            "n_points": self.n_points,
            "coverage": self.coverage,
            "z_min": float(np.percentile(h, 1)),
            "z_median": float(np.median(h)),
            "z_max": float(np.percentile(h, 99)),
            "relief_p99_m": float(np.percentile(h, 99) - np.percentile(h, 1)),
        }


def build_dsm(points_xyz: np.ndarray,
              bounds: tuple,
              *,
              cell_m: float = 0.8,
              min_pts_per_cell: int = 8,
              auto_cell: bool = True,
              target_pts_per_cell: float = 12.0,
              robust_sigma: float = 3.0,
              trim_band_m: float = 4.0,
              smooth_cells: float = 0.8,
              surface: str = "upper",
              plane=None,
              band_m: float = 4.0,
              max_relief_m: float = 8.0,
              min_coverage: float = 0.40) -> DSM | None:
    """BA 점군 → 성긴 DSM.

    Parameters
    ----------
    surface : ``"upper"`` 면 셀 안에서 **상위 분위수**를 취한다. 태양광
        단지에서 한 셀에 패널 상면과 그 아래 지면 점이 섞이면, 정사보정
        기준으로 삼아야 할 것은 **위쪽 표면**이다 (아래는 패널에 가려 보이지
        않는다). ``"median"`` 은 중앙값.
    robust_sigma : 셀 안에서 중앙값으로부터 이 배수의 MAD 를 넘는 점은 버린다.
        BA 이상치가 셀 높이를 끌고 가지 않게 한다.
    auto_cell : ``True`` 면 점 밀도에서 셀 크기를 자동 결정한다.

        ★ 실측 실패의 진짜 원인이 여기 있었다. 0.4 m 셀로 돌렸는데 점 밀도가
          7.9 점/m² 라 **셀당 1.3점**이었다. 중앙값·MAD 이상치 제거가 성립할
          수 없고, 채워진 셀은 이상치 한두 개가 그대로 높이가 된다. 충전율
          25.8%, 기복 35.3 m 가 그 결과다.

          셀당 ``target_pts_per_cell`` 점이 들어가도록 셀을 잡으면 통계가
          성립한다 (이 현장은 약 1.2 m).
    target_pts_per_cell : 자동 결정의 목표 셀당 점 수.
    trim_band_m : **격자에 넣기 전에** 전역 로버스트 평면에서 이 거리를 넘는
        점을 버린다.

        ★ 반복 텍스처의 오매칭은 재투영이 완벽한데 깊이만 틀린 점을 만든다.
          패널 셀 주기 0.23 m, 베이스라인 1.6 m (시선각 2°) 이면 한 칸
          어긋난 매칭이 **6.5 m 깊이오차**를 만들고, 재투영 게이트 3 px 로는
          절대 걸러지지 않는다. 실측 점군의 기복이 35.3 m 로 나온 이유다
          (지면~패널 상면은 2.4 m).

          셀 단위 MAD 만으로는 부족하다 — 한 셀 안의 점이 통째로 이상치일
          수 있기 때문이다. 전역 평면 기준으로 먼저 잘라내야 한다.
    smooth_cells : 마지막 가우시안 다듬기 σ (셀 단위). 과하게 주면 층 경계가
        뭉개져 DSM 의 이점이 사라진다.
    plane : 기준 평면 (``GroundPlane``). 주면 이 평면에서 ``band_m`` 밖의
        점을 **DSM 을 만들기 전에** 버린다.
    band_m : 평면 기준 수용 밴드 (m). 태양광 단지의 실제 기복은 지면~패널
        상면 2~3 m 이므로 ±4 m 면 넉넉하다.
    max_relief_m : 완성된 DSM 의 기복(1~99 백분위 차) 상한. 넘으면 **DSM 을
        폐기**하고 ``None`` 을 돌려준다.
    min_coverage : 점이 들어간 셀 비율 하한. 미만이면 폐기.

    ★ 왜 이런 게이트가 필요한가 (실측)
    ---------------------------------
    231,713 점짜리 BA 점군으로 0.4 m DSM 을 만들었더니:

        기복 24.28 m (기대 2~3 m), 충전율 18.9% (기대 ≥40%),
        표고 중앙값 105.6 m (기대 ≈113.4 m — 지면보다 7.8 m 아래)

    점이 많아졌다고 깨끗한 것은 아니었습니다. 오삼각측량 점이 대량 섞여
    z 산포가 24 m 였고, DSM 은 그것을 그대로 지형으로 믿어 픽셀마다 엉뚱한
    높이로 역투영했습니다. 결과는 단일 평면보다 나빴습니다
    (파손 블록 18.9% → 23.7%).

    **나쁜 점군으로 만든 DSM 은 평면보다 못합니다.** 그래서 통과 기준을
    두고, 못 넘으면 조용히 평면으로 되돌아갑니다.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 100:
        logger.warning("DSM 생성 불가: 유효 점 %d개", len(pts))
        return None

    # ---- 점군 정제: 평면 기준 밴드 밖은 오삼각측량으로 본다 --------------
    n_raw = len(pts)
    if plane is not None:
        z_ref = plane.height_at(pts[:, 0], pts[:, 1]) \
            if hasattr(plane, "height_at") else plane.c
        resid = pts[:, 2] - np.asarray(z_ref, dtype=np.float64)
    else:
        resid = pts[:, 2] - np.median(pts[:, 2])
    keep = np.abs(resid) <= band_m
    pts = pts[keep]
    logger.info("DSM 점군 정제: %d → %d개 (밴드 ±%.1f m, 제거 %.1f%%)",
                n_raw, len(pts), band_m, 100.0 * (1 - len(pts) / max(n_raw, 1)))
    if len(pts) < 100:
        logger.warning("DSM 생성 불가: 정제 후 점 %d개 — 점군 z 분포가 "
                       "기준면과 크게 어긋납니다 (BA 품질 확인 필요).", len(pts))
        return None

    # ---- 전역 로버스트 평면으로 조대 이상치 제거 ----------------------
    n_before = len(pts)
    ctr = pts.mean(axis=0)
    P = pts - ctr
    A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
    coef = np.array([0.0, 0.0, 0.0])
    for _ in range(3):
        resid = P[:, 2] - A @ coef
        med = np.median(resid)
        mad = np.median(np.abs(resid - med)) * 1.4826 + 1e-6
        keep = np.abs(resid - med) <= 3.0 * mad
        if keep.sum() < 100:
            break
        coef, *_ = np.linalg.lstsq(A[keep], P[keep, 2], rcond=None)
    resid = P[:, 2] - A @ coef
    band = np.abs(resid - np.median(resid)) <= trim_band_m
    if band.sum() >= 100:
        pts = pts[band]
    logger.info("DSM 전처리: 전역 평면 ±%.1f m 밖의 점 %d개 제거 "
                "(%d → %d, %.1f%%)", trim_band_m, n_before - len(pts),
                n_before, len(pts), 100.0 * len(pts) / max(n_before, 1))
    if n_before - len(pts) > 0.4 * n_before:
        logger.warning("  점군의 %.0f%% 가 이상치입니다 — 삼각측량 시선각 "
                       "하한(--min-tri-angle)을 올리는 것을 권합니다. "
                       "반복 텍스처에서 짧은 베이스라인은 깊이가 발산합니다.",
                       100.0 * (n_before - len(pts)) / max(n_before, 1))

    x_min, y_min, x_max, y_max = bounds

    if auto_cell:
        area = max((x_max - x_min) * (y_max - y_min), 1.0)
        density = len(pts) / area
        need = float(np.sqrt(target_pts_per_cell / max(density, 1e-9)))
        if need > cell_m * 1.15:
            logger.info("DSM 셀 자동 조정: %.2f → %.2f m "
                        "(점 밀도 %.1f 점/m², 목표 셀당 %.0f점). "
                        "요청한 셀은 셀당 %.1f점뿐이라 이상치 제거가 "
                        "성립하지 않습니다.",
                        cell_m, need, density, target_pts_per_cell,
                        density * cell_m ** 2)
            cell_m = need
        if cell_m > 1.5:
            logger.warning("DSM 셀이 %.2f m 입니다 — 패널 행 폭(약 1.3 m)보다 "
                           "커서 지면/패널 두 층을 분리하지 못합니다. "
                           "지형 경사만 반영되고 층 분리 효과는 없습니다. "
                           "층을 분리하려면 점군 밀도를 높여야 합니다.", cell_m)

    nx = int(np.ceil((x_max - x_min) / cell_m))
    ny = int(np.ceil((y_max - y_min) / cell_m))
    if nx < 4 or ny < 4 or nx * ny > 40_000_000:
        logger.warning("DSM 격자 크기 비정상: %d×%d (cell=%.2f m)", nx, ny, cell_m)
        return None

    j = ((pts[:, 0] - x_min) / cell_m).astype(np.int64)
    i = ((y_max - pts[:, 1]) / cell_m).astype(np.int64)
    keep = (i >= 0) & (i < ny) & (j >= 0) & (j < nx)
    i, j, z = i[keep], j[keep], pts[keep, 2]
    if len(z) < 100:
        logger.warning("DSM 생성 불가: 격자 안 점 %d개", len(z))
        return None

    flat = i * nx + j
    order = np.argsort(flat, kind="stable")
    flat_s, z_s = flat[order], z[order]
    starts = np.concatenate([[0], np.nonzero(np.diff(flat_s))[0] + 1,
                             [len(flat_s)]])

    height = np.full(ny * nx, np.nan)
    q = 0.80 if surface == "upper" else 0.50
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < min_pts_per_cell:
            continue
        zc = z_s[a:b]
        # 로버스트: 중앙값 ± robust_sigma·MAD 밖은 버림 (BA 이상치 방어).
        med = np.median(zc)
        mad = np.median(np.abs(zc - med)) * 1.4826 + 1e-6
        zc = zc[np.abs(zc - med) <= robust_sigma * mad]
        if len(zc) < 1:
            continue
        height[flat_s[a]] = np.quantile(zc, q)

    height = height.reshape(ny, nx)
    filled = np.isfinite(height)
    coverage = float(filled.mean())

    if coverage < min_coverage:
        logger.warning("DSM 폐기: 셀 충전율 %.1f%% < %.0f%% — 점군이 희박합니다. "
                       "--dsm-cell 을 키우거나(예: %.1f m) 단일 평면을 쓰세요.",
                       coverage * 100, min_coverage * 100, cell_m * 2)
        return None

    # 구멍 메우기: inpaint 는 큰 격자에서 느리므로 점진적 팽창 평균.
    h = height.copy()
    mask = ~filled
    if mask.any():
        work = np.where(filled, height, 0.0).astype(np.float32)
        cnt = filled.astype(np.float32)
        for _ in range(6):
            if not np.isnan(h[mask]).any():
                break
            k = np.ones((3, 3), np.float32)
            ws = cv2.filter2D(work, -1, k, borderType=cv2.BORDER_REPLICATE)
            cs = cv2.filter2D(cnt, -1, k, borderType=cv2.BORDER_REPLICATE)
            new = (cs > 0) & (cnt == 0)
            work[new] = ws[new] / cs[new]
            cnt[new] = 1.0
            h[new] = work[new]
        still = ~np.isfinite(h)
        if still.any():
            h[still] = np.nanmedian(height)

    if smooth_cells > 0:
        h = cv2.GaussianBlur(h.astype(np.float32), (0, 0), smooth_cells)

    dsm = DSM(h.astype(np.float32), x_min, y_max, cell_m, len(z), coverage)
    st = dsm.stats()
    if st["relief_p99_m"] > max_relief_m:
        logger.warning(
            "DSM 폐기: 기복 %.2f m 가 상한 %.1f m 를 넘습니다 (표고 %.1f~%.1f m). "
            "태양광 단지의 실제 기복은 지면~패널 상면 2~3 m 입니다 — 점군에 "
            "오삼각측량 점이 섞였다는 뜻이므로 단일 평면을 사용합니다.",
            st["relief_p99_m"], max_relief_m, st["z_min"], st["z_max"])
        return None
    logger.info("DSM 생성: %d×%d 셀 (%.2f m), 점 %d개, 충전율 %.1f%%, "
                "표고 %.2f~%.2f m (기복 %.2f m)",
                ny, nx, cell_m, len(z), coverage * 100,
                st["z_min"], st["z_max"], st["relief_p99_m"])
    if st["relief_p99_m"] < 0.5:
        logger.info("  기복이 %.2f m 로 작습니다 — 단일 평면과 큰 차이가 "
                    "없을 수 있습니다.", st["relief_p99_m"])
    return dsm


def warp_frame_dsm(fh, image: np.ndarray, dsm: DSM,
                   x_min: float, y_max: float, gsd_m: float,
                   u0: int, v0: int, out_w: int, out_h: int):
    """DSM 을 써서 한 프레임을 정사 warp (창 좌표 ``u0, v0`` 기준).

    평면 호모그래피 대신 **픽셀마다 자기 높이로** 공선조건을 푼다::

        diff = (X, Y, Z_dsm) − C
        px = cx − f · (R[0]·diff) / (R[2]·diff)
        py = cy − f · (R[1]·diff) / (R[2]·diff)

    부호 규약은 ``geometry.project_point`` 와 동일하다 (가시 조건
    ``R[2]·diff > 0``).

    Returns ``(warped, valid_mask)``.
    """
    xs = x_min + (u0 + np.arange(out_w, dtype=np.float64) + 0.5) * gsd_m
    ys = y_max - (v0 + np.arange(out_h, dtype=np.float64) + 0.5) * gsd_m
    X, Y = np.meshgrid(xs, ys)
    Z = dsm.sample(X, Y).astype(np.float64)

    C = np.asarray(fh.camera_xyz, dtype=np.float64)
    R = np.asarray(fh.R, dtype=np.float64)
    dX = X - C[0]; dY = Y - C[1]; dZ = Z - C[2]

    den = R[2, 0] * dX + R[2, 1] * dY + R[2, 2] * dZ
    vis = den > 1e-9
    den_safe = np.where(vis, den, 1.0)
    num_x = R[0, 0] * dX + R[0, 1] * dY + R[0, 2] * dZ
    num_y = R[1, 0] * dX + R[1, 1] * dY + R[1, 2] * dZ

    f = fh.intr.f_px
    px = fh.intr.cx - f * num_x / den_safe
    py = fh.intr.cy - f * num_y / den_safe

    inside = (vis & (px >= 0) & (px < fh.intr.width - 1)
              & (py >= 0) & (py < fh.intr.height - 1))
    px = np.where(inside, px, -1.0).astype(np.float32)
    py = np.where(inside, py, -1.0).astype(np.float32)

    warped = cv2.remap(image, px, py, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if warped.ndim == 2:
        warped = warped[:, :, None]
    return warped, inside

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
              min_pts_per_cell: int = 2,
              robust_sigma: float = 3.0,
              smooth_cells: float = 0.8,
              surface: str = "upper") -> DSM | None:
    """BA 점군 → 성긴 DSM.

    Parameters
    ----------
    surface : ``"upper"`` 면 셀 안에서 **상위 분위수**를 취한다. 태양광
        단지에서 한 셀에 패널 상면과 그 아래 지면 점이 섞이면, 정사보정
        기준으로 삼아야 할 것은 **위쪽 표면**이다 (아래는 패널에 가려 보이지
        않는다). ``"median"`` 은 중앙값.
    robust_sigma : 셀 안에서 중앙값으로부터 이 배수의 MAD 를 넘는 점은 버린다.
        BA 이상치가 셀 높이를 끌고 가지 않게 한다.
    smooth_cells : 마지막 가우시안 다듬기 σ (셀 단위). 과하게 주면 층 경계가
        뭉개져 DSM 의 이점이 사라진다.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 100:
        logger.warning("DSM 생성 불가: 유효 점 %d개", len(pts))
        return None

    x_min, y_min, x_max, y_max = bounds
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

    if coverage < 0.15:
        logger.warning("DSM 셀 충전율 %.1f%% — 점군이 너무 희박합니다. "
                       "cell_m 을 키우거나 단일 평면을 쓰세요.", coverage * 100)
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

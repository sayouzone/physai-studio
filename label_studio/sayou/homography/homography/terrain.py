"""저차 다항 지형면 — 경사지에서 단일 평면이 무너지는 것을 막는다.

왜 필요한가 (실측 근거)
-----------------------
EWP-서오창IC-2 는 **경사지**에 조성된 부지입니다. IR 모자이크 가운데 영역을
확대해 보니 패널 행이 크게 찢어져 있었습니다.

```
행 주기 6.19 m 인데
상단 경계 점프: 99% 가 2.01 m, 최대 5.82 m
8개 행이 44개 조각으로 끊김
```

패널 높이(1.5 m)만으로는 설명되지 않습니다. ``k=0.9`` 여도 1.35 m 입니다.
**경사 지형의 국소 기복**이 단일 평면에서 벗어난 것이 주원인입니다
(``Δh=5 m × k=0.4 = 2.0 m`` 로 실측과 맞습니다).

여기에 연직 제한 예외가 **37.2%** (그린환경센터 8.7% 의 4배)라 큰 ``k`` 가
곱해집니다. 경사지라 프레임당 유효 면적이 줄어든 결과입니다.

왜 DSM 도 2층도 답이 아닌가
---------------------------
* **DSM** (셀별 높이): 셀마다 로버스트 추정이 필요해 셀당 10점 이상을
  요구합니다. 그린환경센터에서 셀 충전율 26% 로 실패했고, 이 현장에서도
  ``build_failed`` 였습니다.
* **2층 모델** (지면/패널): 높이가 두 값 중 하나라는 가정인데, **경사지에서는
  지면 자체가 계속 변해** 가정이 깨집니다.

저차 다항이 되는 이유
--------------------
평면은 계수 3개(``a + bx + cy``)입니다. 2차 다항은 6개
(``+ dx² + exy + fy²``)로 경사지의 완만한 곡률을 잡습니다.

**결정적 장점은 전역 적합이라는 점입니다.** 셀별 추정과 달리 계수 6개를
수만 점으로 푸는 것이라, 점이 텍스처에 뭉쳐 있어도(= DSM 이 실패한 조건)
성립합니다. 점 4만 개면 계수당 6,600 점입니다.

한계 — **실측에서 실패했습니다**
--------------------------------
EWP-서오창IC-2 에 적용한 결과 **오히려 나빠졌습니다.**

| 실행 | 지형면 | 99% 단차 |
|---|---|---|
| m50 | 평면 | **1.69 m** |
| m52 | 2차 다항 | 2.22 m |
| m53 | 2차 다항 + 예외끔 | 1.93 m |

잘라낸 이미지에서 **중앙에 매끄러운 타원형 경계**가 나타났고, 그 안쪽만
행이 크게 어긋났습니다. 타원은 시임(다각형)일 수 없고 **2차 다항면의
등고선**입니다.

원인: 2차 다항은 한 방향으로만 휘는 단순한 곡면이라, 실제 경사지의 복잡한
기복을 맞추려다 **가장자리 점들에 끌려 중앙이 들리거나 꺼집니다**
(실측 `max_deviation_from_plane_m = 6.67 m`, `rms_poly` 는 여전히 1.37 m).
평면보다 잔차는 줄지만(gain 37.6%) **그 줄어든 방식이 중앙을 왜곡**합니다.

그래서 **기본값을 꺼 두었습니다.** 지형이 단순한 한 방향 경사이고 다항이
중앙을 왜곡하지 않는 것이 확인된 경우에만 쓰세요. 근본적으로는 저차 다항이
아니라 실제 지형 격자(DSM/MVS)가 필요합니다.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["fit_terrain_surface", "PolyTerrain"]


class PolyTerrain:
    """``GroundPlane`` 과 같은 인터페이스의 2차 다항 지형면.

    ``height_at(x, y)`` 만 쓰이므로 평면 자리에 그대로 끼울 수 있다.
    좌표는 원점을 빼고 스케일을 나눠 조건수를 낮춘다 (절대 좌표를 그대로
    쓰면 x² 항이 1e10 규모가 되어 적합이 무너진다 — 앞서 호모그래피에서
    같은 이유로 국소 원점을 도입했다).
    """

    def __init__(self, coef: np.ndarray, x0: float, y0: float, s: float,
                 degree: int = 2):
        self.coef = np.asarray(coef, dtype=np.float64)
        self.x0 = float(x0); self.y0 = float(y0); self.s = float(s)
        self.degree = int(degree)

    def _design(self, x, y):
        u = (np.asarray(x, dtype=np.float64) - self.x0) / self.s
        v = (np.asarray(y, dtype=np.float64) - self.y0) / self.s
        cols = [np.ones_like(u), u, v]
        if self.degree >= 2:
            cols += [u * u, u * v, v * v]
        return np.stack(cols, axis=-1)

    def height_at(self, x, y):
        return self._design(x, y) @ self.coef

    # 평면과 호환되는 속성 (기울기 점검 등에서 쓰인다)
    @property
    def normal(self):
        b, c = self.coef[1] / self.s, self.coef[2] / self.s
        n = np.array([-b, -c, 1.0])
        return n / np.linalg.norm(n)

    def __repr__(self) -> str:
        return (f"PolyTerrain(degree={self.degree}, "
                f"coef={np.round(self.coef, 4).tolist()})")


def fit_terrain_surface(points_xyz: np.ndarray,
                        plane,
                        *,
                        degree: int = 2,
                        band_m: float = 3.0,
                        min_points: int = 2000,
                        min_gain: float = 0.15,
                        max_extra_relief_m: float = 30.0):
    """BA 점군에 2차 다항면을 적합한다.

    Returns ``(PolyTerrain, info)`` 또는 ``(None, info)``.

    평면 대비 잔차가 ``min_gain`` 이상 줄지 않으면 **채택하지 않는다** —
    평지에서는 평면이 맞고, 계수를 늘리면 외삽만 불안정해진다.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    info: dict = {"n_points": int(len(pts)), "degree": degree}
    if len(pts) < min_points:
        info["reason"] = "points_too_few"
        return None, info

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # ★ **평면 기준으로 밴드를 자르면 안 된다.**
    #   원래는 `|z - 평면| <= band_m` 으로 잘랐는데, 경사지에서는
    #   **지형이 평면에서 벗어난 것 자체가 우리가 찾는 신호**다.
    #   곡률이 클수록 그 곡률을 만드는 점이 먼저 잘려나가(±5 m 휨이면 60%,
    #   ±10 m 면 30% 만 생존) 남은 점은 평면에 가까운 것들뿐이라 gain 이
    #   작게 나온다. 즉 이 필터가 경사지 감지를 스스로 막고 있었다
    #   (실측 EWP 경사지에서 terrain_fit 이 미채택된 원인).
    #
    #   그래서 **다항면을 먼저 대충 맞춘 뒤 그 잔차로** 이상치를 자른다.
    #   구조물·오매칭은 지형면에서 벗어나므로 여전히 걸러진다.
    x0, y0 = float(np.mean(x)), float(np.mean(y))
    s = float(max(np.std(x), np.std(y), 1.0))
    surf = PolyTerrain(np.zeros(6 if degree >= 2 else 3), x0, y0, s, degree)
    A_all = surf._design(x, y)
    coef0, *_ = np.linalg.lstsq(A_all, z, rcond=None)
    r0 = z - A_all @ coef0
    keep = np.abs(r0 - np.median(r0)) <= band_m
    if keep.sum() < min_points:
        info["reason"] = "band_filter_too_few"
        return None, info
    info["band_kept_frac"] = float(keep.mean())
    x, y, z = x[keep], y[keep], z[keep]

    A = surf._design(x, y)

    # IRLS 로 이상치 억제.
    w = np.ones(len(x))
    coef = np.zeros(A.shape[1])
    for _ in range(4):
        Aw = A * w[:, None]
        coef, *_ = np.linalg.lstsq(Aw, z * w, rcond=None)
        r = z - A @ coef
        sc = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w = np.minimum(1.0, 2.0 * sc / np.maximum(np.abs(r), 1e-9))
    surf.coef = coef

    rms_plane = float(np.sqrt(np.mean((z - np.asarray(
        plane.height_at(x, y), dtype=np.float64)) ** 2)))
    rms_poly = float(np.sqrt(np.mean((z - A @ coef) ** 2)))
    gain = 1.0 - rms_poly / max(rms_plane, 1e-9)
    info.update({"rms_plane_m": rms_plane, "rms_poly_m": rms_poly,
                 "gain": gain, "n_used": int(len(x))})

    # 다항이 평면 대비 얼마나 휘었는지 (외삽 폭주 점검)
    extra = float(np.max(np.abs(A @ coef - np.asarray(
        plane.height_at(x, y), dtype=np.float64))))
    info["max_deviation_from_plane_m"] = extra

    if gain < min_gain:
        logger.info("지형면 미채택: 평면 대비 잔차가 %.1f%% 밖에 줄지 않습니다 "
                    "(RMS %.3f → %.3f m, 하한 %.0f%%). 평지에서는 평면이 "
                    "맞습니다.", gain * 100, rms_plane, rms_poly,
                    min_gain * 100)
        info["reason"] = "gain_too_small"
        return None, info
    if extra > max_extra_relief_m:
        logger.warning("지형면 기각: 평면에서 최대 %.1f m 벗어나 외삽이 "
                       "불안정합니다 (한계 %.0f m).", extra,
                       max_extra_relief_m)
        info["reason"] = "deviation_too_large"
        return None, info

    logger.info("지형면 채택 (%d차): 잔차 RMS %.3f → %.3f m (%.0f%% 감소), "
                "평면 대비 최대 %.2f m 휨, 점 %d개 — 경사지에서 단일 평면이 "
                "만드는 기복변위를 줄입니다",
                degree, rms_plane, rms_poly, gain * 100, extra, len(x))
    return surf, info


def terrain_to_dsm(surf, bounds: tuple, cell_m: float = 1.0):
    """다항 지형면을 **기존 ``DSM`` 격자**로 굽는다.

    ★ 왜 이렇게 하는가 — 처음에는 ``plane`` 변수를 ``PolyTerrain`` 으로
      바꿔치기했는데, ``build_frame_homography`` 가 ``GroundPlane`` 의
      내부 구현(``local_coeffs`` 등)에 의존해서
      ``AttributeError: 'PolyTerrain' object has no attribute 'local_coeffs'``
      로 죽었습니다. 그 함수는 제 트리에 없는 ``homography.py`` 안에 있어
      **인터페이스를 확인하지 않고 추측한 것**이 원인이었습니다.

      반면 ``DSM`` 은 이미 있고 ``warp_frame_dsm`` 으로 검증된 경로입니다.
      다항면을 그 격자에 구우면 **알 수 없는 인터페이스를 건드리지 않고**
      비평면 지형을 그대로 쓸 수 있습니다.
    """
    x_min, y_min, x_max, y_max = bounds
    nx = int(np.ceil((x_max - x_min) / cell_m)) + 1
    ny = int(np.ceil((y_max - y_min) / cell_m)) + 1
    if nx < 2 or ny < 2 or nx * ny > 40_000_000:
        return None
    xs = x_min + (np.arange(nx) + 0.5) * cell_m
    ys = y_max - (np.arange(ny) + 0.5) * cell_m
    X, Y = np.meshgrid(xs, ys)
    h = np.asarray(surf.height_at(X, Y), dtype=np.float32)

    from .dsm import DSM
    dsm = DSM(h, x_min, y_max, cell_m, n_points=0, coverage=1.0)
    logger.info("지형면을 DSM 격자로 변환: %d×%d 셀 (%.1f m), 표고 %.2f~%.2f m",
                ny, nx, cell_m, float(h.min()), float(h.max()))
    return dsm


def diagnose_height_scatter(points_xyz: np.ndarray, plane, *,
                            cell_m: float = 5.0) -> dict | None:
    """평면 주위 산포가 **실제 지형**인지 **점군 잡음**인지 가른다.

    ★ 왜 이 판별이 필요한가 — EWP-서오창IC-2 에서 평면 잔차가 RMS 2.19 m
      였습니다. 이것이 실제 지형 기복이면 지형 모델이 답이지만, 점군
      잡음이면 지형 모델은 **잡음에 곡면을 맞추는 것**이라 오히려 해롭습니다.
      실제로 2차 다항을 적용했더니 단차가 1.69 → 2.22 m 로 나빠졌고,
      잘라낸 이미지에 **다항 등고선 모양의 타원 경계**가 나타났습니다.

    판별 방법: 같은 셀 안의 점들끼리 얼마나 흩어져 있는지(셀 내 산포)와,
    셀 중앙값들이 셀 사이에서 얼마나 변하는지(셀 간 변화)를 비교합니다.

    * **셀 간 변화 ≫ 셀 내 산포** → 높이가 위치에 따라 체계적으로 변한다
      = 실제 지형. 지형 모델이 의미 있습니다.
    * **셀 내 산포 ≈ 셀 간 변화** → 같은 자리에서도 높이가 흩어진다
      = 점군 잡음. 지형 모델은 잡음을 좇을 뿐입니다.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 2000:
        return None
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = z - np.asarray(plane.height_at(x, y), dtype=np.float64)

    ix = np.floor((x - x.min()) / cell_m).astype(np.int64)
    iy = np.floor((y - y.min()) / cell_m).astype(np.int64)
    key = ix * (iy.max() + 1) + iy
    order = np.argsort(key)
    key_s, r_s = key[order], r[order]
    bounds_idx = np.flatnonzero(np.diff(key_s)) + 1
    groups = np.split(r_s, bounds_idx)
    groups = [g for g in groups if len(g) >= 10]
    if len(groups) < 20:
        return None

    within = float(np.median([1.4826 * np.median(np.abs(g - np.median(g)))
                              for g in groups]))
    med = np.array([np.median(g) for g in groups])
    between = float(1.4826 * np.median(np.abs(med - np.median(med))))
    ratio = between / max(within, 1e-9)

    verdict = ("real_terrain" if ratio > 2.0
               else "point_noise" if ratio < 1.2 else "mixed")
    out = {"cell_m": cell_m, "n_cells": len(groups),
           "within_cell_scatter_m": within,
           "between_cell_variation_m": between,
           "ratio": ratio, "verdict": verdict}
    logger.info("높이 산포 판별: 셀 내 산포 %.2f m vs 셀 간 변화 %.2f m "
                "(비 %.1f) → %s",
                within, between, ratio,
                {"real_terrain": "실제 지형 기복 — 지형 모델이 의미 있음",
                 "point_noise": "점군 잡음 — 지형 모델은 잡음을 좇습니다",
                 "mixed": "섞여 있음 — 지형 모델 효과가 불확실"}[verdict])
    return out

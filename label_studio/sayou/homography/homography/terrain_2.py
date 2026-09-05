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

한계
----
완만한 곡률만 잡습니다. 계단·옹벽처럼 급격한 단차는 다항으로 표현되지
않으므로 그런 지형에서는 여전히 DSM 이나 조밀 매칭이 필요합니다.
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

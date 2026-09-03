"""2층 높이맵 — 점군 밀도가 아니라 **영상 분할**로 만든다.

왜 이것이 남은 오차의 답인가 (실측 근거)
------------------------------------------
현재 남은 어긋남은 셋 다 **하나의 식**으로 설명됩니다:

    어긋남 ≈ Δh × k        (Δh = 지면과 기준면의 높이차, k = 그 자리 off-nadir)

기준면은 상면 검출로 **패널 상면**에 놓여 있고, 지면은 그보다
``plane_above_ground_m = 1.23 m`` 아래입니다.

| 영역 | 실측 어긋남 | 함의 k = 어긋남/1.23 |
|---|---|---|
| 안쪽 | 0.039 m | 0.032 |
| 중간 | 0.107 m | 0.087 |
| 바깥 | 0.398 m | **0.323** |

바깥의 함의 k 0.323 은 연직 제한 상한 ``k ≤ 0.304`` 와 사실상 같고, 안쪽은
중복이 커서 시임이 연직 근처(k≈0.03)에 놓입니다. **세 영역이 모두 맞습니다.**

즉 **남은 오차의 거의 전부가 '평면 하나로 두 층(지면/패널 상면)을 덮은' 탓**
입니다. BA·초점거리·렌즈왜곡을 아무리 더 손봐도 이 항은 줄지 않습니다 —
그래서 최근 다섯 번의 실행이 정체였습니다.

왜 BA 점군 DSM 은 실패했나
--------------------------
``dsm.py`` 는 점군으로 격자를 채우려 했는데, 점 밀도 13.2 점/m² 에 셀
0.95 m 면 셀당 11.9 개가 기대되지만 실제 충전율이 **26.6%** 였습니다.
점이 균일하지 않고 **텍스처가 있는 곳(패널 모서리)에 뭉쳐** 있기 때문이고,
점을 더 늘려도 빈 셀은 그대로입니다.

이 모듈의 접근
--------------
**높이는 이미 알고 있습니다.** 필요한 것은 "어느 픽셀이 패널인가" 뿐입니다.

* 패널 상면 높이 = 현재 기준면 (상면 검출이 이미 찾아 놓음)
* 지면 높이 = 기준면 − ``plane_above_ground_m`` (LRF 가 독립적으로 측정)

그리고 패널은 **영상에서 잘 보입니다** — 실측 정사영상에서 Otsu 임계만으로
면적의 97% 가 5 ㎡ 초과 덩어리로 잡힙니다. 점군이 희박해도 상관없습니다.

절차:

1. 단일 평면으로 만든 **저해상도 정사영상**(라벨맵 패스에서 이미 만듦)에서
   패널을 분할한다.
2. 패널=상면, 그 외=지면 인 2층 높이 격자를 만든다.
3. 그 높이맵으로 정사보정을 다시 한다 (``dsm.warp_frame_dsm`` 재사용).

한계
----
* 1 회차 분할은 단일 평면 정사영상에서 하므로 패널 경계가 최대 Δh·k ≈
  0.37 m 어긋나 있습니다. 패널 폭이 1.3 m 이므로 마스크는 대체로 맞고,
  ``refine_iterations`` 로 한 번 더 돌리면 줄어듭니다.
* 패널이 아닌 밝은 구조물(콘크리트, 자갈)이 패널로 분류될 수 있습니다.
  면적·형태 필터로 거르지만 완전하지는 않습니다.
* 지면 기복(경사 0.83°)은 기준면 자체가 이미 반영하고 있습니다.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["segment_panels", "build_two_layer_dsm"]


def segment_panels(gray: np.ndarray, valid: np.ndarray, cell_m: float,
                   *, min_area_m2: float = 3.0,
                   close_m: float = 0.30,
                   open_m: float = 0.15) -> np.ndarray:
    """저해상도 정사영상에서 패널 마스크를 만든다.

    밝기 Otsu → 형태학 정리 → 작은 조각 제거. 실측 정사영상에서 이 방법으로
    패널 면적의 97% 가 5 ㎡ 초과 덩어리로 잡혔습니다.
    """
    g = gray.astype(np.uint8)
    vals = g[valid]
    if vals.size < 1000:
        return np.zeros_like(valid)
    thr, _ = cv2.threshold(vals, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((g > thr) & valid).astype(np.uint8)

    kc = max(int(round(close_m / cell_m)), 1)
    ko = max(int(round(open_m / cell_m)), 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((kc, kc), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((ko, ko), np.uint8), iterations=1)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_px = max(int(min_area_m2 / (cell_m * cell_m)), 4)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    out = keep[lab]

    frac = out[valid].mean() if valid.any() else 0.0
    logger.info("패널 분할: 임계 %.0f, 패널 면적비 %.1f%% "
                "(성분 %d개 중 %d㎡ 이상 %d개 유지)",
                thr, frac * 100, n - 1, int(min_area_m2), int(keep.sum()))
    return out


def build_two_layer_dsm(coarse_gray: np.ndarray, coarse_valid: np.ndarray,
                        bounds: tuple, cell_m: float,
                        plane, ground_drop_m: float,
                        *, smooth_m: float = 0.25,
                        min_area_m2: float = 3.0):
    """패널=기준면, 그 외=기준면−``ground_drop_m`` 인 2층 높이맵.

    Parameters
    ----------
    coarse_gray, coarse_valid : 단일 평면으로 만든 저해상도 정사영상과 유효
        마스크. 라벨맵 패스에서 이미 만들어진 것을 그대로 쓴다.
    plane : 현재 기준면 (상면 검출로 패널 상면에 놓여 있음).
    ground_drop_m : 기준면이 지면보다 얼마나 위인지 (로그·summary 의
        ``plane_above_ground_m``). 0 이하면 2층 모델이 성립하지 않는다.

    Returns
    -------
    ``dsm.DSM`` 또는 ``None``.
    """
    from .dsm import DSM

    if ground_drop_m <= 0.15:
        logger.info("2층 높이맵 생략: 기준면이 지면보다 %.2f m 위 — 층 차이가 "
                    "거의 없어 단일 평면과 같습니다.", ground_drop_m)
        return None

    panel = segment_panels(coarse_gray, coarse_valid, cell_m,
                           min_area_m2=min_area_m2)
    frac = panel[coarse_valid].mean() if coarse_valid.any() else 0.0
    if not (0.05 < frac < 0.95):
        logger.warning("2층 높이맵 생략: 패널 면적비 %.1f%% 가 비현실적입니다 "
                       "(5~95%% 밖). 분할이 실패했을 수 있습니다.", frac * 100)
        return None

    x_min, y_min, x_max, y_max = bounds
    ny, nx = coarse_gray.shape[:2]
    xs = x_min + (np.arange(nx) + 0.5) * cell_m
    ys = y_max - (np.arange(ny) + 0.5) * cell_m
    X, Y = np.meshgrid(xs, ys)

    # 기준면 자체가 경사(0.83°)를 담고 있으므로 그대로 평가한 뒤
    # 지면 픽셀만 아래로 내린다.
    Z = plane.height_at(X, Y).astype(np.float32)
    Z_ground = Z - float(ground_drop_m)
    height = np.where(panel, Z, Z_ground).astype(np.float32)

    # 층 경계를 약하게만 다듬는다 — 과하면 2층의 이점이 사라진다.
    sig = max(smooth_m / cell_m, 0.0)
    if sig > 0.1:
        height = cv2.GaussianBlur(height, (0, 0), sig)

    dsm = DSM(height, x_min, y_max, cell_m, int(panel.sum()), 1.0)
    logger.info("2층 높이맵: %d×%d 셀 (%.2f m), 패널 %.1f%% / 지면 %.1f%%, "
                "층 차이 %.2f m — 단일 평면이 남기던 Δh·k 오차를 없앱니다",
                ny, nx, cell_m, frac * 100, (1 - frac) * 100, ground_drop_m)
    return dsm

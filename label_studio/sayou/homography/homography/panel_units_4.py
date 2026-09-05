"""패널 모듈 단위 프레임 배정 — 시임이 패널을 가로지르지 않게 한다.

문제
----
지금 시임은 **픽셀 단위 승자독식**으로 정해집니다. 기하 점수와 차이 영상
비용이 픽셀마다 다르므로, 시임 경계가 패널 한가운데를 자유롭게 가로지릅니다.

패널 상면은 기준면보다 약 1 m 높아 프레임마다 ``Δh·k`` 만큼 다른 위치에
투영됩니다. 그래서 시임이 패널을 가로지르는 순간 **그 크기의 계단 단차**가
그대로 보입니다. 실측(그린환경센터 RGB):

* 패널 상단 경계의 열 간 점프: 99% 가 41 cm, 최대 99 cm
* 3 px 초과 점프가 전체 열의 4.9%

시임 비용에 패널 벌점을 주는 방법을 시도했지만(후속 33~35), 벌점은
"되도록 피하라" 일 뿐 **금지가 아니어서** 전체 기준으로는 효과가 없었습니다.

해법
----
배정 단위를 픽셀에서 **패널 모듈**로 올립니다. 한 모듈은 통째로 한 프레임에서
가져옵니다. 그러면 모듈 안에서는 기하가 일관되므로 **모듈이 찢어지지
않습니다.** 시임은 모듈 사이의 지면·그림자로만 지나갑니다.

모듈 경계에서는 인접 모듈이 다른 프레임일 수 있어 상대 어긋남이 남지만,
그 자리에는 이미 물리적인 틈(어두운 지면)이 있어 훨씬 덜 보입니다.

단위 크기
--------
단위는 **한 프레임이 통째로 덮을 수 있어야** 합니다. ``k ≤ 0.15`` 이고
비행고도 45 m 면 프레임당 유효 반경이 6.7 m 이므로, 3~4 m 단위는 내부에서
안전합니다. 덮지 못하면 그 단위는 픽셀 단위 배정으로 남깁니다.

패널 행은 길이가 30 m 를 넘어 통째로는 한 프레임에 안 들어갑니다. 그래서
연결성분을 그대로 쓰지 않고 **긴 축을 따라 잘라** 단위를 만듭니다.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["consolidate_labels_by_unit"]


def _unit_max_k(fh, x_min: float, y_max: float, g: float,
                u0: int, v0: int, u1: int, v1: int) -> float | None:
    """프레임이 이 사각 단위를 통째로 덮으면 그 안의 **최대 off-nadir 비**,
    못 덮으면 ``None``.

    ★ 처음에는 상수 상한(``max_offnadir_ratio``)으로 통과/탈락만 판정했는데,
      실측에서 단위의 **29.1%** 가 "통째로 덮는 프레임 없음" 으로 처리되지
      못했습니다. 라벨맵은 픽셀별 **완화된** k 상한을 쓰는데(연직 제한으로
      못 덮는 곳은 도달 가능한 최소 k 까지 열어 줌) 여기서는 하드 0.15 로
      잘랐기 때문입니다. 기준이 서로 달랐던 것입니다.

      그래서 통과/탈락 대신 **k 값을 돌려주고**, 후보 중 k 가 가장 작은
      프레임을 고릅니다. 픽셀별 배정과 같은 원리이고, 덮는 프레임이 하나라도
      있으면 반드시 배정됩니다.
    """
    H = fh.ortho_pixel_matrix(x_min, y_max, g)
    pts = np.array([[u0, v0], [u1, v0], [u0, v1], [u1, v1],
                    [(u0 + u1) * 0.5, (v0 + v1) * 0.5]], dtype=np.float64)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = hom[:, 2]
    if np.any(np.abs(w) < 1e-9):
        return None
    xy = hom[:, :2] / w[:, None]
    if not (np.all(xy[:, 0] >= 0) and np.all(xy[:, 0] < fh.intr.width)
            and np.all(xy[:, 1] >= 0) and np.all(xy[:, 1] < fh.intr.height)):
        return None
    h_cam = float(fh.camera_xyz[2]) - float(
        fh.plane.height_at(fh.camera_xyz[0], fh.camera_xyz[1]))
    if h_cam <= 1e-6:
        return None
    X = x_min + pts[:, 0] * g
    Y = y_max - pts[:, 1] * g
    k = np.hypot(X - fh.camera_xyz[0], Y - fh.camera_xyz[1]) / h_cam
    return float(k.max())


def consolidate_labels_by_unit(label_map: np.ndarray,
                               panel_mask: np.ndarray,
                               frames,
                               bounds: tuple,
                               g: float,
                               *,
                               unit_m: float = 3.0,
                               min_unit_px: int = 40,
                               max_k: float = 0.0,
                               hard_k_mult: float = 3.0,
                               coarse_img: np.ndarray | None = None,
                               glint_weight: float = 0.0,
                               glint_thresh: int = 240,
                               k_allow: np.ndarray | None = None
                               ) -> tuple[np.ndarray, dict]:
    """패널 단위로 라벨을 통일한다.

    Parameters
    ----------
    label_map : 픽셀 단위 승자 라벨 (-1 = 미채움). **제자리에서 바꾸지 않고**
        복사본을 돌려준다.
    panel_mask : 패널 영역 bool.
    unit_m : 단위 한 변의 목표 길이 (m). 한 프레임이 덮을 수 있어야 한다.
    max_k : 단위를 배정할 때 허용할 최대 off-nadir 비. 0 이면 검사 안 함.
    k_allow : 픽셀별 허용 k 맵이 있으면 그 중앙값을 단위별 상한으로 쓴다.
    """
    if label_map is None or panel_mask is None:
        return label_map, {}
    x_min, _, _, y_max = bounds[0], bounds[1], bounds[2], bounds[3]
    oh, ow = label_map.shape
    step = max(int(round(unit_m / max(g, 1e-9))), min_unit_px)

    out = label_map.copy()
    n_unit = n_done = n_nocover = n_single = 0

    n_cc, lab_cc, stats, _ = cv2.connectedComponentsWithStats(
        panel_mask.astype(np.uint8), 8)
    for ci in range(1, n_cc):
        if stats[ci, cv2.CC_STAT_AREA] < step * step // 4:
            continue
        cx0 = stats[ci, cv2.CC_STAT_LEFT]; cw = stats[ci, cv2.CC_STAT_WIDTH]
        cy0 = stats[ci, cv2.CC_STAT_TOP]; ch = stats[ci, cv2.CC_STAT_HEIGHT]
        # 긴 축을 따라 자른다 — 패널 행은 한 프레임에 통째로 안 들어간다.
        for v0 in range(cy0, cy0 + ch, step):
            for u0 in range(cx0, cx0 + cw, step):
                u1 = min(u0 + step, ow) - 1
                v1 = min(v0 + step, oh) - 1
                if u1 <= u0 or v1 <= v0:
                    continue
                sub_p = (lab_cc[v0:v1 + 1, u0:u1 + 1] == ci)
                if sub_p.sum() < step * step * 0.15:
                    continue          # 이 단위에 패널이 거의 없음
                n_unit += 1
                sub_l = out[v0:v1 + 1, u0:u1 + 1][sub_p]
                sub_l = sub_l[sub_l >= 0]
                if sub_l.size == 0:
                    continue
                cand, cnt = np.unique(sub_l, return_counts=True)
                if cand.size == 1:
                    n_single += 1
                    continue          # 이미 한 프레임 — 손댈 것 없음
                order = cand[np.argsort(-cnt)]
                _k = max_k
                if k_allow is not None:
                    _k = float(np.median(k_allow[v0:v1 + 1, u0:u1 + 1]))
                # 후보 중 **단위를 통째로 덮으면서 비용이 가장 낮은** 프레임.
                #   비용 = k + glint_weight × (그 프레임이 이 단위에서 만드는
                #   포화 비율). k 만 보면 반사(glint)로 하얗게 날아간 프레임이
                #   뽑힐 수 있는데, 실측에서 **패널의 5.0% 가 포화(240+)** 라
                #   판독이 불가능합니다. 단위 배정이 이미 있으므로 여기서
                #   같이 고르는 것이 자연스럽습니다.
                chosen, best_c, best_k = -1, np.inf, np.inf
                for idx in order[:8]:
                    kk = _unit_max_k(frames[int(idx)], x_min, y_max, g,
                                     u0, v0, u1, v1)
                    if kk is None:
                        continue
                    cost = kk
                    if glint_weight > 0 and coarse_img is not None:
                        # 이 단위에서 그 프레임이 실제로 칠한 픽셀의 포화도.
                        own = (label_map[v0:v1 + 1, u0:u1 + 1] == idx) & sub_p
                        if own.sum() > 20:
                            sub_i = coarse_img[v0:v1 + 1, u0:u1 + 1][own]
                            cost += glint_weight * float(
                                (sub_i >= glint_thresh).mean())
                    if cost < best_c:
                        # ★ 비용(k + 반사)과 실제 k 를 **따로** 들고 간다.
                        #   아래 절대 상한 검사는 k 로 해야 하는데 비용으로
                        #   비교하면 반사 항 때문에 멀쩡한 프레임이 걸린다.
                        chosen, best_c, best_k = int(idx), cost, kk
                if chosen >= 0 and max_k > 0 and best_k > max_k * hard_k_mult:
                    chosen = -1     # 절대 상한을 크게 넘으면 포기
                if chosen < 0:
                    n_nocover += 1
                    continue          # 통째로 덮는 프레임 없음 → 픽셀 배정 유지
                blk = out[v0:v1 + 1, u0:u1 + 1]
                blk[sub_p] = chosen
                n_done += 1

    info = {
        "units": n_unit,
        "unified": n_done,
        "already_single": n_single,
        "no_full_cover": n_nocover,
        "unit_m": float(step * g),
    }
    if n_unit:
        logger.info(
            "패널 단위 프레임 배정: 단위 %d개 (%.1f m), 통일 %d개, "
            "이미 단일 %d개, 통째로 덮는 프레임 없음 %d개 — 시임이 패널을 "
            "가로지르지 않게 합니다",
            n_unit, info["unit_m"], n_done, n_single, n_nocover)
    return out, info

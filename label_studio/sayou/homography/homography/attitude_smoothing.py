"""약한 관측 프레임의 자세를 이웃 프레임에서 보간 — 재촬영 없이 찢어짐을 줄인다.

배경 (확정된 원인)
------------------
IR 은 특징이 RGB 보다 30~100배 적습니다(대응 12~51개 vs 963~1562개).
관측이 20개 근처인 프레임에서 BA 가 자세(특히 kappa, 광축 회전각)를
90~180° 씩 흔드는 것을 실측으로 확인했습니다 (대응 수와 BA 후 인접
프레임 어긋남의 상관계수 −0.86).

RGB 자신도 같은 현상(kappa 급변)이 있지만 무해합니다 — 그 프레임의
X/Y/Z/omega/phi 가 **같은 최적화 안에서** 그 kappa 에 맞춰 함께 풀렸기
때문입니다. 반면 관측이 그 프레임 자체에 너무 적으면, 옆 프레임과 무관하게
재투영오차만 작은 엉뚱한 국소최적해로 혼자 수렴합니다.

Sitemark 등 상업 SfM 이 같은 원본으로 정상 결과를 내는 것은, 비행이
물리적으로 **매끄러운 궤적**이라는 사실을 명시적 제약으로 쓰기 때문입니다
— 관측이 부족한 프레임은 고립된 최적화 대신 이웃의 매끄러운 흐름에서
보간됩니다.

왜 쿼터니언(SO(3))인가 — 오일러가 아니라
-----------------------------------------
앞서 RGB 의 BA 자세(omega/phi/kappa) 를 IR 에 그대로 옮기는 방법을
시도했다가 실패했습니다. 원인은 kappa 를 다른 (X,Y,Z,omega,phi) 조합에
떼어 붙이면 회전행렬 전체가 무효해지기 때문이었습니다.

이번 방법은 **같은 센서 안에서, 회전행렬을 직접(오일러 각으로 안 갔다가)
보간**합니다. 쿼터니언 SLERP 는 ±180° 랩어라운드가 없고, 두 회전 사이의
최단 경로를 항상 찾습니다. 오일러 각 kappa 만 평균 내면 −179° 와 +179°
사이를 최장 경로(358°)로 도는 실수를 할 수 있는데, 쿼터니언은 이 문제가
없습니다.

무엇을 바꾸는가
--------------
* **위치는 건드리지 않습니다.** RTK 로 고정된 값 그대로입니다.
* 관측이 임계값 미만인 프레임의 **회전행렬만** 양옆의 관측 충분한
  프레임 사이에서 SLERP 로 대체합니다.
* 관측이 충분한 프레임은 전혀 건드리지 않습니다.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["smooth_weak_attitudes", "rotmat_to_quat", "quat_slerp",
          "quat_to_rotmat"]


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 회전행렬 → 쿼터니언 (w, x, y, z). 표준 Shepperd 방법."""
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """쿼터니언 (w,x,y,z) → 3×3 회전행렬."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """구면 선형보간. ``t=0`` 이면 q0, ``t=1`` 이면 q1.

    ±180° 랩어라운드가 없다 — 두 쿼터니언의 내적 부호를 맞춰(최단 경로)
    보간하므로 오일러 각처럼 최장 경로로 도는 일이 없다.
    """
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1
        d = -d
    d = min(d, 1.0)
    if d > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / np.linalg.norm(r)
    theta0 = np.arccos(d)
    theta = theta0 * t
    q2 = q1 - q0 * d
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def _quat_angle_deg(qa, qb) -> float:
    """[sayou-patch] 두 쿼터니언 사이 회전각 (deg). 부호 모호성 처리 포함."""
    d = float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb)))
    return float(np.degrees(2.0 * np.arccos(min(abs(d), 1.0))))


def smooth_weak_attitudes(rotations: list,
                          obs_counts: np.ndarray,
                          *,
                          min_obs: int = 60,
                          max_gap: int = 8,
                          ref_rotations=None,          # [sayou-patch]
                          max_anchor_angle_deg: float = 30.0):  # [sayou-patch]
    """관측이 부족한 프레임의 회전행렬을 이웃에서 보간한다.

    Parameters
    ----------
    rotations : 프레임 순서(촬영 순서)대로 나열된 3×3 회전행렬 리스트.
    obs_counts : 같은 순서의 프레임별 tie point 관측 수.
    min_obs : 이 미만이면 '약한 프레임' 으로 보고 보간 대상으로 삼는다.
    max_gap : 양옆으로 이 프레임 수 이내에 관측 충분한 프레임이 없으면
        보간하지 않는다 (너무 먼 보간은 신뢰할 수 없다).

    Returns
    -------
    (new_rotations, info) — ``info`` 에 스무딩된 프레임 수·각도 변화 요약.
    """
    n = len(rotations)
    obs = np.asarray(obs_counts, dtype=np.float64)
    if len(obs) != n:
        raise ValueError(f"rotations({n})와 obs_counts({len(obs)}) 길이 불일치")
    weak = obs < min_obs
    strong_idx = np.flatnonzero(~weak)
    if len(strong_idx) < 2:
        logger.warning("관측 충분한 프레임이 %d개뿐이라 스무딩을 건너뜁니다",
                       len(strong_idx))
        return list(rotations), {"smoothed": 0, "skipped_no_anchor": 0}

    quats = [rotmat_to_quat(R) for R in rotations]
    # [sayou-patch] 앵커 적격성은 **초기 자세(RTK+짐벌)** 로 판단한다.
    #   BA 결과로 판단하면, 지금 의심하고 있는 값으로 의심 대상을 거르는
    #   순환이 된다. ref 가 없으면 게이트를 적용하지 않는다(기존 동작).
    ref_q = None
    if ref_rotations is not None and len(ref_rotations) == n:
        ref_q = [rotmat_to_quat(R) for R in ref_rotations]
    n_anchor_reject = 0
    new_rot = list(rotations)
    n_smoothed = 0
    n_skipped = 0
    angle_changes = []

    for i in np.flatnonzero(weak):
        # 양옆에서 가장 가까운 '강한' 프레임을 찾는다.
        left = strong_idx[strong_idx < i]
        right = strong_idx[strong_idx > i]
        li = left[-1] if len(left) else None
        ri = right[0] if len(right) else None

        if li is not None and (i - li) > max_gap:
            li = None
        if ri is not None and (ri - i) > max_gap:
            ri = None

        # [sayou-patch] 비행선을 넘는 앵커를 배제한다. 왕복 스캔에서
        #   선회를 사이에 둔 앵커로 SLERP 하면 자세가 통째로 틀어진다.
        if ref_q is not None and max_anchor_angle_deg < 180.0:
            if li is not None and _quat_angle_deg(
                    ref_q[li], ref_q[i]) > max_anchor_angle_deg:
                li = None
                n_anchor_reject += 1
            if ri is not None and _quat_angle_deg(
                    ref_q[ri], ref_q[i]) > max_anchor_angle_deg:
                ri = None
                n_anchor_reject += 1
            # 양쪽이 남았는데 서로 다른 비행선이면 보간 자체가 무의미하다.
            if (li is not None and ri is not None
                    and _quat_angle_deg(ref_q[li],
                                        ref_q[ri]) > max_anchor_angle_deg):
                ri = None
                n_anchor_reject += 1

        if li is None and ri is None:
            n_skipped += 1
            continue
        if li is None:
            q_new = quats[ri]
        elif ri is None:
            q_new = quats[li]
        else:
            t = (i - li) / float(ri - li)
            q_new = quat_slerp(quats[li], quats[ri], t)

        R_new = quat_to_rotmat(q_new)
        # 변화량 기록 (회전각, deg)
        R_old = np.asarray(rotations[i], dtype=np.float64)
        dR = R_old.T @ R_new
        cos_a = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
        angle_changes.append(float(np.degrees(np.arccos(cos_a))))
        new_rot[i] = R_new
        n_smoothed += 1

    info = {
        "total_frames": n,
        "weak_frames": int(weak.sum()),
        "smoothed": n_smoothed,
        "skipped_no_anchor": n_skipped,
        "min_obs": min_obs,
        # [sayou-patch]
        "anchor_rejected": int(n_anchor_reject),
        "max_anchor_angle_deg": float(max_anchor_angle_deg),
    }
    if angle_changes:
        info["angle_change_median_deg"] = float(np.median(angle_changes))
        info["angle_change_max_deg"] = float(np.max(angle_changes))
    logger.info(
        "자세 스무딩: 약한 프레임 %d개 중 %d개를 이웃(SLERP)으로 보정 "
        "(%d개는 앵커 없음 — max_gap=%d 프레임 이내에 강한 프레임 없음). "
        "회전각 변화 중앙값 %.2f°, 최대 %.2f°",
        info["weak_frames"], n_smoothed, n_skipped, max_gap,
        info.get("angle_change_median_deg", 0.0),
        info.get("angle_change_max_deg", 0.0))
    return new_rot, info


def detect_trajectory_deviants(rotations: list,
                               *,
                               window: int = 12,
                               angle_thresh_deg: float = 12.0,
                               n_passes: int = 3):
    """관측 수와 무관하게, 이웃과 얼마나 다른가로 이상 프레임을 찾는다.

    ★★ **실측에서 실패했습니다. 파이프라인에서 사용하지 않습니다.** ★★

    EWP-서오창IC-2 IR 909프레임에 적용한 결과 **608개(67%)를 이상으로
    오판**했습니다. 이상이 과반이 되니 앵커(정상 이웃)가 부족해져 308개는
    보정도 못 받고 321개는 오염된 이웃으로 보간돼, 패널 어긋남이
    0.0284 → 0.0493 m 로 나빠졌습니다.

    원인: 이 비행은 왕복 스캔이라 kappa 가 +90° 와 −94° 를 오갑니다
    (비행선 전환 85곳). 인접 프레임끼리는 매우 매끄럽지만(차이 중앙값
    0.1°), ±12프레임 창의 평균과 비교하면 **선회 구간의 정상 프레임도
    32° 이상 벌어집니다**(전체 편차 중앙값 32.6°). 검출기가 선회 자체를
    이상으로 오인한 것입니다.

    제 합성 검증(이상 10%, 직선 궤적)이 실제 비행의 선회를 반영하지 않은
    것이 근본 원인이었습니다. 창 안 평균과의 비교라는 접근 자체가 왕복
    스캔 비행에는 맞지 않습니다.

    아래는 원래 의도였습니다 (참고용):

    ★ 왜 필요한가 — 실측(EWP-서오창IC-2)에서 관측 수 임계값을 60→150 으로
      올려도(대상 59→92개) 결과가 거의 그대로였습니다. 남은 왜곡은
      사이트 진입로(도로) 를 지나는 프레임들에서 나는데, 도로 같은
      **직선·저텍스처 지형**은 SIFT 매칭점이 그 선을 따라서만 분포합니다
      — 점이 몇 개든 도로와 나란한 방향의 위치는 거의 구속되지 않습니다
      (구경문제, aperture problem). 즉 **관측 개수로는 이 프레임들을
      못 찾아냅니다.**

      대신 각 프레임의 회전이 그 프레임을 뺀 주변 프레임들의 회전과
      실제로 얼마나 다른지를 직접 잽니다. 이건 매칭 품질과 무관하게
      "이 프레임의 자세가 비행경로의 매끄러운 흐름에서 벗어났는가" 를
      바로 답합니다.

    Parameters
    ----------
    window : 이웃으로 볼 반경(프레임 수). 자기 자신은 제외한다.
    angle_thresh_deg : 이웃들의 평균 회전과 이 각도 이상 다르면 이상으로
        본다.

    Returns
    -------
    (deviant_mask, deviations_deg)
    """
    # ★ 연속된 이상 구간이 창(window)보다 길면 '이웃 평균' 자체가
    #   이상값에 오염됩니다. 실측 검증에서 6프레임 연속 이상 구간을
    #   window=5 로 봤더니 절반만 잡히고, 그걸로 스무딩하니 더 나빠졌습니다
    #   (이상 프레임이 서로를 정상으로 오인). 창을 넓히고, **이미 이상으로
    #   잡힌 프레임은 다음 패스의 평균에서 제외**하는 반복 정제로
    #   해결합니다.
    n = len(rotations)
    quats = [rotmat_to_quat(R) for R in rotations]
    deviant = np.zeros(n, dtype=bool)
    devs = np.zeros(n, dtype=np.float64)

    for _pass in range(max(n_passes, 1)):
        new_deviant = deviant.copy()
        for i in range(n):
            lo, hi = max(0, i - window), min(n, i + window + 1)
            # 이전 패스에서 이상으로 잡힌 이웃은 평균에서 뺀다.
            neigh = [j for j in range(lo, hi) if j != i and not deviant[j]]
            if len(neigh) < 3:
                neigh = [j for j in range(lo, hi) if j != i]
            if len(neigh) < 2:
                continue
            ref = quats[neigh[0]]
            acc = np.zeros(4)
            for j in neigh:
                qj = quats[j]
                if np.dot(qj, ref) < 0:
                    qj = -qj
                acc += qj
            q_mean = acc / np.linalg.norm(acc)
            qi = quats[i]
            if np.dot(qi, q_mean) < 0:
                qi = -qi
            d = float(np.clip(np.dot(qi, q_mean), -1.0, 1.0))
            angle = np.degrees(2 * np.arccos(d))
            devs[i] = angle
            new_deviant[i] = angle > angle_thresh_deg
        if np.array_equal(new_deviant, deviant):
            deviant = new_deviant
            break
        deviant = new_deviant
    return deviant, devs

"""DJI 짐벌 자세 → 사진측량 회전행렬 R 및 (ω, φ, κ).

파이프라인에서 이 모듈이 메우는 구멍
------------------------------------
``camera_pose.compute_camera_axes_from_gimbal`` 은 짐벌 각을 **OpenCV 카메라
축 (Z=전방, X=우, Y=하)** 의 ENU 표현으로 준다. 반면 ``geometry`` /
``sfm.bundle_adjustment`` / ``sfm.triangulation`` 은 **사진측량 ω-φ-κ** 를
쓴다. 두 세계를 잇는 변환이 코드베이스에 없어서, BA 의 ``initial_cameras``
초기값을 만들 방법이 없었다. 이 모듈이 그 변환이다.

규약 (수치 검증 완료 — ``selftest.py``)
---------------------------------------
``geometry.project_point`` 의 공선조건::

    px = cx − f·(R[0]·diff) / (R[2]·diff)
    py = cy − f·(R[1]·diff) / (R[2]·diff)

이 식이 OpenCV 픽셀 규약 (x 우, y 하) 과 **정확히 일치**하려면 R 의 행이::

    R[0] = −axis_right
    R[1] = −axis_down
    R[2] = +axis_forward   (광축)

이어야 한다 (det R = +1 확인). 즉 광축은 카메라 좌표계 **+Z** 이고,
가시 조건은 ``(R[2]·diff) > 0``.

``rotation_matrix(ω,φ,κ)`` 의 역분해::

    φ = asin(R[0,2])
    κ = atan2(−R[0,1], R[0,0])
    ω = atan2(−R[1,2], R[2,2])

nadir + 정북 (yaw=0, pitch=−90) 은 ``(ω, φ, κ) = (−180°, 0°, −180°)`` 로
떨어진다. ω 가 ±180° 근처인 것은 이 규약의 정상 동작이며, φ 가 0 근처라
짐벌락도 없다. BA 가 최적화하기에 문제 없는 영역이다.

⚠ 짐벌락: ``|φ| → 90°`` (광축이 정확히 동/서 수평) 에서는 ω 와 κ 가 축퇴한다.
nadir 촬영에서는 도달하지 않지만, 사각 촬영 데이터에서는
``check_gimbal_lock()`` 으로 감시할 것.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "camera_axes_enu",
    "rotation_from_gimbal",
    "rotation_from_metadata",
    "decompose_to_opk",
    "opk_from_gimbal",
    "check_gimbal_lock",
    "angle_from_nadir_deg",
]

# 짐벌락 경고 임계값 (deg). |φ| 가 이 값을 넘으면 ω/κ 축퇴 위험.
GIMBAL_LOCK_WARN_DEG = 80.0


# ---------------------------------------------------------------------------
# 짐벌 각 → 카메라 축 (ENU)
# ---------------------------------------------------------------------------
def camera_axes_enu(gimbal_yaw_deg: float,
                    gimbal_pitch_deg: float,
                    gimbal_roll_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DJI 짐벌 각 → (right, down, forward) 축의 ENU 단위벡터.

    DJI 표기::

        GimbalYawDegree   : 정북 기준 시계방향 (N=0, E=90, S=180, W=−90)
        GimbalPitchDegree : 수평 기준 (0=수평, −90=직하방)
        GimbalRollDegree  : 광축 주변 회전 (보통 ≈0)

    광축을 삼각함수로 **직접** 만들고 거기서 나머지 축을 Gram-Schmidt 로
    세운다. 회전행렬 3개를 곱하는 방식보다 부호 실수가 없고 수치적으로도
    안정적이다 (``camera_pose.py`` 와 동일한 전략).

    Returns
    -------
    (axis_right, axis_down, axis_forward) — 각각 ENU 단위벡터.
    """
    cy = np.radians(gimbal_yaw_deg)
    p = np.radians(gimbal_pitch_deg)
    r = np.radians(gimbal_roll_deg)

    fwd = np.array([np.cos(p) * np.sin(cy),
                    np.cos(p) * np.cos(cy),
                    np.sin(p)])
    fwd /= np.linalg.norm(fwd)

    # roll=0 일 때의 수평 우측 방향 (yaw 기준 +90°).
    right0 = np.array([np.cos(cy), -np.sin(cy), 0.0])
    if abs(right0 @ fwd) > 0.9999:
        # 광축이 수평 우측과 평행 — nadir 에서는 도달 불가하지만 방어.
        right0 = np.array([1.0, 0.0, 0.0])

    right0 = right0 - (right0 @ fwd) * fwd
    right0 /= np.linalg.norm(right0)
    down0 = np.cross(fwd, right0)
    down0 /= np.linalg.norm(down0)

    cr, sr = np.cos(r), np.sin(r)
    right = cr * right0 + sr * down0
    down = -sr * right0 + cr * down0
    return right, down, fwd


# ---------------------------------------------------------------------------
# 카메라 축 → 사진측량 R
# ---------------------------------------------------------------------------
def rotation_from_gimbal(gimbal_yaw_deg: float,
                         gimbal_pitch_deg: float,
                         gimbal_roll_deg: float = 0.0) -> np.ndarray:
    """짐벌 각 → 3×3 world(ENU/투영) → camera 회전행렬.

    ``geometry.project_point`` / ``camera_projection_matrix`` 와 부호가
    일치하는 R. ``det R = +1``, ``R·Rᵀ = I`` 를 만족한다.
    """
    right, down, fwd = camera_axes_enu(gimbal_yaw_deg, gimbal_pitch_deg, gimbal_roll_deg)
    R = np.vstack([-right, -down, fwd])
    # 수치 오차 누적 방지 — 가장 가까운 정규직교 행렬로 투영.
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


def rotation_from_metadata(meta) -> np.ndarray:
    """``ImageMetadata`` → R. ``orientation = [yaw, pitch, roll]`` (deg) 사용."""
    return rotation_from_gimbal(meta.gimbal_yaw_deg,
                                meta.gimbal_pitch_deg,
                                meta.gimbal_roll_deg)


# ---------------------------------------------------------------------------
# R ↔ (ω, φ, κ)
# ---------------------------------------------------------------------------
def decompose_to_opk(R: np.ndarray) -> tuple[float, float, float]:
    """3×3 R → (ω, φ, κ) radian. ``geometry.rotation_matrix`` 의 역함수.

    ``rotation_matrix(*decompose_to_opk(R)) == R`` 를 만족 (검증 오차 <1e-15).
    """
    R = np.asarray(R, dtype=np.float64)
    phi = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    kappa = np.arctan2(-R[0, 1], R[0, 0])
    omega = np.arctan2(-R[1, 2], R[2, 2])
    return float(omega), float(phi), float(kappa)


def opk_from_gimbal(gimbal_yaw_deg: float,
                    gimbal_pitch_deg: float,
                    gimbal_roll_deg: float = 0.0) -> tuple[float, float, float]:
    """짐벌 각 → (ω, φ, κ) radian. BA ``initial_cameras`` 초기값용."""
    return decompose_to_opk(
        rotation_from_gimbal(gimbal_yaw_deg, gimbal_pitch_deg, gimbal_roll_deg)
    )


def check_gimbal_lock(phi_rad: float, warn_deg: float = GIMBAL_LOCK_WARN_DEG) -> bool:
    """``|φ|`` 가 짐벌락 근처면 ``True`` (경고 로그 포함)."""
    phi_deg = abs(np.degrees(phi_rad))
    if phi_deg > warn_deg:
        logger.warning(
            "짐벌락 위험: |φ|=%.1f° > %.1f° — ω/κ 가 축퇴해 BA 가 불안정할 수 있음",
            phi_deg, warn_deg,
        )
        return True
    return False


def angle_from_nadir_deg(R: np.ndarray) -> float:
    """광축과 연직 하방(−Up) 사이 각도. nadir 판정/품질 필터용."""
    fwd = np.asarray(R)[2]                      # R[2] = +forward
    cos_a = float(np.clip(fwd @ np.array([0.0, 0.0, -1.0]), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))

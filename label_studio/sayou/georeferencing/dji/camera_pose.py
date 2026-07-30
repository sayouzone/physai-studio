"""
src/georeferencing/camera_pose.py
드론·짐벌 자세 → 카메라 회전 행렬 (ENU 기준)

좌표계 관습
-----------
ENU (월드):   X=East, Y=North, Z=Up
NED (항법):   X=North, Y=East, Z=Down
FRD (기체):   X=Forward, Y=Right, Z=Down
OpenCV (카메라): X=Right, Y=Down, Z=Forward(광축)

DJI 짐벌 표기
-------------
- yaw(compass): 정북 기준 시계방향 (N=0, E=90, S=180, W=-90/270)
- pitch: 수평 기준 (0=수평, -90=직하방/nadir, +90=직상방)
- roll: 광축 주변 회전 (보통 0)

두 가지 경로를 제공하며 결과는 수치적으로 동일하다.
  1) compute_camera_axes_from_gimbal — 광축을 직접 구성 (권장, 부호 혼동 없음)
  2) euler_to_rotation_matrix 기반 행렬 체인 — 드론+짐벌 상대 자세가 필요할 때
"""
import numpy as np
from dataclasses import dataclass

# FRD 기체 좌표 → OpenCV 카메라 좌표
FRD_TO_OPENCV = np.array([
    [0, 1, 0],   # cam X(Right)   = body Y(Right)
    [0, 0, 1],   # cam Y(Down)    = body Z(Down)
    [1, 0, 0],   # cam Z(Forward) = body X(Forward)
], dtype=float)

# NED → ENU (자기역행렬)
NED_TO_ENU = np.array([
    [0, 1,  0],
    [1, 0,  0],
    [0, 0, -1],
], dtype=float)


@dataclass
class DronePose:
    """드론 기체 자세 (IMU 측정, compass yaw / NED 기준)"""
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass
class GimbalPose:
    """짐벌 자세 (절대 또는 기체 상대 — 사용하는 함수에 따라 다름)"""
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


def euler_to_rotation_matrix(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    order: str = "ZYX"
) -> np.ndarray:
    """
    오일러 각 → 회전 행렬 (body → NED)

    order='ZYX': yaw → pitch → roll 순서 적용 (intrinsic), 항공 표준
        R = R_z(yaw) · R_y(pitch) · R_x(roll)

    주의: 반환값은 body→parent(NED) 방향이다. parent→body가 필요하면
    전치(.T)해서 쓸 것.
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    R_z = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,    0, 1]
    ])
    R_y = np.array([
        [ cp, 0, sp],
        [  0, 1,  0],
        [-sp, 0, cp]
    ])
    R_x = np.array([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])

    if order == "ZYX":
        return R_z @ R_y @ R_x
    elif order == "XYZ":
        return R_x @ R_y @ R_z
    else:
        raise ValueError(f"Unsupported order: {order}")


def compute_camera_axes_from_gimbal(
    gimbal_yaw_compass_deg: float,
    gimbal_pitch_deg: float,
    gimbal_roll_deg: float = 0.0
) -> dict:
    """
    DJI 짐벌 절대 자세 → 카메라 좌표축의 ENU 표현 (직접 구성 방식)

        광축(Z) = [cos(p)·sin(cy), cos(p)·cos(cy), sin(p)]

    이 광축에 수평 Right 벡터를 직교화해 X를 만들고, Y = Z × X로 완성한 뒤
    roll을 X-Y 평면에서 적용한다.

    Returns:
        {
            "axis_z_enu": 광축(forward),
            "axis_x_enu": Right,
            "axis_y_enu": Down,
            "R_camera_to_enu": 3x3 (열 = 카메라 축의 ENU 표현)
        }
    """
    cy_rad = np.radians(gimbal_yaw_compass_deg)
    p_rad = np.radians(gimbal_pitch_deg)
    r_rad = np.radians(gimbal_roll_deg)

    axis_z = np.array([
        np.cos(p_rad) * np.sin(cy_rad),
        np.cos(p_rad) * np.cos(cy_rad),
        np.sin(p_rad)
    ])
    axis_z = axis_z / np.linalg.norm(axis_z)

    # roll=0일 때의 수평 Right 벡터
    horizontal_right = np.array([np.cos(cy_rad), -np.sin(cy_rad), 0.0])

    # 광축이 수평 Right와 평행해지는 퇴화 상황 방어
    if abs(np.dot(horizontal_right, axis_z)) > 0.9999:
        horizontal_right = np.array([1.0, 0.0, 0.0])

    axis_x_no_roll = horizontal_right - np.dot(horizontal_right, axis_z) * axis_z
    axis_x_no_roll = axis_x_no_roll / np.linalg.norm(axis_x_no_roll)

    axis_y_no_roll = np.cross(axis_z, axis_x_no_roll)
    axis_y_no_roll = axis_y_no_roll / np.linalg.norm(axis_y_no_roll)

    cos_r, sin_r = np.cos(r_rad), np.sin(r_rad)
    axis_x = cos_r * axis_x_no_roll + sin_r * axis_y_no_roll
    axis_y = -sin_r * axis_x_no_roll + cos_r * axis_y_no_roll

    R_camera_to_enu = np.column_stack([axis_x, axis_y, axis_z])

    return {
        "axis_z_enu": axis_z,
        "axis_x_enu": axis_x,
        "axis_y_enu": axis_y,
        "R_camera_to_enu": R_camera_to_enu
    }


def compute_dji_camera_rotation(gimbal_absolute_pose: GimbalPose) -> np.ndarray:
    """
    DJI EXIF의 짐벌 절대 자세 → R_camera_to_enu

    사용 태그:
        GimbalYawDegree / GimbalPitchDegree / GimbalRollDegree (월드 기준 절대값)
    Nadir 촬영 시 GimbalPitch ≈ -90°

    FlightYaw/Pitch/Roll은 짐벌이 절대 자세를 보고하므로 여기서는 불필요하다.
    """
    return compute_camera_axes_from_gimbal(
        gimbal_absolute_pose.yaw_deg,
        gimbal_absolute_pose.pitch_deg,
        gimbal_absolute_pose.roll_deg
    )["R_camera_to_enu"]


def compute_camera_rotation(
    drone_pose: DronePose,
    gimbal_pose: GimbalPose
) -> np.ndarray:
    """
    드론 자세 + 기체 상대 짐벌 자세 → R_camera_to_enu (행렬 체인 방식)

    체인:
        R_cam→ENU = P_ned→enu · R_body→ned · R_gimbal→body · M_cam→frd

    짐벌이 절대 자세를 보고하는 기종(대부분의 DJI)이라면
    compute_dji_camera_rotation을 쓸 것.
    """
    R_body_to_ned = euler_to_rotation_matrix(
        drone_pose.yaw_deg, drone_pose.pitch_deg, drone_pose.roll_deg
    )
    R_gimbal_to_body = euler_to_rotation_matrix(
        gimbal_pose.yaw_deg, gimbal_pose.pitch_deg, gimbal_pose.roll_deg
    )
    return NED_TO_ENU @ R_body_to_ned @ R_gimbal_to_body @ FRD_TO_OPENCV.T


def verify_nadir_orientation(
    R_camera_to_enu: np.ndarray,
    tolerance_deg: float = 5.0
) -> dict:
    """카메라 광축이 -Up(nadir) 방향에 가까운지 검증"""
    z_axis_enu = R_camera_to_enu[:, 2]

    expected_nadir = np.array([0, 0, -1])
    cos_angle = np.clip(np.dot(z_axis_enu, expected_nadir), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))

    return {
        "z_axis_in_enu": z_axis_enu.tolist(),
        "angle_from_nadir_deg": float(angle_deg),
        "is_nadir": angle_deg < tolerance_deg
    }
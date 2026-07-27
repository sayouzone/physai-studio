"""
DJI 짐벌 자세 → 카메라 좌표계 (ENU 기준)

핵심 원리:
DJI gimbal 표기:
- compass_yaw: 정북에서 시계방향 (N=0, E=90, S=180, W=-90/270)
- pitch: 수평으로부터의 각도 (0=수평, -90=직하방, +90=직상방)
- roll: 광축 주변 회전 (보통 0)

직접 광축을 계산:
    optical_axis_enu = [cos(p)·sin(cy), cos(p)·cos(cy), sin(p)]

이 광축에서 카메라의 Right/Down 축을 구해 회전 행렬 완성.
회전 행렬 곱셈보다 수치적으로 안정적이고 부호 헷갈림 없음.
"""
import numpy as np


def compute_camera_axes_from_gimbal(
    gimbal_yaw_compass_deg: float,
    gimbal_pitch_deg: float,
    gimbal_roll_deg: float = 0.0
) -> dict:
    """
    DJI 짐벌 자세에서 카메라 좌표계의 ENU 표현 계산

    카메라 좌표계 (OpenCV):
        Z: 광축 방향 (앞)
        X: 오른쪽
        Y: 아래

    Returns:
        {
            "axis_z_enu": 광축 (forward) in ENU,
            "axis_x_enu": Right in ENU,
            "axis_y_enu": Down in ENU,
            "R_camera_to_enu": 3x3 rotation matrix
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

    horizontal_right_no_roll = np.array([
        np.cos(cy_rad),
        -np.sin(cy_rad),
        0.0
    ])

    if abs(np.dot(horizontal_right_no_roll, axis_z)) > 0.9999:
        horizontal_right_no_roll = np.array([1.0, 0.0, 0.0])

    axis_x_no_roll = horizontal_right_no_roll - np.dot(horizontal_right_no_roll, axis_z) * axis_z
    axis_x_no_roll = axis_x_no_roll / np.linalg.norm(axis_x_no_roll)

    axis_y_no_roll = np.cross(axis_z, axis_x_no_roll)
    axis_y_no_roll = axis_y_no_roll / np.linalg.norm(axis_y_no_roll)

    cos_r = np.cos(r_rad)
    sin_r = np.sin(r_rad)
    axis_x = cos_r * axis_x_no_roll + sin_r * axis_y_no_roll
    axis_y = -sin_r * axis_x_no_roll + cos_r * axis_y_no_roll

    R_camera_to_enu = np.column_stack([axis_x, axis_y, axis_z])

    return {
        "axis_z_enu": axis_z,
        "axis_x_enu": axis_x,
        "axis_y_enu": axis_y,
        "R_camera_to_enu": R_camera_to_enu
    }


def verify_nadir_orientation(R_camera_to_enu: np.ndarray, tolerance_deg: float = 5.0) -> dict:
    """카메라 광축이 -Up(nadir) 방향에 가까운지 검증"""
    z_axis_enu = R_camera_to_enu[:, 2]

    expected_nadir = np.array([0, 0, -1])
    cos_angle = np.dot(z_axis_enu, expected_nadir)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))

    return {
        "z_axis_in_enu": z_axis_enu.tolist(),
        "angle_from_nadir_deg": float(angle_deg),
        "is_nadir": angle_deg < tolerance_deg
    }
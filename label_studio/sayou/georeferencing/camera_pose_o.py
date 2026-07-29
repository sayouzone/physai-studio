"""
src/georeferencing/camera_pose.py
드론·짐벌 자세를 카메라 회전 행렬로 변환
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class DronePose:
    """드론 자세 (IMU에서 측정)"""
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass
class GimbalPose:
    """짐벌 자세 (드론 기준 상대)"""
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
    오일러 각 → 회전 행렬
    
    order='ZYX': yaw → pitch → roll 순서로 적용 (intrinsic)
    가장 일반적인 항공·드론 표기법
    
    R = R_z(yaw) · R_y(pitch) · R_x(roll)
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


def compute_camera_rotation(
    drone_pose: DronePose,
    gimbal_pose: GimbalPose,
    camera_orientation: str = "FLU_to_camera"
) -> np.ndarray:
    """
    카메라의 월드(ENU) 좌표계 기준 회전 행렬
    
    체인:
        Body (드론 기체) ← R_drone ← ENU
        Gimbal ← R_gimbal ← Body
        Camera ← R_cam_to_gimbal ← Gimbal
    
    최종: R_cam_to_world = R_drone · R_gimbal · R_cam_to_gimbal
    
    Camera 좌표 관습 (OpenCV):
        X: 오른쪽
        Y: 아래
        Z: 앞 (광축)
    """
    R_world_to_body = euler_to_rotation_matrix(
        drone_pose.yaw_deg,
        drone_pose.pitch_deg,
        drone_pose.roll_deg
    )
    
    R_body_to_gimbal = euler_to_rotation_matrix(
        gimbal_pose.yaw_deg,
        gimbal_pose.pitch_deg,
        gimbal_pose.roll_deg
    )
    
    if camera_orientation == "FLU_to_camera":
        R_gimbal_to_camera = np.array([
            [0,  1,  0],
            [0,  0, -1],
            [-1, 0,  0]
        ])
    else:
        raise ValueError(f"Unsupported orientation: {camera_orientation}")
    
    R_world_to_camera = R_gimbal_to_camera @ R_body_to_gimbal @ R_world_to_body
    
    R_camera_to_world = R_world_to_camera.T
    
    return R_camera_to_world


def compute_dji_camera_rotation(
    drone_pose: DronePose,
    gimbal_absolute_pose: GimbalPose
) -> np.ndarray:
    """
    DJI 드론의 짐벌은 보통 절대 자세를 보고
    
    DJI EXIF 태그:
    - FlightYawDegree, FlightPitchDegree, FlightRollDegree (드론)
    - GimbalYawDegree, GimbalPitchDegree, GimbalRollDegree (짐벌, 월드 기준)
    
    Nadir 촬영 시 GimbalPitch ≈ -90°
    """
    R_world_to_camera_frd = euler_to_rotation_matrix(
        gimbal_absolute_pose.yaw_deg,
        gimbal_absolute_pose.pitch_deg,
        gimbal_absolute_pose.roll_deg
    )
    
    R_frd_to_cv = np.array([
        [0,  1,  0],
        [0,  0,  1],
        [1,  0,  0]
    ])
    
    R_world_to_camera_cv = R_frd_to_cv @ R_world_to_camera_frd
    
    return R_world_to_camera_cv.T

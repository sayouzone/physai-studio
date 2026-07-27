"""사진측량 기본 기하: ω-φ-κ 회전행렬, 공선조건, 카메라 투영행렬.

회전행렬 규약
-------------
이 모듈의 모든 회전행렬은 다음 사진측량 ω-φ-κ 규약을 따른다::

    R(ω,φ,κ) = R_ω · R_φ · R_κ
    (ω: roll, φ: pitch, κ: yaw, radian)

공선조건 (project_point) 은::

    x = cx - f_px · (R[0]·diff) / (R[2]·diff)
    y = cy - f_px · (R[1]·diff) / (R[2]·diff)
    diff = world_xyz - camera_xyz

규약 주의사항
-------------
* ``rotation_matrix`` 의 광학축은 카메라 좌표계 +Z 가 아니라 **-Z** 에 대응.
  즉, 픽셀 → 지상점 역투영 시 ``d_cam = [-x/f, -y/f, -1]`` 로 둬야
  nadir 촬영에서 ``d_world[2] < 0`` 이 나와 ground plane 과 만난다.
  자세한 배경은 ``ortho.orthophoto`` 모듈의 주석 참조.
* ``camera_projection_matrix`` 의 ``K_neg`` 는 ``project_point`` 와 동일
  부호를 갖도록 -f_px 를 채워 넣은 행렬. DLT 삼각측량과 BA residual 이
  같은 부호 규약을 공유하도록 만드는 핵심.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 회전행렬 — 단일/배치/CuPy
# ---------------------------------------------------------------------------
def rotation_matrix(omega: float, phi: float, kappa: float) -> np.ndarray:
    """사진측량 ω-φ-κ 회전행렬 (radian)."""
    co, so = np.cos(omega), np.sin(omega)
    cp, sp = np.cos(phi), np.sin(phi)
    ck, sk = np.cos(kappa), np.sin(kappa)
    return np.array([
        [cp * ck,                  -cp * sk,                 sp],
        [co * sk + so * sp * ck,    co * ck - so * sp * sk,  -so * cp],
        [so * sk - co * sp * ck,    so * ck + co * sp * sk,   co * cp],
    ])


def rotation_matrices_batch_np(angles: np.ndarray) -> np.ndarray:
    """벡터화 (N, 3) → (N, 3, 3). angles[:, 0,1,2] = [ω, φ, κ]."""
    om, ph, ka = angles[:, 0], angles[:, 1], angles[:, 2]
    co, so = np.cos(om), np.sin(om)
    cp_, sp = np.cos(ph), np.sin(ph)
    ck, sk = np.cos(ka), np.sin(ka)
    N = angles.shape[0]
    R = np.empty((N, 3, 3), dtype=angles.dtype)
    R[:, 0, 0] = cp_ * ck
    R[:, 0, 1] = -cp_ * sk
    R[:, 0, 2] = sp
    R[:, 1, 0] = co * sk + so * sp * ck
    R[:, 1, 1] = co * ck - so * sp * sk
    R[:, 1, 2] = -so * cp_
    R[:, 2, 0] = so * sk - co * sp * ck
    R[:, 2, 1] = so * ck + co * sp * sk
    R[:, 2, 2] = co * cp_
    return R


def rotation_matrices_batch_cp(angles, cp):
    """CuPy 버전 — ``angles`` 는 ``cp.ndarray`` (N, 3). ``cp`` 모듈 주입.

    CuPy 의존성을 이 파일에 직접 두지 않기 위해 호출자(BA, triangulation)가
    ``cp`` 모듈 핸들을 함께 넘긴다. 이렇게 하면 ``geometry`` 모듈은
    GPU 라이브러리 없이도 import 가능.
    """
    om, ph, ka = angles[:, 0], angles[:, 1], angles[:, 2]
    co, so = cp.cos(om), cp.sin(om)
    cp_, sp = cp.cos(ph), cp.sin(ph)
    ck, sk = cp.cos(ka), cp.sin(ka)
    N = angles.shape[0]
    R = cp.empty((N, 3, 3), dtype=angles.dtype)
    R[:, 0, 0] = cp_ * ck
    R[:, 0, 1] = -cp_ * sk
    R[:, 0, 2] = sp
    R[:, 1, 0] = co * sk + so * sp * ck
    R[:, 1, 1] = co * ck - so * sp * sk
    R[:, 1, 2] = -so * cp_
    R[:, 2, 0] = so * sk - co * sp * ck
    R[:, 2, 1] = so * ck + co * sp * sk
    R[:, 2, 2] = co * cp_
    return R


# ---------------------------------------------------------------------------
# 공선조건 / 카메라 행렬
# ---------------------------------------------------------------------------
def project_point(world_xyz: np.ndarray,
                  camera_xyz: np.ndarray,
                  omega: float, phi: float, kappa: float,
                  f_px: float, cx: float, cy: float) -> np.ndarray:
    """단일 3D 점 → 픽셀 좌표 (공선조건).

    ``f_px`` 는 픽셀 단위 초점거리 (= f_mm × width / sensor_width_mm).
    분모가 0 에 가까우면 ``[nan, nan]`` 반환.
    """
    R = rotation_matrix(omega, phi, kappa)
    diff = world_xyz - camera_xyz
    den = R[2] @ diff
    if abs(den) < 1e-9:
        return np.array([np.nan, np.nan])
    x = cx - f_px * (R[0] @ diff) / den
    y = cy - f_px * (R[1] @ diff) / den
    return np.array([x, y])


def camera_projection_matrix(camera_xyz: np.ndarray,
                             omega: float, phi: float, kappa: float,
                             f_px: float, cx: float, cy: float) -> np.ndarray:
    """``project_point`` 와 동일한 투영을 표현하는 3×4 카메라 행렬 P.

    공선조건::

        x = cx - f_px · (R[0]·diff) / (R[2]·diff)
        y = cy - f_px · (R[1]·diff) / (R[2]·diff)

    는 다음 P 로 표현된다 (검증 완료)::

        K_neg = [[-f_px, 0,     cx],
                 [0,     -f_px, cy],
                 [0,     0,     1 ]]
        P = K_neg · [R | -R·C]

    그러면 ``[u·w, v·w, w]ᵀ = P · [X, Y, Z, 1]ᵀ`` 이고
    ``(u, v) = (u·w / w, v·w / w)`` 가 된다.
    DLT 삼각측량은 이 P 행렬을 그대로 사용.
    """
    R = rotation_matrix(omega, phi, kappa)
    K_neg = np.array([
        [-f_px, 0.0,   cx],
        [0.0,   -f_px, cy],
        [0.0,   0.0,   1.0],
    ])
    Rt = np.hstack([R, (-R @ camera_xyz).reshape(3, 1)])  # [R | -R·C]
    return K_neg @ Rt


def compute_focal_px(meta: ImageMetadata) -> float:
    """픽셀 단위 초점거리. FocalLengthIn35mmFilm 우선 (센서크기 불필요).

    Returns
    -------
    f_px : 픽셀 단위 초점거리. **두 EXIF 필드 모두 0/없음이면 0.0 반환**
        (호출자가 0 체크로 그 사진을 제외해야 함). 0 을 그대로 사용하면
        ``simple_orthophoto`` 의 ``-(px-cx)/f_px`` 에서 0 으로 나눠 NaN 이
        발생하고 ``int(np.ceil(NaN))`` 에서 ``ValueError: cannot convert
        float NaN to integer`` 로 크래시한다.

    노트
    ----
    DJI H20T 의 thermal (Z) 채널 EXIF 는 ``FocalLengthIn35mmFilm`` 과
    ``FocalLength`` 가 모두 비어있는 경우가 있다. RGB 디렉토리만 처리하려는
    의도였다면 thermal 파일이 섞이지 않았는지 확인 (``_Z.JPG``, ``_T.JPG``
    제외 등).
    """
    if meta.focal_length_in_35mm and meta.focal_length_in_35mm > 0:
        return meta.focal_length_in_35mm / 36.0 * meta.width
    if meta.focal_length and meta.focal_length > 0:
        # FocalLengthIn35mmFilm 이 없을 때만 센서폭 13.2mm (1″ 센서) 가정 fallback.
        return meta.focal_length * meta.width / 13.2
    # 두 정보 모두 없음 — 0 반환. 호출자가 필터링해야 함.
    return 0.0


__all__ = [
    "rotation_matrix",
    "rotation_matrices_batch_np",
    "rotation_matrices_batch_cp",
    "project_point",
    "camera_projection_matrix",
    "compute_focal_px",
]
"""부호 규약 및 호모그래피 수식 자기검증.

    python -m sayou.georeferencing.homography.selftest

이 파일이 통과하는 한, ``pose`` 와 ``homography`` 는 ``geometry`` 의 공선조건
규약과 정확히 같은 세계에 있다. 규약을 건드리는 변경을 할 때 **먼저** 여기를
돌려볼 것. 정사영상이 180° 뒤집히거나 좌우 반전되는 부류의 버그는 시각적으로
알아채기 어렵고 (태양광 단지는 회전 대칭에 가깝다) 좌표 오차로만 드러난다.

의존성을 줄이기 위해 ``geometry`` 의 두 함수는 여기에 복제해 두었다.
원본이 바뀌면 ``test_geometry_matches_upstream`` 이 실패한다.
"""

from __future__ import annotations

import sys

import numpy as np

from .homography import GroundPlane, PinholeIntrinsics, build_frame_homography
from .pose import (
    camera_axes_enu, decompose_to_opk, opk_from_gimbal, rotation_from_gimbal,
)

RTOL_PX = 1e-6
RTOL_M = 1e-7


def _rotation_matrix(omega, phi, kappa):
    """``geometry.rotation_matrix`` 의 복제본."""
    co, so = np.cos(omega), np.sin(omega)
    cp, sp = np.cos(phi), np.sin(phi)
    ck, sk = np.cos(kappa), np.sin(kappa)
    return np.array([
        [cp * ck, -cp * sk, sp],
        [co * sk + so * sp * ck, co * ck - so * sp * sk, -so * cp],
        [so * sk - co * sp * ck, so * ck + co * sp * sk, co * cp],
    ])


def _project_point(X, C, om, ph, ka, f, cx, cy):
    """``geometry.project_point`` 의 복제본."""
    R = _rotation_matrix(om, ph, ka)
    d = np.asarray(X) - np.asarray(C)
    den = R[2] @ d
    if abs(den) < 1e-9:
        return np.array([np.nan, np.nan])
    return np.array([cx - f * (R[0] @ d) / den, cy - f * (R[1] @ d) / den])


# ---------------------------------------------------------------------------
def test_geometry_matches_upstream():
    """복제본이 실제 ``geometry`` 모듈과 일치하는지."""
    try:
        from ..geometry import project_point, rotation_matrix
    except ImportError:
        print("  [skip] geometry 모듈 import 불가 (패키지 밖에서 실행 중)")
        return
    rng = np.random.default_rng(7)
    for _ in range(50):
        angles = rng.uniform(-np.pi, np.pi, 3)
        assert np.abs(rotation_matrix(*angles) - _rotation_matrix(*angles)).max() < 1e-14
        X = rng.uniform(-50, 50, 3)
        C = np.array([0.0, 0.0, 100.0])
        a = project_point(X, C, *angles, 3000.0, 2000.0, 1500.0)
        b = _project_point(X, C, *angles, 3000.0, 2000.0, 1500.0)
        if np.all(np.isfinite(a)):
            assert np.abs(a - b).max() < 1e-8, (a, b)
    print("  geometry 규약 일치 확인")


def test_rotation_properties():
    """``rotation_from_gimbal`` 이 정규직교 + det=+1 을 만족하는가."""
    rng = np.random.default_rng(1)
    worst_orth = worst_det = 0.0
    for _ in range(300):
        yaw = rng.uniform(-180, 180)
        pitch = rng.uniform(-90, -30)
        roll = rng.uniform(-5, 5)
        R = rotation_from_gimbal(yaw, pitch, roll)
        worst_orth = max(worst_orth, float(np.abs(R @ R.T - np.eye(3)).max()))
        worst_det = max(worst_det, abs(float(np.linalg.det(R)) - 1.0))
    assert worst_orth < 1e-12, worst_orth
    assert worst_det < 1e-12, worst_det
    print(f"  회전행렬 정규직교 오차 {worst_orth:.2e}, det 오차 {worst_det:.2e}")


def test_opk_roundtrip():
    """R → (ω,φ,κ) → R 왕복."""
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(300):
        R = rotation_from_gimbal(rng.uniform(-180, 180),
                                 rng.uniform(-90, -30),
                                 rng.uniform(-5, 5))
        worst = max(worst, float(np.abs(_rotation_matrix(*decompose_to_opk(R)) - R).max()))
    assert worst < 1e-12, worst
    print(f"  (ω,φ,κ) 왕복 오차 {worst:.2e}")


def test_pixel_convention_matches_opencv():
    """``project_point`` 규약이 OpenCV 픽셀 규약 (x 우, y 하) 과 일치하는가.

    이 테스트가 이 파일의 핵심이다. 여기가 통과해야 짐벌 각으로 만든 R 을
    BA 와 호모그래피에 그대로 쓸 수 있다.
    """
    rng = np.random.default_rng(3)
    f, cx, cy = 3800.0, 2432.0, 1824.0
    worst = 0.0
    for _ in range(200):
        yaw, pitch, roll = (rng.uniform(-180, 180),
                            rng.uniform(-90, -40), rng.uniform(-5, 5))
        right, down, fwd = camera_axes_enu(yaw, pitch, roll)
        om, ph, ka = opk_from_gimbal(yaw, pitch, roll)
        C = np.array([200_000.0, 400_000.0, 160.0])
        for _ in range(10):
            zo = rng.uniform(30, 120)                 # 광축 방향 거리
            xo, yo = rng.uniform(-25, 25, 2)          # right / down 오프셋
            X = C + zo * fwd + xo * right + yo * down
            uv = _project_point(X, C, om, ph, ka, f, cx, cy)
            uv_cv = np.array([cx + f * xo / zo, cy + f * yo / zo])
            worst = max(worst, float(np.abs(uv - uv_cv).max()))
    assert worst < 1e-6, worst
    print(f"  OpenCV 픽셀 규약 일치, 최대 편차 {worst:.2e} px")


def test_visibility_sign():
    """전방의 점은 분모 ``(R[2]·diff) > 0``."""
    for yaw, pitch in [(0, -90), (90, -90), (37, -75), (-140, -60)]:
        R = rotation_from_gimbal(yaw, pitch, 0.0)
        _, _, fwd = camera_axes_enu(yaw, pitch, 0.0)
        C = np.array([0.0, 0.0, 100.0])
        X_front = C + 50 * fwd
        X_back = C - 50 * fwd
        assert R[2] @ (X_front - C) > 0
        assert R[2] @ (X_back - C) < 0
    print("  가시 조건 (R[2]·diff > 0) 확인")


def test_homography_matches_collinearity():
    """호모그래피 forward/inverse 가 공선조건과 일치하는가 (경사평면 포함)."""
    rng = np.random.default_rng(4)
    worst_f = worst_i = 0.0
    for _ in range(30):
        yaw, pitch, roll = (rng.uniform(-180, 180),
                            rng.uniform(-90, -75), rng.uniform(-3, 3))
        R = rotation_from_gimbal(yaw, pitch, roll)
        om, ph, ka = decompose_to_opk(R)
        C = np.array([200_000.0, 400_000.0, 160.0])
        intr = PinholeIntrinsics(f_px=4000.0, cx=2432.0, cy=1824.0,
                                 width=4864, height=3648)

        a, b = rng.uniform(-0.03, 0.03, 2)
        # 원점을 카메라 수평위치로 잡으면 c 는 '카메라 바로 아래 표고'.
        plane = GroundPlane(a=a, b=b, c=112.0,
                            origin_xy=(float(C[0]), float(C[1])))

        fh = build_frame_homography(C, R, intr, plane)
        assert fh is not None

        xs = C[0] + rng.uniform(-40, 40, 100)
        ys = C[1] + rng.uniform(-40, 40, 100)
        X = np.column_stack([xs, ys, plane.height_at(xs, ys)])

        uv_h = fh.ground_to_pixel(np.column_stack([xs, ys]))
        uv_ref = np.array([_project_point(x, C, om, ph, ka,
                                          intr.f_px, intr.cx, intr.cy)
                           for x in X])
        worst_f = max(worst_f, float(np.abs(uv_h - uv_ref).max()))

        back = fh.pixel_to_ground(uv_ref)
        worst_i = max(worst_i, float(np.abs(back - np.column_stack([xs, ys])).max()))

    assert worst_f < 1e-5, worst_f
    assert worst_i < 1e-5, worst_i
    print(f"  호모그래피 forward {worst_f:.2e} px, inverse {worst_i:.2e} m")


def test_homography_equals_projection_matrix():
    """``H_g2i == camera_projection_matrix · S`` (평면 승격 행렬)."""
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(30):
        R = rotation_from_gimbal(rng.uniform(-180, 180),
                                 rng.uniform(-90, -75), rng.uniform(-3, 3))
        C = np.array([200_000.0, 400_000.0, 160.0])
        intr = PinholeIntrinsics(4000.0, 2432.0, 1824.0, 4864, 3648)
        a, b = rng.uniform(-0.02, 0.02, 2)
        plane = GroundPlane(a=a, b=b, c=112.0,
                            origin_xy=(float(C[0]), float(C[1])))
        fh = build_frame_homography(C, R, intr, plane)
        # 국소 좌표계에서 비교 — P 도 국소 카메라 중심으로 세운다.
        o = fh.origin_xy
        C_loc = np.array([C[0] - o[0], C[1] - o[1], C[2]])
        P = intr.K_neg @ np.hstack([R, (-R @ C_loc).reshape(3, 1)])
        worst = max(worst, float(
            np.abs(P @ plane.as_S(o) - fh.H_local_g2i).max()))
    assert worst < 1e-5, worst
    print(f"  P·S 등가성 오차 {worst:.2e}")


def test_gsd_sanity():
    """nadir 촬영 GSD 가 ``고도 / f_px`` 와 일치하는가."""
    R = rotation_from_gimbal(0.0, -90.0, 0.0)
    C = np.array([200_000.0, 400_000.0, 160.0])
    intr = PinholeIntrinsics(4000.0, 2432.0, 1824.0, 4864, 3648)
    plane = GroundPlane.horizontal(112.0)
    fh = build_frame_homography(C, R, intr, plane)
    expected = (C[2] - plane.c) / intr.f_px
    got = fh.gsd_at_principal_point()
    assert abs(got - expected) / expected < 1e-9, (got, expected)

    b = fh.footprint_bounds()
    w_m, h_m = b[2] - b[0], b[3] - b[1]
    assert abs(w_m - intr.width * expected) / w_m < 1e-9
    print(f"  GSD {got*100:.3f} cm/px, footprint {w_m:.2f} × {h_m:.2f} m "
          f"(고도차 {C[2]-plane.c:.1f} m)")


def test_nadir_orientation_maps_north_up():
    """정북 nadir 에서 방위가 올바르게 맺히는가 (동=우, 북=하)."""
    R = rotation_from_gimbal(0.0, -90.0, 0.0)
    C = np.array([0.0, 0.0, 100.0])
    intr = PinholeIntrinsics(1000.0, 500.0, 400.0, 1000, 800)
    fh = build_frame_homography(C, R, intr, GroundPlane.horizontal(0.0))

    east = fh.ground_to_pixel(np.array([[10.0, 0.0]]))[0]
    north = fh.ground_to_pixel(np.array([[0.0, 10.0]]))[0]
    # 정북을 향한 nadir 사진: 동쪽은 이미지 오른쪽, 북쪽은 이미지 위쪽.
    assert east[0] > intr.cx and abs(east[1] - intr.cy) < 1e-6, east
    assert north[1] < intr.cy and abs(north[0] - intr.cx) < 1e-6, north
    # 스케일도 대칭이어야 한다 (등방 GSD).
    assert abs((east[0] - intr.cx) - (intr.cy - north[1])) < 1e-6
    print(f"  방위 확인: 동→+u(우) {east[0]-intr.cx:.1f}px, "
          f"북→−v(위) {intr.cy-north[1]:.1f}px")


def test_yaw_rotates_image():
    """짐벌 yaw 90° (기수 동쪽) 이면 지상 동쪽이 이미지 세로축으로 간다."""
    C = np.array([0.0, 0.0, 100.0])
    intr = PinholeIntrinsics(1000.0, 500.0, 400.0, 1000, 800)
    fh = build_frame_homography(C, rotation_from_gimbal(90.0, -90.0, 0.0),
                                intr, GroundPlane.horizontal(0.0))
    east = fh.ground_to_pixel(np.array([[10.0, 0.0]]))[0]
    # yaw=90 이면 기체 진행방향(동)이 이미지 세로축에 놓인다.
    assert abs(east[0] - intr.cx) < 1e-6, east
    assert abs(east[1] - intr.cy) > 1.0, east
    print("  yaw 회전 반영 확인")


def test_behind_camera_returns_nan():
    """카메라 뒤쪽 지상점은 NaN."""
    C = np.array([0.0, 0.0, 100.0])
    intr = PinholeIntrinsics(1000.0, 500.0, 400.0, 1000, 800)
    # 45° 기울인 카메라 — 지평선 너머가 존재한다.
    fh = build_frame_homography(C, rotation_from_gimbal(0.0, -45.0, 0.0),
                                intr, GroundPlane.horizontal(0.0))
    far_back = fh.ground_to_pixel(np.array([[0.0, -5000.0]]))
    assert np.all(np.isnan(far_back)), far_back
    print("  카메라 후방 점 NaN 처리 확인")


def test_camera_on_plane_is_degenerate():
    """카메라 중심이 지상평면 위에 있으면 특이 → ``None``.

    수평 촬영 자체는 특이하지 않다 (이미지 절반이 지평선 너머일 뿐).
    진짜 축퇴는 세 번째 열이 0 이 되는 이 경우뿐이다.
    """
    intr = PinholeIntrinsics(1000.0, 500.0, 400.0, 1000, 800)
    on_plane = build_frame_homography(
        np.array([0.0, 0.0, 0.0]), rotation_from_gimbal(0.0, -90.0, 0.0),
        intr, GroundPlane.horizontal(0.0))
    assert on_plane is None, on_plane

    # 수평 촬영은 유효하되 지평선 너머는 NaN 이어야 한다.
    tilted = build_frame_homography(
        np.array([0.0, 0.0, 100.0]), rotation_from_gimbal(0.0, -5.0, 0.0),
        intr, GroundPlane.horizontal(0.0))
    assert tilted is not None
    behind = tilted.ground_to_pixel(np.array([[0.0, -3000.0]]))
    assert np.all(np.isnan(behind)), behind
    print("  축퇴 판정 확인 (평면 위 카메라만 None, 저각 촬영은 유효)")


def test_conditioning_with_projected_coordinates():
    """EPSG:5186 절대좌표에서도 조건수가 유지되는가 — 국소 원점의 존재 이유."""
    R = rotation_from_gimbal(0.0, -90.0, 0.0)
    intr = PinholeIntrinsics(4000.0, 2432.0, 1824.0, 4864, 3648)
    worst_cond = 0.0
    worst_rt = 0.0
    for C in (np.array([0.0, 0.0, 160.0]),
              np.array([200_000.0, 400_000.0, 160.0]),
              np.array([1_100_000.0, 2_000_000.0, 160.0])):
        plane = GroundPlane.horizontal(112.0)
        fh = build_frame_homography(C, R, intr, plane)
        assert fh is not None, C
        worst_cond = max(worst_cond, fh.condition_number)
        xy = C[:2] + np.array([[12.0, -7.0], [-30.0, 25.0]])
        rt = fh.pixel_to_ground(fh.ground_to_pixel(xy))
        worst_rt = max(worst_rt, float(np.abs(rt - xy).max()))
    assert worst_cond < 1e7, worst_cond
    assert worst_rt < 1e-6, worst_rt
    print(f"  절대좌표 조건수 최대 {worst_cond:.2e}, 왕복오차 {worst_rt:.2e} m")


def main() -> int:
    tests = [
        test_geometry_matches_upstream,
        test_rotation_properties,
        test_opk_roundtrip,
        test_pixel_convention_matches_opencv,
        test_visibility_sign,
        test_homography_matches_collinearity,
        test_homography_equals_projection_matrix,
        test_gsd_sanity,
        test_nadir_orientation_maps_north_up,
        test_yaw_rotates_image,
        test_behind_camera_returns_nan,
        test_camera_on_plane_is_degenerate,
        test_conditioning_with_projected_coordinates,
    ]
    failed = 0
    print("=" * 66)
    print("RTK 호모그래피 규약 자기검증")
    print("=" * 66)
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {t.__name__}: {type(exc).__name__}: {exc}")
    print("=" * 66)
    print(f"{len(tests) - failed}/{len(tests)} 통과")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

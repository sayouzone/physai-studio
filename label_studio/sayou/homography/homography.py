"""평면유도 호모그래피 (plane-induced homography) — RTK 기반 정사보정의 핵심.

배경
----
``ortho.orthophoto.simple_orthophoto`` 는 출력 래스터의 **모든 픽셀**에 대해
ray-plane 교점을 계산한다 (meshgrid → diff → einsum → 나눗셈 → ``cv2.remap``).
하지만 지표면을 평면으로 가정하는 순간, 지상평면 ↔ 이미지평면 대응은
**3×3 호모그래피 하나로 정확히 닫힌다**. 즉 수천만 번의 광선 교차 계산은
불필요하고, 행렬 하나와 ``cv2.warpPerspective`` 한 번이면 수학적으로 동일한
결과가 나온다.

유도
----
카메라 행렬 (``geometry.camera_projection_matrix`` 와 동일 규약)::

    P = K_neg · [R | -R·C],   K_neg = [[-f, 0, cx], [0, -f, cy], [0, 0, 1]]

지상평면을 ``Z = a·x + b·y + c`` 로 두면 평면 위의 점은::

    [X, Y, Z, 1]ᵀ = S · [x, y, 1]ᵀ,   S = [[1,0,0], [0,1,0], [a,b,c], [0,0,1]]

따라서 지상평면 → 이미지 호모그래피는 단순히::

    H_g2i = P · S = K_neg · [ (r1 + a·r3) | (r2 + b·r3) | (c·r3 − R·C) ]

(``r1, r2, r3`` 은 R 의 **열**). ``a=b=0`` 이면 수평평면 ``Z=c`` 에 대한
익숙한 형태 ``K_neg·[r1 | r2 | c·r3 − R·C]`` 로 환원된다.

경사항 ``a, b`` 가 1·2 열에 들어간다는 점에 주의. 수평평면만 시험하면 이
자리를 틀려도 통과하므로, ``selftest`` 는 반드시 경사평면으로 검증한다.

★ 국소 원점 — 이걸 빠뜨리면 조용히 정밀도가 무너진다
----------------------------------------------------
위 식을 EPSG:5186 절대좌표로 그대로 세우면 안 된다. 한국 중부원점 기준
투영좌표는 (200000, 400000) 근방이라 ``R·C`` 항이 10⁵~10⁶ 스케일인 반면
``r1, r2`` 는 O(1) 이다. 열 간 스케일 차가 10⁶ 이 되어 조건수가 폭증한다.

실측 (H20T, 고도 160 m, f=4000 px)::

    원점 (0, 0)             기준 → cond(H) = 1.1e+05,  왕복오차 1.8e-15 m
    원점 (200000, 400000)   기준 → cond(H) = 1.7e+13,  역행렬 무의미

float64 의 유효자릿수가 ~16 자리이므로 조건수 1e13 은 결과에 3 자리밖에
남기지 않는다. 그래서 이 모듈은 **프레임마다 자기 카메라 위치를 국소 원점**
으로 삼아 호모그래피를 세우고, 공개 API 에서만 절대좌표로 환산한다.
사용자는 항상 절대 투영좌표를 주고받으며, 내부 이동은 신경 쓸 필요가 없다.

부호 규약 — **중요**
--------------------
이 모듈은 ``geometry.project_point`` / ``geometry.camera_projection_matrix``
와 **동일한** 규약을 따른다::

    px = cx − f·(R[0]·diff) / (R[2]·diff)
    py = cy − f·(R[1]·diff) / (R[2]·diff)
    가시 조건: (R[2]·diff) > 0        ← 카메라 전방

이 규약에서 ``R`` 의 행은 ``[-right, -down, forward]`` (``pose.py`` 참조) 이고
광학축은 카메라 좌표계 **+Z** 이다.

⚠ 기존 ``ortho.orthophoto`` 는 ``d_cam = [-(px-cx)/f, -(py-cy)/f, -1]`` 과
``px = cx + f·…`` 를 쓴다. 이 조합은 그 안에서는 왕복 일치하지만
``project_point`` 규약과 **광축 주변 180° 회전만큼 어긋난다**
(``project_point`` 규약의 광선은 ``[-(px-cx)/f, -(py-cy)/f, +1]`` 에 비례).
Bundle Adjustment / triangulation 은 ``project_point`` 규약을 쓰므로,
BA 로 최적화한 자세를 그대로 ``simple_orthophoto`` 에 넣으면 정사영상이
180° 뒤집힌다. 본 모듈은 BA 산출물을 그대로 소비하기 위해 BA 규약에 맞췄다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "GroundPlane",
    "PinholeIntrinsics",
    "FrameHomography",
    "intrinsics_from_metadata",
    "build_frame_homography",
    "ortho_grid_affine",
    "ortho_grid_homography",
    "ground_bounds_from_homography",
    "estimate_gsd",
    "recommend_gsd",
]

# 조건수 상한. 국소 원점을 쓰면 정상 nadir 는 10⁴~10⁶ 수준이므로
# 10⁹ 를 넘으면 자세/평면 추정이 실패한 프레임으로 본다.
MAX_CONDITION_NUMBER = 1e9


# ---------------------------------------------------------------------------
# 지상 평면
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GroundPlane:
    """지상 기준 평면 ``Z = a·(x − x₀) + b·(y − y₀) + c``.

    ``origin_xy`` 를 명시적으로 들고 다니는 이유: 절대 투영좌표 기준으로
    ``c`` 를 쓰면 (경사가 0.01 이어도) ``c ≈ −6000 m`` 같은 물리적 의미 없는
    값이 되고, 부동소수 정밀도도 낭비된다. 원점을 함께 저장하면 ``c`` 는
    **원점에서의 실제 표고**가 되어 로그로 눈검사가 가능하다.

    Attributes
    ----------
    a, b : 경사 (m/m). ``0`` 이면 수평평면.
    c : ``origin_xy`` 위치에서의 표고 (m).
    origin_xy : 경사 기준점 (투영좌표). 보통 점군/카메라 중심.
    inlier_rmse_m : 평면 적합 잔차 RMSE. 진단용.
    n_inliers : 적합에 사용된 점 개수.
    """

    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    origin_xy: tuple[float, float] = (0.0, 0.0)
    inlier_rmse_m: float = float("nan")
    n_inliers: int = 0

    @classmethod
    def horizontal(cls, z: float) -> "GroundPlane":
        return cls(a=0.0, b=0.0, c=float(z))

    def height_at(self, x, y):
        """절대 투영좌표에서의 평면 표고."""
        return (self.a * (np.asarray(x) - self.origin_xy[0])
                + self.b * (np.asarray(y) - self.origin_xy[1]) + self.c)

    def local_coeffs(self, origin_xy) -> tuple[float, float, float]:
        """다른 원점 기준으로 재표현한 ``(a, b, c')``.

        ``Z = a·(x−ox) + b·(y−oy) + c'`` 가 되도록 ``c'`` 를 옮긴다.
        경사(a, b)는 원점에 무관하므로 그대로다.
        """
        ox, oy = float(origin_xy[0]), float(origin_xy[1])
        c_new = (self.a * (ox - self.origin_xy[0])
                 + self.b * (oy - self.origin_xy[1]) + self.c)
        return self.a, self.b, float(c_new)

    @property
    def is_horizontal(self) -> bool:
        return self.a == 0.0 and self.b == 0.0

    @property
    def slope_deg(self) -> float:
        return float(np.degrees(np.arctan(np.hypot(self.a, self.b))))

    def as_S(self, origin_xy=(0.0, 0.0)) -> np.ndarray:
        """4×3 승격 행렬 ``S``: ``[X,Y,Z,1]ᵀ = S · [x',y',1]ᵀ``.

        ``x' = x − ox`` 인 국소 좌표 기준. 절대좌표로 쓰려면
        ``origin_xy=(0,0)`` (기본값). ``selftest`` 의 ``P·S`` 등가성 검증용.
        """
        a, b, c = self.local_coeffs(origin_xy)
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [a, b, c],
            [0.0, 0.0, 1.0],
        ])


# ---------------------------------------------------------------------------
# 내부 파라미터
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PinholeIntrinsics:
    """핀홀 내부 파라미터.

    호모그래피는 ``geometry`` 의 공선조건과 맞추기 위해 **등방 초점거리**를
    가정한다. DewarpData 의 fx/fy 가 다르면 기하평균을 쓰고 비율을
    ``anisotropy`` 에 기록한다.

    렌즈 왜곡(``dist``)은 호모그래피로 표현할 수 없다. 왜곡이 유의하면
    ``undistort_maps()`` 로 원본을 먼저 보정한 뒤 호모그래피를 적용해야 한다.
    """

    f_px: float
    cx: float
    cy: float
    width: int
    height: int
    dist: tuple[float, ...] = ()
    anisotropy: float = 1.0

    @property
    def K_neg(self) -> np.ndarray:
        """``project_point`` 와 부호가 일치하는 내부 행렬."""
        return np.array([
            [-self.f_px, 0.0, self.cx],
            [0.0, -self.f_px, self.cy],
            [0.0, 0.0, 1.0],
        ])

    @property
    def K(self) -> np.ndarray:
        """OpenCV 표준 K (undistort 등 외부 라이브러리 호출용)."""
        return np.array([
            [self.f_px, 0.0, self.cx],
            [0.0, self.f_px, self.cy],
            [0.0, 0.0, 1.0],
        ])

    @property
    def has_distortion(self) -> bool:
        return bool(self.dist) and any(abs(d) > 1e-12 for d in self.dist)

    def undistort_maps(self):
        """``cv2.initUndistortRectifyMap`` 결과 ``(map1, map2)``.

        왜곡계수가 없으면 ``None``. 프레임마다 다시 만들지 말고 카메라별로
        한 번 만들어 캐시할 것 (4864×3648 float32 맵 2장 ≈ 142 MB).
        """
        if not self.has_distortion:
            return None
        import cv2

        d = np.asarray(self.dist, dtype=np.float64).reshape(1, -1)
        return cv2.initUndistortRectifyMap(
            self.K, d, None, self.K,
            (self.width, self.height), cv2.CV_32FC1,
        )


def intrinsics_from_metadata(meta, prefer_dewarp: bool = True) -> PinholeIntrinsics | None:
    """``ImageMetadata`` → ``PinholeIntrinsics``.

    우선순위:

    1. **DewarpData** (DJI 공장 캘리브레이션). fx/fy/cx/cy + [k1,k2,p1,p2,k3].
       ``dewarp_cx/cy`` 는 *이미지 중심으로부터의 오프셋*이므로 절대 주점으로
       환산한다.
    2. ``FocalLengthIn35mmFilm`` → ``f_px = f35 / 36 · width`` (센서폭 불필요).
    3. ``FocalLength`` + 1″ 센서폭 13.2 mm 가정.

    Returns
    -------
    ``None`` 이면 초점거리 정보가 전혀 없는 사진 (호출자가 스킵해야 함).
    H20T thermal 채널에서 두 EXIF 필드가 모두 비어 있는 사례가 있다.
    """
    w, h = int(getattr(meta, "width", 0)), int(getattr(meta, "height", 0))
    if w <= 0 or h <= 0:
        return None

    if prefer_dewarp and getattr(meta, "dewarp_fx", None) and getattr(meta, "dewarp_fy", None):
        fx, fy = float(meta.dewarp_fx), float(meta.dewarp_fy)
        cx = w / 2.0 + float(meta.dewarp_cx or 0.0)
        cy = h / 2.0 + float(meta.dewarp_cy or 0.0)
        aniso = fx / fy if fy else 1.0
        if not (0.9 < aniso < 1.11):
            logger.warning(
                "DewarpData fx/fy 비율 이상 (%.4f) — 호모그래피는 등방 f 를 "
                "가정하므로 기하평균 사용: %s",
                aniso, getattr(meta, "origin_path", "?"),
            )
        return PinholeIntrinsics(
            f_px=float(np.sqrt(fx * fy)), cx=cx, cy=cy, width=w, height=h,
            dist=tuple(float(v) for v in (meta.dewarp_dist or ())),
            anisotropy=aniso,
        )

    f35 = getattr(meta, "focal_length_in_35mm", 0) or 0
    if f35 > 0:
        f_px = float(f35) / 36.0 * w
    elif (getattr(meta, "focal_length", 0) or 0) > 0:
        f_px = float(meta.focal_length) * w / 13.2
    else:
        return None
    return PinholeIntrinsics(f_px=f_px, cx=w / 2.0, cy=h / 2.0, width=w, height=h)


# ---------------------------------------------------------------------------
# 프레임 호모그래피
# ---------------------------------------------------------------------------
@dataclass
class FrameHomography:
    """한 장의 사진에 대한 지상평면 ↔ 이미지 호모그래피.

    내부 행렬 ``H_local_*`` 는 ``origin_xy`` 기준 **국소 좌표**로 세워져 있다.
    공개 메서드는 모두 절대 투영좌표를 받고 돌려준다.
    """

    H_local_g2i: np.ndarray     # 3×3, 국소 지상 (x', y') → 이미지 (px, py)
    H_local_i2g: np.ndarray     # 3×3, 그 역
    origin_xy: np.ndarray       # (2,) 국소 원점 (투영좌표)
    camera_xyz: np.ndarray      # (3,) 절대 투영좌표 카메라 중심
    R: np.ndarray               # 3×3 world → camera
    intr: PinholeIntrinsics
    plane: GroundPlane
    condition_number: float

    # ---- 절대 ↔ 국소 ---------------------------------------------------
    def _to_local(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, dtype=np.float64) - self.origin_xy

    def _to_absolute(self, xy_local: np.ndarray) -> np.ndarray:
        return xy_local + self.origin_xy

    @property
    def H_g2i(self) -> np.ndarray:
        """절대 지상좌표 → 이미지 호모그래피.

        ``H_local_g2i · T`` (``T`` = 원점 이동). 외부 호환용으로만 쓸 것 —
        수치적으로는 국소 행렬이 훨씬 낫다.
        """
        ox, oy = self.origin_xy
        T = np.array([[1.0, 0.0, -ox], [0.0, 1.0, -oy], [0.0, 0.0, 1.0]])
        return self.H_local_g2i @ T

    # ---- 점 변환 --------------------------------------------------------
    def ground_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        """(N, 2) 절대 지상좌표 → (N, 2) 픽셀. 카메라 뒤쪽 점은 NaN."""
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        loc = self._to_local(xy)
        hom = np.column_stack([loc, np.ones(len(loc))])
        p = hom @ self.H_local_g2i.T
        w = p[:, 2]
        # 가시 조건: project_point 규약의 분모 (R[2]·diff) > 0.
        # H 의 3행이 정확히 그 값을 만들므로 w > 0 이 곧 '카메라 전방'.
        bad = w <= 1e-9
        out = np.full((len(xy), 2), np.nan)
        out[~bad] = p[~bad, :2] / w[~bad, None]
        return out

    def pixel_to_ground(self, uv: np.ndarray) -> np.ndarray:
        """(N, 2) 픽셀 → (N, 2) 절대 지상좌표. 지평선 너머 광선은 NaN."""
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        hom = np.column_stack([uv, np.ones(len(uv))])
        g = hom @ self.H_local_i2g.T
        w = g[:, 2]
        bad = np.abs(w) < 1e-12
        loc = np.full((len(uv), 2), np.nan)
        loc[~bad] = g[~bad, :2] / w[~bad, None]

        # 지평선 위의 픽셀은 평면과 '카메라 뒤쪽'에서 만난다. 역행렬만으로는
        # 그 좌표가 조용히 나오므로, 정투영으로 부호를 다시 확인한다.
        ok = ~bad
        if ok.any():
            hom2 = np.column_stack([loc[ok], np.ones(int(ok.sum()))])
            w_fwd = hom2 @ self.H_local_g2i[2]
            idx = np.nonzero(ok)[0]
            loc[idx[w_fwd <= 1e-9]] = np.nan
        return self._to_absolute(loc)

    def pixel_to_ground_xyz(self, uv: np.ndarray) -> np.ndarray:
        """(N, 2) 픽셀 → (N, 3) 절대 지상 3D (평면 위)."""
        xy = self.pixel_to_ground(uv)
        z = self.plane.height_at(xy[:, 0], xy[:, 1])
        return np.column_stack([xy, z])

    # ---- 출력 래스터 결합 -----------------------------------------------
    def ortho_pixel_matrix(self, x_min: float, y_max: float,
                           gsd_m: float) -> np.ndarray:
        """정사영상 픽셀 (u, v) → 원본 이미지 픽셀 호모그래피 (3×3).

        ``cv2.warpPerspective(img, M, (w, h), flags=WARP_INVERSE_MAP)`` 에
        그대로 넣는 행렬. meshgrid 도 remap 맵도 필요 없다.
        """
        ox, oy = self.origin_xy
        A = ortho_grid_affine(x_min - ox, y_max - oy, gsd_m)
        return self.H_local_g2i @ A

    # ---- 파생량 ---------------------------------------------------------
    def image_corners_ground(self) -> np.ndarray:
        """이미지 4모서리의 절대 지상좌표 (TL, TR, BR, BL)."""
        w, h = self.intr.width, self.intr.height
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
        return self.pixel_to_ground(corners)

    def footprint_bounds(self) -> tuple[float, float, float, float] | None:
        """(x_min, y_min, x_max, y_max). 4모서리 중 하나라도 실패하면 None."""
        g = self.image_corners_ground()
        if not np.all(np.isfinite(g)):
            return None
        return (float(g[:, 0].min()), float(g[:, 1].min()),
                float(g[:, 0].max()), float(g[:, 1].max()))

    def gsd_at_principal_point(self) -> float:
        """주점 부근 GSD (m/px)."""
        return estimate_gsd(self, (self.intr.cx, self.intr.cy))

    def nadir_ground_xy(self) -> np.ndarray:
        """카메라 연직 아래점의 지상좌표 (모자이크 가중치용)."""
        return self.camera_xyz[:2].copy()


def build_frame_homography(
    camera_xyz: np.ndarray,
    R: np.ndarray,
    intr: PinholeIntrinsics,
    plane: GroundPlane,
    origin_xy=None,
) -> FrameHomography | None:
    """``H = K_neg · [(r1 + a·r3) | (r2 + b·r3) | (c·r3 − R·C)]`` (국소 좌표).

    Parameters
    ----------
    camera_xyz : (3,) 절대 투영좌표 카메라 중심 (BA 산출물 또는 RTK 관측값).
    R : 3×3 world → camera. ``pose.rotation_from_gimbal`` 또는
        ``geometry.rotation_matrix(ω, φ, κ)`` 의 결과.
    plane : 지상 기준 평면.
    origin_xy : 국소 원점. ``None`` 이면 카메라의 수평 위치를 쓴다 —
        프레임마다 다르지만 각자 자기 원점을 저장하므로 문제 없고,
        조건수가 항상 최적에 가깝다.

    Returns
    -------
    ``None`` 이면 호모그래피가 특이하다. 실제로 특이해지는 경우는
    **카메라 중심이 지상평면 위에 놓일 때**뿐이다 (세 번째 열이 0 이 됨).
    수평 촬영(광축이 평면과 평행)은 특이하지 않다 — 이미지의 절반이 지평선
    너머로 갈 뿐이고, 그 영역은 ``ground_to_pixel`` 이 NaN 으로 걸러낸다.
    """
    C = np.asarray(camera_xyz, dtype=np.float64).reshape(3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    if not (np.all(np.isfinite(C)) and np.all(np.isfinite(R))):
        return None

    origin = (np.asarray(origin_xy, dtype=np.float64).reshape(2)
              if origin_xy is not None else C[:2].copy())
    C_local = np.array([C[0] - origin[0], C[1] - origin[1], C[2]])
    a, b, c = plane.local_coeffs(origin)

    # X_cam = R·(X − C) 를 평면 매개변수 (x', y') 로 전개하면
    #     X_cam = x'·(r1 + a·r3) + y'·(r2 + b·r3) + (c·r3 − R·C)
    # 경사항 a, b 는 **1·2 열**에서 r3 에 곱해진다 (3열이 아니다).
    # a=b=0 인 수평평면에서는 세 열이 [r1 | r2 | c·r3 − R·C] 로 환원되므로
    # 이 자리를 틀려도 수평 데이터로는 드러나지 않는다.
    r1, r2, r3 = R[:, 0], R[:, 1], R[:, 2]
    M = np.column_stack([
        r1 + a * r3,
        r2 + b * r3,
        c * r3 - R @ C_local,
    ])
    H = intr.K_neg @ M

    try:
        cond = float(np.linalg.cond(H))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(cond) or cond > MAX_CONDITION_NUMBER:
        logger.warning("호모그래피 특이 (cond=%.3e) — 카메라가 지상평면 위에 "
                       "있거나 자세/고도가 비정상. 프레임 스킵.", cond)
        return None

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    return FrameHomography(
        H_local_g2i=H, H_local_i2g=H_inv, origin_xy=origin,
        camera_xyz=C, R=R, intr=intr, plane=plane, condition_number=cond,
    )


# ---------------------------------------------------------------------------
# 출력 래스터 기하
# ---------------------------------------------------------------------------
def ortho_grid_affine(x_min: float, y_max: float, gsd_m: float) -> np.ndarray:
    """정사영상 픽셀 (u, v) → 지상 (x, y) 아핀 (3×3 동차).

    북향 정사 규약: ``x = x_min + (u+0.5)·gsd``, ``y = y_max − (v+0.5)·gsd``.
    픽셀 **중심** 기준이라 ``+0.5`` 가 들어간다 — 이걸 빼면 GeoTIFF transform
    (좌상단 모서리 기준) 과 반 픽셀 어긋나 인접 정사영상 접합선에서 계단이
    생긴다.
    """
    return np.array([
        [gsd_m, 0.0, x_min + 0.5 * gsd_m],
        [0.0, -gsd_m, y_max - 0.5 * gsd_m],
        [0.0, 0.0, 1.0],
    ])


def ortho_grid_homography(fh: FrameHomography, x_min: float, y_max: float,
                          gsd_m: float) -> np.ndarray:
    """``FrameHomography.ortho_pixel_matrix`` 의 함수형 별칭."""
    return fh.ortho_pixel_matrix(x_min, y_max, gsd_m)


def ground_bounds_from_homography(
    frames: list[FrameHomography],
) -> tuple[float, float, float, float] | None:
    """여러 프레임의 footprint 를 감싸는 전체 범위."""
    bounds = [b for b in (f.footprint_bounds() for f in frames) if b is not None]
    if not bounds:
        return None
    arr = np.asarray(bounds)
    return (float(arr[:, 0].min()), float(arr[:, 1].min()),
            float(arr[:, 2].max()), float(arr[:, 3].max()))


def estimate_gsd(fh: FrameHomography, uv: tuple[float, float] | None = None) -> float:
    """주어진 픽셀 위치에서의 등가 GSD (m/px).

    호모그래피 국소 야코비안 행렬식의 제곱근. ``|det J|^(1/2)`` 는 면적
    스케일에서 온 등가 변장이라 가로세로 GSD 가 다를 때도 안정적이다.
    유한차분 대신 해석적 미분이라 스텝 크기 선택이 필요 없다.
    """
    if uv is None:
        uv = (fh.intr.cx, fh.intr.cy)
    H = fh.H_local_i2g            # 이동은 야코비안에 영향 없음
    u, v = float(uv[0]), float(uv[1])
    g = H @ np.array([u, v, 1.0])
    w = g[2]
    if abs(w) < 1e-12:
        return float("nan")
    # d(g_i/w)/dp_j = (H[i,j]·w − g_i·H[2,j]) / w²
    J = np.array([[(H[i, j] * w - g[i] * H[2, j]) / (w * w) for j in range(2)]
                  for i in range(2)])
    det = abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
    return float(np.sqrt(det))


def recommend_gsd(frames: list[FrameHomography], percentile: float = 50.0) -> float:
    """프레임 집합에 대한 권장 출력 GSD.

    각 프레임 주점 GSD 의 분위수. 중앙값(기본)은 원해상도를 대체로 보존하면서
    고도가 튄 프레임 하나 때문에 출력 래스터가 과대해지는 것을 막는다.
    """
    vals = np.array([f.gsd_at_principal_point() for f in frames])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        raise ValueError("유효한 GSD 를 계산할 수 있는 프레임이 없음")
    return float(np.percentile(vals, percentile))

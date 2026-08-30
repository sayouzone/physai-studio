"""End-to-End GCP-Free Georeferencing 파이프라인 (RTK 기반 호모그래피).

워크플로우
----------
1. EXIF/XMP 메타데이터 추출 → 사진별 RTK 좌표 + 짐벌 자세
2. 좌표계 변환 (WGS84 → EPSG:5186), 선택적 타원체고 → 정표고
3. RTK 품질 검증 + BA 가중치 공분산 (실측 σ 기반)
4. SfM Tie Point — RTK 인접쌍만 매칭
5. 공선조건 · 회전행렬 (5a track, 5b triangulation)
6. RTK 제약 Bundle Adjustment
7. 지상평면 추정 + 평면유도 호모그래피
8. 정사영상 합성 (프레임별 GeoTIFF 또는 단일 모자이크)

★ 원본 pipeline.py 에서 고친 것 — 반드시 읽을 것
------------------------------------------------
**``_build_initial_state`` 의 자세 변환이 틀려 있었다.**

원본::

    omega = deg2rad(gimbal_roll)
    phi   = deg2rad(gimbal_pitch + 90)      # nadir(-90) → 0 "보정"
    kappa = deg2rad(gimbal_yaw)

이 식은 nadir 정북 촬영 (roll=0, pitch=−90, yaw=0) 에서
``(ω, φ, κ) = (0, 0, 0)`` → ``R = I`` 를 만든다. 그런데
``geometry.project_point`` 규약에서 광축은 ``R[2]`` 이므로 광축이
``(0, 0, +1)`` — **하늘을 향한다.**

결과::

    카메라 (200000, 400000, 160), 지상점 48 m 아래
    → 가시 분모 (R[2]·diff) = −48.00   (전방이어야 하는데 후방)

* ``triangulate_dlt`` 는 cheirality 를 검사하지 않으므로 조용히
  **드론 위쪽**에 3D 점을 만든다.
* BA 는 그 거울상 기하를 최소화하므로 수렴은 하지만 해가 무의미하다.
* ``ortho.simple_orthophoto`` 는 *또 다른* 부호 규약
  (``d_cam=[...,−1]`` + ``px = cx + f·…``) 을 써서 이 오류를 **부분적으로**
  상쇄한다. 두 오류가 겹친 최종 결과는 **동서(E-W)가 거울반전된 정사영상**
  이다. 실측 검증::

      진짜 지상점  : 동 +10.0 m, 북 +7.0 m
      현 파이프라인: 동 −10.0 m, 북 +7.0 m   → 오차 (−20.0, +0.0) m

  태양광 패널 격자는 좌우 대칭에 가까워 육안으로는 거의 식별되지 않는다.
  좌표를 실측과 대조해야만 드러난다.

수정판은 ``homography.pose.opk_from_gimbal`` 을 쓴다. 이 함수는
``selftest`` 에서 ``project_point`` 규약 및 OpenCV 픽셀 규약과의 일치를
수치로 검증한다 (편차 < 1e-8 px).

정사영상도 ``ortho.simple_orthophoto`` 대신 ``homography.orthorectify_frame``
/ ``mosaic_frames`` 를 쓴다. 같은 부호 규약을 공유하므로 BA 산출물을 그대로
소비할 수 있고, 출력 픽셀당 광선교차 계산이 ``warpPerspective`` 한 번으로
대체된다.

GCP-free 정확도 — 오차 예산
---------------------------
"RTK Fixed 니까 2~5 cm" 는 **카메라 위치**의 정확도이지 정사영상 위 지상점의
정확도가 아니다. 지상점 오차는 다른 항이 지배한다.

한 점의 수평 오차를 off-nadir 비 ``k = r/h`` 로 쓰면::

    자세오차 δ  →  h·δ·(1 + k²)        ← 보통 지배항
    평면가정 Δh →  Δh·k                 ← 잘못 잡으면 최대 지배항
    카메라 고도 →  σ_z·k
    카메라 수평 →  σ_xy                 (1:1 전파)
    레버암 잔차 →  약 2 cm

H20T 실측 조건 (고도 45.87 m, 지상범위 26.7 × 35.4 m → 최외곽 k=0.483,
off-nadir 25.8°; RTK Fixed σ_xy≈1 cm, σ_z≈2.5 cm) 에서::

    경로                                        자세     평면    RSS 합
    ------------------------------------------------------------------
    직접 (skip_sfm), 자세 0.3°, 중심           24.0     0.0    24.1 cm
    직접 (skip_sfm), 자세 0.3°, 가장자리        29.6     4.8    30.1 cm
    BA 후, 자세 0.03°, 중심                     2.4     0.0     3.3 cm
    BA 후, 자세 0.03°, 가장자리                 3.0     4.8     6.2 cm
    BA 후, 평면을 '지면' 에 맞춤 (Δh=1.5 m)      3.0    72.5    72.6 cm

읽는 법:

* **``skip_sfm`` 은 dm 급이다.** 짐벌 자세 정확도에 직접 묶여 2~5 cm 가 안
  나온다. 현장에서 눈으로 확인하는 용도지 좌표를 쓰는 용도가 아니다.
* **BA 경로는 중심 3 cm / 가장자리 6 cm 수준**이 현실적인 기대치다.
* **평면을 어디에 맞추느냐가 가장 크다.** 태양광 단지에서 결함은 패널
  상면에 있다. 평면을 지면에 맞추면 가장자리에서 70 cm 가 넘는다. BA 점군은
  패널 상면이 대부분이라 ``estimate_ground_plane`` 이 자연히 상면 평면을
  잡는데, 이것이 **의도된 동작**이다. LRF/메타데이터 폴백 경로로 떨어지면
  지면 평면이 나오므로 ``panel_top_offset_m`` 로 보정할 것.

**수직 정확도는 이 산출물에 정의되지 않는다.** 평면 호모그래피는 점별 높이를
만들지 않는다. 존재하는 수직량은 적합된 평면 ``(a, b, c)`` 하나뿐이고, 그
품질은 ``summary["ground_plane"]["inlier_rmse_m"]`` 로 보고된다. 정사영상에서
표고를 읽으려 하지 말 것 — DSM 이 필요하면 이 파이프라인이 의도적으로 포기한
부분이다.

BA 후 자세 정밀도는 로그에서 역산할 수 있다: ``δ ≈ 재투영RMSE / f_px``
(0.5 px, f_px=4000 → 0.007°). 위 표의 0.03° 는 그보다 보수적인 가정이다.

위 수치는 **계산된 예산이지 측정값이 아니다.** 자세오차 0.3° / 0.03° 는
가정이고, 실제 값은 기체·짐벌 개체차와 비행 조건에 따라 달라진다. 절대
정확도가 중요한 측량/검측용은 최소 1~2 점의 Check Point 로 실측 검증할 것.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

from ..image.metadata import extract_metadata

from .crs import CRSConverter
from .features import build_tie_points, find_neighbor_pairs
from .gpu_backend import gpu_summary
from .homography import (
    MosaicConfig,
    build_frame_homography,
    estimate_ground_plane,
    intrinsics_from_metadata,
    mosaic_frames,
    opk_from_gimbal,
    orthorectify_frame,
    recommend_gsd,
    select_gps_neighbor_pairs,
)
from .homography.homography import GroundPlane
from .homography.pairing import footprint_radius_m
from .rtk import (
    compute_rtk_prior_weights,
    estimate_ground_z,
    validate_rtk_quality,
)
from .sfm import (
    build_tracks,
    rtk_constrained_bundle_adjustment,
    triangulate_tracks,
)
from .utils import fmt_elapsed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 초기 외부표정
# ---------------------------------------------------------------------------
def _build_initial_state(metas, crs: CRSConverter,
                         geoid_undulation_m: float = 0.0):
    """RTK + 짐벌 자세 → 초기 외부표정 + RTK prior 배열.

    ``(ω, φ, κ)`` 는 ``homography.pose.opk_from_gimbal`` 로 만든다.
    nadir 정북에서 ``(−180°, 0°, −180°)`` 가 나오는데, ω 가 ±180° 근처인 것은
    이 규약의 정상 동작이다 (φ 가 0 근처라 짐벌락도 없다).

    Parameters
    ----------
    geoid_undulation_m : 타원체고 → 정표고 변환용 지오이드고 (m).
        한국 내륙은 대략 22~30 m. 정사영상의 평면좌표만 쓸 거라면 전
        프레임 공통이라 상쇄되어 무해하지만, 다른 측량성과나 DEM 과
        비교할 거라면 반드시 넣어야 한다.
    """
    rtk_priors = np.empty((len(metas), 3))
    initial_cameras = np.empty((len(metas), 6))
    for i, m in enumerate(metas):
        X, Y = crs.forward(m.gps.lng, m.gps.lat)
        Z = float(m.gps.altitude) - geoid_undulation_m
        rtk_priors[i] = (X, Y, Z)
        omega, phi, kappa = opk_from_gimbal(
            m.gimbal_yaw_deg, m.gimbal_pitch_deg, m.gimbal_roll_deg
        )
        initial_cameras[i] = (X, Y, Z, omega, phi, kappa)
    return rtk_priors, initial_cameras


def _diagnose_metadata(metas) -> None:
    """정사영상 단계 크래시를 유발하는 메타데이터 결손을 사전 경고."""
    no_focal = [Path(m.origin_path).name for m in metas
                if (not m.focal_length_in_35mm or m.focal_length_in_35mm <= 0)
                and (not m.focal_length or m.focal_length <= 0)]
    if no_focal:
        logger.warning(
            "focal_length 정보가 전혀 없는 사진 %d장: %s%s — 자동 건너뜀. "
            "(DJI H20T 의 _T (thermal) 파일이 섞여있는지 확인)",
            len(no_focal), no_focal[:3], " ..." if len(no_focal) > 3 else "",
        )
    thermal_like = [Path(m.origin_path).name for m in metas
                    if Path(m.origin_path).stem.endswith(("_Z", "_T"))]
    if thermal_like:
        logger.warning(
            "Thermal/Telephoto 패턴 (_Z, _T) 파일명 %d장 감지: %s%s — "
            "RGB 만 처리하려면 디렉토리를 분리하거나 글로브 패턴을 변경.",
            len(thermal_like), thermal_like[:3],
            " ..." if len(thermal_like) > 3 else "",
        )


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def run_homography_pipeline(image_dir: Path,
                 output_dir: Path,
                 target_epsg: int = 5186,
                 gsd_m: float | None = None,
                 device: str = "mps",
                 k_neighbors: int = 8,
                 geoid_undulation_m: float = 0.0,
                 skip_sfm: bool = False,
                 mosaic: bool = True,
                 allow_tilted_plane: bool = True,
                 panel_top_offset_m: float = 0.0,
                 plane_surface: str = "upper",
                 auto_plane: bool = True,
                 coarse_reproj_px: float = 0.0,
                 fine_reproj_px: float = 3.0,
                 ba_stage1_attitude_deg: float = 1.5,
                 seam_optimize: bool = True,
                 seam_cost_weight: float = 1.0,
                 exposure_compensate: bool = True,
                 glint_penalty: float = 0.7,
                 glob_pattern: str = "*.JPG") -> dict:
    """전체 파이프라인 실행.

    Parameters
    ----------
    gsd_m : 출력 픽셀 크기 (m/pixel). ``None`` 이면 프레임 GSD 중앙값을
        자동 사용 (원해상도 보존). 원본은 0.05 고정이었는데, H20T 를 45 m
        고도로 날리면 실제 GSD 는 약 1.2 cm 라 0.05 는 4 배 다운샘플이다.
    skip_sfm : ``True`` 면 특징점 매칭/BA 를 건너뛰고 RTK 자세만으로 정사보정.
        빠른 현장 확인용. 정확도는 RTK+짐벌 정확도에 직접 묶인다.
    mosaic : ``True`` 면 단일 모자이크 GeoTIFF, ``False`` 면 프레임별 GeoTIFF.
    allow_tilted_plane : 지상평면에 경사를 허용. 호모그래피는 임의 평면에
        정확하므로 수평 가정은 근거 없는 제약이다.
    panel_top_offset_m : 지면 대비 패널 상면 높이 (m). LRF/메타데이터 폴백
        경로에서 기준면을 패널 상면으로 올리는 데 쓴다. BA 점군 경로는
        ``plane_surface="upper"`` 가 자동 검출하므로 보통 불필요.
        고정식 가대는 대개 1.5~2.0.
    plane_surface : ``"upper"`` (기본) 는 점군에 두 층이 있으면 위층(패널
        상면) 을 기준면으로 삼는다. 지면 기준 정사영상이 필요하면
        ``"dominant"``.

    Returns
    -------
    실행 요약 dict (``summary.json`` 으로도 저장).
    """
    logger.info("=" * 70)
    logger.info("RTK 기반 호모그래피 파이프라인 (GCP-Free) — 가속: %s",
                gpu_summary())
    logger.info("=" * 70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crs = CRSConverter(target_epsg=target_epsg)

    # ---- 1. 메타데이터 --------------------------------------------------
    images = sorted(Path(image_dir).glob(glob_pattern))
    if not images:
        raise FileNotFoundError(f"{image_dir} 에 {glob_pattern} 파일이 없습니다")
    metas = [extract_metadata(p) for p in images]
    logger.info("이미지 %d장 로딩", len(metas))
    _diagnose_metadata(metas)

    # 내부 파라미터를 못 만드는 사진은 여기서 제외 — 뒤로 흘려보내면
    # triangulation/BA 인덱스가 어긋난다.
    keep = [i for i, m in enumerate(metas)
            if intrinsics_from_metadata(m) is not None]
    if len(keep) < len(metas):
        logger.warning("내부 파라미터 구성 불가 %d장 제외", len(metas) - len(keep))
        metas = [metas[i] for i in keep]
    if not metas:
        raise ValueError("사용 가능한 사진이 없습니다")

    # ---- 2~3. 좌표 변환 + RTK 품질 --------------------------------------
    validate_rtk_quality(metas)
    rtk_priors, initial_cameras = _build_initial_state(
        metas, crs, geoid_undulation_m=geoid_undulation_m)
    rtk_weights = compute_rtk_prior_weights(metas)

    intrinsics_obj = [intrinsics_from_metadata(m) for m in metas]
    intrinsics = [(k.f_px, k.cx, k.cy) for k in intrinsics_obj]

    cams_opt = initial_cameras
    pts_opt = None
    ba_rmse = None

    if not skip_sfm:
        # ---- 4. 인접쌍 + 매칭 -------------------------------------------
        t0 = time.perf_counter()
        radii = np.array([
            footprint_radius_m(float(m.relative_height or 100.0),
                               k.f_px, k.width, k.height)
            for m, k in zip(metas, intrinsics_obj)
        ])
        # 겹침 기하로 후보를 거른 뒤 k-NN 로 상한을 둔다. 원본의 순수 k-NN
        # 은 겹치지 않는 쌍도 포함시켜, 반복 텍스처(패널 격자)에서 가짜
        # 매칭을 만들어 낸다.
        pairs = select_gps_neighbor_pairs(
            rtk_priors, radii, max_pairs_per_image=k_neighbors)
        if not pairs:
            logger.warning("겹침 기반 인접쌍이 0개 — k-NN 폴백")
            pairs = find_neighbor_pairs(metas, crs, k_neighbors=k_neighbors)

        matches, features = build_tie_points(metas, pairs)
        logger.info("[stage] SfM 매칭: %s", fmt_elapsed(time.perf_counter() - t0))

        # ---- 5a. track --------------------------------------------------
        t0 = time.perf_counter()
        tracks = build_tracks(matches, features,
                              min_track_len=2, max_track_len=30)
        logger.info("[stage] track 빌드: %s", fmt_elapsed(time.perf_counter() - t0))

        if len(tracks) < 10:
            logger.warning("track 이 너무 적음(%d). BA 생략, RTK 초기값 사용.",
                           len(tracks))
        else:
            f_rep = float(np.median([k[0] for k in intrinsics]))
            cx_rep = float(np.median([k[1] for k in intrinsics]))
            cy_rep = float(np.median([k[2] for k in intrinsics]))
            spread = np.std([k[0] for k in intrinsics]) / max(f_rep, 1e-9)
            if spread > 0.01:
                logger.warning(
                    "프레임 간 초점거리 산포 %.2f%% — 줌 설정이 섞여 있습니다. "
                    "BA 는 단일 (f, cx, cy) 를 가정하므로 카메라별로 나눠 "
                    "돌리거나 내부 파라미터를 미지수로 두어야 합니다.",
                    spread * 100)

            # ================= 5b~6. 2단계 삼각측량 + BA =================
            # ★ 이 파이프라인이 실데이터에서 실패하던 지점 (실측 근거)
            #
            #   380 장 실행의 summary.json 에서 평면 적합 inlier 가 147 개,
            #   즉 이미지당 0.4 개였다. 고중복 380 장이면 수천~수만 개가
            #   나와야 한다. 원인은 삼각측량 게이트다:
            #
            #     max_reproj_err_px=3.0,  f_px=6768
            #       → 초기 자세오차가 3/6768 rad = 0.025° 를 넘으면 그 점은
            #         버려진다.
            #
            #   짐벌이 보고하는 자세는 0.1° 단위이고 광축 절대 정확도는 그보다
            #   나쁘다. 실제로 모자이크 정렬이 요구한 보정량(최대 1.0 m, 상한에
            #   잘림)을 자세로 환산하면 약 1.3° = 154 px 다. 3 px 게이트는
            #   그런 점을 전부 버린다.
            #
            #   → 삼각측량이 점을 버림 → BA 가 자세를 못 고침 → 다음에도 같은
            #     점이 버려짐. **순환 실패**다. 게이트를 초기 자세 불확실성에
            #     맞춰 열어 주지 않으면 BA 는 영원히 시작하지 못한다.
            #
            # 해법: 느슨한 게이트로 1차 BA(자세를 대략 맞춤) → 그 자세로
            #       다시 삼각측량하되 이번엔 조인 게이트 → 2차 BA(정밀화).
            #       표준 incremental SfM 관행이다.
            loose_px = max(coarse_reproj_px, 4.0 * ba_stage1_attitude_deg
                           * np.pi / 180.0 * f_rep)
            stages = [("1차(느슨)", loose_px, 2.0),
                      ("2차(정밀)", fine_reproj_px, 2.0)]

            cams_cur = initial_cameras
            for si, (label, gate_px, ang) in enumerate(stages):
                t0 = time.perf_counter()
                observations, initial_points, _ = triangulate_tracks(
                    tracks, cams_cur, intrinsics,
                    max_reproj_err_px=gate_px,
                    min_triangulation_angle_deg=ang,
                )
                logger.info("[stage] triangulation %s (게이트 %.0f px): "
                            "점 %d개, 관측 %d개 — %s",
                            label, gate_px, len(initial_points),
                            len(observations),
                            fmt_elapsed(time.perf_counter() - t0))

                if len(initial_points) < 10 or len(observations) < 30:
                    if si == 0:
                        logger.warning(
                            "1차 삼각측량조차 점이 부족합니다(%d). 매칭/track "
                            "단계를 확인하세요. BA 생략.", len(initial_points))
                    break

                pts_per_img = len(initial_points) / max(len(metas), 1)
                if si == 0 and pts_per_img < 5:
                    logger.warning(
                        "1차 점군이 이미지당 %.1f 개로 희박합니다. 게이트를 "
                        "더 열거나(--coarse-reproj) 매칭 품질을 확인하세요.",
                        pts_per_img)

                t0 = time.perf_counter()
                cams_cur, pts_opt, ba_rmse = rtk_constrained_bundle_adjustment(
                    cams_cur, initial_points, observations,
                    rtk_priors, rtk_weights,
                    f_px=f_rep, cx=cx_rep, cy=cy_rep,
                )
                logger.info("[stage] bundle adjustment %s: %s (RMSE %.2f px)",
                            label, fmt_elapsed(time.perf_counter() - t0),
                            ba_rmse)
                att = np.degrees(np.linalg.norm(
                    cams_cur[:, 3:6] - initial_cameras[:, 3:6], axis=1))
                logger.info("  누적 자세 변화: 중앙값 %.3f°, 최대 %.3f° "
                            "(지상 환산 중앙값 %.2f m)",
                            float(np.median(att)), float(att.max()),
                            float(np.median(att)) * np.pi / 180.0 * 45.0)
                cams_opt = cams_cur

    # ---- 7. 지상평면 + 호모그래피 ---------------------------------------
    t0 = time.perf_counter()
    plane = _resolve_ground_plane(metas, cams_opt, pts_opt,
                                  allow_tilted_plane=allow_tilted_plane,
                                  panel_top_offset_m=panel_top_offset_m,
                                  surface=plane_surface)

    frames, frame_metas = [], []
    for i, (m, k) in enumerate(zip(metas, intrinsics_obj)):
        C = cams_opt[i, :3]
        R = _rotation_from_opk(cams_opt[i, 3], cams_opt[i, 4], cams_opt[i, 5])
        z_ground = plane.height_at(C[0], C[1])
        if C[2] <= z_ground:
            logger.warning("카메라가 지상평면 아래 (%.2f ≤ %.2f m) — 제외: %s",
                           C[2], z_ground, m.origin_path)
            continue
        fh = build_frame_homography(C, R, k, plane)
        if fh is None:
            logger.warning("호모그래피 구성 실패 — 제외: %s", m.origin_path)
            continue
        frames.append(fh)
        frame_metas.append(m)

    if not frames:
        raise ValueError("호모그래피를 구성할 수 있는 프레임이 없습니다")

    # ---- 7b. 겹침 시차로 기준면 높이 자동보정 (실패 시 조용히 건너뜀) ----
    if auto_plane and len(frames) >= 4:
        from .homography.plane_calib import calibrate_plane

        keep = list(zip(frames, frame_metas))

        def _rebuild(pl):
            return [build_frame_homography(f.camera_xyz, f.R, f.intr, pl)
                    for f, _ in keep]

        plane_new, cal = calibrate_plane(
            _rebuild, [m.origin_path for _, m in keep], plane)
        if cal is not None and abs(plane_new.c - plane.c) > 1e-6:
            logger.info("기준면 자동보정: %.2f m → %.2f m (Δ%+.2f m)",
                        plane.c, plane_new.c, plane_new.c - plane.c)
            plane = plane_new
            rebuilt = _rebuild(plane)
            frames = [f for f in rebuilt if f is not None]
            frame_metas = [m for f, (_, m) in zip(rebuilt, keep) if f is not None]

    if gsd_m is None:
        gsd_m = recommend_gsd(frames)
        logger.info("GSD 자동 결정: %.4f m/px (프레임 GSD 중앙값)", gsd_m)
    logger.info("[stage] 호모그래피 구성: %s (%d 프레임, 평면 경사 %.3f°)",
                fmt_elapsed(time.perf_counter() - t0), len(frames),
                plane.slope_deg)

    # ---- 8. 정사영상 -----------------------------------------------------
    t0 = time.perf_counter()
    if mosaic:
        out = output_dir / "mosaic.tif"
        stats = mosaic_frames(frames, [m.origin_path for m in frame_metas],
                              out, gsd_m=gsd_m, epsg=target_epsg,
                              cfg=MosaicConfig(
                                  seam_optimize=seam_optimize,
                                  seam_cost_weight=seam_cost_weight,
                                  exposure_compensate=exposure_compensate,
                                  glint_penalty=glint_penalty))
    else:
        n_ok = 0
        for fh, m in zip(frames, frame_metas):
            out = output_dir / f"{Path(m.origin_path).stem}_ortho.tif"
            if orthorectify_frame(fh, m.origin_path, out,
                                  gsd_m=gsd_m, epsg=target_epsg):
                n_ok += 1
        stats = {"frames_ok": n_ok, "frames_failed": len(frames) - n_ok,
                 "gsd_m": gsd_m}
    logger.info("[stage] 정사영상: %s", fmt_elapsed(time.perf_counter() - t0))

    summary = {
        "images": len(metas),
        "frames_georeferenced": len(frames),
        "epsg": target_epsg,
        "gsd_m": gsd_m,
        "ba_reprojection_rmse_px": ba_rmse,
        "ground_plane": {
            "a": plane.a, "b": plane.b, "c": plane.c,
            "origin_xy": list(plane.origin_xy),
            "slope_deg": plane.slope_deg,
            "inlier_rmse_m": plane.inlier_rmse_m,
            "n_inliers": plane.n_inliers,
        },
        "ortho": stats,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def _rotation_from_opk(omega, phi, kappa):
    from .geometry import rotation_matrix
    return rotation_matrix(omega, phi, kappa)


def _resolve_ground_plane(metas, cams_opt, pts_opt, allow_tilted_plane=True,
                          panel_top_offset_m: float = 0.0,
                          surface: str = "upper"):
    """BA 점군 → LRF → 메타데이터 순으로 지상평면 결정.

    ★ 기준면 높이가 정사 모자이크 품질을 지배한다.
      패널 상면은 지면보다 1.5~2 m 높다. 기준면을 지면에 두면 패널이
      프레임마다 ``Δh·tanθ`` 만큼 밀려(relief displacement), 모자이크
      시임에서 패널이 뚝 끊기고 그 자리에 지면이 드러난다.
      H20T 45.9 m 고도 · 패널 1.8 m 기준, 인접 프레임이 반대 방향으로
      밀리면 시임에서 최대 174 cm — 패널 한 장 폭이다.

      따라서 검사 대상인 패널을 기준면으로 삼는다:
      * BA 점군 경로 — ``surface="upper"`` 가 두 층 중 위층(패널 상면) 검출
      * LRF/메타데이터 경로 — 조준점은 지면이므로 ``panel_top_offset_m``
        (지면 대비 패널 상면 높이) 를 더해야 같은 기준면이 된다

    ``estimate_ground_z`` 가 ``None`` 을 주는 사진은 평면 추정 근거에서
    빠지되, 다른 사진이 근거를 주면 그 평면을 공유해서 계속 처리한다.
    """
    # LRF/메타데이터 기준 고도 — BA 점군 평면의 교차검증에 쓴다.
    ref_zs = [z for z in (estimate_ground_z(m) for m in metas) if z is not None]
    ref_z = float(np.median(ref_zs)) if ref_zs else None

    if pts_opt is not None and len(pts_opt) >= 3:
        plane = estimate_ground_plane(
            points_xyz=pts_opt,
            metas=metas, camera_xyz=cams_opt[:, :3],
            allow_tilt=allow_tilted_plane,
            panel_top_offset_m=panel_top_offset_m,
            surface=surface,
        )
        # ★ 교차검증 — BA 점군이 나쁘면 평면이 엉뚱한 고도로 간다.
        #   실측에서 기준면이 LRF 지면보다 6.6 m 아래로 내려간 사례가 있었고,
        #   그 상태로는 어떤 모자이크 설정도 결과를 살리지 못한다.
        #   물리적으로 기준면은 지면 ± 몇 m 안에 있어야 한다.
        if ref_z is not None:
            cx_, cy_ = cams_opt[:, 0].mean(), cams_opt[:, 1].mean()
            gap = float(plane.height_at(cx_, cy_)) - ref_z
            if abs(gap) > 5.0:
                logger.warning(
                    "BA 점군 평면(%.2f m)이 LRF/메타데이터 기준(%.2f m)과 "
                    "%.1f m 어긋납니다 — 점군 품질이 나쁘다는 신호입니다. "
                    "LRF 기준면으로 대체합니다. (BA 재투영 RMSE 와 track 수를 "
                    "확인하세요)",
                    plane.height_at(cx_, cy_), ref_z, gap)
            else:
                return plane
        else:
            return plane

    zs = ref_zs
    if zs:
        z = float(np.median(zs)) + panel_top_offset_m
        if panel_top_offset_m:
            logger.info("지상평면 (수평, %d개 근거): 지면 %.2f m + 패널 오프셋 "
                        "%.2f m = %.2f m", len(zs), float(np.median(zs)),
                        panel_top_offset_m, z)
        else:
            logger.info("지상평면 (수평, %d개 근거): Z=%.2f m", len(zs), z)
            logger.warning(
                "기준면이 지면입니다. 태양광 패널은 지면보다 1.5~2 m 높으므로 "
                "모자이크 시임에서 패널이 끊겨 보일 수 있습니다 — "
                "panel_top_offset_m 에 지면 대비 패널 상면 높이를 주세요.")
        return GroundPlane.horizontal(z)

    return estimate_ground_plane(
        metas=metas, camera_xyz=cams_opt[:, :3],
        allow_tilt=allow_tilted_plane,
        panel_top_offset_m=panel_top_offset_m,
        surface=surface,
    )


def main():
    """CLI 진입점. 데모용 기본 경로 사용.

    (원본은 존재하지 않는 ``run_pipeline_gcp_free`` 를 불러 NameError 로
    죽었다.)
    """
    start = time.perf_counter()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_homography_pipeline(
        image_dir=Path("./data/solar/images/RGB"),
        output_dir=Path("./workspace/output"),
        target_epsg=5186,
        k_neighbors=8,
    )
    print(f"Elapsed: {timedelta(seconds=int(time.perf_counter() - start))}")


if __name__ == "__main__":
    main()


__all__ = ["run_homography_pipeline", "main"]

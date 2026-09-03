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
                 tile_memory_mb: float = 256.0,
                 prefetch_workers: int = 4,
                 max_offnadir_ratio: float = 0.35,
                 offnadir_auto: bool = True,
                 offnadir_frac: float = 0.65,
                 offnadir_fallback: bool = True,
                 offnadir_tolerance: float = 1.15,
                 offnadir_epsilon: float = 0.002,
                 auto_focal: bool = True,
                 focal_tolerance: float = 0.01,
                 focal_max_rounds: int = 4,
                 focal_max_total: float = 0.25,
                 mid_reproj_factor: float = 3.0,
                 estimate_distortion: bool = False,
                 auto_distortion: bool = True,
                 fine_gate_auto: bool = True,
                 fine_gate_max_px: float = 8.0,
                 distortion_max_rounds: int = 1,
                 fix_positions: bool = True,
                 attitude_sigma_deg: float = 1.0,
                 plane_tilt_tolerance_deg: float = 1.0,
                 ba_max_nfev: int = 200,
                 coarse_ba_ftol: float = 1e-4,
                 use_feature_cache: bool = True,
                 tri_z_band_m: float = 25.0,
                 fine_tri_angle_deg: float = 5.0,
                 use_dsm: bool = True,
                 layer_surface: bool = False,
                 layer_gap_m: float = 0.0,
                 layer_cell_m: float = 0.5,
                 use_two_layer: bool = True,
                 two_layer_min_area_m2: float = 3.0,
                 dsm_max_relief_m: float = 6.0,
                 dsm_cell_m: float = 0.8,
                 rtk_boost_max: float = 8.0,
                 plane_lrf_tolerance_m: float = 3.0,
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
    obs_stats = None
    focal_info = None
    distortion_info = None

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

        # ★ SIFT 추출 + 매칭은 전체 시간의 48% 인데(실측 20m49s 중 9m41s),
        #   입력 이미지가 같으면 결과가 항상 같다. 파라미터를 바꿔 가며
        #   반복 실행할 때 매번 다시 계산할 이유가 없다.
        #   초점거리·BA·모자이크 설정을 바꿔도 이 단계 결과는 안 바뀌므로
        #   캐시가 그대로 유효하다 — 재실행이 절반으로 줄어든다.
        from .features.cache import FeatureCache
        _cache = FeatureCache(output_dir, enabled=use_feature_cache)
        _ckey = {"pairs": len(pairs), "k_neighbors": k_neighbors}
        _hit = _cache.load(metas, pairs, _ckey)
        if _hit is not None:
            matches, _feats_ser = _hit
            features = FeatureCache.restore_features(_feats_ser)
        else:
            matches, features = build_tie_points(metas, pairs)
            _cache.save(metas, pairs, _ckey, matches, features)
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
            f_px_exif = f_rep
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
            # ★ 정밀 단계는 시선각 하한을 올린다. 반복 텍스처(패널 셀 주기
            #   0.23 m)에서 짧은 베이스라인의 오매칭은 재투영이 완벽하면서
            #   깊이만 크게 틀린 점을 만든다 — 2°(베이스라인 1.6 m)면 한 칸
            #   오매칭이 6.5 m 깊이오차다. 게이트 3 px 로는 못 거른다.
            #   5°(4.0 m)면 2.6 m, 8°(6.3 m)면 1.6 m 로 줄어든다.
            stages = [("1차(느슨)", loose_px, 2.0),
                      ("2차(정밀)", fine_reproj_px, fine_tri_angle_deg)]
            focal_rounds = 0
            mid_rounds = 0
            dist_rounds = 0
            dist_diag = None

            _rz = [z for z in (estimate_ground_z(m) for m in metas)
                   if z is not None]
            ref_ground_z = float(np.median(_rz)) if _rz else None

            cams_cur = initial_cameras
            si = -1
            while True:
                si += 1
                if si >= len(stages):
                    break
                label, gate_px, ang = stages[si]
                t0 = time.perf_counter()
                observations, initial_points, _ = triangulate_tracks(
                    tracks, cams_cur, intrinsics,
                    max_reproj_err_px=gate_px,
                    min_triangulation_angle_deg=ang,
                )
                # ★ 느슨한 게이트는 Z 가 −52~346 m 인 점까지 통과시킨다
                #   (실측 로그). 그런 점이 BA 를 끌고 다니므로 물리적으로
                #   불가능한 높이는 여기서 잘라낸다. 기준은 LRF/메타데이터.
                if ref_ground_z is not None and len(initial_points):
                    zlo = ref_ground_z - tri_z_band_m
                    zhi = ref_ground_z + tri_z_band_m
                    keep = ((initial_points[:, 2] >= zlo)
                            & (initial_points[:, 2] <= zhi))
                    n_drop = int((~keep).sum())
                    if n_drop:
                        remap = -np.ones(len(initial_points), dtype=np.int64)
                        remap[keep] = np.arange(int(keep.sum()))
                        initial_points = initial_points[keep]
                        observations = [(ci, int(remap[pi]), uv)
                                        for ci, pi, uv in observations
                                        if remap[pi] >= 0]
                        logger.info("  Z 밴드 필터: %d개 제거 (%.1f~%.1f m 밖), "
                                    "%d개 남음", n_drop, zlo, zhi,
                                    len(initial_points))

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

                # ---- 관측수 기반 RTK prior 가중 ------------------------
                # ★ 블록 가장자리 프레임은 이웃이 적어 관측(tie point)이
                #   적다. 그런데 RTK prior 가중치가 전 프레임 동일하면, 관측이
                #   적은 프레임일수록 소수의 관측에 끌려 자유롭게 움직인다.
                #   실측에서 모자이크 국소 어긋남이 안쪽 0.061 m → 바깥
                #   0.321 m 로 5배 커진 이유 중 하나다.
                #
                #   영상 제약이 약하면 RTK 를 더 믿는 것이 옳다. 관측수가
                #   중앙값보다 적은 프레임의 prior 가중치를 비례해서 올린다
                #   (상한 있음 — 완전히 고정해 버리면 BA 가 그 프레임의 자세
                #   오차를 못 고친다).
                obs_cnt = np.bincount(
                    np.fromiter((o[0] for o in observations), dtype=np.int64,
                                count=len(observations)),
                    minlength=len(metas)).astype(float)
                med_obs = max(float(np.median(obs_cnt[obs_cnt > 0])), 1.0)
                boost = np.clip(med_obs / np.maximum(obs_cnt, 1.0), 1.0,
                                rtk_boost_max)
                w_stage = rtk_weights * boost[:, None]
                weak = int((obs_cnt < 0.3 * med_obs).sum())
                if weak:
                    logger.info("  관측이 적은 프레임 %d장(중앙값 %.0f개 대비 "
                                "30%% 미만) — RTK prior 가중치를 최대 %.0f배 "
                                "강화", weak, med_obs, float(boost.max()))
                if int((obs_cnt == 0).sum()):
                    logger.warning(
                        "  tie point 가 하나도 없는 프레임 %d장 — 이 프레임들은 "
                        "RTK/짐벌 값 그대로 남습니다 (자세오차 미보정). "
                        "모자이크 가장자리 품질 저하의 직접 원인입니다.",
                        int((obs_cnt == 0).sum()))

                t0 = time.perf_counter()
                if fix_positions:
                    # ★ 실측 로그가 보여준 발산 (380장):
                    #     RTK prior 대비 카메라 이동 중앙값 1.823 m (σ=1.1 cm → 166σ)
                    #     누적 자세 변화 중앙값 10.728°, 최대 20.758°
                    #       (초기 재투영 110.7 px = 0.94° 인데 11배를 움직였다)
                    #   영상 잔차 1,799,056개 vs RTK 잔차 1,140개 (1578:1) 라
                    #   RTK prior 가 수적으로 무력했고, 자세에는 prior 가 아예
                    #   없었다. 위치를 파라미터에서 빼면 두 문제가 구조적으로
                    #   사라진다 — RTK Fixed 의 1.1 cm 는 BA 가 영상만으로
                    #   도달할 수 있는 정확도보다 훨씬 좋으므로 추정할 이유가
                    #   없다.
                    from .sfm.ba_rtk_fixed import rtk_fixed_bundle_adjustment
                    cams_cur[:, :3] = rtk_priors        # 위치는 RTK 그대로
                    cams_cur, pts_opt, ba_rmse = rtk_fixed_bundle_adjustment(
                        cams_cur, initial_points, observations,
                        f_px=f_rep, cx=cx_rep, cy=cy_rep,
                        attitude_sigma_deg=attitude_sigma_deg,
                        max_nfev=ba_max_nfev,
                        # 느슨/중간 단계만 조기 종료. 정밀 단계는 엄격하게.
                        coarse_ftol=(0.0 if label.startswith("2차")
                                     else coarse_ba_ftol),
                    )
                else:
                    cams_cur, pts_opt, ba_rmse = rtk_constrained_bundle_adjustment(
                        cams_cur, initial_points, observations,
                        rtk_priors, w_stage,
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

                # ---- 잔차의 반경 의존성 진단 (왜곡 판별) -----------------
                # ★ 실측에서 중간 게이트로 RMSE 를 9.43 → 5.21 px 로 낮췄는데도
                #   **중앙값은 2.69 → 2.66 으로 거의 그대로**였다. 이상치를
                #   걷어내도 남는 '바닥' 이 있다는 뜻이다.
                #   그 바닥이 렌즈 왜곡인지(반경의 3승으로 자람) 오매칭인지
                #   (반경과 무관) 는 이 진단 한 줄로 갈린다.
                if label.startswith("2차") or label.startswith("중간"):
                    from .homography.distortion import (
                        diagnose_residual_vs_radius)
                    dist_diag = diagnose_residual_vs_radius(
                        observations, cams_cur, pts_opt, f_rep, cx_rep, cy_rep)

                    # ★ 진단이 왜곡을 확인하면 자동으로 보정한다.
                    #   (--no-auto-distortion 으로 끌 수 있음)
                    #   판정은 '절편 포함 적합' 기준이다 — 절편을 빼먹은 앞선
                    #   버전은 실측에서 R²=0.14 로 오판했고, 절편을 넣자
                    #   R²=0.985, 바닥 1.72 px + 왜곡 3.87 px 로 갈렸다.
                    if ((estimate_distortion
                         or (auto_distortion and dist_diag is not None
                             and dist_diag.get("distortion_detected")))
                            and dist_diag is not None
                            and dist_rounds < distortion_max_rounds):
                        from .homography.distortion import (
                            estimate_radial_distortion, undistort_observations)
                        dres = estimate_radial_distortion(
                            observations, cams_cur, pts_opt,
                            f_rep, cx_rep, cy_rep)
                        if dres is not None:
                            k1d, k2d, dinfo = dres
                            dist_rounds += 1
                            distortion_info = dinfo
                            tracks = [
                                [(ci, ki, *undistort_observations(
                                    [(0, 0, np.array([px_, py_]))],
                                    k1d, k2d, f_rep, cx_rep, cy_rep)[0][2])
                                 for ci, ki, px_, py_ in tk]
                                for tk in tracks]
                            from dataclasses import replace as _dcr
                            intrinsics_obj = [
                                _dcr(k, k1=getattr(k, "k1", 0.0) + k1d,
                                     k2=getattr(k, "k2", 0.0) + k2d)
                                if hasattr(k, "k1") else k
                                for k in intrinsics_obj]
                            loose_px = max(
                                coarse_reproj_px,
                                4.0 * ba_stage1_attitude_deg * np.pi / 180.0
                                * f_rep)
                            stages.insert(si + 1,
                                          ("왜곡보정 후(느슨)", loose_px, 2.0))
                            logger.warning(
                                "방사 왜곡을 보정하고 느슨한 단계부터 다시 "
                                "돌립니다 (k1=%+.5f). 관측 좌표를 펴면 잔차 "
                                "바닥이 사라져 정밀 게이트에서 훨씬 많은 점이 "
                                "살아남습니다.", k1d)

                # ---- 정밀 게이트를 '측정된 잔차 프로파일' 에서 결정 -----
                # ★ 실측: 잔차는 '반경 무관 바닥 1.72 px' 위에 '반경에 따라
                #   자라는 성분 3.87 px' 가 얹힌 형태였다 (절편 포함 적합
                #   R²=0.985). 그런데 정밀 게이트는 상수 3 px 이라
                #   **반경 1814 px 바깥이 통째로 잘렸다** — 프레임 반경의
                #   56% 지점이고, 점 손실 50% 와 정확히 맞는다.
                #
                #   상수 게이트는 이미지 중심 쪽 점만 남겨 점군을 편향시킨다.
                #   원인(왜곡인지 다른 것인지)을 단정하지 않고도, 게이트를
                #   측정된 프로파일에 맞추면 반경 전체에서 고르게 남는다.
                #   r² 와 r³ 은 6구간으로 구분되지 않으므로(0.982 vs 0.985)
                #   멱수를 가정하지 않고 '바닥 + 최대반경 성분' 만 쓴다.
                if (fine_gate_auto and dist_diag is not None
                        and label.startswith("중간")):
                    prof_gate = float(dist_diag["floor_px"]
                                      + dist_diag["radial_at_max_px"])
                    new_fine = float(np.clip(prof_gate, fine_reproj_px,
                                             fine_gate_max_px))
                    if new_fine > fine_reproj_px * 1.2:
                        for _i, _st in enumerate(stages):
                            if _st[0].startswith("2차"):
                                stages[_i] = (_st[0], new_fine, _st[2])
                        logger.info(
                            "  정밀 게이트를 측정 프로파일로 조정: %.1f → "
                            "%.1f px (바닥 %.2f + 반경성분 %.2f). 상수 게이트는 "
                            "반경 바깥을 통째로 잘라 점군을 중심 쪽으로 "
                            "편향시킵니다.", fine_reproj_px, new_fine,
                            dist_diag["floor_px"],
                            dist_diag["radial_at_max_px"])

                # ---- 중간 게이트 단계 자동 삽입 --------------------------
                # ★ 실측 로그: 느슨한 단계가 RMSE 9.26 px 에서 끝났는데 다음
                #   단계 게이트가 3 px 였다. 726 → 3 px 는 242배이고, 3 px 는
                #   그 시점 잔차 중앙값(2.69)보다 겨우 큰 값이라 **정상 점까지
                #   잘렸다** — 재투영으로 91,767개(점의 34%, 관측의 59%) 제거.
                #
                #   '느슨 → 정밀' 2단계 철학을 한 단계 더 이어서, 현재 잔차
                #   수준에 맞춘 중간 게이트를 넣고 BA 를 한 번 더 조인다.
                #   그러면 정밀 게이트에 도달했을 때 잔차가 이미 작아 정상
                #   점이 덜 잘린다.
                if (mid_reproj_factor > 0 and ba_rmse is not None
                        and not label.startswith("중간")
                        and mid_rounds < 1):
                    nxt = stages[si + 1] if si + 1 < len(stages) else None
                    nxt_gate = nxt[1] if nxt else fine_reproj_px
                    mid_gate = float(np.clip(mid_reproj_factor * ba_rmse,
                                             nxt_gate * 2.0, 120.0))
                    if mid_gate > nxt_gate * 2.0:
                        mid_rounds += 1
                        stages.insert(si + 1,
                                      ("중간(게이트 완화)", mid_gate, 2.0))
                        logger.info(
                            "  중간 게이트 단계 삽입: %.0f px "
                            "(현재 RMSE %.2f px × %.1f). 정밀 게이트 %.0f px "
                            "로 바로 가면 정상 점까지 잘립니다.",
                            mid_gate, ba_rmse, mid_reproj_factor, nxt_gate)

                # ---- 초점거리 자기보정 (1차 BA 뒤 한 번) -----------------
                # ★ 초점거리–깊이 축퇴: 점 위치가 자유변수라, f 가 20% 커도
                #   BA 는 점을 20% 깊게 밀어 재투영을 완벽히 맞춘다. RMSE 는
                #   낮게 나오지만 3D 가 통째로 틀린다. RTK 로 카메라 위치를
                #   고정해도 이 축퇴는 안 깨진다 (위치는 맞고 깊이만 스케일).
                #
                #   실측: 재투영 RMSE 1.25 px 로 좋아 보였지만 점군 Z 중앙값이
                #   105.60 m (지면 113.4), 카메라–점군 54.22 m 인데
                #   RelativeAltitude 와 LRF 는 둘 다 44.9 m 였다. 비율 1.207.
                #
                #   축퇴를 깨는 것은 **독립적인 거리 관측**이다 — 기압고도계와
                #   레이저. 둘이 일치하면 그 비율이 곧 f 의 오차 배율이다.
                if (label.startswith("1차") and auto_focal
                        and focal_rounds < focal_max_rounds):
                    from .homography.focal import estimate_focal_scale
                    fres = estimate_focal_scale(metas, cams_cur, pts_opt)
                    if fres is not None:
                        fscale, finfo = fres
                        # EXIF 대비 누적 보정 상한 — 두 실행이 반대 방향으로
                        # 22% 벌어진 적이 있어 물리적 한계를 건다.
                        cum = (f_rep / fscale) / f_px_exif
                        if not (1.0 - focal_max_total <= cum
                                <= 1.0 + focal_max_total):
                            logger.warning(
                                "초점거리 누적 보정이 EXIF 대비 %.1f%% 로 상한 "
                                "±%.0f%% 를 넘어 중단합니다.",
                                (cum - 1.0) * 100, focal_max_total * 100)
                        elif abs(fscale - 1.0) > focal_tolerance:
                            f_rep = f_rep / fscale
                            intrinsics = [(k[0] / fscale, k[1], k[2])
                                          for k in intrinsics]
                            # PinholeIntrinsics 는 frozen dataclass 이므로
                            # 새 인스턴스로 교체한다.
                            from dataclasses import replace as _dc_replace
                            intrinsics_obj = [
                                _dc_replace(k, f_px=k.f_px / fscale)
                                for k in intrinsics_obj]
                            focal_rounds += 1
                            if focal_info is None:
                                focal_info = dict(finfo)
                                focal_info["rounds"] = []
                            focal_info["rounds"].append(
                                {"scale": fscale,
                                 "f_px_after": f_rep,
                                 "d_triangulated_m": finfo["d_triangulated_median_m"],
                                 "d_metadata_m": finfo["d_metadata_median_m"]})
                            focal_info["f_px_final"] = f_rep
                            logger.warning(
                                "초점거리를 %.4f 로 나눠 보정합니다 "
                                "(f_px %.1f → %.1f). EXIF 의 "
                                "FocalLengthIn35mmFilm 이 실제 광학 상태와 "
                                "맞지 않는다는 뜻입니다.",
                                fscale, f_rep * fscale, f_rep)
                            # ★ f 를 바꾸면 재투영이 반경에 비례해 크게 이동한다
                            #   (배율 1.15, r=2000 px 이면 265 px). 옛 f 로
                            #   최적화된 카메라/점을 그대로 두고 정밀 게이트
                            #   (3 px) 로 넘어가면 점이 전부 기각된다 —
                            #   실측에서 231,713 → 76,781 (33%) 로 줄었고
                            #   프레임당 관측은 16% 로 떨어졌다.
                            #   보정된 f 로 **느슨한 단계를 한 번 더** 돌려
                            #   카메라/점을 새 f 에 맞춘 뒤 정밀 단계로 간다.
                            loose_px = max(
                                coarse_reproj_px,
                                4.0 * ba_stage1_attitude_deg * np.pi / 180.0
                                * f_rep)
                            stages.insert(
                                si + 1,
                                (f"1차-재실행#{focal_rounds}(느슨)",
                                 loose_px, 2.0))
                            logger.info(
                                "  보정된 f 로 느슨한 단계를 다시 실행합니다 "
                                "(게이트 %.0f px) — 정밀 단계로 바로 가면 "
                                "점을 전부 잃습니다.", loose_px)
                obs_stats = {
                    "median": float(np.median(obs_cnt)),
                    "min": float(obs_cnt.min()),
                    "p10": float(np.percentile(obs_cnt, 10)),
                    "zero_frames": int((obs_cnt == 0).sum()),
                    "total_points": int(len(initial_points)),
                }

    # ---- 7. 지상평면 + 호모그래피 ---------------------------------------
    t0 = time.perf_counter()
    plane, plane_source, plane_tilt = _resolve_ground_plane(
        metas, cams_opt, pts_opt,
        plane_tilt_tolerance_deg=plane_tilt_tolerance_deg,
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
    # ★ BA 점군이 충분하면 그 평면이 가장 신뢰도가 높다 — 겹침 시차 보정을
    #   덧씌울 이유가 없다. 실데이터에서 보정이 기준면을 8.2 m 아래로 밀어
    #   패널이 기준면보다 8.75 m 위에 놓였고, 시임 단차가 1.75 m 로 커졌다.
    #   (BA 를 고쳐 얻은 이득을 그대로 까먹었다.)
    if auto_plane and plane_source == "ba_points":
        logger.info("기준면 자동보정 생략 — BA 점군 평면을 신뢰합니다.")
        auto_plane = False

    if auto_plane and len(frames) >= 4:
        from .homography.plane_calib import calibrate_plane

        keep = list(zip(frames, frame_metas))

        def _rebuild(pl):
            return [build_frame_homography(f.camera_xyz, f.R, f.intr, pl)
                    for f, _ in keep]

        plane_new, cal = calibrate_plane(
            _rebuild, [m.origin_path for _, m in keep], plane)
        # 사후 검증 — 보정 결과가 LRF/메타데이터 기준에서 크게 벗어나면 되돌린다.
        ref_zs2 = [z for z in (estimate_ground_z(m) for m in metas)
                   if z is not None]
        ok = True
        if ref_zs2 and cal is not None:
            cx2 = float(np.mean([f.camera_xyz[0] for f in frames]))
            cy2 = float(np.mean([f.camera_xyz[1] for f in frames]))
            gap2 = float(plane_new.height_at(cx2, cy2)) - float(np.median(ref_zs2))
            if abs(gap2) > plane_lrf_tolerance_m:
                logger.warning(
                    "기준면 자동보정 결과(%.2f m)가 LRF 기준에서 %+.2f m 벗어나 "
                    "되돌립니다 (허용 ±%.1f m). 기준면은 지면과 구조물 상단 "
                    "사이에만 있을 수 있습니다.",
                    plane_new.height_at(cx2, cy2), gap2, plane_lrf_tolerance_m)
                ok = False
        if ok and cal is not None and abs(plane_new.c - plane.c) > 1e-6:
            logger.info("기준면 자동보정: %.2f m → %.2f m (Δ%+.2f m)",
                        plane.c, plane_new.c, plane_new.c - plane.c)
            plane = plane_new
            plane_source = "overlap_calibrated"
            rebuilt = _rebuild(plane)
            frames = [f for f in rebuilt if f is not None]
            frame_metas = [m for f, (_, m) in zip(rebuilt, keep) if f is not None]

    # ---- 7c. 2층 구조 대응: BA 점군으로 성긴 DSM ------------------------
    # ★ 평면 하나로는 지면(≈113.4 m)과 패널 상면(≈115.8 m)을 동시에 맞출 수
    #   없다. 실측이 그것을 그대로 보여줬다: 기준면 105.45 로 만든 모자이크는
    #   패널이 19% 확대돼 그려졌고(면적비 실측 1.190 = 이론 1.182), 113.67 로
    #   고치자 패널이 제 크기로 줄면서 그림자 띠가 제 폭으로 넓어졌다.
    #   어느 쪽도 두 층을 동시에 맞추지 못한다.
    #
    #   점군이 147개일 때는 평면이 유일한 선택이었지만 이제 231,713개다.
    #   성긴 DSM(0.8 m 격자) 이면 층을 구분하기에 충분하다.
    dsm_obj = None
    dsm_reject = None

    # ---- 2층 표면 모델 (연속 DSM 의 대안) -----------------------------
    # ★ 현재 병목은 BA 가 아니라 단일 평면이다:
    #     정밀 단계 재투영 잔차 중앙값 2.11 px = 지상 1.4 cm
    #     그런데 모자이크 안쪽 어긋남은 3.9 cm
    #   차이는 기준면과 실제 표면의 높이차가 만드는 기복변위 Δh·k 다.
    #   기준면이 패널 상면이므로 **지면층이 0.96 m 아래**에 있고,
    #   k=0.15 에서 14 cm, k=0.304 에서 29 cm 밀린다.
    #
    #   연속 DSM 은 셀당 10개 이상을 못 채워 실패했지만(충전율 26.6%),
    #   이 장면의 높이는 연속이 아니라 **두 값 중 하나**다. 이진 다수결은
    #   셀당 3~5개면 성립한다 (셀 0.5 m 에서 3.6개).
    if layer_surface and pts_opt is not None and len(pts_opt) >= 5000:
        gap = float(getattr(plane, "upper_layer_shift_m", 0.0) or 0.0)
        if gap <= 0.05:
            gap = float(layer_gap_m)
        if gap > 0.05:
            from .homography.dsm import build_layer_surface
            fb = [f.footprint_bounds() for f in frames]
            fb = [x for x in fb if x is not None]
            if fb:
                arr = np.array(fb)
                dsm_obj = build_layer_surface(
                    pts_opt,
                    (float(arr[:, 0].min()), float(arr[:, 1].min()),
                     float(arr[:, 2].max()), float(arr[:, 3].max())),
                    plane, gap, cell_m=layer_cell_m)
                if dsm_obj is None:
                    dsm_reject = {"reason": "layer_surface_failed"}
        else:
            logger.info("2층 표면 생략 — 층 간격을 알 수 없습니다 "
                        "(--layer-gap 으로 지정 가능).")
    if dsm_obj is None and use_dsm and pts_opt is not None \
            and len(pts_opt) >= 5000:
        from .homography.dsm import build_dsm
        fb = [f.footprint_bounds() for f in frames]
        fb = [x for x in fb if x is not None]
        if fb:
            arr = np.array(fb)
            dsm_obj = build_dsm(
                pts_opt,
                (float(arr[:, 0].min()), float(arr[:, 1].min()),
                 float(arr[:, 2].max()), float(arr[:, 3].max())),
                cell_m=dsm_cell_m,
                plane=plane,                  # 밴드 정제의 기준면
                max_relief_m=dsm_max_relief_m)
            # ★ 점군 Z 가 엉망이면 DSM 이 단일 평면보다 나쁘다. 실측에서
            #   기복이 24.3 m 로 나온 적이 있다 (지면~패널 상면은 2.4 m).
            #   그 DSM 을 쓰면 모자이크가 통째로 망가진다.
            if dsm_obj is None and dsm_reject is None:
                dsm_reject = {"reason": "build_failed_see_log"}
            if dsm_obj is not None:
                st = dsm_obj.stats()
                if st["relief_p99_m"] > dsm_max_relief_m:
                    logger.warning(
                        "DSM 기복이 %.1f m 로 비정상입니다 (상한 %.1f m). "
                        "점군 Z 품질이 나쁘다는 뜻이므로 DSM 을 쓰지 않고 "
                        "단일 평면으로 진행합니다. 초점거리 보정 로그와 "
                        "BA track 수를 확인하세요.",
                        st["relief_p99_m"], dsm_max_relief_m)
                    dsm_reject = {"reason": "relief_too_large",
                                  "relief_p99_m": st["relief_p99_m"],
                                  "limit_m": dsm_max_relief_m,
                                  "coverage": st["coverage"]}
                    dsm_obj = None
                elif dsm_obj.coverage < 0.35:
                    logger.warning(
                        "DSM 셀 충전율 %.1f%% 로 낮습니다 — --dsm-cell 을 "
                        "키우는 것을 권합니다.", dsm_obj.coverage * 100)
    elif use_dsm:
        n_pts_have = 0 if pts_opt is None else len(pts_opt)
        logger.info("DSM 생략 — BA 점군이 %d개로 부족합니다 (5000개 이상 필요). "
                    "단일 평면을 사용합니다.", n_pts_have)
        dsm_reject = {"reason": "too_few_points", "n_points": n_pts_have}

    # ---- 2층 높이맵 빌더 -------------------------------------------------
    # ★ 남은 어긋남은 하나의 식으로 설명된다: Δh × k
    #   (Δh = 기준면과 지면의 높이차, k = 그 자리 off-nadir).
    #   실측에서 안쪽 0.039 / 중간 0.107 / 바깥 0.398 m 를 Δh=1.23 m 로 나누면
    #   k = 0.032 / 0.087 / 0.323 이 나오고, 마지막 값은 연직 제한 상한
    #   0.304 와 사실상 같다. **세 영역이 모두 맞는다.**
    #   즉 BA·초점·왜곡이 아니라 '평면 하나로 두 층을 덮은 것' 이 남은
    #   오차의 거의 전부다.
    #
    #   BA 점군 DSM 은 점이 텍스처에 뭉쳐 충전율 26.6% 로 실패했지만,
    #   **높이는 이미 알고 있다** — 패널 상면은 기준면이고 지면은 그보다
    #   plane_above_ground_m 아래다. 필요한 건 "어느 픽셀이 패널인가" 뿐이고
    #   그건 영상에서 분할하면 된다 (실측 정사영상에서 Otsu 만으로 패널
    #   면적의 97% 가 5㎡ 초과 덩어리로 잡힘).
    _ground_drop_hint = 0.0
    if ref_ground_z is not None and plane is not None:
        _cxh = float(np.mean([f_.camera_xyz[0] for f_ in frames]))
        _cyh = float(np.mean([f_.camera_xyz[1] for f_ in frames]))
        _ground_drop_hint = float(plane.height_at(_cxh, _cyh)) - ref_ground_z

    _tl_builder = None
    if use_two_layer and dsm_obj is None:
        # ★ 점군이 아는 '상층(패널) 비율' 을 분할 보정의 목표로 넘긴다.
        #   점군은 높이를 직접 재므로 영상 밝기보다 신뢰할 수 있다.
        _upper_frac = None
        if pts_opt is not None and len(pts_opt) > 1000 and plane is not None:
            _rz = pts_opt[:, 2] - plane.height_at(pts_opt[:, 0], pts_opt[:, 1])
            _up = int(np.sum(np.abs(_rz) <= 0.35))
            _lo = int(np.sum(np.abs(_rz + max(_ground_drop_hint, 0.05)) <= 0.35))
            if _up + _lo > 500:
                _upper_frac = _up / float(_up + _lo)
        _ground_drop = 0.0
        if ref_ground_z is not None:
            _cx = float(np.mean([f_.camera_xyz[0] for f_ in frames]))
            _cy = float(np.mean([f_.camera_xyz[1] for f_ in frames]))
            _ground_drop = float(plane.height_at(_cx, _cy)) - ref_ground_z

        def _tl_builder(coarse_gray, coarse_ok, bnds, cell):
            if coarse_gray is None:
                return None
            from .homography.two_layer import build_two_layer_dsm
            return build_two_layer_dsm(
                coarse_gray, coarse_ok, bnds, cell, plane,
                ground_drop_m=_ground_drop,
                min_area_m2=two_layer_min_area_m2,
                target_fraction=_upper_frac)

    if gsd_m is None:
        gsd_m = recommend_gsd(frames)
        logger.info("GSD 자동 결정: %.4f m/px (프레임 GSD 중앙값)", gsd_m)
    # ---- 초점거리 최종 검증 (GSD 기반) --------------------------------
    if focal_info is not None:
        try:
            d_check = gsd_m * float(np.median([k.f_px for k in intrinsics_obj]))
            d_meta = focal_info.get("d_metadata_median_m")
            # ★ 기준면이 '패널 상면' 으로 올라가면 GSD×f_px 는 카메라→패널
            #   상면 거리인데, 메타데이터(LRF·RelativeAltitude)는 카메라→
            #   **지면** 거리다. 그대로 비교하면 패널 높이만큼(약 1 m)
            #   오차로 잡힌다 — 실측에서 -0.27% 가 -2.4% 로 '악화' 로
            #   보였는데, 그 1.10 m 차이는 상면 검출이 성공해 기준면이
            #   0.96 m 올라간 것이 전부였다. 오차가 아니라 기준 차이다.
            surf_off = 0.0
            if ref_ground_z is not None:
                cxp = float(np.mean([f_.camera_xyz[0] for f_ in frames]))
                cyp = float(np.mean([f_.camera_xyz[1] for f_ in frames]))
                surf_off = max(float(plane.height_at(cxp, cyp)) - ref_ground_z,
                               0.0)
            if d_meta:
                d_meta_eff = d_meta - surf_off
                resid = d_check / d_meta_eff - 1.0
                focal_info["final_check"] = {
                    "gsd_x_f_px_m": d_check, "d_metadata_m": d_meta,
                    "plane_above_ground_m": surf_off,
                    "d_metadata_to_plane_m": d_meta_eff,
                    "residual": resid}
                lvl = logger.warning if abs(resid) > 0.03 else logger.info
                lvl("초점거리 최종 검증: GSD×f_px = %.2f m vs 메타데이터 "
                    "%.2f m (기준면이 지면보다 %.2f m 위 → 보정 후 %.2f m) "
                    "→ 잔차 %+.1f%%%s", d_check, d_meta, surf_off, d_meta_eff,
                    resid * 100,
                    "  (3%% 초과 — --focal-rounds 를 늘려보세요)"
                    if abs(resid) > 0.03 else "")
        except Exception:
            pass

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
                                  glint_penalty=glint_penalty,
                                  tile_memory_mb=tile_memory_mb,
                                  prefetch_workers=prefetch_workers,
                                  max_offnadir_ratio=max_offnadir_ratio,
                                  offnadir_auto=offnadir_auto,
                                  offnadir_frac=offnadir_frac,
                                  offnadir_fallback=offnadir_fallback,
                                  offnadir_tolerance=offnadir_tolerance,
                                  offnadir_epsilon=offnadir_epsilon,
                                  dsm=dsm_obj,
                                  two_layer_builder=_tl_builder))
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

    # ---- 모자이크 품질 자체 진단 --------------------------------------
    # ★ 그동안 품질을 '중심으로부터 거리별 어긋남' 으로 봤는데 그 지표가
    #   오도했다. 바깥 밴드는 잔디·도로이고 극단적 off-nadir 로 채워진
    #   곳이라 검사 품질을 대표하지 못한다. 실제로 픽셀별 상한을 넣었을 때
    #   바깥은 0.427 → 0.563 m 로 나빠졌지만 **패널만 보면 0.087 → 0.078 m
    #   로 좋아졌다.** 검사 대상에서의 어긋남을 직접 재서 남긴다.
    if mosaic and stats.get("output_path"):
        try:
            from .homography.mosaic_qc import assess_mosaic
            qc = assess_mosaic(stats["output_path"], stats.get("gsd_m", gsd_m))
            if qc:
                stats["quality"] = qc
        except Exception as exc:
            logger.warning("품질 진단 실패 (계속 진행): %s", exc)

    summary = {
        "images": len(metas),
        "frames_georeferenced": len(frames),
        "epsg": target_epsg,
        "gsd_m": gsd_m,
        "ba_reprojection_rmse_px": ba_rmse,
        "observations_per_frame": obs_stats,
        "plane_source": plane_source,
        "plane_tilt_check": plane_tilt,
        "focal_calibration": focal_info,
        "distortion": distortion_info,
        "dsm_used": dsm_obj is not None,
        "surface_model": ("layer" if (dsm_obj is not None and layer_surface)
                          else ("dsm" if dsm_obj is not None else "plane")),
        "dsm_rejected": dsm_reject,
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
                          plane_tilt_tolerance_deg: float = 1.0,
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
        gp_a_orig, gp_b_orig = plane.a, plane.b
        plane_tilt_note = None
        # ★ 교차검증 — BA 점군이 나쁘면 평면이 엉뚱한 고도로 간다.
        #   실측에서 기준면이 LRF 지면보다 6.6 m 아래로 내려간 사례가 있었고,
        #   그 상태로는 어떤 모자이크 설정도 결과를 살리지 못한다.
        #   물리적으로 기준면은 지면 ± 몇 m 안에 있어야 한다.
        if ref_z is not None:
            cx_, cy_ = cams_opt[:, 0].mean(), cams_opt[:, 1].mean()
            gap = float(plane.height_at(cx_, cy_)) - ref_z
            # ★ 블록 기울기 교차검증 (실측 근거)
            #   BA 점군 평면이 기울기 2.508° 로 나왔는데 LRF 380점 평면은
            #   0.837° 였다. a 성분은 비슷한데 **b(남북) 부호가 뒤집히고
            #   4배**로 커졌다. 사이트 폭 154 m 기준 양끝 표고차가
            #   2.25 m vs 6.75 m — 가장자리 기복변위로 0.67 m 차이다.
            #
            #   비행선이 동서(yaw ±90°)라 **피치 편향이 가장 관측하기 어렵고**,
            #   그 편향은 정확히 남북 기울기로 나타난다. 영상만으로는 이
            #   자유도를 잡을 수 없다 (근평면 장면의 게이지 문제).
            #
            #   LRF 는 영상 기하와 **완전히 독립**한 레이저 실측이므로 이
            #   기울기를 판정할 수 있다. 크게 어긋나면 LRF 기울기를 채택하고
            #   높이는 BA 것을 유지한다.
            lrf_plane = None
            if metas is not None:
                try:
                    from .homography.plane import plane_from_lrf
                    lrf_plane = plane_from_lrf(metas, cams_opt[:, :3])
                except Exception:
                    lrf_plane = None
            if lrf_plane is not None:
                dtilt = abs(plane.slope_deg - lrf_plane.slope_deg)
                da = abs(plane.a - lrf_plane.a); db = abs(plane.b - lrf_plane.b)
                if dtilt > plane_tilt_tolerance_deg:
                    cxp = float(cams_opt[:, 0].mean())
                    cyp = float(cams_opt[:, 1].mean())
                    z_keep = float(plane.height_at(cxp, cyp))
                    from .homography.homography import GroundPlane as _GP
                    plane = _GP(
                        a=lrf_plane.a, b=lrf_plane.b,
                        c=z_keep - (lrf_plane.a * (cxp - plane.origin_xy[0])
                                    + lrf_plane.b * (cyp - plane.origin_xy[1])),
                        origin_xy=plane.origin_xy,
                        inlier_rmse_m=plane.inlier_rmse_m,
                        n_inliers=plane.n_inliers)
                    ba_slope = float(np.degrees(np.arctan(
                        np.hypot(gp_a_orig, gp_b_orig))))
                    logger.warning(
                        "블록 기울기 교차검증: BA %.3f° vs LRF %.3f° "
                        "(Δa=%.5f, Δb=%.5f) — 차이가 허용 %.1f° 를 넘어 "
                        "LRF 기울기를 채택합니다 (높이는 BA 유지). 동서 비행"
                        "에서 피치 편향은 영상만으로 관측되지 않습니다.",
                        ba_slope, lrf_plane.slope_deg, da, db,
                        plane_tilt_tolerance_deg)
                    plane_tilt_note = {
                        "ba_slope_deg": float(np.degrees(np.arctan(
                            np.hypot(gp_a_orig, gp_b_orig)))),
                        "lrf_slope_deg": lrf_plane.slope_deg,
                        "adopted": "lrf_tilt"}
                else:
                    plane_tilt_note = {
                        "ba_slope_deg": plane.slope_deg,
                        "lrf_slope_deg": lrf_plane.slope_deg,
                        "adopted": "ba"}

            if abs(gap) <= 5.0:
                # ★ 라벨 버그 수정: estimate_ground_plane 은 BA 점군 적합이
                #   실패하면 조용히 LRF 로 폴백한다. 실측 로그에서
                #   "평면 RANSAC inlier 비율 부족: 23.7% < 30% → LRF 폴백"
                #   이 찍혔는데도 summary 에는 plane_source="ba_points" 로
                #   기록됐다. inlier 수가 LRF 점 수(=프레임 수) 규모면
                #   LRF 폴백으로 판정한다.
                src = ("lrf_fallback"
                       if plane.n_inliers <= len(metas) else "ba_points")
                logger.info("기준면 출처: %s (inlier %d개, LRF 대비 %+.2f m)",
                            "LRF 폴백" if src == "lrf_fallback" else "BA 점군",
                            plane.n_inliers, gap)
                return plane, src, plane_tilt_note
            if abs(gap) > 5.0:
                logger.warning(
                    "BA 점군 평면(%.2f m)이 LRF/메타데이터 기준(%.2f m)과 "
                    "%.1f m 어긋납니다 — 점군 품질이 나쁘다는 신호입니다. "
                    "LRF 기준면으로 대체합니다. (BA 재투영 RMSE 와 track 수를 "
                    "확인하세요)",
                    plane.height_at(cx_, cy_), ref_z, gap)
        else:
            return plane, "ba_points", plane_tilt_note

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
        return GroundPlane.horizontal(z), "lrf_metadata", None

    return estimate_ground_plane(
        metas=metas, camera_xyz=cams_opt[:, :3],
        allow_tilt=allow_tilted_plane,
        panel_top_offset_m=panel_top_offset_m,
        surface=surface,
    ), "metadata", None


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

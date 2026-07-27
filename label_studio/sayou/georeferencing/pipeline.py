"""End-to-End GCP-Free Georeferencing 파이프라인.

워크플로우
----------
1. EXIF/XMP 메타데이터 추출 → 사진별 RTK 좌표 + 짐벌 자세
2. RTK Fixed 여부 검증 (RtkFlag=50 만 신뢰)
3. SfM Tie Point 추출 (SIFT/ORB + RANSAC) — GPU 가속
4. 두 시점 초기화 → 점진적 SfM 재구성
5. RTK 제약 Bundle Adjustment (카메라 위치를 GCP 처럼 soft constraint)
6. 정사영상 생성 — DSM 없이 평면 평균 고도 기반 간이 정사보정

GCP-free 한계
-------------
* 수평 정확도: 2~5 cm 가능 (RTK Fixed 기준)
* 수직 정확도: 5~15 cm (안테나 위상 중심 오프셋, GNSS 다중경로 영향)
* 절대 정확도가 중요한 측량/검측용은 최소 1~2 점의 Check Point 권장.
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

from sayou.image.metadata import extract_metadata

from .crs import CRSConverter
from .features import build_tie_points, find_neighbor_pairs
from .geometry import compute_focal_px
from .gpu_backend import gpu_summary
from .ortho import simple_orthophoto
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


def _build_initial_state(metas, crs: CRSConverter):
    """RTK + 짐벌 자세 → 초기 외부표정 + RTK prior 배열.

    ω = roll, φ = pitch + 90° (nadir 보정), κ = yaw  — 모두 radian.
    """
    rtk_priors = []
    initial_cameras = []
    for m in metas:
        X, Y = crs.forward(m.gps.lng, m.gps.lat)
        Z = m.gps.altitude
        rtk_priors.append([X, Y, Z])
        gimbal_yaw, gimbal_pitch, gimbal_roll = m.orientation
        omega = np.deg2rad(gimbal_roll)
        phi = np.deg2rad(gimbal_pitch + 90)  # nadir(-90) → 0 보정
        kappa = np.deg2rad(gimbal_yaw)
        initial_cameras.append([X, Y, Z, omega, phi, kappa])
    rtk_priors = np.array(rtk_priors)
    initial_cameras = np.array(initial_cameras)
    
    return rtk_priors, initial_cameras


def run_pipeline(image_dir: Path,
                          output_dir: Path,
                          target_epsg: int = 5186,
                          gsd_m: float = 0.05,
                          device: str = "mps",
                          k_neighbors: int = 8) -> None:
    """전체 파이프라인 실행.

    Parameters
    ----------
    image_dir : DJI JPG 디렉토리.
    output_dir : 정사영상 GeoTIFF 출력 디렉토리.
    target_epsg : 출력 좌표계 EPSG (한국 측량 표준 5186 권장).
    gsd_m : 출력 픽셀 크기 (Ground Sampling Distance, m/pixel).
        예: DJI Mavic 3E 100m 고도 → 0.05 가 적절.
    k_neighbors : KD-Tree 인접 페어 수.
    """
    logger.info("=" * 70)
    logger.info("Georeferencing Pipeline (GCP-Free) — 가속 백엔드: %s",
                gpu_summary())
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    crs = CRSConverter(target_epsg=target_epsg)

    # 1. 메타데이터 추출.
    images = sorted(image_dir.glob("*.JPG"))
    metas = [extract_metadata(p) for p in images]
    logger.info("이미지 %d장 로딩", len(metas))

    # 1-1. 메타데이터 품질 진단 — 정사영상 단계 크래시 사전 차단.
    no_focal = [Path(m.origin_path).name for m in metas
                if (not m.focal_length_in_35mm or m.focal_length_in_35mm <= 0)
                and (not m.focal_length or m.focal_length <= 0)]
    if no_focal:
        logger.warning(
            "focal_length 정보가 전혀 없는 사진 %d장: %s%s — "
            "이 사진들은 정사영상 단계에서 자동 건너뜀. "
            "(DJI H20T 의 _Z (thermal) 파일이 섞여있는지 확인)",
            len(no_focal), no_focal[:3],
            " ..." if len(no_focal) > 3 else "",
        )
    # 파일명 패턴으로 thermal 혼입 가능성 사전 경고.
    thermal_like = [Path(m.origin_path).name for m in metas
                    if Path(m.origin_path).stem.endswith(("_Z", "_T"))
                    or "_Z_" in Path(m.origin_path).name
                    or "_T_" in Path(m.origin_path).name]
    if thermal_like:
        logger.warning(
            "Thermal/Telephoto 패턴 (_Z, _T) 파일명 %d장 감지: %s%s — "
            "RGB 만 처리하려면 디렉토리를 분리하거나 글로브 패턴을 변경.",
            len(thermal_like), thermal_like[:3],
            " ..." if len(thermal_like) > 3 else "",
        )

    # 2. RTK 품질 검증.
    if not validate_rtk_quality(metas):
        logger.warning("RTK 품질 부족. 결과 정확도 보장 어려움.")

    # 3. RTK → 투영좌표 + 초기 외부표정 + prior 가중치.
    rtk_priors, initial_cameras = _build_initial_state(metas, crs)
    rtk_weights = compute_rtk_prior_weights(metas)

    # 4. 인접 페어 SfM 매칭.
    t0 = time.perf_counter()
    pairs = find_neighbor_pairs(metas, crs, k_neighbors=k_neighbors)
    matches, features = build_tie_points(metas, pairs)
    logger.info("[stage] SfM 매칭: %s", fmt_elapsed(time.perf_counter() - t0))

    # 5. Tie point tracks → 3D 점 (Triangulation 초기화) → RTK 제약 BA
    t0 = time.perf_counter()
    intrinsics = [
        (compute_focal_px(m), m.width / 2.0, m.height / 2.0)
        for m in metas
    ]
    logger.info("[stage] intrinsics: %s", fmt_elapsed(time.perf_counter() - t0))

    # 5a. 페어 매칭들을 track 으로 연결
    t0 = time.perf_counter()
    tracks = build_tracks(matches, features,
                          min_track_len=2, max_track_len=30)
    logger.info("[stage] track 빌드: %s", fmt_elapsed(time.perf_counter() - t0))

    if len(tracks) < 10:
        logger.warning("track 이 너무 적음(%d). BA 생략, RTK 초기값 사용.",
                       len(tracks))
        cams_opt = initial_cameras
    else:
        t0 = time.perf_counter()
        # 5b. track 들을 DLT 삼각측량 → observations + initial_points
        observations, initial_points, _ = triangulate_tracks(
            tracks, initial_cameras, intrinsics,
            max_reproj_err_px=3.0,
            min_triangulation_angle_deg=2.0,
        )
        logger.info("[stage] triangulation: %s", fmt_elapsed(time.perf_counter() - t0))

        if len(initial_points) < 10 or len(observations) < 30:
            logger.warning("삼각측량 결과 부족. BA 생략, RTK 초기값 사용.")
            cams_opt = initial_cameras
        else:
            # 5c. RTK 제약 Bundle Adjustment
            # 카메라마다 f_px 가 다를 수 있으나, rtk_constrained_bundle_adjustment
            # 는 단일 (f_px, cx, cy) 를 받으므로 대표값(중앙값) 사용.
            # (전 사진 동일 카메라/줌이면 정확히 일치)
            f_px_rep = float(np.median([k[0] for k in intrinsics]))
            cx_rep = float(np.median([k[1] for k in intrinsics]))
            cy_rep = float(np.median([k[2] for k in intrinsics]))
            t0 = time.perf_counter()
            cams_opt, pts_opt, rmse = rtk_constrained_bundle_adjustment(
                initial_cameras, initial_points, observations,
                rtk_priors, rtk_weights,
                f_px=f_px_rep, cx=cx_rep, cy=cy_rep,
            )
            logger.info("[stage] bundle adjustment: %s", 
                fmt_elapsed(time.perf_counter() - t0))
            logger.info(
                "BA 후 카메라 위치 평균 이동량: %.3f m",
                float(np.linalg.norm(
                    cams_opt[:, :3] - initial_cameras[:, :3], axis=1
                ).mean()),
            )

    # 6. 정사영상 생성 (사진별).
    # ★ 기존: 모든 사진에 동일한 ground_z_avg 사용 → RelativeAltitude 가
    #   이륙지점 기준이라 부정확. H20T 는 LRF 실측이 있으므로 그게 정확.
    #   → estimate_ground_z() 로 사진마다 개별 추정 (LRF 우선).
    lrf_count = sum(m.has_valid_lrf for m in metas)
    logger.info("LRF 실측 가능 사진: %d/%d", lrf_count, len(metas))

    t0 = time.perf_counter()
    n_success = 0
    n_skipped = 0
    for i, meta in enumerate(metas):
        cam = cams_opt[i]
        ground_z = estimate_ground_z(meta)
        out = output_dir / f"{Path(meta.origin_path).stem}_ortho.tif"
        ok = simple_orthophoto(
            meta=meta,
            camera_xyz=cam[:3],
            omega=cam[3], phi=cam[4], kappa=cam[5],
            ground_z=ground_z,
            gsd_m=gsd_m,
            output_path=out,
            epsg=target_epsg,
        )
        if ok:
            n_success += 1
        else:
            n_skipped += 1
    logger.info("[stage] 정사영상 완료: 성공 %d, 건너뜀 %d: %.1fs",
                n_success, n_skipped, time.perf_counter() - t0)


def main():
    """CLI 진입점. 데모용 기본 경로 사용."""
    start = time.perf_counter()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_pipeline_gcp_free(
        image_dir=Path("./data/solar/images/RGB"),
        output_dir=Path("./workspace/output"),
        target_epsg=5186,
        gsd_m=0.05,
        k_neighbors=8,
    )
    elapsed = time.perf_counter() - start
    print(f"Elapsed: {timedelta(seconds=int(elapsed))}")


if __name__ == "__main__":
    main()


__all__ = ["run_pipeline", "main"]
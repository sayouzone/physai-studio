"""
Homography 파이프라인 CLI 진입점 (RTK 기반 호모그래피).

Usage
-----
::

    # 전체 (SfM + RTK 제약 BA + 정사 모자이크)
    python scripts/run_homography_pipeline.py \
        --image-dir ./data/solar/images/RGB \
        --output-dir ./workspace/output

    # 빠른 현장 확인 (특징점 매칭/BA 생략, RTK 자세만)
    python scripts/run_homography_pipeline.py --skip-sfm ...

    # 프레임별 GeoTIFF (모자이크 대신)
    python scripts/run_homography_pipeline.py --no-mosaic ...

환경변수::

    GEOREF_DISABLE_GPU=1            모든 GPU 경로 비활성화 (벤치마크/디버깅)
    GEOREF_DISABLE_MULTIPROCESS=1   ProcessPool 비활성, 단일 프로세스 강제
    GEOREF_NUM_WORKERS=N            CPU SIFT 워커 수

``--gsd`` 기본값이 ``None`` (자동) 으로 바뀌었다. 기존 0.05 고정값은 H20T 를
45 m 고도로 날릴 때의 실제 GSD (약 1.2 cm) 대비 4 배 다운샘플이다. 자동은
프레임 GSD 중앙값을 써서 원해상도를 보존한다.

본 스크립트는 ``solar_thermal.georeferencing`` 패키지의 얇은 wrapper.
실제 로직은 패키지 내부에 모듈화돼 있으므로 라이브러리로 import 해서
다른 워크플로우에도 재사용 가능.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "label_studio"))

from sayou.homography import run_homography_pipeline


def parse_args() -> argparse.Namespace:
    def valid_filepath(filepath):
        path = os.path.abspath(os.path.expanduser(filepath))
        if os.path.exists(path):
            return path
        raise FileNotFoundError(filepath)

    p = argparse.ArgumentParser(
        description="Homography 파이프라인 (RTK 기반 호모그래피)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image-dir", type=valid_filepath,
                   default="~/Development/sayouzone/solar-thermal/data/solar/그린환경센터/RGB",
                   help="DJI JPG 디렉토리")
    p.add_argument("--output-dir", type=valid_filepath,
                   default="~/Development/sayouzone/solar-thermal/workspace/output",
                   help="GeoTIFF 출력 디렉토리")
    p.add_argument("--glob", dest="glob_pattern", default="*.JPG",
                   help="이미지 글로브 패턴. H20T 는 _T (thermal) 파일이 "
                        "섞이므로 RGB 만 처리하려면 조정할 것")
    p.add_argument("--epsg", type=int, default=5186,
                   help="출력 좌표계 EPSG (기본: Korea 2000 / Central Belt)")
    p.add_argument("--gsd", type=float, default=None,
                   help="출력 픽셀 크기 (m/pixel). 미지정 시 프레임 GSD "
                        "중앙값을 자동 사용")
    p.add_argument("--k-neighbors", type=int, default=8,
                   help="이미지당 매칭 이웃 수 상한")
    p.add_argument("--geoid", dest="geoid_undulation_m", type=float, default=0.0,
                   help="지오이드고 (m). 타원체고 → 정표고 변환용. 한국 내륙 대략 22~30. "
                        "평면좌표만 쓸 거면 0 이어도 무해")
    p.add_argument("--skip-sfm", action="store_true",
                   help="특징점 매칭/BA 생략, RTK 자세만으로 정사보정")
    p.add_argument("--no-mosaic", dest="mosaic", action="store_false",
                   help="단일 모자이크 대신 프레임별 GeoTIFF 출력")
    p.add_argument("--flat-plane", dest="allow_tilted_plane",
                   action="store_false",
                   help="지상평면 경사를 0 으로 강제 (기본은 경사 허용)")
    p.add_argument("--panel-top", dest="panel_top_offset_m", type=float,
                   default=0.0,
                   help="지면 대비 패널 상면 높이 (m). LRF/메타데이터 폴백 "
                        "경로에서 기준면을 패널 상면으로 올린다. 고정식 "
                        "가대는 보통 1.5~2.0. 기준면이 지면이면 모자이크 "
                        "시임에서 패널이 끊겨 보인다")
    p.add_argument("--plane-surface", dest="plane_surface",
                   default="upper", choices=["upper", "dominant"],
                   help="'upper' 는 점군 두 층 중 위층(패널 상면) 을 기준면 "
                        "으로. 지면 기준이 필요하면 'dominant'")
    p.add_argument("--attitude-deg", dest="ba_stage1_attitude_deg", type=float,
                   default=1.5,
                   help="초기 짐벌 자세 불확실성 (deg). 1차 삼각측량 게이트가 "
                        "여기서 나온다. 너무 작으면 삼각측량이 점을 전부 버려 "
                        "BA 가 시작조차 못 한다 (실측 실패 원인). 1.0~1.5 권장")
    p.add_argument("--coarse-reproj", dest="coarse_reproj_px", type=float,
                   default=0.0,
                   help="1차 삼각측량 게이트 (px). 0 이면 --attitude-deg 로 자동")
    p.add_argument("--fixed-fine-gate", dest="fine_gate_auto",
                   action="store_false",
                   help="정밀 게이트를 --fine-reproj 값으로 고정. 기본은 측정된 "
                        "잔차 프로파일(바닥+반경성분)에서 자동 결정 — 상수 3px "
                        "게이트가 반경 1814px 바깥을 통째로 잘라 점의 절반을 "
                        "잃고 있었다")
    p.add_argument("--fine-gate-max", dest="fine_gate_max_px", type=float,
                   default=8.0, help="자동 결정된 정밀 게이트의 상한 (px)")
    p.add_argument("--no-auto-distortion", dest="auto_distortion",
                   action="store_false",
                   help="진단이 왜곡을 확인해도 자동 보정하지 않는다. 기본은 "
                        "자동 — 실측에서 잔차가 '바닥 1.72px + 왜곡 3.87px' 로 "
                        "확인됐고(R²=0.985), 3px 게이트가 반경 1814px 바깥을 "
                        "통째로 버리고 있었다")
    p.add_argument("--distortion-k2", dest="distortion_use_k2",
                   action="store_true",
                   help="왜곡을 k1,k2 두 항으로 푼다. 기본은 k1 단독 — 실측에서 "
                        "두 항을 함께 풀자 k1=-0.0269, k2=+0.3039 로 서로 "
                        "상쇄하다 모서리에서 +44px 폭주하는 병적인 해가 나왔다")
    p.add_argument("--estimate-distortion", dest="estimate_distortion",
                   action="store_true",
                   help="재투영 잔차에서 방사 왜곡(k1,k2)을 추정해 보정한다. "
                        "이 데이터는 DewarpData 가 없어 왜곡이 미보정 상태다. "
                        "로그의 '잔차의 반경 의존성' 진단에서 3승 패턴이 "
                        "확인되면 켤 것. 기본은 꺼짐(미검증)")
    p.add_argument("--mid-reproj-factor", dest="mid_reproj_factor", type=float,
                   default=3.0,
                   help="중간 게이트 = 현재 BA RMSE × 이 배수. 느슨(700px)에서 "
                        "정밀(3px)로 바로 가면 정상 점까지 잘린다 (실측: 점 34%%, "
                        "관측 59%% 손실). 0 이면 중간 단계 없음")
    p.add_argument("--fine-reproj", dest="fine_reproj_px", type=float,
                   default=3.0, help="2차(정밀) 삼각측량 게이트 (px)")
    p.add_argument("--min-tri-angle", dest="fine_tri_angle_deg", type=float,
                   default=5.0,
                   help="정밀 삼각측량의 시선각 하한 (deg). 반복 텍스처에서 "
                        "짧은 베이스라인은 재투영이 완벽하면서 깊이만 크게 "
                        "틀린 점을 만든다. 2°→6.5m, 5°→2.6m, 8°→1.6m 오차")
    p.add_argument("--focal-rounds", dest="focal_max_rounds", type=int,
                   default=4, help="초점거리 자기보정 최대 반복 횟수")
    p.add_argument("--no-offnadir-auto", dest="offnadir_auto",
                   action="store_false",
                   help="연직 제한 자동 조정 끄기 (--max-offnadir 값 고정)")
    p.add_argument("--no-offnadir-fallback", dest="offnadir_fallback",
                   action="store_false",
                   help="연직 제한 예외를 쓰지 않는다. 가장자리에 구멍이 생기는 "
                        "대신 남는 영역의 품질이 균일해진다. 실측: 예외 영역"
                        "(11.3%%)의 어긋남이 0.475 m 로 안쪽 0.039 m 의 12배")
    p.add_argument("--offnadir-frac", dest="offnadir_frac", type=float,
                   default=0.65,
                   help="연직 제한 자동 조정 비율 (프레임 최대 k 대비). "
                        "0.65 면 H20T 기준 k≈0.37")
    p.add_argument("--free-positions", dest="fix_positions",
                   action="store_false",
                   help="BA 에서 카메라 위치를 자유변수로 둔다. 기본은 RTK 에 "
                        "고정 — 실측에서 위치가 166σ(1.8 m) 움직이고 자세가 "
                        "10.7° 돌아가는 발산이 있었다")
    p.add_argument("--plane-tilt-tol", dest="plane_tilt_tolerance_deg",
                   type=float, default=1.0,
                   help="BA 평면 기울기와 LRF 기울기의 허용 차이 (deg). 초과 시 "
                        "LRF 기울기를 채택. 동서 비행에서 피치 편향은 영상만"
                        "으로 관측되지 않으므로 독립 관측이 필요하다")
    p.add_argument("--focal-max-total", dest="focal_max_total", type=float,
                   default=0.25,
                   help="EXIF 대비 초점거리 누적 보정 상한 (0.25 = ±25%%)")
    p.add_argument("--attitude-sigma", dest="attitude_sigma_deg", type=float,
                   default=1.0,
                   help="자세 prior 표준편차 (deg). 짐벌 절대 정확도 수준")
    p.add_argument("--ba-max-nfev", dest="ba_max_nfev", type=int, default=200,
                   help="BA 단계별 최대 함수평가 횟수")
    p.add_argument("--tri-z-band", dest="tri_z_band_m", type=float, default=25.0,
                   help="삼각측량 점의 허용 Z 밴드 (지면 기준 ±m). 느슨한 "
                        "게이트가 통과시킨 물리적으로 불가능한 점을 제거")
    p.add_argument("--no-auto-focal", dest="auto_focal", action="store_false",
                   help="초점거리 자기보정 비활성화. 기본은 켜짐 — EXIF 의 "
                        "FocalLengthIn35mmFilm 이 실제와 다르면 BA 가 그 오차를 "
                        "점군 깊이로 흡수해 RMSE 는 좋은데 3D 가 틀린다")
    p.add_argument("--dsm-max-relief", dest="dsm_max_relief_m", type=float,
                   default=6.0,
                   help="DSM 기복 상한 (m). 초과하면 점군 품질 이상으로 보고 "
                        "단일 평면으로 대체")
    p.add_argument("--no-dsm", dest="use_dsm", action="store_false",
                   help="DSM 사용 안 함(단일 평면). 기본은 BA 점군이 5000개 "
                        "이상이면 DSM 사용 — 지면과 패널 상면이 2.4 m 떨어진 "
                        "2층 구조는 평면 하나로 둘 다 맞출 수 없다")
    p.add_argument("--dsm-cell", dest="dsm_cell_m", type=float, default=0.8,
                   help="DSM 격자 크기 (m). 패널 행 폭 1.3 m 를 분리하려면 "
                        "0.4~0.8 권장. 작을수록 점군 밀도가 더 필요하다")
    p.add_argument("--plane-tolerance", dest="plane_lrf_tolerance_m",
                   type=float, default=3.0,
                   help="기준면이 LRF 기준에서 벗어나도 되는 허용치 (m). "
                        "기준면은 지면과 구조물 상단 사이에만 있을 수 있다")
    p.add_argument("--max-offnadir", dest="max_offnadir_ratio", type=float,
                   default=0.35,
                   help="프레임에서 쓸 영역의 off-nadir 상한 r/h. H20T 최외곽은 "
                        "0.479(25.6°)이고 거기서 기복변위가 최대다. 0 이면 제한 "
                        "없음. 중복이 낮은 현장은 0.40~0.45 로 완화")
    p.add_argument("--rtk-boost", dest="rtk_boost_max", type=float, default=8.0,
                   help="관측이 적은(가장자리) 프레임의 RTK prior 가중 강화 상한")
    p.add_argument("--tile-memory", dest="tile_memory_mb", type=float,
                   default=256.0,
                   help="모자이크 타일 하나의 메모리 예산 (MB). 출력이 아무리 "
                        "커도 메모리는 이 값으로 묶인다")
    p.add_argument("--no-seam-optimize", dest="seam_optimize",
                   action="store_false",
                   help="시임 최적화 비활성화. 기본은 켜짐 — 시임이 패널을 "
                        "가로지르지 않고 잔디/그림자 쪽으로 흐르게 한다")
    p.add_argument("--seam-cost", dest="seam_cost_weight", type=float,
                   default=1.0, help="시임 최적화 강도 (0.5~2.0)")
    p.add_argument("--no-exposure-comp", dest="exposure_compensate",
                   action="store_false",
                   help="프레임 간 노출 보정 비활성화")
    p.add_argument("--glint-penalty", type=float, default=0.7,
                   help="정반사 포화 픽셀 점수 감쇠 (0~1). 0 이면 미사용")
    p.add_argument("--no-auto-plane", dest="auto_plane", action="store_false",
                   help="기준면 높이 자동보정 비활성화")
    p.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    start = time.perf_counter()
    summary = run_homography_pipeline(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        target_epsg=args.epsg,
        gsd_m=args.gsd,
        device=args.device,
        k_neighbors=args.k_neighbors,
        geoid_undulation_m=args.geoid_undulation_m,
        skip_sfm=args.skip_sfm,
        mosaic=args.mosaic,
        allow_tilted_plane=args.allow_tilted_plane,
        panel_top_offset_m=args.panel_top_offset_m,
        plane_surface=args.plane_surface,
        auto_plane=args.auto_plane,
        coarse_reproj_px=args.coarse_reproj_px,
        fine_reproj_px=args.fine_reproj_px,
        mid_reproj_factor=args.mid_reproj_factor,
        estimate_distortion=args.estimate_distortion,
        auto_distortion=args.auto_distortion,
        fine_gate_auto=args.fine_gate_auto,
        fine_gate_max_px=args.fine_gate_max_px,
        ba_stage1_attitude_deg=args.ba_stage1_attitude_deg,
        seam_optimize=args.seam_optimize,
        seam_cost_weight=args.seam_cost_weight,
        exposure_compensate=args.exposure_compensate,
        glint_penalty=args.glint_penalty,
        tile_memory_mb=args.tile_memory_mb,
        max_offnadir_ratio=args.max_offnadir_ratio,
        rtk_boost_max=args.rtk_boost_max,
        plane_lrf_tolerance_m=args.plane_lrf_tolerance_m,
        auto_focal=args.auto_focal,
        fix_positions=args.fix_positions,
        attitude_sigma_deg=args.attitude_sigma_deg,
        plane_tilt_tolerance_deg=args.plane_tilt_tolerance_deg,
        focal_max_total=args.focal_max_total,
        ba_max_nfev=args.ba_max_nfev,
        tri_z_band_m=args.tri_z_band_m,
        focal_max_rounds=args.focal_max_rounds,
        fine_tri_angle_deg=args.fine_tri_angle_deg,
        offnadir_auto=args.offnadir_auto,
        offnadir_frac=args.offnadir_frac,
        offnadir_fallback=args.offnadir_fallback,
        dsm_max_relief_m=args.dsm_max_relief_m,
        use_dsm=args.use_dsm,
        dsm_cell_m=args.dsm_cell_m,
        glob_pattern=args.glob_pattern,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"Elapsed: {timedelta(seconds=int(time.perf_counter() - start))}")


if __name__ == "__main__":
    main()

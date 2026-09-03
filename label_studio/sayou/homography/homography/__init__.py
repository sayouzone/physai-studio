"""RTK 기반 호모그래피 — DSM 없는 고속/간이 정사영상 워크플로우.

배치 위치
---------
``solar_thermal/georeferencing/homography/`` 를 가정한다 (``..geometry``, ``..crs``,
``..sfm`` 상대 import 사용). 다른 곳에 두려면 상대 import 깊이만 맞추면 된다.

핵심 아이디어
-------------
지표면을 평면으로 가정하는 순간, 지상평면 ↔ 이미지 대응은 3×3 호모그래피
하나로 **정확히** 닫힌다. 그러면 DSM 도, 출력 픽셀당 광선 교차 계산도 필요
없다. RTK 가 cm 급 절대위치를 주므로 GCP 도 필요 없다.

    H_g2i = K_neg · [ (r1 + a·r3) | (r2 + b·r3) | (c·r3 − R·C) ]

빠른 시작
---------
::

    from solar_thermal.georeferencing import run_pipeline

    run_pipeline(Path("./data/RGB"), Path("./out"), target_epsg=5186)

라이브러리로 직접 쓰려면::

    from solar_thermal.georeferencing.homography import FrameSet, FrameSetConfig

    fs = FrameSet(FrameSetConfig(target_epsg=5186))
    stats = fs.run_direct(metas, "ortho.tif")     # SfM 없이 RTK 자세만으로

BA 를 포함한 전체 흐름은 ``georeferencing.pipeline.run_pipeline`` 이 이미
엮어 둔다. 단계별로 직접 제어하려면 ``frames`` 모듈 docstring 참조.

부호 규약 (반드시 읽을 것)
--------------------------
이 패키지는 ``geometry.project_point`` 규약 (``px = cx − f·…``, 가시 조건
``R[2]·diff > 0``) 을 따른다. 기존 ``ortho.orthophoto`` 는 이와 광축 주변
180° 만큼 다른 규약을 쓰므로, BA 산출 자세를 그 함수에 넣으면 정사영상이
뒤집힌다. 자세한 내용은 ``homography`` 모듈 docstring 과 ``selftest`` 참조.

검증::

    python -m solar_thermal.georeferencing.homography.selftest
"""

from .homography import (
    FrameHomography,
    GroundPlane,
    PinholeIntrinsics,
    build_frame_homography,
    estimate_gsd,
    ground_bounds_from_homography,
    intrinsics_from_metadata,
    ortho_grid_affine,
    ortho_grid_homography,
    recommend_gsd,
)
from .ortho import MosaicConfig, mosaic_frames, orthorectify_frame, warp_frame
from .pairing import footprint_radius_m, select_gps_neighbor_pairs
from .frames import FrameContext, FrameSet, FrameSetConfig
from .plane import estimate_ground_plane, fit_plane_ransac
from .pose import (
    angle_from_nadir_deg,
    camera_axes_enu,
    decompose_to_opk,
    opk_from_gimbal,
    rotation_from_gimbal,
    rotation_from_metadata,
)
from .quality import (
    RTK_FIXED,
    RTKQuality,
    RTKQualityConfig,
    assess_rtk_quality,
    build_rtk_priors_and_weights,
)

__all__ = [
    # 호모그래피 코어
    "GroundPlane", "PinholeIntrinsics", "FrameHomography",
    "build_frame_homography", "intrinsics_from_metadata",
    "ortho_grid_affine", "ortho_grid_homography",
    "ground_bounds_from_homography", "estimate_gsd", "recommend_gsd",
    # 자세
    "camera_axes_enu", "rotation_from_gimbal", "rotation_from_metadata",
    "decompose_to_opk", "opk_from_gimbal", "angle_from_nadir_deg",
    # 품질
    "RTK_FIXED", "RTKQualityConfig", "RTKQuality",
    "assess_rtk_quality", "build_rtk_priors_and_weights",
    # 인접쌍
    "footprint_radius_m", "select_gps_neighbor_pairs",
    # 평면
    "fit_plane_ransac", "estimate_ground_plane",
    # 정사영상
    "MosaicConfig", "warp_frame", "orthorectify_frame", "mosaic_frames",
    # 파이프라인
    "FrameSetConfig", "FrameContext", "FrameSet",
]

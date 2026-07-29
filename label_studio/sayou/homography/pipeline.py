"""End-to-End GCP-Free 자동화 워크플로우 (파이프라인 8단계 전체).

    1. 메타데이터 추출        → image.metadata / dji_metadata_extractor
    2. 좌표계 변환            → crs.CRSConverter (WGS84 → EPSG:5186)
    3. RTK 품질 검증          → quality.assess_rtk_quality
    4. 인접쌍 선택            → pairing.select_gps_neighbor_pairs
    5. 자세 → R, (ω,φ,κ)      → pose.rotation_from_metadata
    5a/5b. track / 삼각측량   → sfm.tracks / sfm.triangulation
    6. RTK 제약 BA            → sfm.bundle_adjustment
    7. 평면 추정 + 호모그래피 → plane / homography
    8. 정사영상 합성          → ortho.mosaic_frames

4~6 단계 (SfM) 는 **선택사항**이다. 특징점 매칭 없이 RTK 자세만으로도 7~8 이
동작한다 (``run_direct``). 그 경우 정확도는 RTK 위치 정확도와 짐벌 각 정확도에
직접 묶인다 — 실측상 프레임 간 잔차 1~2 m 수준이고, BA 를 거치면 0.1~0.3 m 로
떨어진다. 빠른 현장 확인은 ``run_direct``, 납품물은 BA 경유를 권장한다.

타원체고 → 정표고
-----------------
DJI 의 ``AbsoluteAltitude`` 는 타원체고(WGS84 기준)다. 한국 표고(정표고)로
바꾸려면 지오이드고를 빼야 한다 (KNGeoid: 한국 내륙에서 대략 22~30 m).
정사영상의 **평면 좌표**만 쓸 거라면 이 오프셋은 전 프레임에 공통이라
상쇄되어 무해하다. 하지만 지상 표고를 다른 측량 성과와 비교하거나
DEM 과 겹칠 거라면 반드시 보정해야 한다. ``geoid_undulation_m`` 로 상수
오프셋을 주거나, pyproj 의 지오이드 그리드를 쓸 것.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .homography import (
    FrameHomography, GroundPlane, PinholeIntrinsics,
    build_frame_homography, intrinsics_from_metadata, recommend_gsd,
)
from .ortho import MosaicConfig, mosaic_frames
from .pairing import footprint_radius_m, select_gps_neighbor_pairs
from .plane import estimate_ground_plane
from .pose import decompose_to_opk, rotation_from_metadata
from .quality import (
    RTKQuality, RTKQualityConfig, assess_rtk_quality,
    build_rtk_priors_and_weights,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineConfig", "FrameContext", "RTKHomographyPipeline",
]


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    target_epsg: int = 5186
    geoid_undulation_m: float = 0.0
    rtk: RTKQualityConfig = field(default_factory=RTKQualityConfig)
    mosaic: MosaicConfig = field(default_factory=MosaicConfig)

    # 평면 추정
    allow_tilted_plane: bool = True
    plane_inlier_threshold_m: float = 0.5
    panel_top_offset_m: float = 0.0     # 폴백 경로에서만 사용 (지면 대비 높이)

    # 인접쌍
    pair_margin: float = 1.15
    max_pairs_per_image: int | None = 12

    # 출력
    gsd_m: float | None = None          # None 이면 프레임 GSD 중앙값
    output_epsg: int | None = None      # None 이면 target_epsg


# ---------------------------------------------------------------------------
# 프레임 컨텍스트
# ---------------------------------------------------------------------------
@dataclass
class FrameContext:
    """한 프레임에 대해 파이프라인이 축적하는 모든 상태."""

    index: int
    meta: object
    image_path: str
    intr: PinholeIntrinsics
    camera_xyz: np.ndarray               # 투영좌표계 (X, Y, Z)
    R: np.ndarray                        # world → camera
    quality: RTKQuality | None = None
    homography: FrameHomography | None = None

    @property
    def opk(self) -> tuple[float, float, float]:
        return decompose_to_opk(self.R)

    def ba_row(self) -> np.ndarray:
        """BA ``initial_cameras`` 한 행 ``[X, Y, Z, ω, φ, κ]``."""
        om, ph, ka = self.opk
        return np.array([*self.camera_xyz, om, ph, ka])


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------
class RTKHomographyPipeline:
    """RTK 기반 호모그래피 정사보정 파이프라인."""

    def __init__(self, config: PipelineConfig | None = None):
        self.cfg = config or PipelineConfig()
        from ..crs import CRSConverter
        self.crs = CRSConverter(self.cfg.target_epsg)

    # -- 1~3, 5 단계 -------------------------------------------------------
    def prepare(self, metas: list, image_paths: list[str] | None = None
                ) -> tuple[list[FrameContext], list[FrameContext]]:
        """메타데이터 → 프레임 컨텍스트. ``(accepted, rejected)`` 반환.

        여기서 좌표계 변환, 내부 파라미터 결정, 자세행렬 계산, RTK 품질 게이트
        가 모두 끝난다. 이후 단계는 ``FrameContext`` 만 본다.
        """
        contexts: list[FrameContext] = []
        rotations: list[np.ndarray | None] = []
        kept_metas: list = []

        for i, meta in enumerate(metas):
            intr = intrinsics_from_metadata(meta)
            if intr is None:
                logger.warning("초점거리 정보 없음 — 제외: %s",
                               getattr(meta, "origin_path", i))
                continue

            gps = getattr(meta, "gps", None)
            if gps is None or not np.isfinite(gps.lat) or not np.isfinite(gps.lng):
                logger.warning("GPS 없음 — 제외: %s", getattr(meta, "origin_path", i))
                continue

            x, y = self.crs.forward(float(gps.lng), float(gps.lat))
            z = float(gps.altitude) - self.cfg.geoid_undulation_m
            R = rotation_from_metadata(meta)

            path = (image_paths[i] if image_paths is not None
                    else getattr(meta, "origin_path", ""))
            contexts.append(FrameContext(
                index=len(contexts), meta=meta, image_path=str(path),
                intr=intr, camera_xyz=np.array([x, y, z]), R=R,
            ))
            rotations.append(R)
            kept_metas.append(meta)

        if not contexts:
            raise ValueError("사용 가능한 프레임이 없음")

        qualities = assess_rtk_quality(kept_metas, rotations, self.cfg.rtk)
        for ctx, q in zip(contexts, qualities):
            ctx.quality = q

        accepted = [c for c in contexts if c.quality.accepted]
        rejected = [c for c in contexts if not c.quality.accepted]

        # 인덱스 재부여 — BA 는 연속 인덱스를 가정한다.
        for new_i, ctx in enumerate(accepted):
            ctx.index = new_i
        return accepted, rejected

    # -- 4 단계 -----------------------------------------------------------
    def select_pairs(self, contexts: list[FrameContext]) -> list[tuple[int, int]]:
        """RTK 위치 기반 매칭 후보쌍."""
        C = np.vstack([c.camera_xyz for c in contexts])
        # 지상고도: 절대고도 − 이륙지 지면고도 (없으면 상대고도).
        radii = []
        for c in contexts:
            agl = getattr(c.meta, "relative_height", None)
            if agl is None or not np.isfinite(agl) or agl <= 0:
                agl = 100.0     # 보수적 기본값 — 반경이 커져 후보만 늘 뿐 안전.
            radii.append(footprint_radius_m(
                float(agl), c.intr.f_px, c.intr.width, c.intr.height))
        return select_gps_neighbor_pairs(
            C, np.array(radii),
            margin=self.cfg.pair_margin,
            max_pairs_per_image=self.cfg.max_pairs_per_image,
        )

    # -- 6 단계 -----------------------------------------------------------
    def bundle_adjust(self,
                      contexts: list[FrameContext],
                      tracks: list,
                      initial_points: np.ndarray) -> np.ndarray:
        """RTK 제약 BA 실행. 갱신된 3D 점군을 반환하고 컨텍스트를 in-place 수정.

        ``tracks`` 와 ``initial_points`` 는 ``sfm.tracks.build_tracks`` /
        ``sfm.triangulation`` 산출물. 이 파이프라인은 특징점 추출·매칭을
        직접 하지 않으므로, 그 부분은 기존 SfM 모듈을 그대로 쓴다.
        """
        from ..sfm.bundle_adjustment import rtk_constrained_bundle_adjustment
        from ..geometry import rotation_matrix

        cams0 = np.vstack([c.ba_row() for c in contexts])
        priors, weights = build_rtk_priors_and_weights(
            cams0[:, :3], [c.quality for c in contexts]
        )

        # BA 는 단일 (f_px, cx, cy) 를 받는다. 프레임마다 다르면 중앙값을 쓰되
        # 산포가 크면 카메라별로 나눠 돌려야 한다.
        f_all = np.array([c.intr.f_px for c in contexts])
        if f_all.std() / max(f_all.mean(), 1e-9) > 0.01:
            logger.warning("프레임 간 초점거리 산포 %.2f%% — 줌 설정이 섞여 있다. "
                           "카메라별로 BA 를 분리하거나 내부 파라미터를 "
                           "미지수로 두어야 한다.",
                           100 * f_all.std() / f_all.mean())
        f_px = float(np.median(f_all))
        cx = float(np.median([c.intr.cx for c in contexts]))
        cy = float(np.median([c.intr.cy for c in contexts]))

        observations = [(img_i, t_idx, np.array([px, py]))
                        for t_idx, track in enumerate(tracks)
                        for (img_i, _kp, px, py) in track]

        cams_opt, pts_opt, rmse = rtk_constrained_bundle_adjustment(
            cams0, initial_points, observations, priors, weights, f_px, cx, cy,
        )

        shifts = np.linalg.norm(cams_opt[:, :3] - cams0[:, :3], axis=1)
        logger.info("BA 후 카메라 이동량: 중앙값 %.3f m, 최대 %.3f m "
                    "(재투영 RMSE %.2f px)",
                    float(np.median(shifts)), float(shifts.max()), rmse)
        if shifts.max() > 5.0:
            logger.warning("BA 가 카메라를 %.1f m 옮겼다 — RTK σ 대비 과도하다. "
                           "오매칭 track 또는 가중치 설정을 의심할 것.",
                           float(shifts.max()))

        for ctx, row in zip(contexts, cams_opt):
            ctx.camera_xyz = row[:3].copy()
            ctx.R = rotation_matrix(row[3], row[4], row[5])
        return pts_opt

    # -- 7 단계 -----------------------------------------------------------
    def build_homographies(self,
                           contexts: list[FrameContext],
                           points_xyz: np.ndarray | None = None
                           ) -> tuple[list[FrameContext], GroundPlane]:
        """지상평면 추정 + 프레임별 호모그래피 구성.

        호모그래피 구성에 실패한 프레임은 반환 목록에서 제외된다.
        """
        plane = estimate_ground_plane(
            points_xyz=points_xyz,
            metas=[c.meta for c in contexts],
            camera_xyz=np.vstack([c.camera_xyz for c in contexts]),
            panel_top_offset_m=self.cfg.panel_top_offset_m,
            allow_tilt=self.cfg.allow_tilted_plane,
            inlier_threshold_m=self.cfg.plane_inlier_threshold_m,
        )

        ok: list[FrameContext] = []
        for ctx in contexts:
            # 카메라가 평면 아래에 있으면 기하가 뒤집힌다 — 고도 부호나
            # 지오이드 보정이 틀렸다는 신호.
            z_ground = plane.height_at(ctx.camera_xyz[0], ctx.camera_xyz[1])
            if ctx.camera_xyz[2] <= z_ground:
                logger.warning("카메라가 지상평면 아래 (%.2f m ≤ %.2f m) — 제외: %s",
                               ctx.camera_xyz[2], z_ground, ctx.image_path)
                continue
            fh = build_frame_homography(ctx.camera_xyz, ctx.R, ctx.intr, plane)
            if fh is None:
                logger.warning("호모그래피 구성 실패 — 제외: %s", ctx.image_path)
                continue
            ctx.homography = fh
            ok.append(ctx)

        if not ok:
            raise ValueError("호모그래피를 구성할 수 있는 프레임이 없음")
        logger.info("호모그래피 구성: %d/%d 프레임, 평면 경사 %.3f°, "
                    "GSD 중앙값 %.4f m",
                    len(ok), len(contexts), plane.slope_deg,
                    recommend_gsd([c.homography for c in ok]))
        return ok, plane

    # -- 8 단계 -----------------------------------------------------------
    def run_direct(self,
                   metas: list,
                   output_path: str | Path,
                   image_paths: list[str] | None = None,
                   points_xyz: np.ndarray | None = None) -> dict:
        """SfM 없이 RTK 자세만으로 정사 모자이크 생성 (빠른 현장 확인용).

        ``points_xyz`` 를 주면 (이전 실행의 BA 점군 등) 평면 추정에 사용한다.
        """
        accepted, rejected = self.prepare(metas, image_paths)
        logger.info("품질 게이트: %d 통과 / %d 제외", len(accepted), len(rejected))

        frames, plane = self.build_homographies(accepted, points_xyz)
        stats = mosaic_frames(
            [c.homography for c in frames],
            [c.image_path for c in frames],
            output_path,
            gsd_m=self.cfg.gsd_m,
            epsg=self.cfg.output_epsg or self.cfg.target_epsg,
            cfg=self.cfg.mosaic,
        )
        stats["frames_rejected"] = len(rejected)
        stats["mode"] = "direct (SfM 미사용)"
        return stats

    # -- 유틸: 검출 결과 지오코딩 ------------------------------------------
    def pixels_to_wgs84(self,
                        ctx: FrameContext,
                        pixels: np.ndarray) -> np.ndarray:
        """(N, 2) 이미지 픽셀 → (N, 2) WGS84 ``(lon, lat)``.

        YOLO 검출 bbox 모서리나 폴리곤 정점을 그대로 넣으면 된다.
        ``ctx.homography`` 가 구성되어 있어야 한다. 카메라 뒤쪽으로 떨어지는
        픽셀은 NaN 으로 나온다.
        """
        if ctx.homography is None:
            raise ValueError("호모그래피가 아직 구성되지 않음 — "
                             "build_homographies() 를 먼저 호출할 것")
        xy = ctx.homography.pixel_to_ground(pixels)
        out = np.full_like(xy, np.nan)
        ok = np.all(np.isfinite(xy), axis=1)
        for k in np.nonzero(ok)[0]:
            lon, lat = self.crs.inverse(float(xy[k, 0]), float(xy[k, 1]))
            out[k] = (lon, lat)
        return out

    def bbox_to_wgs84_polygon(self,
                              ctx: FrameContext,
                              bbox: tuple[float, float, float, float]) -> np.ndarray:
        """``(x1, y1, x2, y2)`` 픽셀 bbox → (4, 2) WGS84 폴리곤 (TL,TR,BR,BL).

        ⚠ 이미지에서 축정렬 사각형이던 bbox 는 지상에서 **사각형이 아니다**
        (호모그래피는 평행성을 보존하지 않는다). 중심점 하나만 변환하고 지상에서
        사각형을 그리면 가장자리 검출일수록 오차가 커진다. 4모서리를 각각
        변환해 사변형으로 두는 것이 옳다.
        """
        x1, y1, x2, y2 = bbox
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)
        return self.pixels_to_wgs84(ctx, corners)

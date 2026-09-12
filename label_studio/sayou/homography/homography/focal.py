"""초점거리 자기보정 — EXIF 값이 틀렸을 때 이를 검출하고 바로잡는다.

왜 필요한가 (실측 근거)
-----------------------
그린환경센터 380장 실행에서 BA 는 훌륭해 보였습니다 — 점 231,713개,
재투영 RMSE 1.25 px. 그런데 그 점군으로 만든 DSM 이 이상했습니다:

* 점군 Z 중앙값 **105.60 m** (지면은 약 113.4 m)
* 기복 p1~p99 가 **24.3 m** (지면~패널 상면은 2.4 m 여야 함)

카메라 Z 를 역산하면 159.81 m 이므로 **카메라–점군 거리 = 54.22 m** 입니다.
그런데 메타데이터가 말하는 카메라–지면 거리는:

* ``RelativeAltitude`` ≈ 44.9 m (기압/GNSS)
* ``LRFTargetDistance`` ≈ 44~45 m (레이저)

**서로 독립인 두 센서가 일치**하는데 영상기하만 54.2 m 라고 합니다.
비율 1.207.

원인: **``f_px`` 가 약 21% 큽니다.** ``compute_focal_px`` 는
``FocalLengthIn35mmFilm / 36 × width`` 로 계산하는데, H20T 줌 카메라
(5184×3888, 35mm 환산 31.7~556.2 mm) 의 EXIF 값이 실제 광학 상태와 맞지
않으면 그대로 틀립니다.

왜 BA 가 이걸 못 잡나
---------------------
**초점거리–깊이 축퇴** 때문입니다. 점 위치가 자유변수이므로, BA 는 f 가
21% 커도 점을 21% 깊게 밀어 넣어 재투영을 완벽히 맞춥니다. RMSE 는 낮게
나오지만 3D 는 틀립니다. 합성 검증에서 그대로 재현됐습니다:

    진짜 f_px 5605 → 점군 Z 115.60, 카메라–점군 42.70 m, RMSE 0.39 px
    틀린 f_px 6768 → 점군 Z 106.72, 카메라–점군 51.58 m, RMSE 0.40 px
                                                    비율 1.208 = f 비율

RTK 가 카메라 위치를 미터로 고정해도 이 축퇴는 깨지지 않습니다. 위치는
맞고 깊이만 스케일되기 때문입니다.

해법
----
**독립적인 거리 관측**을 쓰면 축퇴가 깨집니다. LRF 와 RelativeAltitude 가
바로 그것입니다. 두 값이 일치하면 신뢰할 수 있고, 삼각측량이 낸 거리와의
비율이 곧 ``f_px`` 의 오차 배율입니다.

    scale = (카메라–점군 거리) / (메타데이터 거리)
    f_px_corrected = f_px / scale

한 스칼라이므로 추정이 안정적입니다. 보정 후 삼각측량+BA 를 다시 돌리면
점군이 제 높이로 돌아옵니다.

주의
----
* LRF 와 RelativeAltitude 가 **서로 다르면** 둘 중 하나가 이미 틀린
  것이므로 보정하지 않습니다 (경고).
* 보정 배율은 ``[0.6, 1.6]`` 로 제한합니다. 이를 벗어나면 f 문제가 아니라
  다른 것이 잘못된 것입니다.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["metadata_camera_height", "estimate_focal_scale"]


def metadata_camera_height(meta) -> float | None:
    """메타데이터가 말하는 카메라–지면 수직거리 (m).

    LRF 와 RelativeAltitude 를 모두 쓸 수 있으면 서로 검증한다. 크게 다르면
    ``None`` (둘 중 하나가 이미 틀렸다는 뜻).
    """
    rel = getattr(meta, "relative_height", None)
    rel = float(rel) if rel else None

    lrf = None
    alt = getattr(getattr(meta, "gps", None), "altitude", None)
    tgt = getattr(meta, "lrf_target_abs_alt", None)
    if alt is not None and tgt is not None and np.isfinite(tgt) and tgt != 0:
        lrf = float(alt) - float(tgt)

    # [sayou-patch] LRF 가 있으면 그것만 쓴다. RelativeAltitude 와 평균내면
    #   지형 정보가 없는 값이 절반 섞여 기준이 통째로 편향된다.
    if lrf is not None and 1.0 < lrf < 1000.0:
        return float(lrf)

    vals = [v for v in (rel, lrf) if v is not None and 1.0 < v < 1000.0]
    if not vals:
        return None
    if len(vals) == 2 and abs(vals[0] - vals[1]) > 0.25 * max(vals):
        return None                       # 두 센서 불일치 — 신뢰 불가
    return float(np.median(vals))


def estimate_focal_scale(metas, cams_opt, points_xyz,
                         *,
                         min_frames: int = 20,
                         min_points_per_frame: int = 50,
                         search_radius_m: float = 12.0,
                         clip: tuple[float, float] = (0.6, 1.6),
                         max_single_correction: float = 0.30,
                         damping: float = 0.7
                         ) -> tuple[float, dict] | None:
    """``f_px`` 오차 배율 추정. ``f_px_corrected = f_px / scale``.

    각 카메라 아래(수평 ``search_radius_m`` 안)의 점들의 중앙 Z 를 그 프레임의
    지표면으로 보고, 카메라와의 수직거리를 삼각측량 거리로 삼는다. 이를
    메타데이터 거리와 비교한다.

    Returns ``(scale, 진단 dict)`` 또는 ``None`` (근거 부족).
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 500:
        return None

    C = np.asarray(cams_opt, dtype=np.float64)[:, :3]
    ratios, d_ba_l, d_meta_l = [], [], []

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts[:, :2])
        query = lambda xy: tree.query_ball_point(xy, r=search_radius_m)  # noqa: E731
    except ImportError:                                   # pragma: no cover
        query = None

    for i, meta in enumerate(metas):
        if i >= len(C):
            break
        d_meta = metadata_camera_height(meta)
        if d_meta is None:
            continue
        if query is not None:
            idx = query(C[i, :2])
        else:
            idx = np.nonzero(np.hypot(pts[:, 0] - C[i, 0],
                                      pts[:, 1] - C[i, 1]) <= search_radius_m)[0]
        if len(idx) < min_points_per_frame:
            continue
        z_surf = float(np.median(pts[np.asarray(idx), 2]))
        d_ba = float(C[i, 2]) - z_surf
        if not (1.0 < d_ba < 1000.0):
            continue
        ratios.append(d_ba / d_meta)
        d_ba_l.append(d_ba); d_meta_l.append(d_meta)

    if len(ratios) < min_frames:
        logger.info("초점거리 자기보정: 근거 프레임 %d장으로 부족 (최소 %d) — 생략",
                    len(ratios), min_frames)
        return None

    arr = np.array(ratios)
    raw = float(np.median(arr))
    spread = float(np.median(np.abs(arr - raw)))
    # ★ 감쇠: 두 실행이 서로 반대 방향으로 22% 어긋난 적이 있다
    #   (f_px 5743 vs 7016). 한 번에 전량 보정하면 점군 품질에 따라
    #   진동한다. 부분 보정 후 다음 라운드에서 재측정하는 편이 안전하다.
    scale = 1.0 + damping * (raw - 1.0)

    info = {
        "scale": scale,
        "scale_raw": raw,
        "damping": damping,
        "frames_used": len(ratios),
        "mad": spread,
        "d_triangulated_median_m": float(np.median(d_ba_l)),
        "d_metadata_median_m": float(np.median(d_meta_l)),
    }

    logger.info("초점거리 자기보정: 삼각측량 거리 %.2f m vs 메타데이터 %.2f m "
                "→ 배율 %.4f (감쇠 전 %.4f, 프레임 %d장, 산포 ±%.4f)",
                info["d_triangulated_median_m"], info["d_metadata_median_m"],
                scale, raw, len(ratios), spread)

    if spread > 0.08:
        logger.warning("  프레임별 배율 산포가 ±%.3f 로 큽니다 — 지형 기복이 "
                       "크거나 점군에 이상치가 많습니다. 보정을 건너뜁니다.",
                       spread)
        return None
    # ★ 메타데이터 거리 자체가 틀렸을 때를 거른다.
    #   이 보정은 메타데이터 거리를 **정답으로 가정**한다. DJI
    #   ``RelativeAltitude`` 는 **이륙 지점 기준** 상대고도이므로, 이륙점이
    #   촬영 부지와 높이가 다르면 그 차이만큼 통째로 틀린다.
    #
    #   실측(극동대): 메타데이터가 33.19 m 라고 했지만 실제는 약 40 m 였다.
    #   보정 루프가 그 틀린 값을 목표로 4회 돌며 f_px 를 2704 → 3232
    #   (+19.5%) 로 부풀렸고 최종 검증 잔차가 +20.3% 로 남았다. 세
    #   센서(광각/줌/열화상)가 독립적으로 BA 평면을 -7.8~-8.0 m 로 가리킨
    #   것이 같은 이야기다.
    #
    #   ★ 되돌림 기록 — 이 가드를 0.08 로 걸었다가 **품질이 나빠져** 0.30 으로
    #     완화했다. 극동대에서 Wide 0.677 → 0.775 m, TM 0.666 → 0.749 m 로
    #     악화했다 (Zoom 은 산포 가드로 원래 보정을 안 해 변화 없음 = 대조군).
    #
    #     당시 제 진단은 "메타데이터 고도가 8 m 틀렸다" 였는데 **거꾸로**였다.
    #     보정을 막자 BA 평면과 LRF 의 차이가 -7.75 m → -0.51 m 로 사라졌다.
    #     즉 그 8 m 는 메타데이터 오차가 아니라 **부풀려진 f_px 가 만든 것**
    #     이었고(초점-깊이 축퇴), 그런데도 그 보정이 최종 품질에는 도움이
    #     되고 있었다.
    #
    #     교훈: 중간 지표(평면 일치도)가 나빠 보여도 **최종 품질로 판정**해야
    #     한다. 이 가드는 명백한 폭주만 막는 안전핀으로 남긴다.
    #   판정은 **감쇠 전 원값(raw)** 으로 한다. 감쇠(0.7배)를 거친 값으로 재면
    #   실측 극동대가 12.1% → 8.5% 로 줄어 한계를 통과해 버린다.
    if abs(1.0 - raw) > max_single_correction:
        _dt = info.get("d_triangulated_median_m", float("nan"))
        _dm = info.get("d_metadata_median_m", float("nan"))
        logger.warning(
            "  초점거리 보정 기각: 한 번에 %.1f%% 보정이 필요합니다 "
            "(한계 %.0f%%). 이 크기는 렌즈가 아니라 **메타데이터 고도 기준**이 "
            "틀렸다는 신호입니다 — DJI RelativeAltitude 는 이륙 지점 기준이라 "
            "이륙점이 촬영 부지와 높이가 다르면 그만큼 통째로 어긋납니다. "
            "삼각측량 %.2f m vs 메타데이터 %.2f m (차이 %.2f m). "
            "실제 비행고도를 확인하세요.",
            abs(1.0 - raw) * 100, max_single_correction * 100,
            _dt, _dm, _dm - _dt)
        return None

    if not (clip[0] <= scale <= clip[1]):
        logger.warning("  배율 %.3f 가 타당 범위 %.1f~%.1f 를 벗어납니다 — "
                       "초점거리가 아니라 다른 것이 잘못됐을 수 있습니다. "
                       "보정을 건너뜁니다.", scale, *clip)
        return None
    return scale, info

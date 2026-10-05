"""쌍 단위 RTK 일관성 검사 — 반복 격자의 '한 줄 건너뛴' 오매칭을 거른다.

왜 반경으로는 안 되는가 (실측)
------------------------------
처음에는 RTK 예측 주변 **반경 안에서만 대응을 찾는** 방식(`guided_match`)을
썼습니다. 실패했습니다.

```
줄무늬 주기 160 px → 반경은 80 px 미만이어야 오매칭이 걸러짐
그런데 예측 오차는:
  짐벌 자세 0.9°  →  11 px
  지형 기복 5 m   →  71 px      (평면 가정에서 벗어난 만큼)
  합계            →  82 px  > 80 px
```

**여유가 없습니다.** 반경 64 px 로 돌린 결과 정상 대응까지 잘려
`zero_frames` 가 25 → 37 로 늘고 총 점이 11% 줄었습니다.

이 현장은 경사지라 지형 기복이 **줄무늬 주기와 맞먹는 예측 오차**를
만듭니다. 반경 하나로 둘을 분리할 수 없습니다.

쌍 단위로 보면 분리됩니다
--------------------------
개별 대응 대신 **쌍 전체의 변위**를 봅니다.

* 정상 매칭: 쌍의 변위가 RTK 예측과 수십 px 차이 (예측 오차 수준)
* 한 줄 오매칭: 쌍의 변위가 RTK 예측과 **줄무늬 주기만큼**(160 px) 차이

쌍 안의 대응 수십~수백 개가 함께 어긋나므로, 중앙값을 쓰면 개별 예측
오차에 둔감해집니다. **오차 82 px 와 주기 160 px 는 충분히 벌어져 있습니다.**

실측 근거: 원본 10장에서 쌍별 변위를 주기로 나누니 10쌍 중 7쌍이 정수배
±0.25 이내였습니다 (무작위면 50%). 그 7쌍이 걸러야 할 대상입니다.

검증 상태 — **끝까지 확인하지 못했습니다**
------------------------------------------
오매칭이 실재한다는 것(정수배 70%)은 원본으로 확인했습니다. 그러나 이
필터가 실제로 그 7쌍만 골라내는지는 **검증하지 못했습니다** — RTK 예측을
계산하려면 homography.py 가 필요한데 그 파일이 이 트리에 없습니다.

그래서 안전장치를 두었습니다:

* min_pairs_keep (기본 0.3) — 남는 쌍이 30% 미만이면 **필터를 적용하지
  않고** 기존 매칭을 유지합니다. 예측이 통째로 틀린 상황에서 전부 버리는
  것을 막습니다.
* 기본값은 꺼짐입니다.

앞선 guided_match 시도가 정상 대응까지 잘라 zero_frames 를 25 → 37
로 늘린 전례가 있으므로, **실행 후 관측 수를 반드시 확인**하십시오.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["filter_matches_by_rtk", "estimate_stripe_pitch_px"]


def estimate_stripe_pitch_px(image_paths, n_sample: int = 5) -> float | None:
    """원본 몇 장에서 패널 줄무늬 주기(px)를 잰다."""
    import cv2
    vals = []
    for p in list(image_paths)[:max(n_sample, 1)]:
        try:
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            for prof in (im.mean(axis=0), im.mean(axis=1)):
                pr = prof.astype(float) - float(prof.mean())
                sp = np.abs(np.fft.rfft(pr))
                if len(sp) < 8:
                    continue
                k = int(np.argmax(sp[3:min(60, len(sp))])) + 3
                vals.append(len(pr) / k)
        except Exception:
            continue
    if not vals:
        return None
    return float(np.median(vals))


def filter_matches_by_rtk(matches, frames, *,
                          pitch_px: float,
                          reject_frac: float = 0.5,
                          min_pairs_keep: float = 0.3):
    """쌍 단위로 RTK 예측과 비교해 '한 줄 어긋난' 쌍을 버린다.

    Parameters
    ----------
    matches : ``{(i, j): (pts_i, pts_j, idx_i, idx_j)}``
    frames : RTK+짐벌+평면으로 만든 초기 ``FrameHomography`` 목록.
    pitch_px : 줄무늬 주기.
    reject_frac : 예측과의 차이가 ``pitch_px × reject_frac`` 을 넘으면 기각.
        0.5 면 '주기의 절반 이상 어긋나면 버린다' 는 뜻이다.
    min_pairs_keep : 남는 쌍이 이 비율 미만이면 **필터를 적용하지 않는다**
        (예측이 통째로 틀린 상황에서 전부 버리는 것을 막는다).
    """
    if not matches or pitch_px is None or pitch_px <= 0:
        return matches, {}

    thr = pitch_px * reject_frac
    kept, diffs, rejected = {}, [], []
    for (i, j), val in matches.items():
        try:
            pts_i, pts_j = np.asarray(val[0]), np.asarray(val[1])
            if len(pts_i) < 5:
                kept[(i, j)] = val
                continue
            fh_i, fh_j = frames[i], frames[j]
            # RTK 예측: i 의 점을 지상으로 → j 의 영상으로
            from .guided_match import predict_correspondence
            pred = predict_correspondence(fh_i, fh_j, pts_i)
            if pred is None or len(pred) != len(pts_j):
                kept[(i, j)] = val
                continue
            d = np.median(np.hypot(*(pts_j - pred).T))
            diffs.append(d)
            if d > thr:
                rejected.append(((i, j), d))
            else:
                kept[(i, j)] = val
        except Exception:
            kept[(i, j)] = val

    info = {"pitch_px": pitch_px, "threshold_px": thr,
            "pairs_in": len(matches), "pairs_kept": len(kept),
            "pairs_rejected": len(rejected)}
    if diffs:
        info["median_deviation_px"] = float(np.median(diffs))

    if not matches:
        return matches, info
    keep_ratio = len(kept) / len(matches)
    if keep_ratio < min_pairs_keep:
        logger.warning(
            "RTK 쌍 검사 미적용: 남는 쌍이 %.0f%% 뿐입니다 (기준 %.0f%%). "
            "RTK 예측 자체가 크게 틀렸을 수 있어 기존 매칭을 유지합니다.",
            keep_ratio * 100, min_pairs_keep * 100)
        info["applied"] = False
        return matches, info

    info["applied"] = True
    logger.info(
        "RTK 쌍 검사: %d쌍 중 %d쌍 기각 (변위가 예측과 %.0f px 초과 차이). "
        "줄무늬 주기 %.0f px 의 %.0f%% 를 기준으로 '한 줄 건너뛴' 오매칭을 "
        "거릅니다. 예측 차이 중앙값 %.0f px",
        len(matches), len(rejected), thr, pitch_px, reject_frac * 100,
        info.get("median_deviation_px", float("nan")))
    return kept, info



def choose_prediction_plane(zxy, cam_center_xy, *, env=None):
    """RTK 예측에 쓸 평면을 고른다 — 지정 기준면 → 레이저 지면 평면 → 수평면.

    ★ 예전에는 레이저 지면 높이 중앙값의 **수평면**만 썼다. 안내 매칭을 기각할 때
      (EWP 열화상) 예측 오차 82 px 가운데 71 px 가 지형 기복 5 m 에서 왔고, 경사
      부지(Site-1 13°, Site-2 9°)에서는 쌍 검사의 예측 자체가 크게 틀린다. 실행할
      때 SAYOU_PLANE_OVERRIDE 로 확정 기준면을 넘기고 있으니 그것을 먼저 쓴다.
      기준면을 정하는 함수는 번들조정 뒤에 돌아 이 단계에서는 값을 받을 수 없어,
      같은 환경변수를 여기서 직접 읽는다.

    Parameters
    ----------
    zxy : [(x, y, 지면 z)] — 레이저로 지면 높이를 구한 프레임들의 카메라 위치.
    cam_center_xy : 카메라 중심 (지정 기준면의 원점이 이 부지 것인지 검사).
    env : 환경변수 사전 (시험용). 기본은 os.environ.
        SAYOU_RTK_CHECK_PLANE=horizontal 이면 예전처럼 수평면을 강제한다.

    Returns ``(GroundPlane, info)``.
    """
    import os
    from .homography import GroundPlane
    env = os.environ if env is None else env
    zs = np.array([z for _, _, z in zxy], dtype=float) if zxy else np.array([])
    z0 = float(np.median(zs)) if zs.size else 0.0
    horiz = GroundPlane.horizontal(z0)
    mode = str(env.get("SAYOU_RTK_CHECK_PLANE", "auto")).strip().lower()
    cx, cy = float(cam_center_xy[0]), float(cam_center_xy[1])

    def _out(pl, src, **kw):
        info = {"source": src,
                "slope_deg": float(np.degrees(np.arctan(np.hypot(pl.a, pl.b)))),
                "z_at_center": float(pl.height_at(cx, cy))}
        info.update(kw)
        logger.info("RTK 쌍 검사 예측 평면: %s (경사 %.2f°, 카메라 중심 표고 %.2f m)%s",
                    {"override": "지정 기준면", "lrf_fit": "레이저 지면 평면",
                     "horizontal": "수평면", "horizontal_forced": "수평면 (SAYOU_RTK_CHECK_PLANE=horizontal)"}[src],
                    info["slope_deg"], info["z_at_center"],
                    ("  — %s" % kw["note"]) if kw.get("note") else "")
        return pl, info

    if mode == "horizontal":
        return _out(horiz, "horizontal_forced")

    # 1) 지정 기준면 — 원점이 이 부지 것일 때만 (다른 부지 값이 셸에 남은 경우 차단)
    ov = str(env.get("SAYOU_PLANE_OVERRIDE", "") or "").strip()
    note = ""
    if ov:
        try:
            v = [float(t) for t in ov.replace(" ", "").split(",")]
            if len(v) == 5:
                dz = v[0] * (cx - v[3]) + v[1] * (cy - v[4])
                if abs(dz) <= 2.0:
                    return _out(GroundPlane(a=v[0], b=v[1], c=v[2], origin_xy=(v[3], v[4])), "override")
                note = "지정 기준면 원점이 카메라 중심에서 표고차 %+.1f m — 다른 부지 값으로 보고 쓰지 않음" % dz
        except ValueError:
            note = "SAYOU_PLANE_OVERRIDE 를 읽지 못함"

    # 2) 레이저 지면 평면 — 옥상 · 나무 같은 이상치를 깎아 가며 적합
    if zs.size >= 10:
        P = np.array([(x, y) for x, y, _ in zxy], dtype=float)
        x0, y0 = float(np.median(P[:, 0])), float(np.median(P[:, 1]))
        A = np.c_[P[:, 0] - x0, P[:, 1] - y0, np.ones(len(zs))]
        keep = np.ones(len(zs), bool)
        coef = None
        for _ in range(3):
            if keep.sum() < 6:
                break
            coef, *_ = np.linalg.lstsq(A[keep], zs[keep], rcond=None)
            r = zs - A @ coef
            mad = float(np.median(np.abs(r[keep] - np.median(r[keep])))) * 1.4826 + 1e-6
            keep = np.abs(r) < max(2.5 * mad, 0.5)
        if coef is not None and keep.sum() >= 6:
            r = zs - A @ coef
            rms_fit = float(np.sqrt(np.mean(r[keep] ** 2)))
            rms_h = float(np.sqrt(np.mean((zs[keep] - np.median(zs[keep])) ** 2)))
            slope = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
            if slope > 0.5 and rms_fit < 0.8 * rms_h:
                pl = GroundPlane(a=float(coef[0]), b=float(coef[1]), c=float(coef[2]), origin_xy=(x0, y0))
                return _out(pl, "lrf_fit", rms_m=rms_fit, rms_horizontal_m=rms_h,
                            frames=int(keep.sum()), note=note)
    return _out(horiz, "horizontal", note=note)

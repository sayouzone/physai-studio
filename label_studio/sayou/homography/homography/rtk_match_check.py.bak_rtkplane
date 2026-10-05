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

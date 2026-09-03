"""모자이크 품질 자체 진단 — 패널 영역의 국소 어긋남을 잰다.

.. note::
   모듈 이름이 ``mosaic_qc`` 인 이유: 같은 패키지에 이미 ``quality.py``
   (RTK 품질 평가 — ``RTKQuality``, ``build_rtk_priors_and_weights``) 가
   있습니다. 처음에 ``quality.py`` 로 만들었다가 그 파일을 덮어써
   ``ImportError: cannot import name 'RTKQuality'`` 를 냈습니다.
   이 모듈은 **모자이크 산출물**의 품질을, ``quality.py`` 는 **입력 RTK**
   의 품질을 다룹니다.

왜 이것이 필요한가
------------------
지금까지 품질을 "중심으로부터 거리별 국소 어긋남" 으로 판단했는데, **그
지표가 오도했습니다.**

| | 바깥(전체) | 패널만 | 패널 커버 |
|---|---|---|---|
| 예외 끔 | 0.146 m | 0.063 m | 40.6 Mpx |
| 예외 1단계 | 0.427 m | 0.087 m | 64.3 Mpx |
| 픽셀별 상한 | **0.563 m** | **0.078 m** | 55.0 Mpx |

방사형 "바깥" 은 픽셀별 상한에서 나빠졌지만(0.427 → 0.563) **패널만 보면
오히려 좋아졌습니다**(0.087 → 0.078). 바깥 밴드는 대부분 잔디·도로이고
극단적 off-nadir 로 채워진 곳이라 검사 품질을 대표하지 못합니다.

그래서 **검사 대상(패널)에서의 어긋남**을 실행할 때마다 직접 재서
``summary.json`` 에 남깁니다. 다음부터는 결과 파일만 보고 판단할 수
있습니다.

측정 방법
---------
1. 밝기 Otsu 로 패널 마스크를 만든다.
2. 출력을 블록으로 나누고, **패널이 30% 이상인 블록**만 고른다.
3. 각 블록의 좌우 절반에서 세로 밝기 프로파일을 뽑아 상호상관으로
   세로 어긋남을 잰다 (패널 행이 가로로 길어 이 방향이 민감하다).
4. 어긋남의 중앙값과 '상관 0.6 미만' 블록 비율을 보고한다.

이 값은 **절대 정확도가 아니라 프레임 간 정합의 일관성** 입니다. GCP 가
없으므로 절대 정확도는 RTK 가 담당하고, 여기서 보는 것은 이어붙임 품질
입니다.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["assess_mosaic"]


def assess_mosaic(path, gsd_m: float, *,
                  block_m: float = 3.8,
                  min_panel_frac: float = 0.30,
                  max_shift_m: float = 0.40,
                  max_blocks: int = 4000) -> dict | None:
    """모자이크를 읽어 패널 영역의 국소 어긋남을 잰다."""
    try:
        import rasterio
    except ImportError:                                   # pragma: no cover
        return None

    try:
        with rasterio.open(str(path)) as ds:
            n_band = min(ds.count, 3)
            # 메모리 보호: 긴 변 6000 px 로 축소해 읽는다.
            scale = max(max(ds.width, ds.height) / 6000.0, 1.0)
            oh = int(ds.height / scale); ow = int(ds.width / scale)
            bands = [np.squeeze(ds.read(i + 1, out_shape=(1, oh, ow)))
                     for i in range(n_band)]
    except Exception as exc:
        logger.warning("품질 진단을 건너뜁니다 (%s)", exc)
        return None

    a = np.stack(bands, axis=-1).astype(np.uint8)
    eff_gsd = gsd_m * scale
    g = (cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.shape[2] == 3
         else a[:, :, 0])
    valid = a.astype(np.int32).sum(axis=2) > 0
    if valid.sum() < 10000:
        return None

    thr, _ = cv2.threshold(g[valid], 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    panel = (g > thr) & valid

    gl = cv2.GaussianBlur(g.astype(np.float32), (0, 0),
                          max(0.8 / eff_gsd / 3.0, 0.8))
    step = max(int(block_m / eff_gsd), 40)
    lim = max(int(max_shift_m / eff_gsd), 4)
    H, W = g.shape

    shifts, corrs = [], []
    for y in range(0, H - step, step):
        for x in range(0, W - step, step):
            if panel[y:y + step, x:x + step].mean() < min_panel_frac:
                continue
            if valid[y:y + step, x:x + step].mean() < 0.95:
                continue
            blk = gl[y:y + step, x:x + step]
            if blk.std() < 12:
                continue
            p1 = blk[:, :step // 2].mean(axis=1)
            p2 = blk[:, step // 2:].mean(axis=1)
            p1 = p1 - p1.mean(); p2 = p2 - p2.mean()
            best, bl = -9.0, 0
            for L in range(-lim, lim + 1):
                A = p1[max(0, -L):len(p1) - max(0, L)]
                B = p2[max(0, L):len(p2) - max(0, -L)]
                if len(A) < 40:
                    continue
                c = float(np.dot(A, B) /
                          (np.linalg.norm(A) * np.linalg.norm(B) + 1e-9))
                if c > best:
                    best, bl = c, L
            shifts.append(abs(bl) * eff_gsd); corrs.append(best)
            if len(shifts) >= max_blocks:
                break
        if len(shifts) >= max_blocks:
            break

    if len(shifts) < 20:
        logger.info("품질 진단: 패널 블록이 %d개로 부족해 생략", len(shifts))
        return None

    sh = np.array(shifts); co = np.array(corrs)
    out = {
        "panel_misalign_median_m": float(np.median(sh)),
        "panel_misalign_p90_m": float(np.percentile(sh, 90)),
        "broken_block_ratio": float((co < 0.6).mean()),
        "blocks": int(len(sh)),
        "panel_area_px": int(panel.sum() * scale * scale),
        "eff_gsd_m": float(eff_gsd),
    }
    logger.info("모자이크 품질(패널 영역): 어긋남 중앙값 %.3f m "
                "(%.1f px), 90%% %.3f m, 파손 블록 %.1f%%, 블록 %d개",
                out["panel_misalign_median_m"],
                out["panel_misalign_median_m"] / max(gsd_m, 1e-9),
                out["panel_misalign_p90_m"],
                out["broken_block_ratio"] * 100, out["blocks"])
    if out["panel_misalign_median_m"] > 0.15:
        logger.warning("  패널 어긋남이 %.2f m 로 큽니다 — --offnadir-frac 을 "
                       "낮추거나(예: 0.32) BA 로그를 확인하세요.",
                       out["panel_misalign_median_m"])
    return out

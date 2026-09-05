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
                  max_shift_m: float = 1.00,
                  read_px: int = 12000,
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
            scale = max(max(ds.width, ds.height) / float(read_px), 1.0)
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
    logger.debug("품질 진단 해상도: 유효 GSD %.4f m, 블록 %d px, 탐색 ±%d",
                 eff_gsd, step, lim)
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
            cs = {}
            best, bl = -9.0, 0
            for L in range(-lim, lim + 1):
                A = p1[max(0, -L):len(p1) - max(0, L)]
                B = p2[max(0, L):len(p2) - max(0, -L)]
                if len(A) < 40:
                    continue
                c = float(np.dot(A, B) /
                          (np.linalg.norm(A) * np.linalg.norm(B) + 1e-9))
                cs[L] = c
                if c > best:
                    best, bl = c, L
            # ★ 서브픽셀 보정. 정수 이동만 쓰면 분해능이 유효 GSD 로 묶여
            #   실측에서 0.051 m 가 **정확히 2칸**으로 나왔다 — 실제 값이
            #   0.03 이든 0.06 이든 같은 숫자가 찍힌다. 상호상관 최댓값
            #   주변에 포물선을 맞춰 소수점 이동을 얻는다.
            sub = float(bl)
            if bl - 1 in cs and bl + 1 in cs:
                c0, c1, c2 = cs[bl - 1], cs[bl], cs[bl + 1]
                den = c0 - 2.0 * c1 + c2
                if abs(den) > 1e-12:
                    sub = bl + float(np.clip(0.5 * (c0 - c2) / den, -0.5, 0.5))
            shifts.append(abs(sub) * eff_gsd); corrs.append(best)
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
        "valid_ratio": float(valid.mean()),
    }

    # ★ 눈에 보이는 것은 평균이 아니라 **패널 경계의 계단 단차**다.
    #   실측(그린환경센터 RGB)에서 중앙값은 0.06 m 로 멀쩡한데 패널 상단
    #   경계의 열 간 점프는 99% 가 64 px(41 cm), 최대 155 px(99 cm) 였다.
    #   시임이 패널 위를 지날 때 Δh·k 만큼 끊기기 때문이다.
    #   중앙값·p90 과 별개로 이 값을 직접 재서 남긴다.
    out.update(_panel_edge_steps(panel, eff_gsd))

    # ★ 반사(포화) 비율 — 판독 가능성의 직접 지표.
    #   실측에서 패널의 5.0% 가 포화(240+)라 그 자리는 결함을 볼 수 없습니다.
    #   정합 오차(중앙값 5 cm = 8 px)보다 판독에 해로우므로 함께 잽니다.
    #   참고: 원본 이미지 자체가 밝은 영역의 7~11% 가 포화이고 모자이크는
    #   5.0% 이므로, 프레임 선택은 이미 제 역할을 하고 있습니다. 이 값이
    #   크게 오르면 촬영 조건(태양 고도·시각)을 의심해야 합니다.
    if panel.sum() > 1000:
        gp = g[panel]
        out["panel_saturated_240"] = float((gp >= 240).mean())
        out["panel_saturated_250"] = float((gp >= 250).mean())
        logger.info("  패널 반사(포화): 240+ %.2f%%, 250+ %.2f%% — 그 자리는 "
                    "결함 판독이 불가능합니다. 크게 오르면 촬영 시각·태양 "
                    "고도를 확인하세요.",
                    out["panel_saturated_240"] * 100,
                    out["panel_saturated_250"] * 100)
    logger.info("모자이크 품질(패널 영역): 어긋남 중앙값 %.3f m "
                "(%.1f px), 90%% %.3f m, 파손 블록 %.1f%%, 블록 %d개",
                out["panel_misalign_median_m"],
                out["panel_misalign_median_m"] / max(gsd_m, 1e-9),
                out["panel_misalign_p90_m"],
                out["broken_block_ratio"] * 100, out["blocks"])
    # ★ 커버리지가 다른 결과끼리 이 값을 직접 비교하면 안 된다.
    #   실측에서 tolerance 1.0 이 0.045 m 로 최고로 보였지만 그때 충전율이
    #   0.760 뿐이었고, **버려진 픽셀이 하필 가장 나쁜 것들**이었다. ε 로
    #   되살리자 0.058 m 로 올라갔다. 즉 좋은 값이 아니라 나쁜 데이터를
    #   뺀 값이었다. 공통 영역으로 맞춰 재면 순서가 뒤집힌다.
    logger.info("  (유효 영역 %.1f%% — 충전율이 다른 결과끼리 이 값을 직접 "
                "비교하지 마세요. 덜 채운 쪽이 나쁜 픽셀을 빼서 좋아 보입니다.)",
                out["valid_ratio"] * 100)
    if out["panel_misalign_median_m"] > 0.15:
        logger.warning("  패널 어긋남이 %.2f m 로 큽니다 — --offnadir-frac 을 "
                       "낮추거나(예: 0.32) BA 로그를 확인하세요.",
                       out["panel_misalign_median_m"])
    return out


def _panel_edge_steps(panel: np.ndarray, gsd: float,
                      max_step_m: float = 1.5) -> dict:
    """패널 상단 경계선의 열 간 점프 = 시임 단차.

    각 열에서 패널이 처음 나타나는 행을 찾고, 이웃 열과의 차이를 본다.
    정합이 완벽하면 경계선이 매끄러워 차이가 0~1 px 이고, 시임이 패널을
    자르면 그 자리에서 크게 튄다.
    """
    if panel.ndim != 2 or panel.sum() < 1000:
        return {}

    # ★ 전체 마스크에서 '열별 첫 패널 행' 을 쓰면 **패널 행 사이를 건너뛰어**
    #   의미가 없어진다 (실측 전체 모자이크에서 최대 36 m 가 나왔다 — 그건
    #   단차가 아니라 다른 행으로 넘어간 것이다).
    #   패널 덩어리(연결성분)마다 그 안에서만 경계선을 따라간다.
    n, lab, st, _ = cv2.connectedComponentsWithStats(
        panel.astype(np.uint8), 8)
    diffs = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 2000:
            continue
        x0 = st[i, cv2.CC_STAT_LEFT]; w = st[i, cv2.CC_STAT_WIDTH]
        y0 = st[i, cv2.CC_STAT_TOP]; h = st[i, cv2.CC_STAT_HEIGHT]
        if w < 50:
            continue
        sub = (lab[y0:y0 + h, x0:x0 + w] == i)
        tops = []
        for x in range(sub.shape[1]):
            idx = np.flatnonzero(sub[:, x])
            tops.append(idx[0] if idx.size else -1)
        tops = np.asarray(tops, dtype=np.int64)
        ok = tops >= 0
        # 성분 안에서 연속된 구간만 차분한다.
        run = []
        for x in range(len(tops)):
            if ok[x]:
                run.append(tops[x])
            else:
                if len(run) > 20:
                    diffs.append(np.abs(np.diff(np.asarray(run))))
                run = []
        if len(run) > 20:
            diffs.append(np.abs(np.diff(np.asarray(run))))
    if not diffs:
        return {}
    d = np.concatenate(diffs)
    if d.size < 200:
        return {}

    # ★ 물리적 상한을 넘는 점프는 시임 단차가 아니다.
    #   단차 = 패널 높이 × k 인데, 패널 1.5 m 에 k=1.0 이어도 1.5 m 다.
    #   실측에서 최대 13.87 m 가 나왔는데, 그건 패널 연결성분이 L자·계단
    #   모양이라 상단 경계가 성분 안에서도 **정당하게** 크게 뛴 것이다
    #   (행이 합쳐지거나 꺾이는 곳). 정합 오차가 아니라 배치의 형태다.
    #   그런 열은 빼고 재고, 몇 개였는지만 따로 보고한다.
    d_m = d * gsd
    plausible = d_m <= max_step_m
    n_excl = int((~plausible).sum())
    dd = d_m[plausible]
    if dd.size < 100:
        return {}
    res = {
        "edge_step_p95_m": float(np.percentile(dd, 95)),
        "edge_step_p99_m": float(np.percentile(dd, 99)),
        "edge_step_over3px_ratio": float((dd > 3 * gsd).mean()),
        "edge_step_columns": int(dd.size),
        "edge_step_excluded": n_excl,
    }
    logger.info("  패널 경계 단차: 95%% %.3f m, 99%% %.3f m, 3px 초과 %.1f%% "
                "(열 %d개, 형태성 점프 %d개 제외) — 시임이 패널을 자르면 "
                "여기서 커집니다 (--seam-panel-penalty 로 조절)",
                res["edge_step_p95_m"], res["edge_step_p99_m"],
                res["edge_step_over3px_ratio"] * 100,
                res["edge_step_columns"], n_excl)
    return res

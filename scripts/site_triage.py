#!/usr/bin/env python3
"""부지 사전 진단 — 파이프라인을 돌리기 **전에** 튜닝 가능한 부지인지 판정한다.

    exiftool -q -n -T -GPSLatitude -GPSLongitude -GPSAltitude \\
        -RelativeAltitude -LRFTargetDistance \\
        -FocalLengthIn35mmFormat -ImageWidth <RGB>/*.JPG > exif.tsv
    python scripts/site_triage.py exif.tsv

★ 초점거리 열(6·7번)을 반드시 넣으십시오. 없으면 광각을 가정하는데,
  줌(_Z)·열화상(_T) 데이터에서 프레임 크기가 통째로 틀립니다. 실측:
  어느 부지를 줌(f35 61mm)으로 찍었는데 광각(24mm)으로 가정해 덮힘
  반경을 50 m 로 잡았고(실제 15 m), 그 탓에 "촬영 부족"이라는 반대
  판정이 나왔다.

★ 왜 필요한가
  옥산_1호에서 30분짜리 실행을 아홉 번 돌린 뒤에야 "촬영이 부족해서
  후처리로 못 고친다"는 결론에 도달했다. 그 판단에 필요한 숫자는 전부
  EXIF 안에 있었고 30초면 나왔다.

  판정 기준은 EWP-서오창IC-2(성공)와 옥산_1호(실패)의 실측 대비다.

                        EWP TM    EWP RGB   옥산 RGB
    촬영 밀도            12.5      12.5       7.5    프레임/1000m2
    최소 k 중앙값        0.06      0.09       0.15
    카메라밴드/모자이크   ~1.0      ~1.0       0.54
    LRF 평면적합 RMSE    0.82      0.82       3.70   m
    LRF 기복             13        13         38     m
    → 결과              성공      부분성공    실패

  최소 k 가 이 중 가장 강력하다. k 는 기복변위 배율이므로
  패널 높이 1.5 m 기준 지상 오차 = 1.5 x k 다.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from datetime import datetime
import sys

import math
import numpy as np

# EWP-서오창IC-2 실측 기준값 (튜닝이 통했던 부지)
REF = dict(density=12.5, min_k=0.09, band_ratio=0.95, lrf_rmse=0.82)


def load(path, lat_col=0, lon_col=1):
    """exiftool TSV. 열: lat lon GPSAlt RelAlt LRF [f35 ImageWidth]"""
    d = np.genfromtxt(path, dtype=float)
    if d.ndim == 1:
        d = d[None, :]
    if d.shape[1] < 3:
        sys.exit("열이 %d개뿐입니다 — lat lon GPSAlt [RelAlt] [LRF] 순서로 "
                 "뽑으셨는지 확인하십시오" % d.shape[1])
    ncol = d.shape[1]
    lat, lon, alt = d[:, lat_col], d[:, lon_col], d[:, 2]
    rel = d[:, 3] if d.shape[1] > 3 else np.full(len(d), np.nan)
    lrf = d[:, 4] if d.shape[1] > 4 else np.full(len(d), np.nan)
    f35 = float(np.nanmedian(d[:, 5])) if d.shape[1] > 5 else float("nan")
    wpx = float(np.nanmedian(d[:, 6])) if d.shape[1] > 6 else float("nan")
    # 국소 평면 근사 — 부지 규모(<2 km)에서 오차 무시 가능
    lat0 = np.nanmedian(lat)
    x = np.radians(lon - np.nanmedian(lon)) * 6378137.0 * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * 6356752.0
    return x, y, alt, rel, lrf, f35, wpx, ncol


def load_npz(cam_path, alt_path):
    """검증용 — cameras.npz(투영좌표) + alt.tsv(RelAlt LRF GPSAlt)."""
    z = np.load(cam_path, allow_pickle=True)
    c = z["cams_opt"]
    a = np.genfromtxt(alt_path, dtype=float)
    if len(a) != len(c):
        sys.exit("cameras.npz %d장 vs alt.tsv %d행 — 개수가 다릅니다"
                 % (len(c), len(a)))
    return (c[:, 0], c[:, 1], a[:, 2], a[:, 0], a[:, 1],
            float("nan"), float("nan"), 7)


def _detect_two_level(z, min_gap=3.0, min_frac=0.15, sep_ratio=1.5):
    """지면 표고가 두 덩어리로 갈리는지 본다 (옥상·계단식 조성).

    가장 큰 빈 구간(gap)이 min_gap 이상이고, 양쪽에 각각 min_frac 이상이
    있고, **간극이 각 층 내부 산포보다 충분히 커야** 두 층으로 본다.

    ★ 세 번째 조건이 없어서 경사면을 옥상으로 오진했다. 실측(같은 부지
      두 회차): 표고가 88→115 m 로 거의 연속인 경사면인데 중간에 빈
      구간 하나(2.35 m)가 있어 "두 층, 위층 폭 4.1 m" 로 판정했고,
      옥상 기준면(113.8 m)을 적용했더니 전체 평면보다 나을 게 없었다
      (14.55 vs 13.76%, 구간 겹침). 진짜 옥상(199장)은 아래층 σ 1.65 m,
      위층 σ 1.25 m 에 간극 5.4 m 로 뚜렷이 갈렸다.

      판정: 간극 > sep_ratio × max(아래층 σ, 위층 σ)
        진짜 옥상   5.4 > 1.5 × 1.65 = 2.5   → 두 층
        경사면       2.4 > 1.5 × 5.9  = 8.8  → 두 층 아님
    """
    v = np.sort(np.asarray(z, dtype=float))
    v = v[np.isfinite(v)]
    if len(v) < 30:
        return None
    d = np.diff(v)
    i = int(np.argmax(d))
    gap = float(d[i])
    lo_n, hi_n = i + 1, len(v) - i - 1
    if gap < min_gap or min(lo_n, hi_n) < min_frac * len(v):
        return None
    lo, hi = v[:i + 1], v[i + 1:]
    lo_sd, hi_sd = float(np.std(lo)), float(np.std(hi))
    if gap < sep_ratio * max(lo_sd, hi_sd):
        return None      # 연속 분포 속의 작은 빈틈 — 경사면
    lo_med, hi_med = float(np.median(lo)), float(np.median(hi))
    # gap = 값이 하나도 없는 구간의 폭, step = 두 층 중앙값의 차이.
    return dict(gap=gap, step=hi_med - lo_med,
                cut=float(0.5 * (v[i] + v[i + 1])),
                lo_med=lo_med, hi_med=hi_med, lo_n=lo_n, hi_n=hi_n,
                lo_sd=lo_sd, hi_sd=hi_sd, n=len(v))


def _report_roof(x, y, gz, ok, t, frame_w, frame_h, edge_m=5.0):
    """두 층 부지에서 위층(옥상)의 크기·가장자리 비율·프레임 적합도.

    ★ 작은 옥상은 가장자리가 면적의 대부분을 차지한다. 가장자리 바로
      바깥이 낙차(높이차)만큼 아래이고 기준면은 옥상 안쪽에서만 맞으므로,
      가장자리 픽셀을 비스듬한 프레임이 채우면 벽면·난간·아래층이
      옥상 기준면으로 투영되며 `낙차 × k` 만큼 밀린다. 실측:
        극동대 옥상   34 x 27 m   가장자리 5 m 띠 ≈ 56%   낙차 14.2 m
        (비교: Site-2 두 회차는 옥상이 아니라 연속 경사면이었다)

    ★ 프레임이 옥상보다 크면 한 장의 대부분이 아래층이다. 같은 비행에서
        광각 60x45 m  교차 매칭 29%, tie point 0개 55%
        줌   30x22 m             52%,                23%
        열화상 18x15 m           100%,                 0%   ← 옥상 안에 들어감
      다만 이 매칭 차이가 시임 품질로 그대로 옮겨지지는 않았다(광각·줌
      qc_tear 구분 안 됨). 매칭 지표만으로 카메라를 고르지 말 것.

    ★ plane_sweep 을 **모든 프레임**으로 돌리면 특징이 많은 층(대개 지면)에
      맞춘다. 실측: 열화상으로 옥상 기준면을 재니 12곳 중 9곳이 -6 m 범위
      끝(아래층 쪽)에 붙었다.
    ★ 그러나 **옥상 위 연속 9장**에서만 재면 옥상 기준면을 잴 수 있다.
      극동대 실측: 열화상(프레임 18x15 m, 옥상 안에 들어감)으로 재니 신뢰
      9곳이 LRF 옥상 평면보다 +3.8 m 위로 모였고(잔차 0.51 m), 두 후보를
      다시 재 경사 1.58°·c 141.52 가 수렴(잔차 0.35 m)했다. 그 평면을 줌에
      넣자 원래 해상도에서 패널 모듈선이 곧게 이어졌다. 한때 "LRF 옥상 평면을
      쓰라"고 안내했는데, 그 평면은 패널 면보다 약 4 m 낮았다.
    ★ 프레임이 옥상보다 큰 센서는 옥상 프레임만 골라도 지면이 섞여 측정이
      성립하지 않는다(극동대 광각 11/12 · 줌 6/12 곳이 범위 끝, 기복비율
      0.00). 프레임이 옥상 안에 드는 센서로 재고 그 평면을 옮긴다.
    ★ 판정은 원래 해상도로 패널 위를 잘라 눈으로 한다. 옥상 창의 qc_tear 는
      바닥·난간이 섞여, 패널이 맞고 난간이 밀린 결과를 구분하지 못했다
      (10.21 대 11.89%, 구간 겹침).
    """
    hi = ok & (gz > t["cut"])
    if hi.sum() < 5:
        return
    rx, ry = float(np.ptp(x[hi])), float(np.ptp(y[hi]))
    ra = rx * ry
    inner = max(rx - 2 * edge_m, 0.0) * max(ry - 2 * edge_m, 0.0)
    edge_frac = 1.0 - inner / ra if ra > 0 else 1.0
    print()
    print("      위층(옥상) 궤적 %.0f x %.0f m,  가장자리 %.0f m 띠가 면적의 %.0f%%"
          % (rx, ry, edge_m, 100 * edge_frac))
    if edge_frac > 0.4:
        print("      ★ 옥상이 작습니다 — 면적의 대부분이 가장자리입니다.")
        print("        가장자리 바로 밖이 %.1f m 아래라 그 근처 시임이 가장"
              % t["step"])
        print("        나쁩니다(벽면·난간·아래층이 %.1f m × k 만큼 밀림)."
              % t["step"])
        print("        육안 품질은 옥상 안쪽만 보고 판단하고, 결과는")
        print("        옥상 범위로 잘라내십시오.")
    if frame_w:
        print("      프레임 %.0f x %.0f m  vs  옥상 %.0f x %.0f m"
              % (frame_w, frame_h, rx, ry))
        if frame_w > rx or frame_h > ry:
            cover = min(1.0, ra / (frame_w * frame_h))
            print("        → 프레임이 옥상보다 큽니다. 한 장 중 옥상은 최대 %.0f%%."
                  % (100 * cover))
            print("          아래층이 함께 찍혀 매칭이 불리해집니다(시임 품질로")
            print("          그대로 옮겨지지는 않았으니 매칭만으로 판단하지 말 것).")
            print("          같은 비행에 더 좁은 화각(줌·열화상)이 있으면 그쪽으로")
            print("          기준면을 재십시오(아래).")
        else:
            print("        → 프레임이 옥상 안에 들어갑니다.")
    print()
    fits = bool(frame_w) and frame_w <= rx and frame_h <= ry
    # 옥상 위 연속 9장 — plane_sweep 은 시작 프레임부터 연속 8쌍을 쓴다
    idx = np.arange(len(gz))
    runs = [int(i) for i in idx[:-8] if hi[i:i + 9].all()]
    pick = [runs[k] for k in np.linspace(0, len(runs) - 1, min(12, len(runs))).astype(int)] if runs else []
    print()
    print("      ★ 옥상 기준면은 **옥상 위 프레임만으로** plane_sweep 을 돌려 재십시오.")
    print("        모든 프레임으로 재면 지면 쪽에 맞춥니다. LRF 옥상 평면은 출발점일")
    print("        뿐입니다 — 극동대에서는 패널 면보다 약 4 m 낮았습니다.")
    if frame_w:
        if fits:
            print("        → 이 센서는 프레임이 옥상 안에 들어가 **잴 수 있습니다.**")
        else:
            print("        → 이 센서는 프레임이 옥상보다 커서, 옥상 프레임만 골라도 지면이")
            print("          섞여 측정이 성립하지 않습니다(극동대 광각·줌 실측). 같은 비행에서")
            print("          프레임이 옥상 안에 드는 센서(대개 열화상)로 재고 그 평면을")
            print("          이 센서에 옮기십시오. 세 센서는 순번이 같아 아래 목록을 그대로 씁니다.")
    print("        옥상 위 연속 9장 구간 %d곳%s" % (len(runs), "" if len(pick) >= 5 else
          " — ★ 5곳 미만이라 측정이 약합니다. 옥상 위 비행이 짧습니다."))
    if pick:
        print("          export ROOF_STARTS=%s" % ",".join(map(str, pick)))
        print("        (TSV 행 순서 = 사진 파일명 순서일 때의 프레임 번호입니다)")
    m = hi.copy()
    if m.sum() >= 5:
        coef, (ox, oy), sl, rm, mm = fit_plane_trimmed(x, y, gz, m)
        print("        출발점: 위층 LRF 적합 경사 %.2f도, RMSE %.2f m (%d점)"
              % (sl, rm, int(mm.sum())))
        print("          (국소 좌표 근사 — override 는 cameras.npz 좌표로 다시 적합)")
    print("        절차: LRF 옥상 평면으로 한 번 돌림 → sweep <실행> <폴더> -2 10 \"$ROOF_STARTS\"")
    print("              → ovr → 두 후보를 돌려 다시 잼 → 수렴한 평면을 다른 센서에도")
    print("        판정: 원래 해상도로 패널 위를 잘라 눈으로. 옥상 창 qc_tear 는 바닥·")
    print("              난간이 섞여 판정에 쓸 수 없습니다.")


def _report_raised(x, y, gz, ok, frame_w, frame_h, gap=12.0, min_frames=5, edge=15.0):
    """경사지 위 옥상 — 땅 평면을 뺀 높이로 솟은 무리를 찾는다.

    ★ Site-2-29696 · 29719: 경사 26 m 부지 위 건물 옥상의 패널이 땅 기준면보다
      13~18 m 높았다. 절대 높이로는 경사면 높은 쪽 땅과 옥상이 섞여 두 층이 보이지
      않았고(_detect_two_level 불통과), 땅 평면을 빼자 +12~+18 m 에 옥상 무리가
      뚜렷했다. 땅 기준면으로 만든 결과는 옥상 패널이 조각났고, 옥상 기준면(+17 m)을
      윤곽 안에만 덮자(session_tools overlay) 두 센서 모두 이어졌다.
    ★ 촬영 경계에 걸친 옥상은 레이저가 지붕을 한두 번만 맞혀 무리가 되지 않고,
      영상도 비스듬한 프레임에서만 온다 — 판독 제한으로 알린다.
    """
    coef, (ox, oy), sl, rm, m = fit_plane_trimmed(x, y, gz, ok)
    r = gz - (coef[0] * (x - ox) + coef[1] * (y - oy) + coef[2])
    rin = float(np.sqrt((r[m] ** 2).mean())) if m.any() else 0.0
    thr = max(8.0, 4.0 * rin)
    hi = np.where(ok & (r > thr))[0]
    if len(hi) < 1:
        return None
    par = list(range(len(hi)))
    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(len(hi)):
        for j in range(i + 1, len(hi)):
            if np.hypot(x[hi[i]] - x[hi[j]], y[hi[i]] - y[hi[j]]) < gap:
                par[find(i)] = find(j)
    groups = {}
    for i in range(len(hi)):
        groups.setdefault(find(i), []).append(int(hi[i]))
    groups = sorted(groups.values(), key=len, reverse=True)
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    print()
    print("   ★ 땅 평면을 뺀 높이로 보면 **솟은 무리**가 있습니다 (땅 평면보다 %.0f m 넘게 높은 프레임 %d장)."
          % (thr, len(hi)))
    print("     경사지 위 건물 옥상일 수 있습니다 — 절대 높이로는 경사면 높은 쪽 땅과 섞여 두 층으로 보이지 않습니다.")
    main = []
    for k, g in enumerate(groups):
        cx, cy = float(np.mean(x[g])), float(np.mean(y[g]))
        dedge = min(cx - xmin, xmax - cx, cy - ymin, ymax - cy)
        limited = len(g) < min_frames or dedge < edge
        why = []
        if len(g) < min_frames: why.append("레이저가 지붕을 %d번만 맞힘" % len(g))
        if dedge < edge: why.append("촬영 경계에서 %.0f m" % dedge)
        print("     무리 %d: %2d장, 땅 평면보다 %+.1f m (%+.1f ~ %+.1f), 궤적 %.0f x %.0f m%s"
              % (k + 1, len(g), np.median(r[g]), r[g].min(), r[g].max(), np.ptp(x[g]), np.ptp(y[g]),
                 ("  — ★ 판독 제한: " + ", ".join(why)) if limited else ""))
        if not limited:
            main.append(g)
    if not main:
        print("     → 모두 판독 제한입니다. 그 옥상 위를 지나는 비행을 더하는 것이 근본 대책입니다.")
        return dict(thr=thr, groups=groups, main=[])
    sel = np.zeros(len(gz), bool)
    for g in main: sel[g] = True
    for need in (9, 6, 5):
        starts = [int(i) for i in range(len(gz) - need + 1) if sel[i:i + need].all()]
        if len(starts) >= 5: break
    pick = sorted(set(starts[int(round(i))] for i in np.linspace(0, len(starts) - 1, min(12, len(starts))))) if starts else []
    dh = float(np.median(r[sel]))
    print()
    print("     → 땅 기준면은 옥상 패널에 맞지 않습니다(땅 평면보다 %+.0f m). 옥상은 따로 재서 덮으십시오." % dh)
    print("       옥상 위 연속 %d장 구간 %d곳%s" % (need, len(starts), "" if len(pick) >= 5 else " — ★ 5곳 미만, 측정이 약합니다"))
    if pick:
        print("         export ROOF_STARTS=%s" % ",".join(map(str, pick)))
    if frame_w and np.ptp(x[sel]) and (frame_w > np.ptp(x[sel]) + 2 * 5 or frame_h > np.ptp(y[sel]) + 2 * 5):
        print("       프레임 %.0f x %.0f m 가 옥상보다 커서 한 장에 땅이 섞입니다. 기복비율이 낮아 '신뢰 낮음'이 많아도"
              % (frame_w, frame_h))
        print("       최적 높이가 여러 시작 번호에서 한곳에 모이면 그 값을 쓰고, 판정은 패널 타일로 하십시오(29696 실측).")
    print("       절차: 땅 기준 실행 <R> 을 만든 뒤")
    print("         sweep <R> <폴더> %.0f %.0f $ROOF_STARTS     # 땅 기준면에 더할 오프셋을 넓게 탐색"
          % (max(2.0, dh - 10), dh + 8))
    print("         piece <R>_roof <폴더> \"<땅 기준면에서 c 만 최적 오프셋만큼 올린 평면>\"")
    print("         session_tools.py overlay --base <R>/mosaic.tif --top <R>_roof/mosaic.tif \\")
    print("             --exif <이 TSV> --plane=\"<땅 기준면>\" --out <R>_2layer/mosaic.tif")
    print("       다른 센서는 같은 옥상 평면과 같은 윤곽으로 합친 뒤 열화상에 정렬(session_tools align).")
    return dict(thr=thr, groups=groups, main=main, dh=dh, starts=pick)


def fit_plane_trimmed(x, y, z, ok):
    """대칭 절단 최소제곱 — 법면·수목 이상치를 걷어낸다."""
    ox, oy = float(np.mean(x)), float(np.mean(y))
    X, Y = x - ox, y - oy
    m = ok.copy()
    coef = np.zeros(3)
    for _ in range(15):
        if m.sum() < 10:
            break
        A = np.column_stack([X[m], Y[m], np.ones(m.sum())])
        coef, *_ = np.linalg.lstsq(A, z[m], rcond=None)
        r = z - (coef[0] * X + coef[1] * Y + coef[2])
        s = 1.4826 * np.median(np.abs(r[m] - np.median(r[m])))
        if not np.isfinite(s) or s <= 0:
            break
        nm = ok & (np.abs(r - np.median(r[m])) < 2.5 * s)
        if nm.sum() == m.sum():
            m = nm
            break
        m = nm
    r = z - (coef[0] * X + coef[1] * Y + coef[2])
    slope = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
    rmse = float(np.sqrt((r[m] ** 2).mean())) if m.any() else float("nan")
    return coef, (ox, oy), slope, rmse, m


def min_k_map(x, y, height, half_diag, cell=4.0):
    """격자마다 '가장 가까운 카메라까지의 수평거리 / 고도' = 도달 가능한 최소 k.

    ★ **카메라 궤적 안팎을 반드시 나눠 봐야 한다.** 처음에는 모자이크
      전체의 중앙값 하나만 냈는데, 어느 부지에서 0.465 가 나와 "촬영
      부족" 으로 판정했다. 나눠 보니 궤적 안은 0.052(EWP 0.09 보다 우수),
      밖은 0.654 였고 밖이 면적의 75% 였다. 촬영이 부족한 게 아니라
      모자이크가 촬영 범위보다 넓게 잡힌 것이었다.

      판정은 **궤적 안쪽 값**으로 한다. 밖은 잘라내면 되는 문제다.
    """
    try:
        from scipy.spatial import cKDTree, ConvexHull, Delaunay
    except Exception:
        return None
    gx = np.arange(x.min() - half_diag, x.max() + half_diag + cell, cell)
    gy = np.arange(y.min() - half_diag, y.max() + half_diag + cell, cell)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    cam = np.column_stack([x, y])
    d, _ = cKDTree(cam).query(pts, k=1)
    covered = d <= half_diag              # 어떤 프레임에라도 덮이는 자리
    try:
        hull = ConvexHull(cam)
        tri = Delaunay(cam[hull.vertices])
        in_hull = tri.find_simplex(pts) >= 0
        hull_area = float(hull.volume)
    except Exception:
        in_hull = covered
        hull_area = float("nan")
    return dict(k_in=d[covered & in_hull] / height,
                k_out=d[covered & ~in_hull] / height,
                n_cov=int(covered.sum()), hull_area=hull_area)


def analyze_sessions(names, x, y):
    """파일명(DJI_YYYYMMDDhhmmss_NNNN)으로 촬영 세션을 가르고 경계를 본다.

    ★ 아홉 부지 중 네 곳(EWP · K_Demo · 그린환경 · 옥산)이 두 비행이었다.
      EWP 는 배터리 교체 뒤 **정확히 다음 줄**에서 재개해, 경계 두 줄
      (세션 0 마지막 Y 458899, 세션 1 첫 줄 458906, 간격 7.1 m = 평소 간격)이
      각 블록의 가장자리가 됐다. 두 세션을 함께 풀면 경계 줄끼리 약 20분
      차이라 대응이 약하고 노출 이득이 양쪽 clip 에 붙어 이음매가 조각났고,
      세션 하나만 풀면 경계 줄이 가장자리라 기운 복제가 생겼다. 경계만
      좁게 함께 풀었을 때만 깨끗했다.
      K_Demo 는 열화상만 2장(0024·0025)이 빠져 이름순 짝이 2장씩 밀렸다.

    ★ 비행선은 '연속된 직선 구간(leg)'으로 잡는다. 수직 좌표만으로 묶었더니
      217 m 폭에서 휜 줄이 쪼개지고 이륙 뒤 이동 구간이 경계 줄에 섞여,
      EWP 경계 간격을 12.4 m(실제 7.1 m), 시각 차이를 5분(실제 약 20분)으로
      틀리게 냈다. 시각 차이는 경계 두 줄에서 **같은 자리를 찍은 프레임끼리**
      비교한다.
    """
    pat = re.compile(r"DJI_(\d{14})_(\d{4})")
    t, q, base = [], [], []
    for n in names:
        b = os.path.basename(n)
        m = pat.search(b)
        if not m:
            return None
        t.append(datetime.strptime(m.group(1), "%Y%m%d%H%M%S"))
        q.append(int(m.group(2)))
        base.append(b)
    N = len(names)
    sess = np.zeros(N, int)
    missing = []
    for i in range(1, N):
        gap = (t[i] - t[i-1]).total_seconds()
        new = q[i] <= q[i-1] or gap > 120
        sess[i] = sess[i-1] + (1 if new else 0)
        if not new and q[i] > q[i-1] + 1:
            missing.append((int(sess[i]), q[i-1] + 1, q[i] - 1))
    ns = int(sess.max()) + 1

    # 주 비행 방향 — 이웃 프레임 이동 방향의 원형 평균 (180° 접기)
    dx, dy = np.diff(x), np.diff(y)
    use = (np.hypot(dx, dy) > 0.5) & (np.diff(sess) == 0)
    th = np.arctan2(dy[use], dx[use]) * 2
    hrad = 0.5 * np.arctan2(np.sin(th).mean(), np.cos(th).mean()) if use.sum() else 0.0
    hdom = float(np.degrees(hrad)) % 180
    ux, uy = np.cos(hrad), np.sin(hrad)
    u = x * ux + y * uy                                  # 비행 방향 좌표
    v = -x * uy + y * ux                                 # 비행선에 수직인 좌표

    def legs_of(k):
        ids = np.where(sess == k)[0]
        if len(ids) < 6:
            return []
        hx, hy = np.diff(x[ids]), np.diff(y[ids])
        ang = np.degrees(np.arctan2(hy, hx)) % 180
        dev = np.abs(((ang - hdom) + 90) % 180 - 90)
        on = (dev < 20) & (np.hypot(hx, hy) > 0.5)
        legs, cur = [], []
        for j in range(len(on)):
            if on[j]:
                if not cur:
                    cur = [ids[j]]
                cur.append(ids[j+1])
            else:
                if len(cur) >= 5:
                    legs.append(np.array(cur))
                cur = []
        if len(cur) >= 5:
            legs.append(np.array(cur))
        return [(float(np.median(v[L])), L) for L in legs]

    L = {k: legs_of(k) for k in range(ns)}
    cen = np.sort([c for k in L for c, _ in L[k]])
    if len(cen) > 1:
        # 같은 줄의 조각(정지·흔들림으로 끊긴 구간)을 합쳐 줄 중심만 남긴 뒤 간격
        merged = [cen[0]]
        for c in cen[1:]:
            if c - merged[-1] > 1.5:
                merged.append(c)
        d = np.diff(merged)
        spacing = float(np.median(d)) if len(d) else float("nan")
    else:
        spacing = float("nan")

    info = dict(n=ns, sess=sess, heading=0.0 if hdom >= 179.5 else hdom,
                spacing=spacing, missing=missing, pairs=[],
                thermal=sum(b.upper().endswith("_T.JPG") for b in base) > N // 2)
    rows = []
    for k in range(ns):
        m = sess == k
        i0, i1 = np.where(m)[0][[0, -1]]
        rows.append((k, int(m.sum()), t[i0], t[i1], base[i0], base[i1]))
    info["rows"] = rows

    # ★ 복귀·이동 구간 — 주 방향에서 45° 넘게 벗어난 이동이 3장 이상 이어지고
    #   비행선 간격의 1.5배를 넘게 가로지른 구간. 선회(한 칸 이동, 3~5장)는 제외.
    #   EWP: 세션 1 이 끝나고 부지 한가운데를 남북으로 가로지른 24장(0372~0395)이
    #   RGB·열화상 모두 가운데 이음매를 깨뜨렸다. 빼고 한 번에 풀자 둘 다 풀림.
    #   (처음엔 '상대 세션 쪽으로 넘어간 프레임'으로 4장만 잡아 기각했었다.)
    off = set()
    for a, b in zip(range(N - 1), range(1, N)):
        if sess[a] != sess[b]:
            continue
        ddx, ddy = x[b] - x[a], y[b] - y[a]
        if np.hypot(ddx, ddy) < 0.5:
            continue
        ang = np.degrees(np.arctan2(ddy, ddx)) % 180
        if abs(((ang - hdom) + 90) % 180 - 90) > 45:
            off.update((a, b))
    runs, cur = [], []
    for i in range(N):
        if i in off and (not cur or sess[cur[-1]] == sess[i]):
            cur.append(i)
        else:
            if len(cur) >= 3: runs.append(cur)
            cur = [i] if i in off else []
    if len(cur) >= 3: runs.append(cur)
    tr = []
    umin, umax = float(np.min(u)), float(np.max(u))
    for r in runs:
        cross = abs(v[r[-1]] - v[r[0]])
        if np.isfinite(spacing) and cross <= 1.5 * spacing:
            continue
        uf = (np.median(u[r]) - umin) / max(umax - umin, 1e-6)     # 비행 방향으로 부지의 몇 % 지점
        tr.append(dict(sess=int(sess[r[0]]), first=base[r[0]], last=base[r[-1]], n=len(r),
                       t0=t[r[0]].strftime("%H:%M:%S"), t1=t[r[-1]].strftime("%H:%M:%S"),
                       cross=float(cross), mid=bool(0.2 <= uf <= 0.8)))
    info["transit"] = tr
    # 세션 지도 (북쪽이 위): 칸마다 어느 세션의 카메라가 있나
    if ns >= 2:
        st = np.hypot(np.diff(x), np.diff(y)); st = st[(st > 0.5) & (np.diff(sess) == 0)]
        step = float(np.median(st)) if len(st) else 7.0
        sp_ = spacing if np.isfinite(spacing) else step
        nc = int(np.clip(np.ptp(x) / (1.6 * max(step, sp_ * 0.5)), 12, 48))
        nr = int(np.clip(np.ptp(y) / (1.6 * max(step, sp_ * 0.5)), 6, 20))
        xe = np.linspace(x.min(), x.max() + 1e-6, nc + 1)
        ye = np.linspace(y.max() + 1e-6, y.min(), nr + 1)
        ci = np.clip(np.searchsorted(xe, x, side="right") - 1, 0, nc - 1)
        ri = np.clip(np.searchsorted(-ye, -y, side="right") - 1, 0, nr - 1)
        cell = [[set() for _ in range(nc)] for _ in range(nr)]
        for i in range(N):
            cell[ri[i]][ci[i]].add(int(sess[i]))
        sym = lambda st: " " if not st else ("*" if len(st) > 1 else str(min(st)))
        info["map"] = ["".join(sym(cell[r][c]) for c in range(nc)) for r in range(nr)]

    for k in range(ns - 1):
        A, Bl = L[k], L[k+1]
        if not A or not Bl or not np.isfinite(spacing):
            continue
        best = None
        for ca, la in A:
            ua = u[la]
            for cb, lb in Bl:
                ub = u[lb]
                ov = min(ua.max(), ub.max()) - max(ua.min(), ub.min())
                if ov < 20:                     # 비행 방향으로 20 m 이상 함께 지나야 이웃 줄
                    continue
                dv = abs(ca - cb)
                if best is None or dv < best[0]:
                    best = (dv, la, lb)
        if best is None:
            continue
        dv, la, lb = best
        # 다시 찍은 줄: 수직 좌표가 0.35 간격 안이고 **비행 방향으로도 20 m 이상 겹치는** 것.
        # (옥산: 수직 좌표만 보면 1줄이었는데 경계 간격은 8.7 m — 같은 줄 위치를 두 세션이
        #  서로 다른 구간에서 나눠 찍은 것. 경계가 ㄱ자로 꺾인 경우다.)
        rep, split_same = 0, 0
        for cb, lb in Bl:
            ub = u[lb]
            for ca, la in A:
                if abs(ca - cb) >= 0.35 * spacing:
                    continue
                ua = u[la]
                ov = min(ua.max(), ub.max()) - max(ua.min(), ub.min())
                if ov >= 20:
                    rep += 1
                else:
                    split_same += 1
                break
        # 두 세션의 수직 좌표 범위가 두 줄 넘게 겹치면 경계가 비행선과 나란하지 않다
        va = np.array([c for c, _ in A]); vb = np.array([c for c, _ in Bl])
        vov = min(va.max(), vb.max()) - max(va.min(), vb.min())
        # 같은 자리를 찍은 프레임끼리 시각 차이
        dts = []
        for j in lb:
            i = la[np.argmin(np.abs(u[la] - u[j]))]
            dts.append(abs((t[j] - t[i]).total_seconds()))
        gap_min = (t[np.where(sess == k+1)[0][0]] - t[np.where(sess == k)[0][-1]]).total_seconds() / 60
        la, lb = np.sort(la), np.sort(lb)
        info["pairs"].append(dict(
            a=k, b=k+1, repeated=int(rep), split_same=int(split_same),
            v_overlap=float(vov), gap_m=float(dv), minutes=gap_min,
            line_minutes=float(np.median(dts)) / 60,
            a_line=(base[la[0]], base[la[-1]]), b_line=(base[lb[0]], base[lb[-1]])))
    return info


_PIPE_NAME = re.compile(r"_(\d{14})_(\d{4})_")


def pipeline_transit(names, x, y, min_run=3, dev_deg=45.0, cross_factor=1.5):
    """homography_pipeline.py 의 detect_transit 과 **같은 계산** (위도·경도 대신 m 좌표를 받음).

    ★ 파이프라인은 이 판정으로 비행선 밖 구간을 **기본으로 빼고** 처리한다(--drop-transit).
      site_triage 의 기존 판정(6번 위쪽)은 비행 방향 · 간격을 다르게 구해 경계 사례에서
      갈릴 수 있으므로, 처리 전에 '파이프라인이 실제로 뺄 사진'을 같은 계산으로 보여 준다.
    """
    n = len(names)
    seq, tt = [], []
    for nm in names:
        m = _PIPE_NAME.search(os.path.basename(nm))
        seq.append(int(m.group(2)) if m else None)
        tt.append(datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None)
    sess = [0] * n
    for i in range(1, n):
        brk = (seq[i] is not None and seq[i - 1] is not None and seq[i] <= seq[i - 1]) or \
              (tt[i] is not None and tt[i - 1] is not None and (tt[i] - tt[i - 1]).total_seconds() > 120)
        sess[i] = sess[i - 1] + (1 if brk else 0)
    cs = ss = 0.0; moves = []
    for i in range(n - 1):
        if sess[i] != sess[i + 1]: continue
        dx, dy = x[i + 1] - x[i], y[i + 1] - y[i]
        d = math.hypot(dx, dy)
        if d < 0.5: continue
        a_ = math.atan2(dy, dx)
        cs += math.cos(2 * a_) * d; ss += math.sin(2 * a_) * d
        moves.append((i, a_))
    if not moves:
        return None
    h = 0.5 * math.atan2(ss, cs)
    ux, uy = math.cos(h), math.sin(h)
    v = [-xi * uy + yi * ux for xi, yi in zip(x, y)]
    off = set(); on_v = []
    for i, a_ in moves:
        dv = abs(((math.degrees(a_ - h)) + 90) % 180 - 90)
        if dv > dev_deg: off.update((i, i + 1))
        else: on_v.extend((v[i], v[i + 1]))
    # ★ 비행선 간격은 **줄(비행 방향으로 이어진 구간)마다 수직 좌표 중앙값 하나**로 잰다.
    #   예전에는 비행 방향 이동의 모든 점을 1.5 m 안에서 묶었는데, 갈평 줌 비행처럼 줄 안에서
    #   옆으로 흔들리면 한 줄이 여러 조각으로 갈라져 간격이 2.2 m (실제 15.6 m)로 나왔고,
    #   문턱이 3.3 m 가 되어 줄 끝 선회(12 m) 아홉 곳 27장이 복귀 구간으로 빠졌다.
    legs, run = [], []
    for k, (i, a_) in enumerate(moves):
        dv_ = abs(((math.degrees(a_ - h)) + 90) % 180 - 90)
        ok_ = dv_ <= 20.0
        if ok_ and run and moves[k - 1][0] == i - 1 and sess[i] == sess[run[-1]]:
            run.append(i + 1)
        else:
            if len(run) >= 5: legs.append(run)
            run = [i, i + 1] if ok_ else []
    if len(run) >= 5: legs.append(run)
    leg_v = sorted(sorted(v[t] for t in r)[len(r) // 2] for r in legs)
    lines = []; cur = []
    for val in (leg_v if len(leg_v) >= 2 else on_v):
        if cur and val - cur[-1] > 1.5: lines.append(sum(cur) / len(cur)); cur = []
        cur.append(val)
    if cur: lines.append(sum(cur) / len(cur))
    gaps = sorted(b - a_ for a_, b in zip(lines, lines[1:]) if b - a_ > 1.5)
    spacing = gaps[len(gaps) // 2] if gaps else None
    runs, cur = [], []
    for i in range(n):
        if i in off and (not cur or sess[cur[-1]] == sess[i]):
            cur.append(i)
        else:
            if len(cur) >= min_run: runs.append(cur)
            cur = [i] if i in off else []
    if len(cur) >= min_run: runs.append(cur)
    out = []
    for r in runs:
        cross = abs(v[r[-1]] - v[r[0]])
        # ★ 하한 20 m — 간격을 잘못 재도(갈평 2.2 m) 줄 끝 선회(한 칸, 5~16 m)는 걸리지 않게.
        #   EWP 복귀 구간은 110 m 를 가로질러 그대로 잡힌다.
        if cross > max(cross_factor * spacing if spacing else 0.0, 20.0):
            out.append(dict(sess=sess[r[0]], i0=r[0], i1=r[-1], n=len(r), cross=cross,
                            s0=seq[r[0]], s1=seq[r[-1]],
                            first=os.path.basename(names[r[0]]), last=os.path.basename(names[r[-1]])))
    return dict(runs=out, heading=math.degrees(h) % 180, spacing=spacing, nsess=max(sess) + 1)


def print_pipeline_transit(pt, own):
    """파이프라인이 기본으로 뺄 사진과, 직접 지정할 때 쓸 --exclude-frames 값."""
    if pt is None:
        return
    runs = pt["runs"]
    total = sum(r["n"] for r in runs)
    print()
    print("   파이프라인이 기본으로 뺄 사진 (homography_pipeline --drop-transit, 같은 계산 — 비행선 간격 %s):"
          % ("%.1f m" % pt["spacing"] if pt.get("spacing") else "?"))
    if not runs:
        print("     없음 — 기본값이 이 부지의 결과를 바꾸지 않습니다")
    else:
        for r in runs:
            print("     세션 %d  %04d ~ %04d  (%d장, %.0f m 가로지름)   %s ~ %s"
                  % (r["sess"], r["s0"], r["s1"], r["n"], r["cross"], r["first"], r["last"]))
        spec = ",".join(("%d:%d-%d" % (r["sess"], r["s0"], r["s1"])) if pt["nsess"] > 1 else ("%d-%d" % (r["s0"], r["s1"]))
                        for r in runs)
        print("     모두 %d장.  빼지 않으려면 --keep-transit,  같은 사진만 직접 빼려면 --keep-transit --exclude-frames %s"
              % (total, spec))
    mine = sorted((r["first"], r["last"]) for r in (own or []))
    theirs = sorted((r["first"], r["last"]) for r in runs)
    if mine != theirs:
        print("     ★ 위의 비행선 밖 구간 목록과 다릅니다 — 파이프라인은 이 목록대로 뺍니다."
              " 차이가 나는 구간을 눈으로 확인하십시오.")


def print_sessions(info):
    print()
    print("6. 촬영 세션 (파일명의 촬영 시각 · 순번)")
    if info is None:
        print("   파일명이 DJI_YYYYMMDDhhmmss_NNNN 형식이 아니라 판정 불가")
        return
    for k, cnt, t0, t1, b0, b1 in info["rows"]:
        print("   세션 %d  %4d장  %s ~ %s" % (k, cnt, t0.strftime("%H:%M"), t1.strftime("%H:%M")))
    if np.isfinite(info["spacing"]):
        print("   비행선 간격 %.1f m,  비행 방향 %.0f°  (0° = 동서, 90° = 남북)"
              % (info["spacing"], info["heading"]))
    for s0, a, b in info["missing"]:
        print("   ★ 세션 %d 에서 번호 %04d~%04d 누락 (%d장) — 다른 센서와 이름순으로"
              " 짝지으면 그 뒤가 밀립니다" % (s0, a, b, b - a + 1))
    tr = info.get("transit") or []
    if tr:
        print("   ★ 비행선 밖 구간(복귀·이동) %d곳, 모두 %d장:" % (len(tr), sum(r["n"] for r in tr)))
        for r in tr:
            print("     세션 %d  %s ~ %s  (%d장, %s ~ %s, %.0f m 가로지름%s)"
                  % (r["sess"], r["first"], r["last"], r["n"], r["t0"], r["t1"], r["cross"],
                     ", 부지 한가운데" if r["mid"] else ", 가장자리 쪽"))
        if any(r["mid"] for r in tr):
            print("     부지 한가운데를 지난 구간은 모자이크에 자주 뽑혀 패널을 1~2 m 밀어냅니다")
            print("     (EWP 실측). 빼고 처리하십시오:")
            print("       session_tools.py transit --cameras <실행>/cameras.npz --image-dir <사진> --out <새 폴더>")
    elif info["n"] >= 1:
        print("   비행선 밖 구간(복귀·이동) 없음")
    if info["n"] == 1:
        print("   세션 1개 — 경계 문제 없음")
        return
    if "map" in info:
        print("   세션 지도 (북쪽이 위, 숫자 = 세션, * = 두 세션이 한 칸에)")
        for line in info["map"]:
            print("     |%s|" % line)
    for p in info["pairs"]:
        print()
        print("   세션 %d ↔ %d   재개까지 %.0f분,  다시 찍은 줄 %d개,  경계 줄 간격 %.1f m"
              % (p["a"], p["b"], p["minutes"], p["repeated"], p["gap_m"]))
        print("     경계 두 줄의 촬영 시각 차이 %.0f분" % p["line_minutes"])
        print("     경계 줄  세션 %d: %s ~ %s" % (p["a"], *p["a_line"]))
        print("              세션 %d: %s ~ %s" % (p["b"], *p["b_line"]))
        # ★ 옥산: 경계는 곧은데 경계 줄 하나를 두 세션이 구간을 나눠 찍었다(1줄).
        #   이런 들쭉날쭉함은 경계 띠 안에 들어가므로 세 조각 처리가 된다.
        #   두 선 자르기가 안 되는 것은 여러 줄에 걸쳐 꺾일 때(ㄱ자)다.
        partial = p["split_same"] > 0 and p["split_same"] <= 2 and p["v_overlap"] <= 3 * info["spacing"]
        nonpar = (p["split_same"] > 2) or (p["v_overlap"] > 3 * info["spacing"])
        if partial:
            print("   → 경계 줄 %d개를 두 세션이 구간을 나눠 찍었습니다. 경계는 곧고, 나뉜 줄은"
                  % p["split_same"])
            print("     경계 띠 안에 들어가므로 세 조각 처리가 됩니다.")
        elif nonpar:
            print("   ★ 경계가 비행선과 나란하지 않습니다 — %d줄에 걸쳐 두 세션이 구간을 나눠"
                  % p["split_same"])
            print("     찍었습니다. 위 지도에서 경계가 꺾이는지 보십시오. session_tools.py 의")
            print("     두 선 자르기는 이 경우 맞지 않습니다.")
        if p["repeated"] == 0:
            print("   ★ 두 비행이 **겹친 줄 없이** 맞닿았습니다. 경계 두 줄이 각 블록의")
            print("     가장자리라, 한 번에 풀면 경계 이음매가 조각나고 세션별로 풀면")
            print("     경계 쪽이 기울어 보입니다 (EWP 실측).")
            if not nonpar:
                print("     처리: 세션별 모자이크 + 경계 띠(양쪽 3~6줄)만 함께 푼 모자이크를")
                print("           줄 사이 땅에서 잘라 붙이십시오.")
            print("     촬영: 재개할 때 앞 비행의 마지막 한두 줄을 다시 찍으십시오.")
        else:
            print("   → 재개 때 %d줄을 다시 찍었습니다. 세션별로 만들어 겹친 줄 가운데서"
                  % p["repeated"])
            print("     잘라 붙이면 됩니다.")
        if p["line_minutes"] > 10:
            print("   ★ 경계 두 줄이 %.0f분 차이 — 햇빛·반사가 달라 경계에서 색이 조각나고"
                  % p["line_minutes"])
            print("     노출 이득이 양 끝(clip)으로 밀릴 수 있습니다.")
            if info["thermal"]:
                print("     열화상은 패널 온도도 달라지니 세션을 섞지 마십시오.")


def bar(v, ref, better_low, width=22):
    """기준 대비 막대. 기준=1.0 위치에 | 를 찍는다."""
    ratio = (ref / max(v, 1e-9)) if better_low else (v / max(ref, 1e-9))
    n = int(np.clip(ratio, 0, 2) / 2 * width)
    return "#" * n + "." * (width - n) + ("  %.2fx" % ratio)


def main():
    ap = argparse.ArgumentParser(
        description="파이프라인 실행 전 부지 진단 (EXIF 만으로 30초)")
    ap.add_argument("tsv", nargs="?", help="exiftool TSV (lat lon GPSAlt RelAlt LRF)")
    ap.add_argument("--cameras", help="검증용: cameras.npz (투영좌표)")
    ap.add_argument("--alt", help="검증용: alt.tsv (RelAlt LRF GPSAlt)")
    ap.add_argument("--panel-h", type=float, default=1.5,
                    help="패널 높이 (기복변위 환산용, 기본 1.5 m)")
    ap.add_argument("--cell", type=float, default=4.0, help="최소 k 격자 (m)")
    ap.add_argument("--f35", type=float, default=None,
                    help="35mm 환산 초점거리(mm). TSV 에 열이 없을 때 한 장만 "
                         "읽어 지정: exiftool -FocalLengthIn35mmFormat <파일>")
    ap.add_argument("--f-px", type=float, default=None,
                    help="초점거리(px)를 직접 지정. 기존 실행이 있으면 "
                         "summary.json 의 focal_calibration.f_px_final")
    ap.add_argument("--image-w", type=float, default=None,
                    help="영상 가로 픽셀 (기본 4000)")
    ap.add_argument("--image-dir", default=None,
                    help="TSV 를 만든 사진 폴더 — 주면 촬영 세션·누락·경계를 검사")
    a = ap.parse_args()

    if a.cameras and a.alt:
        x, y, alt, rel, lrf, f35, wpx, ncol = load_npz(a.cameras, a.alt)
    elif a.tsv:
        x, y, alt, rel, lrf, f35, wpx, ncol = load(a.tsv)
    else:
        ap.error("TSV 를 주거나 --cameras/--alt 를 함께 주십시오")

    n = len(x)
    ok = np.isfinite(lrf) & (lrf > 5) & (lrf < 300)
    gz = alt - lrf                       # LRF 가 맞힌 지점의 표고
    height = float(np.nanmedian(lrf[ok])) if ok.sum() > 20 else \
        float(np.nanmedian(rel))
    if not np.isfinite(height) or height <= 0:
        sys.exit("비행고도를 구할 수 없습니다 — LRF 도 RelativeAltitude 도 "
                 "읽히지 않았습니다")

    print("=" * 68)
    print("부지 사전 진단 — 프레임 %d장, 비행고도 %.1f m" % (n, height))
    print("=" * 68)

    # ── 1. 촬영 밀도 ────────────────────────────────────────────────
    W = x.max() - x.min()
    H = y.max() - y.min()
    # ── 프레임 지상 크기 — 초점거리에 따라 3배까지 달라진다 ──────────
    if a.image_w:
        wpx = a.image_w
    if not np.isfinite(wpx) or wpx <= 0:
        wpx = 4000.0
    if a.f_px:
        f_px, src = a.f_px, "지정값"
    elif a.f35:
        f_px, src = a.f35 / 36.0 * wpx, "지정 f35 %.1fmm" % a.f35
    elif np.isfinite(f35) and f35 > 0:
        f_px, src = f35 / 36.0 * wpx, "EXIF f35 %.1fmm" % f35
    else:
        # ★ 경고만 내면 계속 놓친다. 실측에서 같은 실수가 세 번 반복됐고,
        #   줌(f35 47mm)을 광각(24mm)으로 가정해 프레임 면적을 6배로
        #   잡는 바람에 3번 항목이 매번 틀린 답을 냈다. 멈춘다.
        # ★ 초점거리에 의존하지 않는 항목은 그대로 내준다. 멈추기만 하고
        #   "유효하다"고 말만 하면 쓸모가 없다. 의존하는 것은 프레임 크기
        #   (덮힘 반경) → 3번 궤적 밖 비율뿐이다.
        f_px, src, frame_w, frame_h = None, None, None, None
        half_diag = None
    if f_px:
        frame_w, frame_h = wpx * height / f_px, 0.75 * wpx * height / f_px
        half_diag = 0.5 * float(np.hypot(frame_w, frame_h))
    print()
    print("0. 카메라")
    if not f_px:
        print("   ★ 초점거리를 읽을 수 없습니다 (TSV %d열, 7열 필요)." % ncol)
        print("     아래 1·2·4·5번은 초점거리와 무관하므로 유효합니다.")
        print("     3번(궤적 밖 비율)만 생략합니다.")
        print()
        print("     전체를 보시려면 아래 중 하나:")
        print("       exiftool ... -FocalLengthIn35mmFormat -ImageWidth ... > exif.tsv")
        print("       site_triage.py exif.tsv --f35 <mm> --image-w <px>")
        print("       site_triage.py exif.tsv --f-px <summary.json 의 f_px_final>")
        # 격자 범위만 정하는 용도의 임시값 — 3번은 생략하므로 판정에 안 쓰인다
        half_diag = 60.0
    else:
        print("   f_px %.0f (%s),  영상 %.0f px" % (f_px, src, wpx))
        print("   프레임 지상 %.1f x %.1f m,  덮힘 반경 %.1f m"
              % (frame_w, frame_h, half_diag))
    if frame_w and frame_w < 50.0:
        print("   ★ 프레임이 좁습니다 (성공 사례 EWP 는 83 m).")
        print("     같은 비행의 광각(_W) 파일이 있으면 그쪽을 쓰십시오.")
        print("     화각이 좁으면 한 장에 담기는 특징이 줄어 매칭이 무너집니다.")
        print("     실측(줌 f35 47mm, 프레임 33 m): 비행선 교차 매칭 12%,")
        print("     프레임의 19%가 tie point 0개, 어긋남 0.65 m.")
        print("     ※ --k-neighbors 를 16 으로 올리는 것은 **해롭습니다**")
        print("       (실측: 내부 51->34%, 교차 12->8%, 어긋남 0.652->0.680 m).")
        print("       후보를 늘리면 겹침 적은 먼 쌍이 성공률을 희석합니다.")
        print("     → 줌이어도 지면이 평탄하면(4번 RMSE < 0.5 m) 결과는")
        print("       좋습니다. 실측: 교차 매칭 1.7% 인데 어긋남 0.047 m.")


    mk = min_k_map(x, y, height, half_diag, a.cell)
    hull_area = mk["hull_area"] if mk else float("nan")
    # ★ 밀도는 **카메라 궤적 면적** 기준. 바운딩박스로 재면 가늘고 긴
    #   궤적에서 과소평가된다.
    dens = 1000.0 * n / hull_area if np.isfinite(hull_area) and hull_area > 0 \
        else float("nan")
    print()
    print("1. 촬영 밀도 (카메라 궤적 면적 기준)")
    if f_px:
        print("   카메라 범위 %.0f x %.0f m,  궤적 면적 %.0f m2,  덮힘 반경 %.0f m"
              % (W, H, hull_area, half_diag))
    else:
        print("   카메라 범위 %.0f x %.0f m,  궤적 면적 %.0f m2"
              % (W, H, hull_area))
    print("   %.1f 프레임/1000m2   %s" % (dens, bar(dens, REF["density"], False)))

    # ── 2. 최소 k — 가장 중요 ──────────────────────────────────────
    print()
    print("2. 도달 가능한 최소 k  ★ 가장 중요")
    if mk is None:
        print("   scipy 가 없어 건너뜁니다 (pip install scipy)")
        med_k = float("nan")
        frac_out = float("nan")
    else:
        ki, ko = mk["k_in"], mk["k_out"]
        med_k = float(np.median(ki)) if len(ki) else float("nan")
        frac_out = len(ko) / max(mk["n_cov"], 1)
        print("   격자 %.1f m,  덮히는 칸 %d" % (a.cell, mk["n_cov"]))
        print("   궤적 안  %5d칸 (%2.0f%%)   중앙값 %.3f  p90 %.3f   %s"
              % (len(ki), 100 * (1 - frac_out), med_k,
                 np.percentile(ki, 90) if len(ki) else float("nan"),
                 bar(med_k, REF["min_k"], True)))
        if len(ko):
            print("   궤적 밖  %5d칸 (%2.0f%%)   중앙값 %.3f  p90 %.3f"
                  % (len(ko), 100 * frac_out,
                     np.median(ko), np.percentile(ko, 90)))
        print("   → 궤적 안 기복변위 중앙값 %.2f m (패널 높이 %.1f m 기준)"
              % (a.panel_h * med_k, a.panel_h))
        if f_px and frac_out > 0.35:
            print("   ★ 모자이크의 %.0f%% 가 카메라가 지나간 적 없는 자리입니다."
                  % (100 * frac_out))
            print("     그 자리의 원형 무늬는 잘라내는 것이 답입니다 — "
                  "설정으로 못 고칩니다.")

    # ── 3. 카메라 밴드 vs 부지 ─────────────────────────────────────
    print()
    print("3. 카메라가 부지를 덮는가")
    if not f_px:
        print("   생략 — 초점거리를 몰라 프레임 크기를 정할 수 없습니다.")
        ratio = float("nan")
        frac_out = float("nan")
    else:
        ratio = 1.0 - frac_out if np.isfinite(frac_out) else float("nan")
    if f_px:
        print("   모자이크 중 카메라 궤적 안 비율 %.2f   %s"
              % (ratio, bar(ratio, REF["band_ratio"], False)))
        print("   → 궤적 + 여유 %.0f m 로 잘라내면 나머지가 사라집니다:"
              % half_diag)
        print("      gdal_translate -projwin <xmin-%.0f> <ymax+%.0f> "
              "<xmax+%.0f> <ymin-%.0f> in.tif out.tif"
              % (half_diag * 0.25, half_diag * 0.25,
                 half_diag * 0.25, half_diag * 0.25))

    # ── 4. LRF 지형 ────────────────────────────────────────────────
    print()
    print("4. 지형 — 단일 평면으로 표현되는가")
    if ok.sum() < 20:
        print("   LRF 유효 %d장뿐 — 판정 불가" % ok.sum())
        rmse = relief = float("nan")
        two_level = None
    else:
        # ★ 계단/옥상 부지: 지면이 **연속 경사**가 아니라 **두 층**일 수 있다.
        #   실측(옥상 부지): 옥상 137.6 m 97장 + 주변 지면 123.4 m 102장,
        #   높이차 14.2 m. 전부 한 평면으로 적합하면 두 층 사이를 대각선으로
        #   가로질러 **경사 24.1도, RMSE 4.85 m** 라는 존재하지 않는 값이
        #   나온다. 각 층을 따로 적합하면 1.8도/1.22 m 와 2.6도/1.59 m 다.
        #   처방이 정반대(경사 보정 vs 한 층만 기준면으로)라 반드시 가른다.
        two_level = _detect_two_level(gz[ok])
        _, _, slope, rmse, m = fit_plane_trimmed(x, y, gz, ok)
        relief = float(np.percentile(gz[ok], 97) - np.percentile(gz[ok], 3))
        print("   LRF 지면 표고 %.1f ~ %.1f m (p3~p97),  기복 %.1f m"
              % (np.percentile(gz[ok], 3), np.percentile(gz[ok], 97), relief))
        print("   전체 범위 %.1f ~ %.1f m" % (gz[ok].min(), gz[ok].max()))
        print("   평면적합 경사 %.3f도,  RMSE %.2f m (inlier %d/%d)   %s"
              % (slope, rmse, m.sum(), ok.sum(),
                 bar(rmse, REF["lrf_rmse"], True)))
        if two_level:
            t = two_level
            print()
            print("   ★★ 지면이 **두 층**으로 갈립니다 — 연속 경사가 아닙니다.")
            print("      아래층 %.1f m (%d장)  |  위층 %.1f m (%d장)"
                  % (t["lo_med"], t["lo_n"], t["hi_med"], t["hi_n"]))
            print("      층 높이차 %.1f m  (그 사이 %.1f m 구간에는 지면이 "
                  "전혀 없음)" % (t["step"], t["gap"]))
            for nm2, sel in (("아래층", gz <= t["cut"]), ("위층", gz > t["cut"])):
                m2 = ok & sel
                if m2.sum() >= 10:
                    _, _, sl2, rm2, _ = fit_plane_trimmed(x, y, gz, m2)
                    print("      %s만 따로 적합: 경사 %.2f도, RMSE %.2f m"
                          % (nm2, sl2, rm2))
            print()
            print("      → 위 '경사 %.3f도 / RMSE %.2f m' 는 두 층 사이를"
                  % (slope, rmse))
            print("        가로지른 **허수입니다.** 경사 보정을 하지 마십시오.")
            print("      → 패널이 있는 층 하나를 기준면으로 잡으십시오.")
            print("        (옥상 설치면 위층. 그 층의 평면을 옥상 프레임으로 재서 P5 override)")
            _report_roof(x, y, gz, ok, t, frame_w, frame_h)
        elif rmse > 1.5:
            print("   ★ RMSE 가 1.5 m 를 넘습니다 — 단일 평면 모델이 "
                  "성립하지 않습니다. 기준면을 어떻게 잡아도 이만큼 남습니다.")
        elif rmse < 0.5:
            print("   → 평면 모델이 잘 맞습니다. 다만 **경사는 반드시 "
                  "plane_sweep 으로 확인하십시오.**")
        else:
            print("   → 평면 모델은 성립하지만 경사는 확인이 필요합니다.")
        raised = None if two_level else _report_raised(x, y, gz, ok, frame_w, frame_h)
        # ★ 한때 "RMSE<1.0 이면 LRF 경사를 쓰라" 고 안내했으나 반증됐다.
        #   실측: LRF 1.288°, 파이프라인 채택 0.830°, plane_sweep 실측
        #   2.237°. 후자를 적용하니 어긋남 0.158 → 0.102 m (35% 개선).
        #   LRF 경사가 맞은 부지도 있었으나(EWP: LRF 1.561 vs 채택 1.504)
        #   일반 규칙으로 쓸 수 없다. plane_sweep 으로 재는 것이 유일하다.

    # ── 5. P7 필요 여부 ────────────────────────────────────────────
    print()
    print("5. RelativeAltitude vs LRF  (P7 패치 필요 여부)")
    if ok.sum() > 20 and np.isfinite(rel).sum() > 20:
        sp_rel = float(np.percentile(rel, 90) - np.percentile(rel, 10))
        sp_lrf = float(np.percentile(lrf[ok], 90) - np.percentile(lrf[ok], 10))
        print("   RelativeAltitude 폭 %.2f m   LRF 폭 %.2f m" % (sp_rel, sp_lrf))
        if sp_rel < 0.5 < sp_lrf:
            print("   ★ RelativeAltitude 에 지형 정보가 없습니다 — "
                  "**P7 필수**. 없으면 초점거리가 잘못된 방향으로 보정됩니다.")
        else:
            print("   두 값이 비슷하게 변합니다 — P7 영향은 작습니다.")
    else:
        print("   판정 불가")

    # ── 6. 촬영 세션 ───────────────────────────────────────────────
    sinfo = None
    if a.image_dir:
        names = sorted(glob.glob(os.path.join(a.image_dir, "*.JPG")) +
                       glob.glob(os.path.join(a.image_dir, "*.jpg")),
                       key=lambda p: os.path.basename(p))
        if len(names) != n:
            print()
            print("6. 촬영 세션")
            print("   ★ 사진 %d장과 TSV %d행이 다릅니다 — 같은 폴더로 TSV 를 만드십시오"
                  % (len(names), n))
        else:
            sinfo = analyze_sessions(names, x, y)
            print_sessions(sinfo)
            ptr = pipeline_transit(names, list(x), list(y))
            print_pipeline_transit(ptr, (sinfo or {}).get("transit"))
            # ★ 비행선 사이 겹침이 작으면 2차 안내 매칭을 권한다 (갈평 RGB 40%: 교차 매칭 2%,
            #   tie point 없는 프레임 81 → --guided-rematch 로 10). 프레임 짧은 변 기준 (보수적).
            try:
                _sp = (ptr or {}).get("spacing")
                if _sp and frame_w and frame_h:
                    _cross = float(min(frame_w, frame_h)); _ov = 1.0 - float(_sp) / _cross
                    print("   비행선 사이 겹침 %.0f%% (간격 %.1f m, 프레임 짧은 변 %.1f m)" % (100 * _ov, _sp, _cross))
                    if _ov < 0.5:
                        print("   ★ 겹침이 50%% 미만입니다 — 비행선 사이 매칭이 약해 tie point 없는 프레임이 많이 생깁니다.")
                        print("     homography_pipeline 에 --guided-rematch 를 주십시오 (갈평 RGB 겹침 40%%: tie point 없는 프레임 81 → 10).")
            except Exception:
                pass
            if sinfo is not None:
                sinfo["pipe_transit"] = ptr

    def _session_tail():
        pt = (sinfo or {}).get("pipe_transit")
        if pt and pt["runs"]:
            print()
            print("  ※ 파이프라인은 비행 방향이 다른 구간 %d장을 기본으로 빼고 처리합니다 (6번 목록)."
                  % sum(r["n"] for r in pt["runs"]))
        if sinfo is not None and any(r["mid"] for r in (sinfo.get("transit") or [])):
            print()
            print("  ※ 부지 한가운데를 가로지른 비행선 밖 구간(복귀·이동)이 있습니다 — 6번의")
            print("    목록을 빼고 처리하십시오. EWP 가운데 이음매가 깨진 주원인이었습니다.")
        if sinfo is None or sinfo["n"] < 2:
            return
        flat = [p for p in sinfo["pairs"] if p["repeated"] == 0]
        print()
        print("  ※ 촬영 세션 %d개%s — 경계 비행선에서 이음매를 따로 확인하십시오."
              % (sinfo["n"], " (겹친 줄 없이 맞닿음)" if flat else ""))
        print("    기준면은 세션마다 plane_sweep 으로 재십시오 (EWP 실측: 자동 기준면이")
        print("    세션마다 2.7~3.7 m 틀렸고, 재면 경계에서 같은 높이로 모였습니다).")

    # ── 종합 ───────────────────────────────────────────────────────
    print()
    print("=" * 68)
    # ★ 촬영 문제와 지형 문제를 나눠야 한다. 처방이 정반대다.
    #   촬영 부족  → 후처리로 못 고침. 기본 설정이 최선 (옥산 실측).
    #   지형 비평면 → 경사 보정이 실제로 듣는다. 실측(경사 21도 산지):
    #     촬영은 좋았는데(밀도 24.0, 최소 k 0.057) 지형만 걸렸고,
    #     평면을 0도 → 13.6도 로 고쳐 어긋남 0.392 → 0.265 m (32%).
    #     그런데 "촬영 부족, 기본이 최선" 으로 판정해 정반대를 권했다.
    shoot, terrain = [], []
    if np.isfinite(med_k) and med_k > 0.12:
        shoot.append("궤적 안 최소 k %.3f (기준 0.09, 한계 0.12)" % med_k)
    if np.isfinite(dens) and dens < 9.0:
        shoot.append("촬영 밀도 %.1f (기준 12.5, 한계 9.0)" % dens)
    if two_level:
        terrain.append("지면이 두 층 (높이차 %.1f m) — 한 층만 기준면으로"
                       % two_level["step"])
    elif np.isfinite(rmse) and rmse > 1.5:
        terrain.append("LRF 평면 RMSE %.2f m (한계 1.5)" % rmse)
    fails = shoot + terrain
    crop = bool(f_px) and np.isfinite(frac_out) and frac_out > 0.35

    if not fails:
        print("★ 판정: 촬영은 충분합니다. 기준면·초점 조정으로 개선 여지가 "
              "있습니다.")
        if crop:
            print("  단, 모자이크의 %.0f%% 가 궤적 밖입니다 — **잘라내십시오.**"
                  % (100 * frac_out))
            print("  그 부분의 원형 무늬는 설정으로 고쳐지지 않습니다.")
        print("  절차 — 기본 실행 → plane_sweep(--starts 10곳 이상) →")
        print("         신뢰지점 오프셋이 ±0.3 m 밖이면 P5 override →")
        print("         qc_tear --window 로 확정")
    elif shoot and terrain:
        print("★ 판정: **촬영도 지형도 한계 밖**. 후처리로 고칠 수 없습니다.")
        for f in fails:
            print("   - " + f)
        print()
        print("  기본 설정으로 한 번 돌리고 궤적 밖은 잘라내십시오.")
        print("  (옥산_1호 실측: 9가지 설정 중 기본을 이긴 것이 없었습니다)")
    elif shoot:
        print("★ 판정: **촬영 부족**. 후처리로 고칠 수 없습니다.")
        for f in shoot:
            print("   - " + f)
        print()
        print("  기본 설정이 최선이었습니다. 한 번 돌리고 궤적 밖은")
        print("  잘라내십시오.")
        print()
        print("  재촬영 조건: 비행선 간격을 절반으로, 부지 짧은 축 전체를")
        print("  덮도록. 그러면 최소 k 가 0.06 대로 내려갑니다.")
    else:
        print("★ 판정: **촬영은 충분, 지형이 평면이 아님.**")
        for f in terrain:
            print("   - " + f)
        print()
        if two_level:
            print("  **전체 경사 보정을 하지 마십시오.** 이 부지는 기울어진 것이")
            print("  아니라 두 층으로 나뉜 것입니다. 패널이 있는 층(옥상이면 위층)의")
            print("  평면을 **그 층 위 프레임만으로** plane_sweep 해서 재고 P5 override")
            print("  로 지정하십시오(4번의 ROOF_STARTS). 프레임이 옥상보다 큰 센서는")
            print("  열화상으로 잰 평면을 옮기십시오. 다른 층은 잘라내는 편이 낫습니다.")
            _session_tail()
            print("=" * 68)
            return
        print("  기본 설정을 쓰지 마십시오 — **경사 보정이 실제로 듣습니다.**")
        print("  실측(경사 21도 산지): 평면 0도 → 13.6도 로 어긋남")
        print("  0.392 → 0.265 m (32%). 다만 잔차는 남습니다.")
        print()
        print("  절차 — 기본 실행 → plane_sweep(--starts 10곳, --z 범위를")
        print("         기복만큼 넓게) → P5 override → 재측정해 수렴 확인")
        print()
        print("  그래도 남는 잔차는 평면 모델의 한계입니다. DSM 기반")
        print("  도구(WebODM 등) 비교를 검토하십시오.")
        if np.isfinite(relief) and relief > 10:
            print()
            print("  촬영 쪽 개선: 기복 %.0f m 부지에서 수평 비행을 하면"
                  % relief)
            print("  고도가 자리마다 달라져 충전율이 떨어집니다. 지형추종")
            print("  (terrain-follow) 비행을 검토하십시오.")
    _session_tail()
    if not a.image_dir:
        print()
        print("  ※ --image-dir <사진 폴더> 를 주면 촬영 세션·누락도 검사합니다.")
    print("=" * 68)


if __name__ == "__main__":
    main()

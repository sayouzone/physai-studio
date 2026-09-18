#!/usr/bin/env python3
"""모자이크의 **국소 불연속**을 이미지 내용으로 직접 잰다.

앞 버전(마스크 기반)을 왜 버렸는가
----------------------------------
패널 마스크의 경계 y 를 열마다 비교하는 방식이었는데, 마스크를 모든
실행에 공통으로 고정했더니 **세 실행의 값이 소수점까지 같아졌습니다.**
지표가 이미지가 아니라 마스크 모양만 재고 있었다는 뜻입니다. 각 실행이
자기 이미지로 마스크를 새로 잡으니 값이 달라 보였을 뿐이고, 실측에서
패널비가 24.1~38.8% 로 흔들리자 순위가 통째로 뒤집혔습니다.

무엇을 재는가
------------
서로 붙어 있는 두 세로 띠(기본 0.72 m)의 밝기 프로파일을 상호상관해서,
오른쪽 띠가 왼쪽 띠에 대해 세로로 얼마나 밀려 있는지 잽니다. 가로 방향도
같은 방식으로 잽니다.

* 정합이 맞으면 이웃한 띠는 거의 안 밀립니다 (0 근처).
* 프레임이 갈리는 자리에서만 계단이 생깁니다.

이미지 내용을 쓰므로 마스크가 필요 없고, 탐색 범위를 **행 피치의 절반
미만**으로 잘라 반복 격자 오매칭(한 줄 건너)에 걸리지 않습니다.

읽는 법
-------
* ``>0.5m`` 비율이 주지표입니다. 눈에 보이는 단차와 가장 잘 맞습니다.
* ``p99`` 와 ``최대`` 를 함께 보십시오. 중앙값이 좋아져도 이 둘이 안
  움직이면 보기에는 그대로입니다.
* ``유효`` 는 상관 신뢰도가 충분해 실제로 잰 지점의 비율입니다. 이 값이
  실행마다 10%p 이상 다르면 비교하지 마십시오.

사용법
------
```bash
python qc_tear.py ab_rgb/*/mosaic.tif --x 237150 --y 458960 --pitch-m 6.0
```

**여러 결과를 비교할 때는 반드시 같은 ``--x/--y`` 구역**을 쓰십시오.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


_last_angle = [0.0]


def read_window(path, X, Y, w, h):
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(path) as ds:
        b, g = ds.bounds, abs(ds.transform[0])
        if X is None or Y is None:
            x = max(0, ds.width // 2 - w // 2)
            y = max(0, ds.height // 2 - h // 2)
        else:
            x = int((X - b.left) / g)
            y = int((b.top - Y) / g)
        w, h = min(w, ds.width - x), min(h, ds.height - y)
        if w < 200 or h < 200:
            return None, None
        a = np.transpose(ds.read(indexes=[1, 2, 3],
                                 window=Window(x, y, w, h)), (1, 2, 0))
    return a, g


def detect_grid_angle(a, gsd):
    """패널 격자가 영상 축에서 몇 도 돌아가 있는지 FFT 로 잰다.

    ★ 옥상 부지 실측: 건물이 비스듬히 앉아 격자가 36도 돌아가 있었다.
      행 프로파일을 영상 가로로 평균내면 그 구조가 뭉개져 상호상관이
      엉뚱한 곳에 붙는다. 같은 모자이크를 0도로 재면 57.6%, 격자
      방향으로 돌려 재면 전혀 다른 값이 나온다.
    """
    n = min(a.shape[0], a.shape[1], 2048)
    y0 = (a.shape[0] - n) // 2
    x0 = (a.shape[1] - n) // 2
    g = cv2.cvtColor(a[y0:y0 + n, x0:x0 + n], cv2.COLOR_RGB2GRAY).astype(np.float32)
    if not np.isfinite(g).all() or g.std() < 1e-3:
        return 0.0
    g = g - g.mean()
    g *= np.outer(np.hanning(n), np.hanning(n))
    F = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    c = n // 2
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - c, xx - c)
    th = np.degrees(np.arctan2(yy - c, xx - c)) % 180
    band = (r > 20) & (r < n * 0.25)
    prof = np.array([F[band & (np.abs(th - t) < 1.0)].sum()
                     for t in range(180)])
    if prof.max() <= 0:
        return 0.0
    best = float(np.argmax(prof))
    # 0~90 으로 접는다 (직교 격자라 90도 차이는 같은 방향)
    return best - 90.0 if best > 45.0 and best <= 135.0 else (
        best - 180.0 if best > 135.0 else best)


def rotate_keep(a, deg):
    """중심 기준 회전. 빈 자리는 0 (유효 마스크가 알아서 걸러낸다)."""
    if abs(deg) < 0.5:
        return a
    h, w = a.shape[:2]
    Mrot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(a, Mrot, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def panel_mask(a):
    """청회색(패널) 대 녹색(식생)."""
    R, G, B = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    m = (a.sum(2) > 0) & (B >= G - 4) & (G >= R - 6)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((7, 7), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                            np.ones((9, 9), np.uint8)).astype(bool)


def _shift_along(g, strip, block, lag, min_corr, mask=None, need=0.5):
    """이웃한 두 띠 사이의 밀림을 상호상관으로 잰다.

    반환: (밀림 픽셀 배열, 시도 횟수)
    """
    n0, n1 = g.shape
    out, tried = [], 0
    if n0 < block + 2 * lag + 4:
        return np.zeros(0), 0
    for i in range(0, n0 - block + 1, max(1, block // 2)):
        blk = g[i:i + block]
        for j in range(0, n1 - 2 * strip + 1, strip):
            tried += 1
            if mask is not None:
                # ★ 패널 위 지점만 센다. 부지 면적의 대부분이 수목·법면인데
                #   거기는 어떤 설정으로도 안 고쳐지므로(18개 실행이 53~62%
                #   안에 몰림) 전체를 재면 패널의 차이가 묻힌다.
                if mask[i:i + block, j:j + 2 * strip].mean() < need:
                    continue
            p = blk[:, j:j + strip].mean(axis=1)
            q = blk[:, j + strip:j + 2 * strip].mean(axis=1)
            if p.std() < 3.0 or q.std() < 3.0:
                continue          # 평탄한 곳은 밀림을 알 수 없다
            p = p - p.mean()
            q = q - q.mean()
            pc = p[lag:len(p) - lag]
            if pc.size < 8:
                continue
            c = np.correlate(q, pc, mode="valid")
            nrm = np.linalg.norm(pc) * np.linalg.norm(q) + 1e-9
            c = c / nrm
            k = int(np.argmax(c))
            if c[k] < min_corr:
                continue          # 상관이 약하면 신뢰할 수 없다
            out.append(abs(k - lag))
    return np.asarray(out, dtype=np.float64), tried


def measure_full(path, pitch_m, strip_m, block_m, min_corr,
                 band_px=3000, panel_only=True, axis="row"):
    """모자이크 **전체**를 가로 밴드로 나눠 훑는다.

    ★ 한 창(3000×2500)만 재면 지점이 350~400개뿐이고, >0.5m 2% 는 지점
      7~8개, >1.0m 0.28% 는 **지점 1개**를 뜻한다. 실측(EWP RGB)에서
      상위 8개 실행이 1.97~2.75% 안에 몰렸는데 그 차이가 지점 두세
      개였고, --panel-top 스윕도 단조롭지 않았다 — 도구 분해능을 넘어
      순위를 매기고 있었다. 전체를 훑어 지점을 수만 개로 늘린다.
    """
    import rasterio
    from rasterio.windows import Window
    parts = []
    tried = 0
    with rasterio.open(path) as ds:
        g = abs(ds.transform[0])
        strip = max(4, int(round(strip_m / g)))
        block = max(48, int(round(block_m / g)))
        lag = max(4, int(round(pitch_m / 3.0 / g)))
        over = block + 2 * lag
        y = 0
        while y < ds.height - over:
            hh = min(band_px, ds.height - y)
            a = np.transpose(ds.read(indexes=[1, 2, 3],
                                     window=Window(0, y, ds.width, hh)),
                             (1, 2, 0))
            y += max(1, hh - over)
            valid = a.sum(2) > 0
            if valid.mean() < 0.15:
                continue
            gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gray[~valid] = float(np.median(gray[valid]))
            pm = panel_mask(a) if panel_only else None
            # ★ 가로 방향은 기본으로 재지 않는다. 탐색 범위를 pitch/3 로
            #   자르는데 그 pitch 는 **행 피치(6 m)** 다. 가로로는 패널 모듈
            #   격자가 약 1 m 주기라 탐색 2 m 가 주기의 두 배가 되어 옆
            #   모듈에 붙는다. 실측(EWP RGB, g_top*): 두 방향을 함께 재면
            #   >0.5m 가 42~44% 로 나왔는데 대부분 그 오매칭이었다.
            got = []
            if axis in ("row", "both"):
                d1, t1 = _shift_along(gray, strip, block, lag, min_corr, pm)
                got.append(d1); tried += t1
            if axis in ("col", "both"):
                d2, t2 = _shift_along(np.ascontiguousarray(gray.T), strip,
                                      block, lag, min_corr,
                                      None if pm is None
                                      else np.ascontiguousarray(pm.T))
                got.append(d2); tried += t2
            parts.append(np.concatenate(got) if got else np.zeros(0))
    if not parts:
        return None
    d = np.concatenate(parts) * g
    if d.size < 500:
        return None
    return _stats(d, tried, g)


def _stats(d, tried, g):
    rng = np.random.default_rng(0)
    # 부트스트랩으로 >0.5m 의 95% 구간을 낸다 — 차이가 잡음인지 도구가 말한다
    bs = np.array([(d[rng.integers(0, d.size, d.size)] > 0.5).mean()
                   for _ in range(200)])
    return dict(n=int(d.size), eff=d.size / max(tried, 1), gsd=g,
                o05=float((d > 0.5).mean()), o10=float((d > 1.0).mean()),
                lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)),
                p50=float(np.percentile(d, 50)),
                p99=float(np.percentile(d, 99)), mx=float(d.max()))


def measure(path, X, Y, w, h, pitch_m, strip_m, block_m, min_corr,
            angle_deg=None):
    a, g = read_window(path, X, Y, w, h)
    if a is None:
        return None
    if angle_deg is None:
        angle_deg = detect_grid_angle(a, g)
    if abs(angle_deg) >= 0.5:
        a = rotate_keep(a, angle_deg)
    _last_angle[0] = float(angle_deg)
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    valid = a.sum(2) > 0
    if valid.mean() < 0.2:
        return None
    gray[~valid] = float(np.median(gray[valid]))

    strip = max(4, int(round(strip_m / g)))
    block = max(48, int(round(block_m / g)))
    # ★ 탐색을 행 피치의 절반 미만으로 자른다. 그보다 넓으면 옆 행에
    #   붙어 버려 계단이 아니라 주기를 재게 된다.
    lag = max(4, int(round(pitch_m / 3.0 / g)))

    dy, t1 = _shift_along(gray, strip, block, lag, min_corr)
    dx, t2 = _shift_along(np.ascontiguousarray(gray.T), strip, block,
                          lag, min_corr)
    d = np.concatenate([dy, dx]) * g
    if d.size < 100:
        return None
    return _stats(d, t1 + t2, g)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mosaics", nargs="+", type=Path)
    ap.add_argument("--x", type=float, default=None)
    ap.add_argument("--y", type=float, default=None)
    ap.add_argument("--size", type=int, nargs=2, default=[4000, 3000],
                    metavar=("W", "H"))
    ap.add_argument("--pitch-m", type=float, default=6.0,
                    help="패널 행 피치 (m). 탐색 범위를 이것의 2.4분의 1로 "
                         "잘라 한 줄 건너 오매칭을 막는다")
    ap.add_argument("--strip-m", type=float, default=0.72)
    ap.add_argument("--block-m", type=float, default=20.0)
    ap.add_argument("--min-corr", type=float, default=0.60)
    ap.add_argument("--axis", choices=["row", "col", "both"], default="row",
                    help="어느 방향의 밀림을 잴지. 기본 row = 행을 가로지르는 "
                         "세로 밀림만. 패널 모듈 격자(약 1 m 주기) 때문에 "
                         "가로 방향은 오매칭에 걸린다")
    ap.add_argument("--all-area", action="store_true",
                    help="패널뿐 아니라 수목·법면까지 전부 잰다. 기본은 "
                         "패널 영역만 — 법면은 어떤 설정으로도 안 고쳐져서 "
                         "함께 재면 패널의 차이가 묻힌다")
    ap.add_argument("--angle-deg", type=float, default=None,
                    help="패널 격자가 영상 축에서 기울어진 각도. 생략하면 "
                         "FFT 로 자동 검출한다. 0 을 주면 회전하지 않는다 "
                         "(--window 모드에서만 동작)")
    ap.add_argument("--window", action="store_true",
                    help="--x/--y 창 하나만 잰다 (빠르지만 지점이 적어 "
                         "0.5%%p 미만 차이는 구분되지 않는다). 기본은 "
                         "모자이크 전체를 훑는다")
    a = ap.parse_args()

    rows = []
    for p in a.mosaics:
        if not p.exists():
            print(f"  없음: {p}")
            continue
        if a.window:
            r = measure(p, a.x, a.y, a.size[0], a.size[1],
                        a.pitch_m, a.strip_m, a.block_m, a.min_corr,
                        angle_deg=a.angle_deg)
            if r is not None:
                r["angle"] = _last_angle[0]
        else:
            r = measure_full(p, a.pitch_m, a.strip_m, a.block_m,
                             a.min_corr, panel_only=not a.all_area,
                             axis=a.axis)
        if r is None:
            print(f"  잴 수 있는 지점이 부족합니다: {p}")
            continue
        r["name"] = p.parent.name or p.stem
        r["path"] = p
        rows.append(r)
    if not rows:
        return 2

    # ★ 이름이 겹치면 어느 행이 어느 파일인지 알 수 없다. 실측: 두 파일을
    #   모두 /tmp 에 두고 비교했더니 둘 다 "tmp" 로 찍혀 판정이 불가능했다.
    #   겹치는 이름은 파일명(stem)으로, 그것도 겹치면 부모/파일명으로 바꾼다.
    seen = {}
    for r in rows:
        seen.setdefault(r["name"], []).append(r)
    for nm, grp in seen.items():
        if len(grp) < 2:
            continue
        for r in grp:
            r["name"] = r["path"].stem
        if len({r["name"] for r in grp}) < len(grp):
            for r in grp:
                r["name"] = f'{r["path"].parent.name}/{r["path"].stem}'

    w = max(len(r["name"]) for r in rows)
    print(f"  {'run':<{w}} {'지점':>9} {'유효':>7} {'>0.5m':>8}"
          f" {'95% 구간':>16} {'>1.0m':>8} {'p99':>7}")
    print("  " + "-" * (w + 60))
    rows = sorted(rows, key=lambda x: x["o05"])
    ang = [r.get("angle") for r in rows if r.get("angle") is not None]
    if ang and max(abs(v) for v in ang) >= 0.5:
        print("  ※ 패널 격자가 영상 축에서 %.0f도 기울어져 있어 그만큼 "
              "회전해 쟀습니다." % np.median(ang))
    for r in rows:
        print(f"  {r['name']:<{w}} {r['n']:9d} {r['eff']*100:6.1f}%"
              f" {r['o05']*100:7.2f}%"
              f"   {r['lo']*100:5.2f}~{r['hi']*100:5.2f}%"
              f" {r['o10']*100:7.2f}% {r['p99']:6.2f}m")
    # 최선과 구간이 겹치는 실행은 구분되지 않는다
    b = rows[0]
    tie = [r["name"] for r in rows[1:] if r["lo"] <= b["hi"]]
    if tie:
        print("")
        print(f"  ★ {b['name']} 과 구간이 겹쳐 **구분되지 않는** 실행: "
              f"{', '.join(tie[:8])}" + (" …" if len(tie) > 8 else ""))
        print("    이들 사이의 순위에 의미를 두지 마십시오.")

    if min(r["eff"] for r in rows) < 0.03:
        print("")
        print("  ★ 유효 지점 비율이 3%% 미만입니다. 표본이 너무 적어")
        print("    구간이 넓어집니다. --block-m 을 줄이거나 --all-area 로")
        print("    범위를 넓히십시오.")
    eff = [r["eff"] for r in rows]
    if max(eff) - min(eff) > 0.10:
        print("")
        print("  ★ 유효 지점 비율이 실행마다 10%p 이상 다릅니다 — 이미지")
        print("    내용 자체가 달라졌다는 뜻이라 비교가 흔들립니다.")
    print("")
    print("  ※ >0.5m 가 주지표입니다. p50 이 좋아져도 >0.5m 와 최대가")
    print("    안 움직이면 보기에는 그대로입니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

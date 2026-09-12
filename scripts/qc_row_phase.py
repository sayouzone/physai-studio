#!/usr/bin/env python3
"""모자이크에서 **패널 행 위상**을 재서 어긋난 자리를 지도로 찍는다.

왜 필요한가
-----------
현재 ``mosaic_qc.assess_mosaic`` 의 ``panel_misalign_median_m`` 는 이
장면의 실패를 **구조적으로 볼 수 없다.**

* ``block_m = 3.8 m`` 인데 실측 패널 행 피치는 **5.99 m** 다. 한 블록이
  한 주기도 담지 못하므로 상호상관이 주기 앨리어싱에 걸린다.
* ``max_shift_m = 1.00 m`` 로 탐색을 자른다. 눈에 보이는 어긋남은
  2~3 m 라서 애초에 측정 범위 밖이다.
* 블록의 **좌/우 절반**을 비교하므로 블록 내부의 전단만 본다. 블록
  전체가 통째로 밀린 경우는 0 으로 나온다.

그래서 "QC 0.035 m, 눈으로는 수 m" 라는 모순이 생긴다. 이 스크립트는
모자이크 전체에서 행 패턴의 **위상**을 직접 재기 때문에 통째로 밀린
구역을 그대로 잡아낸다.

원리
----
패널 행은 거의 완전한 주기 신호다. 세로 방향으로 주기 ``p`` 의 복소
지수를 곱해 국소 평균하면(complex demodulation) 각 위치에서 행 패턴의
위상을 얻는다. 위상 × p / 2π = 그 자리의 행 오프셋(m).

정합이 맞으면 위상은 부지 전체에서 완만하게 변한다. 특정 구역이 통째로
밀렸으면 그 경계에서 **계단**이 생긴다.

사용법
------
```bash
python qc_row_phase.py mosaic.tif --out phase.png
python qc_row_phase.py mosaic.tif --pitch-m 5.99 --out phase.png
```

읽는 법
-------
* 색이 넓게 균일 → 그 구역은 서로 정합돼 있다(절대 정확도와는 무관).
* 좁은 띠를 경계로 색이 바뀜 → 그 띠가 어긋난 프레임 무리다.
* 보고되는 ``step_p95_m`` 이 행 피치의 30% 를 넘으면 반복 패턴
  오매칭(옆 줄에 붙음)을 먼저 의심한다.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np


def load_gray(path: str, max_px: int = 2200):
    import rasterio
    with rasterio.open(path) as ds:
        sc = max(1, int(max(ds.height, ds.width) / max_px))
        a = ds.read(out_shape=(ds.count, ds.height // sc, ds.width // sc))
        gsd = abs(ds.transform[0]) if ds.crs else 1.0
    a = np.transpose(a, (1, 2, 0))
    valid = a[..., :3].sum(axis=2) > 0
    g = cv2.cvtColor(a[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return g.astype(np.float32), valid, gsd * sc


def estimate_pitch(g: np.ndarray, valid: np.ndarray):
    """세로 자기상관 최대점으로 행 피치(px)와 **주기 강도**를 잰다.

    강도 = 자기상관의 피치 지점 값 / 0 지점 값. 깨끗한 사각파(열화상 패널)는
    0.4~0.8 이 나오고, 주기가 묻힌 신호는 0.1 아래로 떨어진다. 이 값이
    낮으면 위상 지표 자체가 성립하지 않는다.
    """
    col = valid.mean(axis=0) > 0.8
    if col.sum() < 50:
        col = valid.mean(axis=0) > 0.5
    p = g[:, col].mean(axis=1)
    p = p - cv2.GaussianBlur(p.reshape(-1, 1), (1, 101), 0).ravel()
    f = np.correlate(p, p, "full")[len(p) - 1:]
    lo = max(4, int(len(p) * 0.005))
    f0 = float(f[0])
    f[:lo] = -np.inf
    k = int(np.argmax(f[:len(f) // 3]))
    strength = float(f[k] / f0) if f0 > 0 else 0.0
    return float(k), strength


def phase_map(g, valid, pitch_px, mask_pct: float = 70.0,
              sigma_x: float = 3.0):
    y = np.arange(g.shape[0])[:, None]
    z = g * np.exp(-1j * 2 * np.pi * y / pitch_px)
    s = cv2.GaussianBlur(np.stack([z.real, z.imag], -1).astype(np.float32),
                         (0, 0), sigmaX=sigma_x, sigmaY=pitch_px * 0.7)
    zc = s[..., 0] + 1j * s[..., 1]
    amp = np.abs(zc)
    off = np.angle(zc) / (2 * np.pi) * pitch_px          # px
    strong = valid & (amp > np.percentile(amp[valid], mask_pct))
    return off, amp, strong


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mosaic")
    ap.add_argument("--out", default="row_phase.png")
    ap.add_argument("--pitch-m", type=float, default=0.0,
                    help="행 피치(m). 0 이면 자동 측정")
    ap.add_argument("--sigma-x", type=float, default=3.0,
                    help="행 방향 평활 (px). 기본 3. RGB 처럼 패널 안 모듈 "
                         "격자가 해상되는 경우 행 피치의 절반쯤(예: 20) 을 "
                         "주면 하부 구조가 눌려 위상이 살아난다")
    ap.add_argument("--mask-pct", type=float, default=70.0,
                    help="주기 진폭 상위 몇 %% 만 볼지 (기본 70). 올리면 "
                         "패턴이 뚜렷한 곳만 남는다")
    ap.add_argument("--max-px", type=int, default=2200,
                    help="분석 해상도 상한 (긴 변 픽셀)")
    a = ap.parse_args()

    g, valid, gsd = load_gray(a.mosaic, a.max_px)
    auto_pitch, strength = estimate_pitch(g, valid)
    pitch_px = (a.pitch_m / gsd) if a.pitch_m > 0 else auto_pitch
    if not (4 < pitch_px < g.shape[0] / 4):
        print(f"행 피치 추정 실패 (={pitch_px:.1f}px). --pitch-m 로 직접 주세요.")
        return 2
    print(f"해상도 {g.shape[1]}x{g.shape[0]}, 유효 GSD {gsd:.3f} m/px")
    print(f"행 피치 {pitch_px:.1f} px = {pitch_px * gsd:.2f} m")
    print(f"주기 강도 {strength:.3f}  (0.25 미만이면 위상 지표가 성립하지 않음)")

    sx = a.sigma_x if a.sigma_x > 0 else 3.0
    off, amp, strong = phase_map(g, valid, pitch_px, a.mask_pct, sx)
    off_m = off * gsd

    # 위상의 국소 계단(gradient) — 어긋남 경계
    b = max(4, int(round(pitch_px * 0.5)))
    ph = np.angle(np.exp(1j * 2 * np.pi * off / pitch_px))
    gx = np.abs(np.angle(np.exp(1j * (np.roll(ph, -b, 1) - ph))))
    gy = np.abs(np.angle(np.exp(1j * (np.roll(ph, -b, 0) - ph))))
    step = np.maximum(gx, gy) / (2 * np.pi) * pitch_px * gsd
    s = step[strong]
    print(f"블록 경계 계단  p50 {np.percentile(s, 50):.3f} m  "
          f"p95 {np.percentile(s, 95):.3f} m  p99 {np.percentile(s, 99):.3f} m")
    frac = float((s > pitch_px * gsd * 0.15).mean())
    print(f"행 피치의 15% 를 넘는 계단 비율: {frac * 100:.2f} %")

    # ---- 유효성 판정 -------------------------------------------------------
    # ★ 위상은 ±pitch/2 밖을 볼 수 없다. 위상이 무작위면 |Δ| 가 [0, π]
    #   균등분포가 되어 p95 → 0.95·pitch/2 로 **포화**한다. 그 상태의
    #   숫자는 어긋남이 아니라 잡음이다. 실측(EWP RGB): p95 2.53,
    #   포화 예측 2.79 — 여섯 실행 전부 붙어 있었고 서로 구분되지 않았다.
    ceil_m = pitch_px * gsd / 2.0
    sat = float(np.percentile(s, 95)) / (0.95 * ceil_m)
    if abs(sx - 3.0) > 1e-6:
        print(f"★ --sigma-x {sx:g} 로 잰 값입니다. 이 옵션은 계단 자체를")
        print("  평활하므로 값의 크기가 달라집니다 (실측: TM 에서 sigma-x 3 →")
        print("  p95 0.197 m, sigma-x 24 → 0.072 m, 같은 모자이크). **같은")
        print("  --sigma-x 로 잰 결과끼리만** 비교하십시오.")
    print(f"포화도 {sat:.2f}  (p95 / 무작위 예측 {0.95 * ceil_m:.2f} m)")
    if strength < 0.25 or sat > 0.70:
        print("")
        print("★ 이 지표를 쓸 수 없습니다.")
        if strength < 0.25:
            print(f"  주기 강도가 {strength:.3f} 로 낮습니다 — 이 모자이크에는")
            print("  행 패턴 자체가 잡히지 않습니다. 피치를 --pitch-m 으로")
            print("  직접 주거나, 이 지표를 포기하십시오.")
        if sat > 0.70:
            print(f"  p95 가 포화 상한({0.95 * ceil_m:.2f} m)의 {sat * 100:.0f}%")
            print("  입니다. 위상이 사실상 무작위라는 뜻이고, 이 상태에서는")
            print("  실행끼리의 차이가 전부 잡음입니다.")
        if strength >= 0.25 and sat > 0.70:
            print("")
            print("  ★ 강도는 충분한데 포화했습니다. 행 패턴은 있지만 그 위에")
            print("    더 잔 구조(RGB 의 모듈 격자·프레임 선)가 얹혀 국소")
            print("    위상을 흔드는 경우입니다. 열화상에서는 안 생기고")
            print("    RGB 에서 생깁니다. --sigma-x 를 올려 보십시오.")
        print("")
        print("  다음을 시도하십시오.")
        print(f"   1) --sigma-x {pitch_px * 0.25:.0f} --mask-pct 85 로 다시.")
        print("      과하게 올리면 진짜 계단까지 뭉갭니다 — 피치의 1/4 부터")
        print("      시작해서 포화도가 0.7 아래로 내려오는 최소값을 쓰십시오.")
        print("   2) 그래도 안 되면 이 지표를 버리고 **직접 측정**으로 가십시오:")
        print("        python scripts/debug_single_frame.py \\")
        print("            --image-dir <RGB> --output-dir /tmp/chk \\")
        print("            --use-ba <출력>/cameras.npz --measure --count 12")
        print("      인접 프레임 간 지상 어긋남을 m 단위로 직접 잽니다.")
        print("      포화가 없고 실행 비교에 그대로 쓸 수 있습니다.")
        print("")
        print("  그림은 저장하니 눈으로도 확인하십시오.")
    elif frac > 0.02:
        print("\n★ 통째로 밀린 구역이 있습니다. 그림에서 색이 갈라지는 띠를")
        print("  보세요. 계단이 행 피치의 30% 이상이면 반복 패턴 오매칭")
        print("  (옆 줄에 붙음)을 먼저 의심하십시오.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lim = pitch_px * gsd / 2
    fig, ax = plt.subplots(1, 2, figsize=(15, 9))
    ax[0].imshow(g, cmap="gray")
    ax[0].set_title("mosaic")
    im = ax[1].imshow(np.where(strong, off_m, np.nan), cmap="twilight",
                      vmin=-lim, vmax=lim)
    ax[1].set_title(f"row phase offset (m), pitch={pitch_px * gsd:.2f} m")
    fig.colorbar(im, ax=ax[1], shrink=0.7)
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.tight_layout()
    fig.savefig(a.out, dpi=100)
    print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

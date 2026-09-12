#!/usr/bin/env python3
"""프레임별 **대응점 배치의 조건수**를 재서 자세가 구속되지 않는 프레임을 찾는다.

왜 개수로는 안 되는가
--------------------
지금까지 약한 프레임을 ``관측 수`` 로만 판정했고, 그 기준으로 한 조치가
모두 빗나갔습니다 (``--min-frame-obs``, 자세 prior 차등 — 후자는 자세오차가
0.366° → 0.748° 로 **나빠졌다**고 코드에 기록돼 있습니다).

이유는 개수가 문제가 아니기 때문입니다. 진입로·도로 위에 중심이 놓인
프레임은 SIFT 대응이 그 **직선을 따라서만** 분포합니다. 개수는 200개여도
도로와 나란한 방향의 위치는 사실상 구속되지 않습니다 (구경문제, aperture
problem). BA 의 자세 prior 는 σ=1° = 지상 0.84 m 이므로, 재투영 비용이
평평한 방향으로 2~3 m 미끄러져도 페널티가 거의 없습니다.

실측 모자이크에서 어긋남이 부지 중앙을 대각선으로 가로지르는 폭 20~25 m
줄기에 몰려 있고, 그 줄기가 진입로 선과 겹칩니다. 이 스크립트는 그 대응
관계를 **숫자와 지도로** 확인합니다.

무엇을 재는가
------------
프레임 *i* 의 관측 영상좌표 집합의 2×2 공분산을 고유분해해서

* ``spread_major`` / ``spread_minor`` = √λ (px) — 대응점 구름의 장축/단축
* ``ratio`` = minor / major — **0 에 가까울수록 한 직선에 몰려 있다**
* ``minor_px`` — 가장 약한 방향의 실제 퍼짐 (px)

``ratio < 0.15`` 이면 퇴화로 봅니다. 개수가 아무리 많아도 이 값이 낮으면
그 프레임의 자세는 영상이 잡아 주지 못합니다.

준비
----
``patch_sayou.py`` 의 **P3** 이 적용된 상태로 파이프라인을 한 번 돌려야
``cameras.npz`` 안에 ``obs_cam`` / ``obs_uv`` 가 생깁니다. 없으면 이
스크립트는 무엇이 빠졌는지 알려주고 멈춥니다.

사용법
------
```bash
python frame_conditioning.py <출력폴더>/cameras.npz \\
    --out-dir <출력폴더>/cond
```

산출물
------
* ``conditioning.csv`` — 프레임별 수치 (전부)
* ``degenerate_frames.txt`` — 퇴화 프레임의 이미지 경로 목록
* ``conditioning_map.png`` — 카메라 위치를 조건수로 칠한 지도.
  **퇴화 프레임이 모자이크의 찢어진 줄기와 겹치는지** 를 여기서 봅니다.
  겹치면 진단이 맞고, 안 겹치면 원인이 다른 데 있습니다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEGEN_RATIO = 0.15


def load(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)
    missing = {"obs_cam", "obs_uv"} - keys
    if missing:
        print(f"cameras.npz 에 {sorted(missing)} 가 없습니다.")
        print("")
        print("  이 파일은 patch_sayou.py 의 P3 이 적용된 파이프라인으로")
        print("  다시 한 번 돌려야 생깁니다.")
        print("")
        print("    python patch_sayou.py --root <설치루트>/sayou --apply")
        print("    python scripts/homography_pipeline.py ... (재실행)")
        print("")
        print(f"  현재 들어 있는 항목: {sorted(keys)}")
        return None
    return (np.asarray(z["cams_opt"], dtype=np.float64),
            np.asarray(z["initial"], dtype=np.float64),
            np.asarray(z["paths"]),
            np.asarray(z["obs_cam"], dtype=np.int64),
            np.asarray(z["obs_uv"], dtype=np.float64))


def conditioning(obs_cam, obs_uv, n_cam):
    """프레임별 (n, major_px, minor_px, ratio)."""
    out = np.zeros((n_cam, 4))
    order = np.argsort(obs_cam, kind="stable")
    oc = obs_cam[order]
    uv = obs_uv[order]
    edges = np.searchsorted(oc, np.arange(n_cam + 1))
    for i in range(n_cam):
        a, b = edges[i], edges[i + 1]
        n = b - a
        out[i, 0] = n
        if n < 3:
            continue
        p = uv[a:b]
        c = np.cov((p - p.mean(axis=0)).T)
        w = np.linalg.eigvalsh(c)
        major, minor = float(np.sqrt(max(w[1], 0))), float(np.sqrt(max(w[0], 0)))
        out[i, 1] = major
        out[i, 2] = minor
        out[i, 3] = minor / major if major > 1e-9 else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cameras_npz", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--ratio", type=float, default=DEGEN_RATIO,
                    help="이 비율 미만이면 퇴화로 본다 (기본 0.15)")
    ap.add_argument("--min-obs", type=int, default=20,
                    help="관측이 이보다 적으면 개수 부족으로 따로 센다")
    a = ap.parse_args()

    if not a.cameras_npz.exists():
        print(f"파일 없음: {a.cameras_npz}")
        return 2
    got = load(a.cameras_npz)
    if got is None:
        return 2
    cams, init, paths, obs_cam, obs_uv = got
    n = len(cams)
    out_dir = a.out_dir or a.cameras_npz.parent / "cond"
    out_dir.mkdir(parents=True, exist_ok=True)

    t = conditioning(obs_cam, obs_uv, n)
    cnt, major, minor, ratio = t[:, 0], t[:, 1], t[:, 2], t[:, 3]

    few = cnt < a.min_obs
    degen = (~few) & (ratio < a.ratio)

    print(f"프레임 {n}장, 관측 {len(obs_cam):,}개")
    print("")
    print(f"  관측 부족 (<{a.min_obs}개)         : {int(few.sum())}장")
    print(f"  퇴화 배치 (ratio<{a.ratio})       : {int(degen.sum())}장"
          "   ← 개수로는 안 잡히는 프레임")
    ok = ~(few | degen)
    if ok.any():
        print(f"  정상                        : {int(ok.sum())}장")
    print("")
    print(f"  ratio  p10 {np.percentile(ratio[cnt > 0], 10):.3f}"
          f"   중앙값 {np.median(ratio[cnt > 0]):.3f}"
          f"   p90 {np.percentile(ratio[cnt > 0], 90):.3f}")

    if degen.any():
        print("")
        print("  퇴화 프레임 상위 15 (ratio 낮은 순)")
        idx = np.argsort(np.where(degen, ratio, 9.0))[:15]
        for i in idx:
            if not degen[i]:
                break
            print(f"    #{i:4d}  관측 {int(cnt[i]):5d}  "
                  f"장축 {major[i]:6.0f}px  단축 {minor[i]:6.0f}px  "
                  f"ratio {ratio[i]:.3f}  {Path(str(paths[i])).name}")

    # ---- 파일 --------------------------------------------------------------
    csv = out_dir / "conditioning.csv"
    with open(csv, "w") as f:
        f.write("frame,n_obs,major_px,minor_px,ratio,degenerate,X,Y,Z,path\n")
        for i in range(n):
            f.write(f"{i},{int(cnt[i])},{major[i]:.2f},{minor[i]:.2f},"
                    f"{ratio[i]:.4f},{int(degen[i])},"
                    f"{cams[i,0]:.3f},{cams[i,1]:.3f},{cams[i,2]:.3f},"
                    f"{paths[i]}\n")
    lst = out_dir / "degenerate_frames.txt"
    with open(lst, "w") as f:
        for i in np.flatnonzero(degen | few):
            f.write(str(paths[i]) + "\n")
    print("")
    print(f"  저장: {csv}")
    print(f"  저장: {lst}  ({int((degen | few).sum())}장)")

    # ---- 지도 --------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        s = ax[0].scatter(cams[:, 0], cams[:, 1], c=np.clip(ratio, 0, 0.6),
                          cmap="viridis", s=14)
        fig.colorbar(s, ax=ax[0], label="minor/major ratio")
        ax[0].set_title("frame conditioning (low = collinear tie points)")
        ax[1].scatter(cams[:, 0], cams[:, 1], c="lightgrey", s=10,
                      label="ok")
        if few.any():
            ax[1].scatter(cams[few, 0], cams[few, 1], c="tab:orange", s=26,
                          label=f"few obs ({int(few.sum())})")
        if degen.any():
            ax[1].scatter(cams[degen, 0], cams[degen, 1], c="tab:red", s=26,
                          label=f"degenerate ({int(degen.sum())})")
        ax[1].legend(loc="best", fontsize=9)
        ax[1].set_title("problem frames vs. torn corridor in the mosaic")
        for x in ax:
            x.set_aspect("equal")
            x.set_xlabel("X (m)")
            x.set_ylabel("Y (m)")
            x.grid(alpha=0.3)
        fig.tight_layout()
        png = out_dir / "conditioning_map.png"
        fig.savefig(png, dpi=100)
        print(f"  저장: {png}")
    except Exception as exc:
        print(f"  지도 생략 ({exc})")

    print("")
    print("★ 다음 단계")
    if degen.sum() >= 5:
        print("  지도에서 빨간 점이 모자이크가 찢어진 줄기 위에 늘어서 있으면")
        print("  진단이 확정됩니다. 그 경우 대책은 두 가지입니다.")
        print("")
        print("  (1) 그 프레임들의 자세 prior 만 조인다 — 근본 대책이지만")
        print("      코드 수정이 필요합니다 (ba_rtk_fixed 의 sig_vec 을")
        print("      개수가 아니라 이 ratio 로 정하도록).")
        print("  (2) 모자이크 승자에서 뺀다 — 즉시 가능합니다. 겹침이")
        print("      충분하면 옆 프레임이 그 자리를 채웁니다.")
        print("      degenerate_frames.txt 의 파일들을 별도 폴더로 옮기고")
        print("      원본 폴더를 --image-dir 로 다시 돌리면 됩니다.")
    else:
        print("  퇴화 프레임이 거의 없습니다. 구경문제 가설은 기각이고,")
        print("  반복 패턴 오매칭(--rtk-match-check) 쪽을 먼저 보십시오.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

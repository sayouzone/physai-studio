#!/usr/bin/env python3
"""기준면 높이를 훑어가며 인접 프레임 어긋남을 재서 **원인을 가른다.**

무엇을 가르는가
--------------
`debug_single_frame --measure` 로 잰 결과가 이랬습니다.

```
BA 이전  0.746 m
BA 이후  0.560 m   (−25%, 최악 쌍 2.563 → 0.501)
```

BA 는 제 몫을 했습니다. 문제는 남은 0.560 m 입니다.

```
ba_reprojection_rmse  2.18 px × 프레임 GSD 0.0161 m/px = 0.035 m
실측 인접 프레임 어긋남                              0.560 m   (16배)
```

재투영은 맞는데 지상은 16배 틀립니다. 삼각측량된 점은 Z 가 자유롭기
때문에 **재투영 오차는 기준면을 전혀 보지 못합니다.** 모자이크가 그
점들을 단일 평면에 눌러 붙일 때 비로소 오차가 나타납니다.

두 가설이 남습니다.

* **자세 오차**  — 겹침 전체가 같은 방향으로 같은 만큼 밀린다. 기준면
  높이를 바꿔도 어긋남이 거의 변하지 않는다. 곡선이 **평평**하다.
* **기복변위**  — 기준면에서 Δh 떨어진 점이 Δh × k 만큼 밀린다. 인접
  프레임의 k 차이는 기선/고도 ≈ 0.23 이므로, 기준면을 맞는 높이로
  옮기면 어긋남이 줄어든다. 곡선에 **최솟값**이 생긴다.

이 스크립트는 기준면 Z 를 훑으며 그 곡선을 그립니다. 한 번에 판별됩니다.

읽는 법
-------
* **뚜렷한 최솟값이 있다** → 기복변위입니다. 최솟값 위치가 실제로 맞는
  기준면 높이이고, 곡선의 깊이가 평면 하나로 얻을 수 있는 상한입니다.
  그 상한이 여전히 크면 평면으로는 안 되고 2층/DSM 이 필요합니다.
* **평평하다 (최고-최저 차이가 20% 미만)** → 기준면이 원인이 아닙니다.
  자세나 시임 쪽을 봐야 합니다.

같이 나오는 ``p90/중앙값`` 도 보십시오. 이 비가 2 를 넘으면 어긋남이
위치마다 다르다는 뜻이라 자세 오차로는 설명되지 않습니다.

사용법
------
```bash
python plane_sweep.py \\
    --image-dir .../RGB \\
    --use-ba .../cameras.npz \\
    --z-min 166 --z-max 174 --z-step 1.0 --count 8
```

``--use-ba`` 를 빼면 초기값(RTK+짐벌)으로 훑습니다.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("plane_sweep")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "label_studio"))


def fit_correction(samples, bp):
    """지점별 최적 오프셋에서 평면 보정 (a, b, c) 을 적합한다.

    오프셋이 위치와 무관하게 일정하면 **높이만** 틀린 것이고, 위치에 따라
    기울어져 있으면 평면의 **경사**도 틀린 것이다. 후자면 높이만 고쳐서는
    부지 한쪽이 반드시 남는다.
    """
    ox, oy = bp["origin_xy"]
    # ★ 계수 3개를 지점 3개로 적합하면 자유도가 0 이라 잔차가 **항상 정확히
    #   0** 이다. 실측(EWP RGB, starts 300/600/800)에서 "잔차 0.00 m — 평면
    #   하나로 부지 전체가 맞습니다" 라는 잘못된 판정이 나왔고, 경사를
    #   1.504° → 5.100° 로 올리라고 권고했다. 지점별 최적 오프셋의 재현
    #   오차가 ±1 m 이므로 최소 5곳은 있어야 잔차가 의미를 갖는다.
    # ★ 가중만으로는 부족했다. 기복비율이 낮은 지점을 **적합에서 완전히
    #   빼야** 한다. 실측(EWP RGB, 10지점): 전부 쓰면 잔차 2.62 m / RMSE
    #   1.30 m, 신뢰 6곳만 쓰면 0.68 m / 0.39 m 로 떨어졌다. 낮은 지점이
    #   경사를 끌어당긴다.
    edge = [s_ for s_ in samples if len(s_) > 6 and s_[6]]
    fit = [s_ for s_ in samples if s_[5] >= 0.6
           and not (len(s_) > 6 and s_[6])]
    dof = len(fit) - 3
    # ★ 적합·판정보다 **먼저** 막아야 한다. 실측(release, 6지점)에서
    #   신뢰 지점이 2곳뿐인데 자유도 -1 로 적합을 수행하고 "잔차 0.00 m,
    #   기준면은 이미 최적" 까지 출력했다. 2점으로 3계수를 적합하면
    #   잔차는 항상 0 이고 아무 정보가 없다.
    if len(fit) < 5:
        logger.info("")
        logger.info("=" * 66)
        logger.info("★ 여러 지점 종합 (%d곳 중 신뢰 %d곳)", len(samples),
                    len(fit))
        # ★ 튜플에 7번째(범위 끝 여부)를 추가했을 때 이 경로의 언패킹을
        #   못 고쳐 ValueError 가 났다. 슬라이스로 받아 길이 변화에 견딘다.
        for s_ in samples:
            x, y, o, m0_, mb_, rel = s_[:6]
            tag = "   ← 범위 끝(제외)" if (len(s_) > 6 and s_[6]) else (
                "   ← 신뢰 낮음" if rel < 0.6 else "")
            logger.info("   X %.0f  Y %.0f  →  최적 %+.2f m  "
                        "(%.3f → %.3f m)  기복비율 %.2f%s",
                        x, y, o, m0_, mb_, rel, tag)
        logger.info("")
        logger.info("  ★ 기복비율 0.6 이상인 지점이 %d곳뿐입니다. 계수 3개를",
                    len(fit))
        logger.info("    적합하려면 최소 5곳이 필요합니다 — **적합도 경사")
        logger.info("    판정도 하지 않습니다.**")
        if fit:
            logger.info("    신뢰 지점의 오프셋: %s  (중앙값 %+.2f m)",
                        ", ".join("%+.2f" % g[2] for g in fit),
                        float(np.median([g[2] for g in fit])))
        logger.info("")
        logger.info("    --starts 를 10곳 이상으로 늘리십시오. 같은 평면에서도")
        logger.info("    지점을 늘리면 신뢰 지점이 늘어납니다 (실측: 6곳→2,")
        logger.info("    10곳→6).")
        logger.info("    경사는 LRF 로 확인하는 편이 낫습니다 — 909점 실측:")
        logger.info("      exiftool -q -n -T -LRFTargetDistance -GPSAltitude "
                    "<RGB>/*.JPG")
        logger.info("=" * 66)
        return

    A = np.array([[s[0] - ox, s[1] - oy, 1.0] for s in fit])
    z = np.array([s[2] for s in fit])
    if len(fit) >= 3:
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        da, db, dc = map(float, coef)
    else:
        da = db = 0.0
        dc = float(np.mean(z)) if len(z) else 0.0
    resid = z - A @ np.array([da, db, dc])

    logger.info("")
    logger.info("=" * 66)
    logger.info("★ 여러 지점 종합 (%d곳)", len(samples))
    for s_ in samples:
        x, y, o, m0, mb, rel = s_[:6]
        tag = "   ← 범위 끝(제외)" if (len(s_) > 6 and s_[6]) else (
            "   ← 신뢰 낮음" if rel < 0.6 else "")
        logger.info("   X %.0f  Y %.0f  →  최적 %+.2f m  "
                    "(%.3f → %.3f m)  기복비율 %.2f%s",
                    x, y, o, m0, mb, rel, tag)
    if edge:
        logger.info("")
        logger.info("  ※ 최적이 탐색 범위 끝인 지점이 %d곳입니다 — "
                    "적합에서 뺐습니다.", len(edge))
        logger.info("    그 지점들은 아직 바닥을 안 지났습니다. "
                    "--z-min/--z-max 를 넓혀 다시 재십시오.")
    weak = sum(1 for s_ in samples if s_[5] < 0.6
               and not (len(s_) > 6 and s_[6]))
    if weak:
        logger.info("")
        logger.info("  ※ 기복비율 0.6 미만이 %d곳입니다. 그 지점은 오차의",
                    weak)
        logger.info("    절반 이상이 기준면과 무관해 최적 오프셋이 흔들립니다.")
    spread = float(np.ptp(z)) if len(z) else 0.0
    rmax = float(np.abs(resid).max()) if len(resid) else 0.0
    logger.info("")
    logger.info("  신뢰 지점 %d곳만으로 적합 — 오프셋 중앙값 %+.2f m, "
                "편차 %.2f m", len(fit), float(np.median(z)), spread)
    logger.info("  적합 잔차 최대 %.2f m, RMSE %.2f m (자유도 %d)",
                rmax, float(np.sqrt((resid ** 2).mean())), dof)
    # 개선 여지 — 이것이 작으면 더 만질 이유가 없다
    m0 = float(np.median([s_[3] for s_ in fit]))
    mb = float(np.median([s_[4] for s_ in fit]))
    logger.info("  신뢰 지점 어긋남: 현재 %.3f m → 각 지점 최적 %.3f m "
                "(%.0f%%)", m0, mb, 100 * (1 - mb / m0) if m0 > 0 else 0)
    if m0 > 0 and (m0 - mb) < 0.05:
        logger.info("")
        logger.info("  ★ 지점마다 최적 평면을 따로 써도 %.3f m 개선입니다.",
                    m0 - mb)
        logger.info("    단일 평면으로는 그보다 못합니다 — **기준면은 이미")
        logger.info("    최적에 가깝고, 더 만질 이유가 없습니다.**")
        logger.info("=" * 66)
        return
    # ★ 기복비율 0.6 미만 지점은 오차의 절반 이상이 기준면과 무관해 최적
    #   오프셋이 ±1 m 흔들린다. 그런 점으로 경사를 적합하면 잡음을 증폭한다.
    #   실측(EWP RGB)에서 이 도구가 1.504° → 3.119° → 5.100° → 2.143° 로
    #   매번 다른 경사를 권고했고, LRF 909점 실측은 1.561° 로 파이프라인의
    #   1.504° 를 지지했다. 경사는 신뢰 지점 5곳 이상일 때만 판정한다.
    if dof < 2:
        logger.info("")
        logger.info("  ★ 지점이 %d곳뿐입니다. 계수 3개를 적합하므로 자유도가",
                    len(samples))
        logger.info("    %d 이고, 잔차 %.2f m 는 **적합이 잘 됐다는 증거가",
                    dof, rmax)
        logger.info("    아닙니다** — 지점 3곳이면 잔차는 항상 0 입니다.")
        logger.info("    경사 보정은 판단하지 않습니다. --starts 로 최소")
        logger.info("    5~6곳을 주고 다시 돌리십시오.")
        logger.info("    지금 쓸 수 있는 것은 오프셋의 중앙값뿐입니다: %+.2f m",
                    float(np.median(z)))
        logger.info("=" * 66)
        return
    if spread < 1.0:
        logger.info("  → 위치와 무관하게 일정합니다. **높이만** 틀렸습니다.")
        logger.info("     ground_plane.c  %.3f → %.3f  (%+.2f m)",
                    bp["c"], bp["c"] + dc, dc)
    else:
        ns = np.degrees(np.arctan(np.hypot(bp["a"] + da, bp["b"] + db)))
        logger.info("  → 위치에 따라 변합니다. **경사도** 틀렸습니다.")
        logger.info("     현재  a %+.6f  b %+.6f  c %.3f   경사 %.3f°",
                    bp["a"], bp["b"], bp["c"], bp.get("slope_deg", 0.0))
        logger.info("     보정  a %+.6f  b %+.6f  c %.3f   경사 %.3f°",
                    bp["a"] + da, bp["b"] + db, bp["c"] + dc, ns)
        logger.info("")
        # ★ 편차가 크다고 곧바로 "단일 평면 불가" 가 아니다. 기울어진
        #   평면 하나로 설명되면 잔차가 작게 남는다. 판정은 **잔차**로 한다.
        if rmax < 0.5:
            logger.info("  잔차가 %.2f m 로 작습니다 — **기울어진 평면 하나로",
                        rmax)
            logger.info("  부지 전체가 맞습니다.** 위 보정 계수를 적용하십시오:")
            logger.info("")
            logger.info("    export SAYOU_PLANE_OVERRIDE=\"%.9f,%.9f,%.6f,"
                        "%.3f,%.3f\"", bp["a"] + da, bp["b"] + db,
                        bp["c"] + dc, ox, oy)
            logger.info("")
            logger.info("  (patch_sayou.py 의 P5 가 적용돼 있어야 합니다)")
        else:
            logger.info("  잔차가 %.2f m 로 큽니다 — 평면 하나로는 부지 한쪽이",
                        rmax)
            logger.info("  남습니다. --layer-surface 를 보십시오.")
    logger.info("=" * 66)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--use-ba", type=Path, default=None)
    p.add_argument("--smoothed", action="store_true",
                   help="cameras.npz 의 cams_smoothed (자세 스무딩 반영) 를 쓴다. "
                        "P4 패치가 적용된 실행에만 있다")
    p.add_argument("--summary", type=Path, default=None,
                   help="파이프라인 summary.json. 주면 그 경사 평면 위에서 "
                        "오프셋을 훑는다. 안 주면 수평면을 훑는데, 경사 1.5° "
                        "부지에서는 위치마다 3~4 m 틀려 값이 부풀려진다")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--starts", default="",
                   help="여러 지점을 한 번에 (예: 100,300,600,800). 지점별 최적 "
                        "오프셋으로 평면 보정 계수를 적합해 준다")
    p.add_argument("--z-min", type=float, default=0.0)
    p.add_argument("--z-max", type=float, default=0.0)
    p.add_argument("--z-step", type=float, default=1.0)
    p.add_argument("--pitch-m", type=float, default=0.0,
                   help="패널 행 피치(m). 주면 그 2.2분의 1을 정답 판정 거리로 "
                        "쓴다 — 한 줄 건너 붙은 대응을 걸러낸다")
    p.add_argument("--inlier-m", type=float, default=1.5)
    p.add_argument("--gsd", type=float, default=0.05)
    p.add_argument("--epsg", type=int, default=5186)
    p.add_argument("--glob", default="*.JPG")
    p.add_argument("--plot", type=Path, default=None)
    a = p.parse_args()

    try:
        import cv2
        from sayou.homography.crs import CRSConverter
        from sayou.image.metadata import extract_metadata
        from sayou.homography.homography import (
            build_frame_homography, intrinsics_from_metadata)
        from sayou.homography.homography.homography import GroundPlane
        from sayou.homography.rtk import estimate_ground_z
        from sayou.homography.pipeline import (
            _build_initial_state, _rotation_from_opk)
    except Exception as exc:
        logger.error("모듈을 불러오지 못했습니다: %s", exc)
        return 2

    images = sorted(Path(a.image_dir).glob(a.glob))
    if not images:
        logger.error("%s 에 %s 가 없습니다", a.image_dir, a.glob)
        return 2
    metas = [extract_metadata(x) for x in images]
    crs = CRSConverter(target_epsg=a.epsg)
    _, cams = _build_initial_state(metas, crs)

    tag = "BA 이전 초기값(RTK+짐벌)"
    if a.use_ba:
        if not a.use_ba.exists():
            logger.error("cameras.npz 가 없습니다: %s", a.use_ba)
            return 2
        z = np.load(a.use_ba, allow_pickle=True)
        key = "cams_smoothed" if a.smoothed else "cams_opt"
        if key not in z.files:
            logger.error("cameras.npz 에 %s 가 없습니다 (있는 항목: %s). "
                         "P4 패치가 적용된 실행이어야 합니다.", key, z.files)
            return 2
        ba = np.asarray(z[key], dtype=np.float64)
        if len(ba) != len(cams):
            logger.error("프레임 수 불일치: npz %d, 이미지 %d", len(ba), len(cams))
            return 2
        cams = ba
        tag = "BA 결과 + 자세 스무딩" if a.smoothed else "BA 결과 (스무딩 전)"

    intr = [intrinsics_from_metadata(m) for m in metas]
    zs = [v for v in (estimate_ground_z(m) for m in metas) if v is not None]
    z_med = float(np.median(zs)) if zs else 0.0
    lim_m = (a.pitch_m / 2.2) if a.pitch_m > 0 else a.inlier_m

    base_plane = None
    if a.summary and a.summary.exists():
        import json
        gp = json.load(open(a.summary)).get("ground_plane") or {}
        if "a" in gp:
            base_plane = gp

    if base_plane is not None:
        grid = (np.arange(a.z_min, a.z_max + 1e-9, a.z_step)
                if (a.z_min or a.z_max)
                else np.arange(-4.0, 8.0 + 1e-9, a.z_step))
        logger.info("기준 평면: summary.json (경사 %.3f°, 원점 표고 %.2f m)",
                    base_plane.get("slope_deg", 0.0), base_plane["c"])
        logger.info("훑는 값: 그 평면에 더할 오프셋 (m)")
    else:
        logger.warning("경사 평면 정보가 없어 **수평면**을 훑습니다 — 경사 "
                       "부지에서는 값이 부풀려집니다. --summary 를 주십시오.")
        zmin = a.z_min or z_med - 4.0
        zmax = a.z_max or z_med + 4.0
        grid = np.arange(zmin, zmax + 1e-9, a.z_step)

    def make_plane(zv):
        if base_plane is None:
            return GroundPlane.horizontal(zv)
        return GroundPlane(a=base_plane["a"], b=base_plane["b"],
                           c=base_plane["c"] + zv,
                           origin_xy=tuple(base_plane["origin_xy"]))

    logger.info("기준: %s", tag)
    logger.info("정답 판정 거리 %.2f m%s", lim_m,
                f"  (행 피치 {a.pitch_m:.2f} / 2.2)" if a.pitch_m > 0 else "")

    sift = cv2.SIFT_create(nfeatures=6000)
    bf = cv2.BFMatcher()
    cache: dict[int, np.ndarray] = {}

    def warp(idx, plane):
        fh = build_frame_homography(cams[idx, :3],
                                    _rotation_from_opk(*cams[idx, 3:6]),
                                    intr[idx], plane)
        b = fh.footprint_bounds()
        if b is None:
            return None, None
        x0, y0, x1, y1 = b
        nx, ny = int((x1 - x0) / a.gsd), int((y1 - y0) / a.gsd)
        if nx < 10 or ny < 10 or nx * ny > 40_000_000:
            return None, None
        if idx not in cache:
            im = cv2.imread(str(metas[idx].origin_path), cv2.IMREAD_UNCHANGED)
            if im is None:
                return None, None
            if im.ndim == 3 and im.shape[2] >= 3:
                im = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY)
            cache[idx] = im
        H = fh.ortho_pixel_matrix(x0, y1, a.gsd)
        return cv2.warpPerspective(
            cache[idx], H.astype(np.float64), (nx, ny),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0), (x0, y1)

    def sweep_one(sel, plot_path=None):
        rows = []
        for zv in grid:
            plane = make_plane(float(zv))
            meds, p90s, fracs, npair = [], [], [], 0
            for i, j in zip(sel[:-1], sel[1:]):
                wa, oa = warp(i, plane)
                wb, ob = warp(j, plane)
                if wa is None or wb is None:
                    continue
                k1, d1 = sift.detectAndCompute(wa, None)
                k2, d2 = sift.detectAndCompute(wb, None)
                if d1 is None or d2 is None:
                    continue
                good = [x for x, y in bf.knnMatch(d1, d2, k=2)
                        if x.distance < 0.7 * y.distance]
                if len(good) < 60:
                    continue
                src = np.float32([k1[x.queryIdx].pt for x in good])
                dst = np.float32([k2[x.trainIdx].pt for x in good])
                d = np.hypot(oa[0] + src[:, 0] * a.gsd
                             - (ob[0] + dst[:, 0] * a.gsd),
                             oa[1] - src[:, 1] * a.gsd
                             - (ob[1] - dst[:, 1] * a.gsd))
                # ★ 중앙값을 그대로 쓰면 안 된다. 반복 격자에서는 대응이 두
                #   덩어리(정답 ≈ 0, 한 줄 건너 ≈ 행 피치)로 갈라지고 중앙값은
                #   어느 쪽이 과반이냐만 알려준다. 실측(EWP RGB, 피치 6.00 m):
                #   오프셋 0 에서 중앙값 5.24 m 가 나왔는데 그건 어긋남이
                #   아니라 오매칭이 과반이라는 뜻이었다.
                inl = d < lim_m
                if inl.sum() < 15:
                    continue
                meds.append(float(np.median(d[inl])))
                p90s.append(float(np.percentile(d[inl], 90)))
                fracs.append(float(inl.mean()))
                npair += 1
            if not meds:
                logger.warning("  오프셋 %+.2f: 잴 수 있는 쌍이 없습니다", zv)
                continue
            rows.append((float(zv), float(np.median(meds)),
                         float(np.median(p90s)), npair, float(np.median(fracs))))
            logger.info("  오프셋 %+6.2f m   정답비율 %5.1f%%   "
                        "정답 중앙값 %.3f m   p90 %.3f m   쌍 %d",
                        zv, rows[-1][4] * 100, rows[-1][1], rows[-1][2], npair)
        if len(rows) < 3:
            logger.error("판정할 만큼 결과가 모이지 않았습니다.")
            return None

        zs_ = np.array([r[0] for r in rows])
        ms = np.array([r[1] for r in rows])
        frs = np.array([r[4] for r in rows])
        # ★ 판정은 **항상 정답 중앙값**으로 한다. 정답비율을 기준으로
        #   쓰면 안 된다 — 어긋남이 클수록 정답비율이 떨어지는 종속 변수라
        #   독립 판별자가 못 된다. 실측(EWP RGB start=100)에서 비율 기준이
        #   진짜 최소 +0.7 을 두고 +7.0 을 골랐다. 비율은 매칭이 아예
        #   깨졌는지 보는 위생 지표로만 쓴다.
        best = int(np.argmin(ms))
        i0 = int(np.argmin(np.abs(zs_)))
        opt = float(zs_[best])
        if 0 < best < len(ms) - 1:      # 포물선 정점으로 보간
            den = ms[best - 1] - 2 * ms[best] + ms[best + 1]
            if abs(den) > 1e-9:
                opt += float((ms[best - 1] - ms[best + 1]) / (2 * den)) \
                    * float(zs_[1] - zs_[0])

        # 좌측 가지의 기울기를 **진짜 최소 기준으로** 잰다.
        sl = (abs(float(np.polyfit(zs_[:best + 1], ms[:best + 1], 1)[0]))
              if best >= 2 else float("nan"))
        # 기하가 예측하는 기울기 = 기선 / 비행고도
        pos = np.array([cams[i, :3] for i in sel])
        base_len = float(np.median(np.hypot(np.diff(pos[:, 0]),
                                            np.diff(pos[:, 1]))))
        zpl = make_plane(0.0).height_at(pos[:, 0].mean(), pos[:, 1].mean())
        H = float(pos[:, 2].mean() - zpl)
        pred = base_len / H if H > 0 else float("nan")
        relief = sl / pred if pred == pred and pred > 0 else float("nan")

        logger.info("")
        logger.info("  최적 오프셋 %+.2f m — 정답 중앙값 %.3f m (정답비율 %.1f%%)",
                    opt, ms[best], frs[best] * 100)
        logger.info("  현재 평면(오프셋 0) %.3f m → %.1f 배 개선 여지",
                    ms[i0], ms[i0] / ms[best] if ms[best] > 0 else float("inf"))
        logger.info("  기울기 실측 %.4f  예측(기선 %.2f m / 고도 %.1f m) %.4f  "
                    "→ 기복 비율 %.2f", sl, base_len, H, pred, relief)
        if relief == relief and relief < 0.6:
            logger.info("    ※ 기복 비율이 낮습니다 — 이 구간 오차의 %.0f%% 는",
                        (1 - relief) * 100)
            logger.info("      기준면과 무관한 성분(자세 등)입니다. 최적 "
                        "오프셋의 신뢰도도 그만큼 낮습니다.")
        if best in (0, len(rows) - 1):
            logger.info("  ★ 최적이 범위 끝입니다. --z-min/--z-max 를 넓히십시오.")
        if frs.max() < 0.5:
            logger.info("  ★ 정답비율이 어디서도 50%% 를 못 넘습니다 — 대응이")
            logger.info("    한 줄 건너 붙고 있습니다. 기준면 문제가 아닙니다.")

        if plot_path:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(zs_, ms, "o-", label="inlier median (m)")
                ax.plot(zs_, [r[2] for r in rows], "s--", alpha=.5,
                        label="inlier p90 (m)")
                ax2 = ax.twinx()
                ax2.plot(zs_, frs * 100, "^-", color="tab:green",
                         label="inlier ratio (%)")
                ax2.set_ylabel("inlier ratio (%)")
                ax2.set_ylim(0, 100)
                ax.axvline(zs_[best], color="tab:red", ls=":",
                           label=f"best {opt:+.2f} m")
                ax.set_xlabel("plane offset (m)")
                ax.set_ylabel("adjacent-frame misalignment (m)")
                ax.grid(alpha=.3)
                ax.legend(loc="upper center")
                fig.tight_layout()
                fig.savefig(plot_path, dpi=100)
                logger.info("  그림: %s", plot_path)
            except Exception as exc:
                logger.info("  그림 생략 (%s)", exc)

        cx = float(np.mean([cams[i, 0] for i in sel]))
        cy = float(np.mean([cams[i, 1] for i in sel]))
        # ★ 최적이 탐색 범위 끝이면 그 값은 '거기서 잘렸다' 는 뜻이지
        #   최적이 아니다. 실측(K_Demo): --z-max 2 에서 세 지점이 +2.00 에
        #   붙었고, 그 세 점이 경사를 끌어당겨 "경사도 틀렸다"(1.442°,
        #   잔차 2.97 m) 는 오판을 만들었다. 같은 부지를 범위만 넓혀
        #   재니 9곳 전부가 -1.8~-2.6 에 모였고 "높이만 틀렸다"(잔차
        #   0.46 m) 가 나왔다. 적합에서 빼야 한다.
        return (cx, cy, opt, float(ms[i0]), float(ms[best]),
                float(relief) if relief == relief else 0.0,
                bool(best in (0, len(rows) - 1)))

    ok = [i for i, k in enumerate(intr) if k is not None]
    starts = [int(v) for v in a.starts.split(",") if v.strip()] or [a.start]
    samples = []
    for st in starts:
        sel = ok[st:st + a.count]
        if len(sel) < 2:
            logger.warning("start=%d: 프레임이 부족합니다", st)
            continue
        if len(starts) > 1:
            logger.info("")
            logger.info("--- start=%d (프레임 %d~%d) ---", st, sel[0], sel[-1])
        r = sweep_one(sel, a.plot if len(starts) == 1 else None)
        if r is not None:
            samples.append(r)

    if len(samples) >= 2 and base_plane is not None:
        fit_correction(samples, base_plane)
    return 0 if samples else 2


if __name__ == "__main__":
    sys.exit(main())

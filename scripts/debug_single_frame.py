#!/usr/bin/env python3
"""프레임 한 장씩 정사보정해 저장 — 워프 문제와 스티칭 문제를 가른다.

왜 필요한가
-----------
EWP-서오창IC-2 에서 **모순**이 계속됐습니다.

```
QC 패널 어긋남 중앙값 : 0.035 m
눈으로 보이는 단차      : 수 m
```

100배 차이입니다. 저는 지금까지 "프레임끼리 어긋난다" 고 **전제**하고
매칭·평면·지형·연직제한을 차례로 의심했고, 제안한 옵션이 모두 빗나갔습니다
(`--terrain-fit`, `--min-frame-obs`, `--guided-matching`).

확인하지 않은 것이 하나 있습니다 — **한 장짜리 워프 자체가 이미 찢어져
있는가.**

* 한 장 안에서 이미 찢어짐 → **워프 문제**. 기준면·기복이 원인이고
  스티칭은 무관합니다.
* 한 장은 멀쩡한데 합치면 찢어짐 → **스티칭 문제**. 매칭·시임을 봐야 합니다.

이 도구는 파이프라인 전체(20분)를 돌리지 않고 **몇 초 만에** 그 판별을
해 줍니다.

사용법
------
```bash
python scripts/debug_single_frame.py \\
    --image-dir .../TM --output-dir /tmp/single \\
    --count 6
```

각 프레임의 정사보정 결과가 개별 GeoTIFF 로 저장됩니다. 두세 장을 열어
**한 장 안에서 패널 행이 곧게 이어지는지** 보십시오.

인접한 두 장을 같은 좌표계에 겹쳐 저장하는 `--pair` 모드도 있습니다.
겹침 영역에서 패널이 어긋나면 그 자리가 곧 스티칭 오차입니다.
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
logger = logging.getLogger("debug_single_frame")
_USE_BA = [False]

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "label_studio"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--count", type=int, default=6,
                   help="저장할 프레임 수 (기본 6)")
    p.add_argument("--start", type=int, default=0,
                   help="몇 번째 프레임부터 (중앙 영역을 보려면 조정)")
    p.add_argument("--gsd", type=float, default=0.0,
                   help="정사영상 GSD (m). 0 이면 자동")
    p.add_argument("--pair", action="store_true",
                   help="인접 두 장을 같은 좌표계에 겹쳐 저장 (스티칭 확인용)")
    p.add_argument("--use-ba", type=Path, default=None,
                   help="파이프라인이 저장한 cameras.npz 를 읽어 **BA 결과**로 "
                        "워프한다. 지정하지 않으면 초기값(RTK+짐벌)을 쓴다. "
                        "--measure 와 함께 쓰면 BA 전후 어긋남을 비교할 수 있다")
    p.add_argument("--measure", action="store_true",
                   help="인접 프레임 간 **지상 어긋남을 직접 측정**해 출력한다. "
                        "정사보정된 두 장에서 SIFT 대응을 찾아 지리좌표 차이를 "
                        "잰다. 이미지를 눈으로 보지 않고 숫자로 확인할 때 사용")
    p.add_argument("--epsg", type=int, default=5186)
    p.add_argument("--glob", default="*.JPG",
                   help="이미지 글로브 패턴 (기본 *.JPG)")
    args = p.parse_args()

    try:
        import cv2
        import rasterio
        from rasterio.transform import from_origin
        # ★ 임포트 경로는 pipeline.py 가 실제로 쓰는 것과 동일하게 맞췄다.
        #   (앞서 추측으로 썼다가 인터페이스 불일치로 실패한 적이 있다.)
        from sayou.homography.crs import CRSConverter
        from sayou.image.metadata import extract_metadata
        from sayou.homography.homography import (
            build_frame_homography, intrinsics_from_metadata, recommend_gsd)
        from sayou.homography.homography.homography import (
            GroundPlane)
        from sayou.homography.rtk import estimate_ground_z
        from sayou.homography.pipeline import (
            _build_initial_state, _rotation_from_opk)
    except Exception as exc:
        logger.error("모듈을 불러오지 못했습니다: %s", exc)
        logger.error("이 스크립트는 파이프라인과 같은 환경에서 실행해야 합니다.")
        return 2

    images = sorted(Path(args.image_dir).glob(args.glob))
    if not images:
        logger.error("%s 에 %s 파일이 없습니다", args.image_dir, args.glob)
        return 2
    metas = [extract_metadata(p) for p in images]
    logger.info("이미지 %d장", len(metas))

    crs = CRSConverter(target_epsg=args.epsg)
    _, cams = _build_initial_state(metas, crs)

    # ★ BA 결과가 있으면 그것으로 워프한다. 초기값과 나란히 재면
    #   "BA 가 자세 오차를 실제로 잡았는가" 가 바로 나온다.
    if args.use_ba:
        # ★ 조용히 초기값으로 되돌아가지 않는다. --use-ba 를 명시했는데
        #   파일이 없거나 안 맞으면 **멈춘다.**
        #   처음엔 경고만 내고 넘어갔는데, 그 결과 before/after 두 실행이
        #   글자 그대로 같은 계산을 해서 "BA 전후가 동일" 이라는 무의미한
        #   결과를 냈다. 조용한 폴백은 시간만 버리게 만든다.
        if not args.use_ba.exists():
            logger.error("cameras.npz 가 없습니다: %s", args.use_ba)
            logger.error("")
            logger.error("이 파일은 **새 pipeline.py 로 파이프라인을 한 번 "
                         "돌려야** 생깁니다.")
            logger.error("경로도 확인하세요 — 파이프라인의 --output-dir 안에 "
                         "생성됩니다.")
            logger.error("  예: <output-dir>/cameras.npz")
            return 2
        try:
            z = np.load(args.use_ba, allow_pickle=True)
            ba = np.asarray(z["cams_opt"], dtype=np.float64)
        except Exception as exc:
            logger.error("cameras.npz 를 읽지 못했습니다: %s", exc)
            return 2
        if len(ba) != len(cams):
            logger.error("cameras.npz 의 프레임 수(%d)가 이미지 수(%d)와 "
                         "다릅니다 — 같은 --image-dir 로 만든 파일인지 "
                         "확인하세요.", len(ba), len(cams))
            return 2
        cams = ba
        _USE_BA[0] = True
        logger.info("BA 결과로 워프합니다 (%s)", args.use_ba.name)
    intr = [intrinsics_from_metadata(m) for m in metas]

    zs = [z for z in (estimate_ground_z(m) for m in metas) if z is not None]
    plane = GroundPlane.horizontal(float(np.median(zs)) if zs else 0.0)
    logger.info("기준면 Z = %.2f m (메타데이터 중앙값)", plane.height_at(0, 0))

    frames = []
    for i, (m, k) in enumerate(zip(metas, intr)):
        if k is None:
            frames.append(None)
            continue
        C = cams[i, :3]
        R = _rotation_from_opk(*cams[i, 3:6])
        frames.append(build_frame_homography(C, R, k, plane))

    ok = [i for i, f in enumerate(frames) if f is not None]
    if not ok:
        logger.error("호모그래피를 만들 수 있는 프레임이 없습니다")
        return 2

    gsd = args.gsd if args.gsd > 0 else recommend_gsd(
        [frames[i] for i in ok])
    logger.info("GSD = %.4f m/px", gsd)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sel = ok[args.start:args.start + args.count]

    def warp_one(idx, bounds=None):
        fh = frames[idx]
        b = bounds or fh.footprint_bounds()
        if b is None:
            return None, None
        x0, y0, x1, y1 = b
        nx = int((x1 - x0) / gsd); ny = int((y1 - y0) / gsd)
        if nx < 10 or ny < 10 or nx * ny > 80_000_000:
            return None, None
        im = cv2.imread(str(metas[idx].origin_path), cv2.IMREAD_UNCHANGED)
        if im is None:
            return None, None
        if im.ndim == 3 and im.shape[2] >= 3:
            im = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2RGB)
        H = fh.ortho_pixel_matrix(x0, y1, gsd)
        w = cv2.warpPerspective(
            im, H.astype(np.float64), (nx, ny),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return w, (x0, y1)

    if args.pair:
        # 인접 두 장을 같은 창에 겹쳐 저장 — 겹침에서의 어긋남이 스티칭 오차
        for a, b in zip(sel[:-1], sel[1:]):
            ba = frames[a].footprint_bounds()
            bb = frames[b].footprint_bounds()
            if ba is None or bb is None:
                continue
            box = (min(ba[0], bb[0]), min(ba[1], bb[1]),
                   max(ba[2], bb[2]), max(ba[3], bb[3]))
            wa, org = warp_one(a, box)
            wb, _ = warp_one(b, box)
            if wa is None or wb is None:
                continue
            # 두 장을 R / G 채널로 겹쳐 어긋남이 색으로 보이게 한다
            ga = cv2.cvtColor(wa, cv2.COLOR_RGB2GRAY) if wa.ndim == 3 else wa
            gb = cv2.cvtColor(wb, cv2.COLOR_RGB2GRAY) if wb.ndim == 3 else wb
            comp = np.dstack([ga, gb, np.zeros_like(ga)])
            out = args.output_dir / f"pair_{a:04d}_{b:04d}.tif"
            _save(out, comp, org, gsd, args.epsg, rasterio, from_origin)
            logger.info("겹침 저장: %s  (빨강=%d번, 초록=%d번 — 색이 갈라지면 "
                        "그 자리가 스티칭 오차)", out.name, a, b)
    else:
        for idx in sel:
            w, org = warp_one(idx)
            if w is None:
                continue
            out = args.output_dir / f"frame_{idx:04d}.tif"
            _save(out, w, org, gsd, args.epsg, rasterio, from_origin)
            logger.info("저장: %s  (%s)", out.name,
                        Path(metas[idx].origin_path).name)

    if args.measure:
        _measure(sel, frames, metas, gsd, warp_one, cv2)

    logger.info("")
    logger.info("★ 판별 방법")
    logger.info("  한 장 안에서 패널 행이 곧게 이어지면 → 워프는 정상,")
    logger.info("     문제는 스티칭(매칭·시임)입니다.")
    logger.info("  한 장 안에서 이미 찢어져 있으면 → 워프 문제이고,")
    logger.info("     기준면·기복을 봐야 합니다. 스티칭은 무관합니다.")
    return 0


def _measure(sel, frames, metas, gsd, warp_one, cv2):
    """인접 프레임 간 지상 어긋남을 SIFT 대응으로 직접 잰다.

    ★ 위상상관(phaseCorrelate)은 이 장면에서 신뢰도가 0.00~0.06 으로
      의미가 없었다 (반복 격자라 상관 피크가 뭉개진다). SIFT 대응의
      지리좌표 차이를 중앙값으로 보는 것이 확실하다.
    """
    sift = cv2.SIFT_create(nfeatures=6000)
    bf = cv2.BFMatcher()
    logger.info("")
    logger.info("인접 프레임 간 지상 어긋남 (SIFT 대응 기준)")
    meds, weak = [], []
    for a, b in zip(sel[:-1], sel[1:]):
        wa, oa = warp_one(a)
        wb, ob = warp_one(b)
        if wa is None or wb is None:
            continue
        ga = cv2.cvtColor(wa, cv2.COLOR_RGB2GRAY) if wa.ndim == 3 else wa
        gb = cv2.cvtColor(wb, cv2.COLOR_RGB2GRAY) if wb.ndim == 3 else wb
        sc = 0.35
        gas = cv2.resize(ga, None, fx=sc, fy=sc)
        gbs = cv2.resize(gb, None, fx=sc, fy=sc)
        k1, d1 = sift.detectAndCompute(gas, None)
        k2, d2 = sift.detectAndCompute(gbs, None)
        if d1 is None or d2 is None:
            continue
        m = bf.knnMatch(d1, d2, k=2)
        good = [x for x, y in m if x.distance < 0.7 * y.distance]
        if len(good) < 20:
            logger.info("  %d-%d: 대응 부족 (%d개)", a, b, len(good))
            continue
        # ★ 대응이 적으면 중앙값 자체가 흔들린다. 실측 IR 에서 대응이
        #   12~51개였고(RGB 는 963~1562개) 그 상태의 어긋남 1.11 m 는
        #   신뢰하기 어렵다. 몇 개로 잰 값인지 항상 함께 본다.
        if len(good) < 60:
            weak.append((a, b, len(good)))
        src = np.float32([k1[x.queryIdx].pt for x in good]) / sc
        dst = np.float32([k2[x.trainIdx].pt for x in good]) / sc
        gx1 = oa[0] + src[:, 0] * gsd; gy1 = oa[1] - src[:, 1] * gsd
        gx2 = ob[0] + dst[:, 0] * gsd; gy2 = ob[1] - dst[:, 1] * gsd
        d = np.hypot(gx2 - gx1, gy2 - gy1)
        meds.append(float(np.median(d)))
        logger.info("  %d-%d: 대응 %d개, 어긋남 중앙값 %.3f m, p90 %.3f m",
                    a, b, len(good), np.median(d), np.percentile(d, 90))
    if weak:
        logger.warning("  ★ 대응이 60개 미만인 쌍이 %d개 있습니다 %s — "
                       "그런 쌍의 어긋남 값은 흔들립니다. 열화상은 특징이 "
                       "적어 흔한 일이며, 이 경우 숫자보다 이미지를 "
                       "보십시오.", len(weak),
                       [f"{a}-{b}({n})" for a, b, n in weak])
    if meds:
        med = float(np.median(meds))
        logger.info("  → 전체 중앙값 %.3f m", med)
        logger.info("")
        logger.info("  기준: %s",
                    "BA 결과" if _USE_BA[0] else "BA 이전 초기값(RTK+짐벌)")
        logger.info("  지상 %.2f m 는 자세 오차 약 %.2f° 에 해당합니다.",
                    med, np.degrees(np.arctan(med / 48.0)))
        logger.info("  최종 모자이크가 이보다 나쁘면 BA 가 이 오차를 "
                    "못 잡고 있는 것입니다.")


def _save(path, arr, origin, gsd, epsg, rasterio, from_origin):
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    tr = from_origin(origin[0], origin[1], gsd, gsd)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w,
                       count=c, dtype=arr.dtype, crs=f"EPSG:{epsg}",
                       transform=tr, compress="deflate") as ds:
        for i in range(c):
            ds.write(arr[:, :, i], i + 1)


if __name__ == "__main__":
    sys.exit(main())

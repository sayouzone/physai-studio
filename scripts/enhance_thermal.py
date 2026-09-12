#!/usr/bin/env python3
"""열화상 대비 향상(CLAHE) — BA 가 자세를 잡을 수 있도록 특징점을 늘린다.

왜 필요한가 (실측으로 확정된 원인)
-----------------------------------
EWP-서오창IC-2 IR 에서 **BA 가 자세 오차를 못 잡고 있습니다.**

`debug_single_frame --measure` 로 BA 전후를 직접 쟀습니다.

| 쌍 | 대응 수 | BA 전 | BA 후 |
|---|---|---|---|
| 0-1 | **51** | 0.411 m | **0.177 m** |
| 3-4 | 24 | 1.108 m | 0.798 m |
| 4-5 | 24 | 1.135 m | 0.919 m |
| 5-6 | 21 | — | **1.585 m** |
| 전체 | | 1.108 m | 0.859 m (−23%) |

**대응 수와 BA 후 어긋남의 상관계수가 −0.86** 입니다. 대응이 51개인 쌍은
0.18 m 까지 잡혔지만, 21~24개인 쌍은 0.8~1.6 m 가 그대로 남습니다.

**BA 는 대응이 충분한 곳만 고칩니다.** 관측이 20개 근처인 프레임은 자세를
붙잡을 근거가 없어 짐벌 초기 오차(약 1.3°)가 거의 그대로 남고, 그런
프레임이 모자이크에서 승자로 뽑힌 자리가 곧 **눈에 보이는 단차**입니다.

RGB 는 같은 측정에서 대응이 963~1,562개였습니다. **열화상은 대비가 낮아
특징이 30~100배 적습니다.** 이것이 근본 원인입니다.

CLAHE 가 얼마나 도움이 되는가
------------------------------
원본 10장의 연속 9쌍에서 측정했습니다.

| 전처리 | 대응 중앙값 | **최소** | 최대 |
|---|---|---|---|
| 원본 | 118 | **37** | 312 |
| **CLAHE(3.0, 8×8)** | **137** | **52** | 484 |
| CLAHE(2.0, 4×4) | 117 | 38 | 491 |

중앙값은 +16% 지만 **최소값이 37 → 52 로 +41%** 입니다. 문제가 되는 것은
평균이 아니라 **대응이 바닥인 쌍**이므로 이쪽이 중요합니다.

사용법
------
```bash
# 1) 전처리본 생성
python scripts/enhance_thermal.py \\
    --input-dir .../TM --output-dir .../TM_clahe

# 2) 그 폴더로 파이프라인 실행
python scripts/homography_pipeline.py \\
    --image-dir .../TM_clahe --output-dir .../out_clahe \\
    --offnadir-frac 0.32 --panel-unit 2
```

주의 — 온도값이 바뀝니다
------------------------
CLAHE 는 국소 대비를 늘리므로 **픽셀값과 온도의 관계가 깨집니다.**

* **정합(BA·모자이크 기하)에는 써도 됩니다** — 특징점만 쓰기 때문입니다.
* **결함 판독용 모자이크로는 쓰지 마십시오** — 핫스팟 판정이 왜곡됩니다.

권장 사용법: 전처리본으로 `cameras.npz` 를 얻고, **원본으로 모자이크를
만들 때 그 카메라 해를 재사용**하는 것입니다. 다만 현재 파이프라인에는
그 옵션이 없으므로, 우선은 **정합이 실제로 좋아지는지 확인**하는 용도로
쓰십시오.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("enhance_thermal")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--glob", default="*.JPG")
    p.add_argument("--clip", type=float, default=3.0,
                   help="CLAHE clipLimit (기본 3.0 — 실측 최적)")
    p.add_argument("--grid", type=int, default=8,
                   help="CLAHE tileGridSize (기본 8 — 실측 최적)")
    p.add_argument("--keep-exif", action="store_true", default=True,
                   help="EXIF/XMP 를 원본에서 복사 (RTK·짐벌 값 보존). "
                        "기본 켜짐 — 이게 없으면 파이프라인이 못 읽습니다")
    args = p.parse_args()

    try:
        import cv2
    except Exception as exc:
        logger.error("OpenCV 를 불러오지 못했습니다: %s", exc)
        return 2

    # ★ EXIF 복사 수단을 **먼저** 확인한다. 이미지를 다 만든 뒤에 실패하면
    #   메타데이터 없는 폴더가 남아, 그것을 파이프라인에 넣으면
    #   `모든 프레임의 footprint 계산 실패` 로 죽는다 (실제로 그렇게 됐다).
    have_exiftool = shutil.which("exiftool") is not None
    # ★ piexif 는 **쓸 수 없다.** EXIF 는 옮기지만 DJI 의 자세·고도는
    #   **XMP** 에 들어 있는데 piexif 는 XMP 를 못 옮긴다. 실측 확인:
    #     EXIF        출력본=True  원본=True
    #     DJI XMP     출력본=False 원본=True   ← 없음
    #     GimbalYaw   출력본=False 원본=True   ← 없음
    #   그 상태로 파이프라인을 돌리면 자세를 못 읽어
    #   `모든 프레임의 footprint 계산 실패` 로 죽는다.
    #   exiftool 만이 XMP 를 통째로 옮길 수 있다.
    if args.keep_exif and not have_exiftool:
        logger.error("exiftool 이 없습니다 — **중단합니다.**")
        logger.error("")
        logger.error("  이대로 진행하면 메타데이터 없는 이미지가 생기고,")
        logger.error("  파이프라인이 '모든 프레임의 footprint 계산 실패' 로")
        logger.error("  죽습니다. (실제로 그렇게 됐습니다)")
        logger.error("")
        logger.error("  DJI 의 짐벌 자세·상대고도는 **XMP** 에 있어서")
        logger.error("  exiftool 이 아니면 옮길 수 없습니다 (piexif 로는 안 됨).")
        logger.error("")
        logger.error("    brew install exiftool")
        return 2

    files = sorted(Path(args.input_dir).glob(args.glob))
    if not files:
        logger.error("%s 에 %s 파일이 없습니다", args.input_dir, args.glob)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("이미지 %d장 → %s", len(files), args.output_dir)

    clahe = cv2.createCLAHE(clipLimit=args.clip,
                            tileGridSize=(args.grid, args.grid))
    n_ok = 0
    for f in files:
        im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if im is None:
            logger.warning("읽지 못함: %s", f.name)
            continue
        if im.ndim == 3:
            # 컬러맵이 입혀진 열화상 — 밝기 채널에만 적용해 색상은 보존
            lab = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            out = clahe.apply(im)
        dst = args.output_dir / f.name
        cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 97])
        n_ok += 1

    logger.info("변환 완료: %d장", n_ok)

    if args.keep_exif:
        # ★ EXIF/XMP 가 없으면 파이프라인이 RTK·짐벌을 못 읽어 **아무것도
        #   되지 않는다.** OpenCV 는 메타데이터를 버리므로 반드시 복사한다.
        if have_exiftool:
            import subprocess
            logger.info("exiftool 로 EXIF/XMP 복사 중...")
            r = subprocess.run(
                ["exiftool", "-TagsFromFile", str(args.input_dir) + "/%f.JPG",
                 "-all:all", "-overwrite_original", str(args.output_dir)],
                capture_output=True, text=True)
            if r.returncode == 0:
                logger.info("EXIF/XMP 복사 완료")
            else:
                logger.error("exiftool 실패: %s", r.stderr.strip()[:300])
                return 2
        else:
            logger.error("")
            logger.error("★ exiftool 이 없습니다. **이대로는 파이프라인이 "
                         "동작하지 않습니다** — RTK·짐벌 메타데이터가 "
                         "사라졌기 때문입니다.")
            logger.error("  설치: brew install exiftool")
            logger.error("  설치 후 이 스크립트를 다시 실행하십시오.")
            return 2

    _verify(args.output_dir, args.glob, logger)
    return _done(args, logger)


def _verify(out_dir, glob_pat, logger):
    """EXIF 가 실제로 들어갔는지 **확인한다.** 복사 명령이 성공을 보고해도
    실제로 안 들어가는 경우가 있어, 파일을 직접 열어 본다."""
    from pathlib import Path as _P
    # ★ EXIF 만 보면 안 된다. DJI 자세·고도는 XMP 에 있고, 그게 없으면
    #   파이프라인이 죽는다. 실제로 piexif 는 EXIF 만 옮겨서 통과해 놓고
    #   XMP 가 빠져 있었다. 둘 다 확인한다.
    bad = []
    for f in sorted(_P(out_dir).glob(glob_pat))[:20]:
        try:
            b = open(f, "rb").read(65536)
            if b"Exif" not in b or b"drone-dji" not in b:
                bad.append(f.name)
        except Exception:
            bad.append(f.name)
    if bad:
        logger.error("★ EXIF 또는 DJI XMP 가 빠진 파일이 있습니다: %s",
                     bad[:5])
        logger.error("  이대로 파이프라인에 넣으면 'footprint 계산 실패' 로 "
                     "죽습니다. 이 폴더를 쓰지 마십시오.")
    else:
        logger.info("EXIF + DJI XMP 확인 완료 — 파이프라인에 넣어도 됩니다")


def _done(args, logger):
    logger.info("")
    logger.info("★ 다음 단계")
    logger.info("  python scripts/homography_pipeline.py \\\\")
    logger.info("      --image-dir %s \\\\", args.output_dir)
    logger.info("      --output-dir <출력> --offnadir-frac 0.32 --panel-unit 2")
    logger.info("")
    logger.info("  확인할 것: observations_per_frame 의 median 과 p10 이")
    logger.info("  올라가고 zero_frames 가 줄어드는지.")
    logger.info("")
    logger.info("  ※ CLAHE 는 온도값을 바꿉니다. 결함 판독용 모자이크가 아니라")
    logger.info("    **정합이 좋아지는지 확인하는 용도**로만 쓰십시오.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

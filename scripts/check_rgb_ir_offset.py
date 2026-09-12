#!/usr/bin/env python3
"""RGB → IR 자세 전이가 성립하는지 **측정만** 한다.

배경
----
EWP-서오창IC-2 IR 에서 **BA 가 대응 부족한 프레임의 자세를 못 잡습니다.**

| 대응 수 | BA 후 인접 프레임 어긋남 |
|---|---|
| 51 | 0.177 m |
| 24 | 0.798 m |
| 24 | 0.919 m |
| 21 | 1.585 m |

**상관계수 −0.86.** RGB 는 같은 측정에서 대응이 963~1,562개라 이 문제가
없습니다 (열화상은 대비가 낮아 특징이 30~100배 적습니다).

그래서 **RGB 로 카메라 해를 풀고 IR 에 옮기는** 방안이 나왔습니다. 같은
비행에서 두 센서가 함께 찍히므로 위치는 동일하고, 자세는 센서 간 고정
오프셋만 다를 것이라는 가정입니다.

**이 스크립트는 그 가정이 참인지 확인만 합니다.** 참이 아니면 전이는
성립하지 않으므로, 확인 전에 구현하지 않습니다.

(이번 세션에서 검증 없이 만든 옵션 — `--terrain-fit`, `--min-frame-obs`,
`--guided-matching`, CLAHE — 이 모두 무관하거나 해로웠습니다. 같은 실수를
반복하지 않기 위한 순서입니다.)

무엇을 재는가
------------
1. **짝 맞추기**: 파일명 시퀀스 번호로 RGB(`_W`) ↔ IR(`_T`) 를 대응시킵니다.
   (실측 확인: 시퀀스가 1:1 이고 촬영 시각 차이 0~2초)
2. **위치 차이**: 두 센서의 GPS 가 같은지.
3. **자세 오프셋**: `IR 짐벌각 − RGB 짐벌각` 이 프레임마다 얼마나 흔들리는지.

판정
----
* 오프셋 **표준편차가 0.1° 미만** → 일정하다고 볼 수 있습니다. 전이가
  성립하며, RGB 해에 오프셋을 더해 IR 에 쓰면 됩니다.
* **0.1~0.5°** → 애매합니다. 0.3° 는 지상 0.25 m 오차이므로, 지금 IR 의
  0.86 m 보다는 낫지만 완벽하지 않습니다.
* **0.5° 이상** → 오프셋이 일정하지 않습니다. 전이는 성립하지 않으며
  다른 방법을 찾아야 합니다.

사용법
------
```bash
python scripts/check_rgb_ir_offset.py \\
    --rgb-dir .../RGB --ir-dir .../TM
```
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("check_rgb_ir_offset")

_PAT = re.compile(r"DJI_(\d{14})_(\d{4})_(\w+)")

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "label_studio"))


def _seq(path: Path):
    m = _PAT.match(path.stem)
    return (m.group(2), m.group(1)) if m else (None, None)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rgb-dir", type=Path, required=True)
    p.add_argument("--ir-dir", type=Path, required=True)
    p.add_argument("--glob", default="*.JPG")
    p.add_argument("--limit", type=int, default=0,
                   help="검사할 짝 수 상한 (0 이면 전부)")
    args = p.parse_args()

    try:
        from sayou.image.metadata import extract_metadata
    except Exception as exc:
        logger.error("메타데이터 모듈을 불러오지 못했습니다: %s", exc)
        logger.error("파이프라인과 같은 환경에서 실행하십시오.")
        return 2

    rgb = {}
    for f in sorted(Path(args.rgb_dir).glob(args.glob)):
        s, _ = _seq(f)
        if s:
            rgb[s] = f
    ir = {}
    for f in sorted(Path(args.ir_dir).glob(args.glob)):
        s, _ = _seq(f)
        if s:
            ir[s] = f
    common = sorted(set(rgb) & set(ir))
    if args.limit:
        common = common[:args.limit]
    logger.info("RGB %d장, IR %d장, 시퀀스 짝 %d개",
                len(rgb), len(ir), len(common))
    if len(common) < 20:
        logger.error("짝이 %d개뿐입니다 — 파일명 규칙이 다를 수 있습니다.",
                     len(common))
        return 2

    d_yaw, d_pitch, d_roll, d_pos, d_alt = [], [], [], [], []
    n_bad = 0
    for s in common:
        try:
            a = extract_metadata(rgb[s])
            b = extract_metadata(ir[s])
            ya, pa, ra = a.gimbal_yaw_deg, a.gimbal_pitch_deg, a.gimbal_roll_deg
            yb, pb, rb = b.gimbal_yaw_deg, b.gimbal_pitch_deg, b.gimbal_roll_deg
            if None in (ya, pa, ra, yb, pb, rb):
                n_bad += 1
                continue
            # yaw 는 ±180 경계를 넘을 수 있으므로 wrap
            dy = (float(yb) - float(ya) + 180.0) % 360.0 - 180.0
            d_yaw.append(dy)
            d_pitch.append(float(pb) - float(pa))
            d_roll.append(float(rb) - float(ra))
            if a.gps and b.gps:
                d_pos.append(np.hypot(float(b.gps.lat) - float(a.gps.lat),
                                      float(b.gps.lng) - float(a.gps.lng)))
                if a.gps.altitude is not None and b.gps.altitude is not None:
                    d_alt.append(float(b.gps.altitude) - float(a.gps.altitude))
        except Exception:
            n_bad += 1

    if len(d_yaw) < 20:
        logger.error("자세를 읽은 짝이 %d개뿐입니다 (실패 %d)",
                     len(d_yaw), n_bad)
        return 2

    logger.info("")
    logger.info("짝 %d개에서 측정 (읽기 실패 %d)", len(d_yaw), n_bad)
    logger.info("")
    logger.info("위치 차이 (RGB vs IR)")
    if d_pos:
        logger.info("  GPS 거리 중앙값 %.2e도, 최대 %.2e도",
                    float(np.median(d_pos)), float(np.max(d_pos)))
    if d_alt:
        logger.info("  고도 차이 중앙값 %.3f m, 표준편차 %.3f m",
                    float(np.median(d_alt)), float(np.std(d_alt)))

    logger.info("")
    logger.info("자세 오프셋 (IR − RGB)")
    verdict_std = 0.0
    for name, arr in (("yaw", d_yaw), ("pitch", d_pitch), ("roll", d_roll)):
        a = np.asarray(arr, dtype=np.float64)
        med = float(np.median(a))
        # 로버스트 산포 (이상치에 둔감)
        rstd = float(1.4826 * np.median(np.abs(a - med)))
        verdict_std = max(verdict_std, rstd)
        logger.info("  %-6s 중앙값 %+8.4f°   산포(로버스트) %.4f°   "
                    "p5~p95 %+.3f ~ %+.3f°",
                    name, med, rstd,
                    float(np.percentile(a, 5)), float(np.percentile(a, 95)))

    logger.info("")
    logger.info("★ 판정")
    h = 48.0
    err_m = np.degrees(0) if False else h * np.tan(np.radians(verdict_std))
    if verdict_std < 0.1:
        logger.info("  오프셋이 **일정합니다** (최대 산포 %.3f° = 지상 %.2f m).",
                    verdict_std, err_m)
        logger.info("  RGB 해에 이 오프셋을 더해 IR 에 쓰는 방식이 성립합니다.")
        logger.info("  현재 IR 의 BA 후 어긋남 0.86 m 보다 훨씬 낫습니다.")
    elif verdict_std < 0.5:
        logger.info("  오프셋이 **어느 정도 일정합니다** "
                    "(최대 산포 %.3f° = 지상 %.2f m).", verdict_std, err_m)
        logger.info("  현재 IR 의 0.86 m 보다는 낫지만 완벽하지 않습니다.")
        logger.info("  전이를 시도해 볼 가치는 있습니다.")
    else:
        logger.info("  오프셋이 **일정하지 않습니다** "
                    "(최대 산포 %.3f° = 지상 %.2f m).", verdict_std, err_m)
        logger.info("  RGB 해를 그대로 옮기는 방식은 성립하지 않습니다.")
        logger.info("  이 결과를 보내주시면 다른 방법을 찾겠습니다.")
    logger.info("")
    logger.info("  참고: 현재 IR 은 BA 후에도 인접 프레임이 0.86 m 어긋납니다")
    logger.info("        (지상 0.86 m = 자세 오차 약 1.03°).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

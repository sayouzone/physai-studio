#!/usr/bin/env python3
"""여러 실행을 나란히 놓고 비교한다 — summary.json + 독립 계측기.

왜 summary.json 만 보면 안 되는가
---------------------------------
``panel_misalign_median_m`` 은 이 장면의 실패를 구조적으로 못 봅니다
(블록 3.8 m < 행 피치 6.03 m, 탐색 상한 1.00 m < 실제 어긋남 2~3 m).
``patch_sayou.py`` P1 을 적용하면 나아지지만, 블록 수가 100 대로 떨어져
실행 간 변동이 큽니다.

그래서 이 스크립트는 ``qc_row_phase.py`` 의 **행 위상 계측**을 같이
돌립니다. 이쪽은 모자이크에서 직접 재는 값이라 파이프라인 설정과
독립이고, 통째로 밀린 구역을 그대로 잡습니다. 두 지표가 같은 방향을
가리킬 때만 개선으로 봅니다.

사용법
------
```bash
python compare_runs.py out_a out_b out_c
python compare_runs.py ab/run_*            # 셸 글로브
```

각 폴더에 ``summary.json`` 과 ``mosaic.tif`` 가 있어야 합니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _phase_metrics(mosaic: Path, pitch_m: float = 0.0):
    """qc_row_phase 의 계측만 재사용한다 (그림은 안 그린다)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from qc_row_phase import estimate_pitch, load_gray, phase_map
    except Exception as exc:
        return {"err": f"qc_row_phase 를 못 불러옴 ({exc})"}
    try:
        g, valid, gsd = load_gray(str(mosaic))
        auto_p, strength = estimate_pitch(g, valid)
        p = (pitch_m / gsd) if pitch_m > 0 else auto_p
        if not (4 < p < g.shape[0] / 4):
            return {"err": "행 피치 추정 실패"}
        off, amp, strong = phase_map(g, valid, p)
        ph = np.angle(np.exp(1j * 2 * np.pi * off / p))
        b = max(4, int(round(p * 0.5)))
        gx = np.abs(np.angle(np.exp(1j * (np.roll(ph, -b, 1) - ph))))
        gy = np.abs(np.angle(np.exp(1j * (np.roll(ph, -b, 0) - ph))))
        step = np.maximum(gx, gy) / (2 * np.pi) * p * gsd
        s = step[strong]
        ceil_m = p * gsd / 2.0
        p95 = float(np.percentile(s, 95))
        return {
            "pitch_m": p * gsd,
            "strength": strength,
            "sat": p95 / (0.95 * ceil_m),
            "step_p95_m": p95,
            "step_p99_m": float(np.percentile(s, 99)),
            "torn_ratio": float((s > p * gsd * 0.15).mean()),
            "valid": bool(strength >= 0.25 and p95 / (0.95 * ceil_m) <= 0.70),
        }
    except Exception as exc:
        return {"err": str(exc)}


def _get(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--pitch-m", type=float, default=0.0,
                    help="행 피치(m). 0 이면 자동. 실행마다 다르게 추정되면 "
                         "직접 고정하십시오 (실측 6.03)")
    ap.add_argument("--no-phase", action="store_true",
                    help="행 위상 계측을 건너뛴다 (빠름)")
    a = ap.parse_args()

    rows = []
    for d in a.runs:
        sj = d / "summary.json"
        if not sj.exists():
            print(f"  건너뜀 (summary.json 없음): {d}")
            continue
        s = json.load(open(sj))
        r = {"name": d.name}
        q = _get(s, "ortho", "quality", default={}) or {}
        r["misalign_med"] = q.get("panel_misalign_median_m")
        r["misalign_p90"] = q.get("panel_misalign_p90_m")
        r["broken"] = q.get("broken_block_ratio")
        r["blocks"] = q.get("blocks")
        r["step_p99"] = q.get("edge_step_p99_m")
        r["obs_med"] = _get(s, "observations_per_frame", "median")
        r["obs_p10"] = _get(s, "observations_per_frame", "p10")
        r["zero"] = _get(s, "observations_per_frame", "zero_frames")
        r["rmse"] = s.get("ba_reprojection_rmse_px")
        r["surface"] = s.get("surface_model")
        r["match_check"] = "on" if s.get("match_diagnosis") is not None \
            and s.get("rtk_match_check") is not None else "off"
        sm = s.get("attitude_smoothing") or {}
        r["smoothed"] = sm.get("smoothed")
        r["anchor_rej"] = sm.get("anchor_rejected")
        r["ang_med"] = sm.get("angle_change_median_deg")
        r["excluded"] = s.get("excluded_weak_frames")
        r["fill"] = _get(s, "ortho", "fill_ratio")
        r["cover"] = q.get("coverage_in_site")
        r["plane_src"] = s.get("plane_source")
        r["plane_c"] = _get(s, "ground_plane", "c")
        r["plane_rmse"] = _get(s, "ground_plane", "inlier_rmse_m")
        r["k_max"] = _get(s, "ortho", "max_offnadir_ratio")
        r["sat240"] = q.get("panel_saturated_240")
        r["frames"] = _get(s, "ortho", "frames_ok")

        mos = d / "mosaic.tif"
        if not a.no_phase and mos.exists():
            print(f"  계측 중: {d.name} ...", flush=True)
            r.update({("ph_" + k): v for k, v in
                      _phase_metrics(mos, a.pitch_m).items()})
        rows.append(r)

    if not rows:
        print("비교할 실행이 없습니다.")
        return 2

    def fmt(v, f="{:.3f}"):
        if v is None:
            return "-"
        if isinstance(v, str):
            return v
        return f.format(v)

    def table(title, cols):
        print("")
        print(title)
        w = max(len(r["name"]) for r in rows)
        head = f"  {'run':<{w}}" + "".join(f"{c[0]:>14}" for c in cols)
        print(head)
        print("  " + "-" * (len(head) - 2))
        for r in rows:
            line = f"  {r['name']:<{w}}"
            for _, key, f in cols:
                line += f"{fmt(r.get(key), f):>14}"
            print(line)

    table("[1] 독립 계측 — 행 위상", [
        ("pitch_m", "ph_pitch_m", "{:.2f}"),
        ("strength", "ph_strength", "{:.3f}"),
        ("saturation", "ph_sat", "{:.2f}"),
        ("step_p95_m", "ph_step_p95_m", "{:.3f}"),
        ("step_p99_m", "ph_step_p99_m", "{:.3f}"),
        ("torn_ratio", "ph_torn_ratio", "{:.4f}"),
    ])
    table("[2] summary.json QC (보조 — P1 패치 후에만 의미)", [
        ("misalign_med", "misalign_med", "{:.4f}"),
        ("misalign_p90", "misalign_p90", "{:.4f}"),
        ("broken", "broken", "{:.4f}"),
        ("blocks", "blocks", "{:.0f}"),
        ("edge_p99", "step_p99", "{:.4f}"),
    ])
    table("[3] 비교 가능성 — 이 값들이 다르면 [1][2] 를 비교하지 말 것", [
        ("fill_ratio", "fill", "{:.4f}"),
        ("coverage", "cover", "{:.4f}"),
        ("frames_ok", "frames", "{:.0f}"),
        ("excluded", "excluded", "{:.0f}"),
        ("k_max", "k_max", "{:.4f}"),
    ])
    table("[3b] 기준면", [
        ("plane_source", "plane_src", "{}"),
        ("plane_c_m", "plane_c", "{:.3f}"),
        ("plane_rmse_m", "plane_rmse", "{:.3f}"),
        ("surface", "surface", "{}"),
        ("saturated240", "sat240", "{:.3f}"),
    ])
    table("[4] 정합 상태", [
        ("ba_rmse_px", "rmse", "{:.3f}"),
        ("obs_median", "obs_med", "{:.0f}"),
        ("obs_p10", "obs_p10", "{:.0f}"),
        ("zero_frames", "zero", "{:.0f}"),
        ("excluded", "excluded", "{:.0f}"),
    ])
    table("[5] 자세 스무딩", [
        ("smoothed", "smoothed", "{:.0f}"),
        ("anchor_rej", "anchor_rej", "{:.0f}"),
        ("ang_med_deg", "ang_med", "{:.2f}"),
    ])

    # ---- 자동 경고 ---------------------------------------------------------
    warn = []
    valids = [r.get("ph_valid") for r in rows if "ph_valid" in r]
    if valids and not any(valids):
        warn.append(
            "[1] 이 **전부 무효**입니다 (주기 강도 부족 또는 포화). 이 표의\n"
            "    숫자는 어긋남이 아니라 잡음이고, 실행 간 차이도 잡음입니다.\n"
            "    qc_row_phase.py 를 --mask-pct 85 로 다시 시도하고, 그래도\n"
            "    무효면 debug_single_frame --measure 로 갈아타십시오.")
    noqc = [r["name"] for r in rows if r.get("blocks") is None]
    if noqc:
        warn.append(
            f"{', '.join(noqc)} 에 QC 가 없습니다 — assess_mosaic 이 블록을\n"
            "    하나도 만들지 못했습니다. max_shift_m 이 block_m 보다 크면\n"
            "    모든 블록이 걸러집니다 (max_shift 는 block 의 1/3 이하로).\n"
            "    patch_sayou.py 의 P1b 를 --revert 하십시오.")

    fills = [r["fill"] for r in rows if r.get("fill") is not None]
    if fills and (max(fills) - min(fills)) > 0.02:
        warn.append(
            f"충전율이 실행마다 다릅니다 ({min(fills):.3f} ~ {max(fills):.3f}).\n"
            "    덜 채운 쪽은 나쁜 픽셀이 측정에서 빠져 좋아 보입니다.\n"
            "    [1][2] 를 이 실행들 사이에서 비교하지 마십시오.")
    pitches = [round(r["ph_pitch_m"], 2) for r in rows
               if r.get("ph_pitch_m") is not None]
    if pitches and (max(pitches) - min(pitches)) > 0.05:
        warn.append(
            f"행 피치 추정이 실행마다 다릅니다 ({min(pitches):.2f} ~ "
            f"{max(pitches):.2f} m).\n"
            "    다른 피치로 잰 [1] 값은 서로 비교할 수 없습니다.\n"
            "    --pitch-m 으로 고정해 다시 돌리십시오.")

    keys = ("misalign_med", "broken", "blocks", "rmse", "obs_med", "excluded")
    seen = {}
    for r in rows:
        sig = tuple(r.get(k) for k in keys)
        if all(v is None for v in sig):
            continue          # QC 가 아예 없는 실행끼리는 비교 불가 (오탐 방지)
        seen.setdefault(sig, []).append(r["name"])
    dup = [v for v in seen.values() if len(v) > 1]
    for d in dup:
        warn.append(
            f"{' == '.join(d)} 의 결과가 **완전히 같습니다** — 그 옵션이\n"
            "    실제로는 아무것도 바꾸지 않았다는 뜻입니다. 다른 옵션이\n"
            "    덮어쓰고 있는지 확인하십시오.")
    if any(r.get("anchor_rej") is None for r in rows):
        warn.append(
            "anchor_rej 가 비어 있는 실행이 있습니다 — patch_sayou.py 의 P2 가\n"
            "    실제로 임포트되는 설치본에 적용되지 않았습니다.")
    if warn:
        print("")
        print("★ 경고")
        for w in warn:
            print("  · " + w)

    print("")
    print("★ 읽는 법")
    print("  · [1] 은 strength >= 0.25 이고 saturation <= 0.70 일 때만")
    print("    유효하다. 그 조건을 못 채우면 숫자를 읽지 말 것 — 위상이")
    print("    무작위여서 p95 가 pitch/2 에 붙어 버린 상태다.")
    print("  · 유효할 때의 주지표는 torn_ratio 와 step_p99 다. 낮을수록 좋다.")
    print("  · [2] 는 블록 수가 적어 실행마다 흔들린다. 차이가 20% 이상일")
    print("    때만 의미를 두라.")
    print("  · [5] 의 ang_med_deg 는 **클수록 나쁘다는 뜻이 아니다.** 앵커")
    print("    게이트가 켜지면 잘못된 SLERP 대신 올바른 큰 보정이 들어가")
    print("    이 값이 오히려 커질 수 있다. anchor_rej 와 [1] 을 함께 보라.")
    print("  · ba_rmse_px 는 좋아 보여도 신뢰하지 말 것. 반복 격자 오매칭은")
    print("    자기일관적이라 RMSE 를 낮게 유지한 채로 틀린다.")

    best = [r for r in rows if r.get("ph_torn_ratio") is not None
            and r.get("ph_valid")]
    if len(best) > 1:
        best.sort(key=lambda r: r["ph_torn_ratio"])
        print("")
        print(f"  현재 최선: {best[0]['name']} "
              f"(torn_ratio {best[0]['ph_torn_ratio']:.4f}, "
              f"기준선 대비 참고용)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

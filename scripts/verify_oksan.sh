#!/usr/bin/env bash
# ============================================================================
# 옥산_1호 검증 — 수정된 소스(P10 매칭 축소 포함)로 다시 확인한다
# ============================================================================
#
# 이 부지의 기존 판정: 지형 RMSE 3.70 m(평면 불가), 궤적 안 최소 k 0.197,
# 촬영 밀도 7.5. 아홉 가지 설정을 시험해 기본을 이긴 것이 없었다.
#
# 그런데 이후 P10(매칭 단계 영상 축소)이 추가됐고, 이 부지는 그 대상
# 조건(교차 매칭 30% 근처, zero_frames 존재)에 해당한다. 아직 이 부지에
# 시험한 적이 없으므로 편견 없이 다시 잰다.
#
# ★ 결론을 먼저 정하지 않는다. 이 스크립트는 측정만 하고, 판단은 마지막
#   qc_tear 결과로 한다. 사천에서 P10 이 매칭은 크게 개선했지만(교차
#   12%→70%) qc_tear 는 오히려 소폭 나빠졌던 전례가 있다 — "대응이
#   늘어나는 것"과 "정사영상이 좋아지는 것"은 다르다.
#
# 절차 (다른 부지에서 확정된 순서를 그대로 따른다):
#   1. 환경 초기화 + 부지 확인
#   2. 기본 실행 (P10 없이) — 대조군
#   3. P10 실행 (매칭 축소) — 실험군
#   4. 두 실행 모두 plane_sweep 으로 기준면 확인 — 절대 건너뛰지 않는다
#      (사천에서 이 단계를 뒤로 미뤄 시간을 낭비한 전례가 있다)
#   5. 필요하면 P5 override 적용 (수동 계산 필요 — 6절 참고)
#   6. qc_tear 로 최종 판정 — 신뢰구간이 겹치면 순위를 매기지 않는다
#
# 사용법:
#   export SAYOU=<sayou 소스 경로>
#   bash verify_oksan.sh
#
# 각 단계는 앞 단계 결과에 의존하므로 순서대로 실행하는 것을 전제로
# 작성했다. set -e 로 중간 실패 시 멈춘다.
# ============================================================================
set -euo pipefail

IMAGE_DIR="${IMAGE_DIR:-$HOME/Development/sayouzone/solar-thermal/data/solar/옥산_1호}"
SAYOU="${SAYOU:?SAYOU=<sayou 소스 경로> 를 지정하십시오}"

# 이 부지의 확정된 좌표·측정값 (site_triage / 이전 실행에서 얻은 값)
SITE_X_MIN=230479; SITE_X_MAX=231044
SITE_Y_MIN=457617; SITE_Y_MAX=457859
QC_X=230520; QC_Y=457825; QC_W=20000; QC_H=7000; QC_BLOCK=12
PANEL_PITCH=5.65     # panel_unit_info.unit_m — 다른 값이면 아래에서 자동 갱신

echo "======================================================================"
echo "0. 환경 초기화 — 부지 전환 시 잔류 환경변수가 새어 들어간 전례가 있다"
echo "======================================================================"
unset SAYOU_PLANE_OVERRIDE SAYOU_OFFNADIR_CEILING SAYOU_MATCH_SCALE \
      SAYOU_ANCHOR_ANGLE_DEG SAYOU_PANEL_FRACTION 2>/dev/null || true
env | grep '^SAYOU_' && { echo "★ SAYOU_* 환경변수가 남아 있습니다 — 위 값을 확인하십시오"; exit 1; } || echo "SAYOU_* 없음 — 정상"
echo "IMAGE_DIR=$IMAGE_DIR"
echo

echo "======================================================================"
echo "1. 부지 사전 확인 (초점거리 포함 7열 TSV)"
echo "======================================================================"
if [ ! -f "$IMAGE_DIR/exif.tsv" ]; then
  exiftool -q -n -T -GPSLatitude -GPSLongitude -GPSAltitude \
      -RelativeAltitude -LRFTargetDistance \
      -FocalLengthIn35mmFormat -ImageWidth \
      "$IMAGE_DIR"/RGB/*.JPG > "$IMAGE_DIR/exif.tsv"
fi
NCOL=$(awk 'NR==1{print NF}' "$IMAGE_DIR/exif.tsv")
echo "exif.tsv 열 수: $NCOL (7이어야 초점거리 포함)"
python scripts/site_triage.py "$IMAGE_DIR/exif.tsv"
echo
echo "  ↑ 이 판정은 이미 알려진 것과 같아야 정상이다 (지형 RMSE > 1.5 m 로"
echo "    걸림). 여기서 다르게 나오면 부지 데이터 자체가 바뀐 것이니 먼저"
echo "    그 이유를 확인하라."
echo

echo "======================================================================"
echo "2. 패치 적용 — P2 P3 P4 P7 (기본). P1 은 블록 수 확인 후 별도 판단"
echo "======================================================================"
python scripts/patch_sayou.py --root "$SAYOU" --revert || true
python scripts/patch_sayou.py --root "$SAYOU" --only P2 P3 P4 P7 --apply
echo

echo "======================================================================"
echo "3. 대조군 — 기본 실행 (P10 없음, SAYOU_MATCH_SCALE 미설정)"
echo "======================================================================"
BASE_DIR="$IMAGE_DIR/verify_base"
if [ ! -f "$BASE_DIR/summary.json" ]; then
  python scripts/homography_pipeline.py \
      --image-dir "$IMAGE_DIR/RGB" --output-dir "$BASE_DIR" \
      --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude
else
  echo "  기존 결과 재사용: $BASE_DIR (다시 돌리려면 폴더를 지우십시오)"
fi
echo

echo "======================================================================"
echo "4. 실험군 — P10 적용 (SAYOU_MATCH_SCALE=0.25)"
echo "======================================================================"
echo "  ※ 이 부지 조건: probe 실측(축소 0.25배)에서 교차 inlier 중앙값이"
echo "    8 → 78 로 늘었었다(별도 측정). 파이프라인 전체로 같은 효과가"
echo "    나는지 지금 처음 확인한다."
python scripts/patch_sayou.py --root "$SAYOU" --revert
python scripts/patch_sayou.py --root "$SAYOU" --only P2 P3 P4 P7 P10 --apply

P10_DIR="$IMAGE_DIR/verify_p10"
if [ ! -f "$P10_DIR/summary.json" ]; then
  SAYOU_MATCH_SCALE=0.25 python scripts/homography_pipeline.py \
      --image-dir "$IMAGE_DIR/RGB" --output-dir "$P10_DIR" \
      --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude
else
  echo "  기존 결과 재사용: $P10_DIR (다시 돌리려면 폴더를 지우십시오)"
fi
echo

echo "======================================================================"
echo "5. 매칭·BA 비교 — P10 이 실제로 무엇을 바꿨는지 먼저 본다"
echo "======================================================================"
python3 - "$BASE_DIR/summary.json" "$P10_DIR/summary.json" <<'PYEOF'
import json, sys
b = json.load(open(sys.argv[1]))
p = json.load(open(sys.argv[2]))

def row(label, fn):
    print("  %-22s base %-12s p10 %-12s" % (label, fn(b), fn(p)))

print("  %-22s %-17s %-17s" % ("", "base(scale 1.0)", "P10(scale 0.25)"))
row("BA RMSE (px)", lambda s: "%.3f" % s["ba_reprojection_rmse_px"])
row("교차 매칭률", lambda s: "%.1f%%" % (100*s["match_diagnosis"]["cross_line_rate"]))
row("내부 매칭률", lambda s: "%.1f%%" % (100*s["match_diagnosis"]["same_line_rate"]))
row("교차 대응 중앙값", lambda s: "%s" % s["match_diagnosis"]["cross_line_median_pts"])
row("관측 중앙값", lambda s: "%s" % s["observations_per_frame"]["median"])
row("관측 p10", lambda s: "%s" % s["observations_per_frame"]["p10"])
row("zero_frames", lambda s: "%s" % s["observations_per_frame"]["zero_frames"])
row("총 점", lambda s: "%s" % s["observations_per_frame"]["total_points"])
row("기준면 출처", lambda s: s["plane_source"])
row("misalign 중앙값", lambda s: "%.3f" % s["ortho"]["quality"]["panel_misalign_median_m"])
row("파손 블록", lambda s: "%.1f%%" % (100*s["ortho"]["quality"]["broken_block_ratio"]))
row("블록 수", lambda s: "%s" % s["ortho"]["quality"]["blocks"])
row("포화 240+", lambda s: "%.1f%%" % (100*s["ortho"]["quality"]["panel_saturated_240"]))
row("eff_gsd", lambda s: "%.4f" % s["ortho"]["quality"]["eff_gsd_m"])

print()
n_base = b["images"]
zf_b = b["observations_per_frame"]["zero_frames"]
zf_p = p["observations_per_frame"]["zero_frames"]
print("  zero_frames 비율: base %.1f%% → p10 %.1f%%"
      % (100*zf_b/n_base, 100*zf_p/n_base))
if p["match_diagnosis"]["cross_line_rate"] > 1.5 * (b["match_diagnosis"]["cross_line_rate"] or 0.01):
    print("  → 교차 매칭이 크게 개선됐습니다. 6단계로 진행해 기준면을 잡으십시오.")
else:
    print("  → 교차 매칭 개선이 크지 않습니다. 이 부지는 매칭이 아니라")
    print("    다른 요인(촬영 밀도·정반사·지형)이 병목일 수 있습니다.")
PYEOF
echo

PITCH_BASE=$(python -c "import json;print(round(json.load(open('$BASE_DIR/summary.json'))['ortho']['panel_unit_info']['unit_m'],2))")
PITCH_P10=$(python -c "import json;print(round(json.load(open('$P10_DIR/summary.json'))['ortho']['panel_unit_info']['unit_m'],2))")
echo "패널 피치: base=$PITCH_BASE  p10=$PITCH_P10  (아래 plane_sweep 에 사용)"
echo

echo "======================================================================"
echo "6. 기준면 확인 — 판정과 무관하게 항상 실행한다"
echo "   (사천에서 이 단계를 뒤로 미뤘다가 순서를 되짚어야 했다)"
echo "======================================================================"
echo
echo "--- 6a. base 기준면 ---"
python scripts/plane_sweep.py --image-dir "$IMAGE_DIR/RGB" \
    --use-ba "$BASE_DIR/cameras.npz" --summary "$BASE_DIR/summary.json" \
    --pitch-m "$PITCH_BASE" --count 6 --z-min -10 --z-max 10 \
    --starts 30,120,210,300,390,480,570,660,750,850,940,1010 2>&1 | tail -35
echo
echo "--- 6b. P10 기준면 ---"
python scripts/plane_sweep.py --image-dir "$IMAGE_DIR/RGB" \
    --use-ba "$P10_DIR/cameras.npz" --summary "$P10_DIR/summary.json" \
    --pitch-m "$PITCH_P10" --count 6 --z-min -10 --z-max 10 \
    --starts 30,120,210,300,390,480,570,660,750,850,940,1010 2>&1 | tail -35
echo
echo "  ★★ 수동 판단 필요 지점 ★★"
echo "  위 두 결과의 마지막 블록을 읽고 판단하십시오:"
echo "    - '더 만질 이유가 없습니다' → 기준면 이미 최선, 7단계로"
echo "    - '높이만 틀렸습니다' 또는 '경사도 틀렸습니다' → override 필요"
echo "    - 신뢰 지점 5곳 미만 → --starts 를 더 촘촘히 하거나 --z 범위 확장"
echo "    - '최적이 범위 끝' 경고 다수 → --z-min/--z-max 를 그 방향으로 확장"
echo
echo "  override 가 필요하면 다음 형식으로 계산하십시오 (신뢰 지점의"
echo "  X,Y,최적오프셋 을 채워 넣는다 — plane_sweep 출력에서 그대로 복사):"
cat <<'PYEOF2'

  python3 - <<'PY'
  import numpy as np, json
  s = json.load(open("<base 또는 p10>/summary.json"))
  gp = s["ground_plane"]
  a0, b0, c0 = gp["a"], gp["b"], gp["c"]
  oX, oY = gp["origin_xy"]

  # plane_sweep 출력에서 "기복비율 0.6 이상"인 신뢰 지점만 옮겨 적는다
  S = [
      # (X, Y, 최적오프셋)
      (0, 0, 0.0),   # ← 여기를 실제 값으로 교체
  ]
  P = np.array(S, float)
  ox, oy = P[:, 0].mean(), P[:, 1].mean()
  A = np.column_stack([P[:, 0]-ox, P[:, 1]-oy, np.ones(len(P))])
  c, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
  r = P[:, 2] - A @ c
  na, nb = a0 + c[0], b0 + c[1]
  nc = c0 + a0*(ox-oX) + b0*(oy-oY) + c[2]
  print("잔차 최대 %.2f m  RMSE %.2f m  (자유도 %d)"
        % (np.abs(r).max(), np.sqrt((r**2).mean()), len(P)-3))
  print('SAYOU_PLANE_OVERRIDE="%.9f,%.9f,%.4f,%.3f,%.3f"'
        % (na, nb, nc, ox, oy))
  PY

PYEOF2
echo
echo "  값을 얻으면 아래처럼 재실행하십시오 (P10 켰던 쪽이면 SAYOU_MATCH_SCALE도):"
echo '    export SAYOU_PLANE_OVERRIDE="<계산값>"'
echo '    python scripts/homography_pipeline.py --image-dir $IMAGE_DIR/RGB \'
echo '        --output-dir $IMAGE_DIR/verify_<base|p10>_plane \'
echo '        --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude'
echo
echo "  기준면을 고친 뒤에는 unset SAYOU_PLANE_OVERRIDE 하고 plane_sweep 을"
echo "  다시 돌려 '더 만질 이유가 없습니다' 가 나오는지 확인하십시오."
echo

echo "======================================================================"
echo "7. 최종 판정 — qc_tear (신뢰구간이 겹치면 순위를 매기지 않는다)"
echo "======================================================================"
echo "  기준선(이전 확정값): >0.5m 10.03%  (8.92~10.87)"
echo
echo "  6단계에서 override 를 적용했다면 <위 산출물 폴더>를, 안 했다면"
echo "  base/p10 폴더를 그대로 아래에 대입해 실행하십시오:"
echo
cat <<EOF
  python scripts/qc_tear.py \\
      "$BASE_DIR/mosaic.tif" "$P10_DIR/mosaic.tif" \\
      --window --x $QC_X --y $QC_Y --size $QC_W $QC_H --block-m $QC_BLOCK
EOF
echo
echo "  GSD 가 다르면(한쪽만 --gsd 를 줬다면) 같은 GSD 로 맞춘 뒤 비교하십시오:"
echo "    python -c \"import rasterio as r; print(abs(r.open('$BASE_DIR/mosaic.tif').transform[0]))\""
echo "    python -c \"import rasterio as r; print(abs(r.open('$P10_DIR/mosaic.tif').transform[0]))\""
echo
echo "======================================================================"
echo "8. 판단 기준 요약"
echo "======================================================================"
cat <<'EOF'
  qc_tear 구간이 겹치면
    → P10 이 이 부지에서 효과 없음. base 를 산출물로 유지.
      (사천에서도 같은 결과였다 — 매칭 개선이 시임 품질로 안 옮겨짐)

  P10 이 유의하게 낫고 zero_frames 도 크게 줄었으면
    → P10 을 이 부지의 기본으로 채택. 다만 BA RMSE 가 얼마나
      나빠졌는지 반드시 함께 보고하십시오 (5단계 표).

  base 가 유의하게 나으면
    → 이 부지에서는 P10 을 쓰지 마십시오. 매칭 축소의 정밀도 손실이
      대응 증가의 이득보다 큰 경우입니다.

  둘 다 기준선(10.03%)보다 나쁘면
    → 이 부지는 여전히 기본 설정이 최선입니다. 아홉 번째 반례가
      아니라 열 번째 확인입니다. WebODM 비교로 넘어가십시오
      (단, feature-quality high, resize-to 2048 이상으로 재실행 필요 —
      이전 ODM 실행은 512x383 축소로 돌아 무효였습니다).
EOF

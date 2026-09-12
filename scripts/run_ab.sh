#!/usr/bin/env bash
# A-B 실행 드라이버 — 제안한 조치를 한 번에 하나씩만 바꿔 돌린다.
#
# 왜 한 번에 하나인가
# -------------------
# 이 사이트는 패널 블록 수가 적어 QC 값이 실행마다 흔들린다 (같은 설정의
# 두 실행이 0.073 vs 0.111 m 로 52% 차이났다고 코드에 기록돼 있다).
# 두 가지를 동시에 바꾸면 어느 쪽이 들었는지 영영 알 수 없다.
#
# 사용법
# ------
#   ./run_ab.sh 0          # base (지금 설정 재현) — 기준선
#   ./run_ab.sh 1          # + --rtk-match-check
#   ./run_ab.sh 2          # + 앵커 게이트 30° (patch 필요)
#   ./run_ab.sh 3          # + 스무딩 끔
#   ./run_ab.sh 4          # + --min-frame-obs 60
#   ./run_ab.sh 5          # + 2층 표면
#   ./run_ab.sh 6          # + 시임 패널 회피
#   ./run_ab.sh all        # 0~6 전부 (오래 걸린다)
#
# 끝나면:
#   python compare_runs.py "$OUT_ROOT"/run_*
#
set -euo pipefail

# ---- 여기만 환경에 맞게 고치십시오 -----------------------------------------
PIPE="${PIPE:-scripts/homography_pipeline.py}"
IMAGE_DIR="${IMAGE_DIR:-$HOME/Development/sayouzone/solar-thermal/data/solar/EWP-서오창IC-2/TM}"
OUT_ROOT="${OUT_ROOT:-$HOME/Development/sayouzone/solar-thermal/data/solar/EWP-서오창IC-2/ab}"
PY="${PY:-python}"

# summary.json 에서 읽은 현재 설정. 기준선이 지금 결과를 재현해야 비교가 성립한다.
BASE_ARGS=(
  --epsg 5186
  --offnadir-frac 0.32
  --panel-unit 2
  --smooth-weak-attitude
)
# ---------------------------------------------------------------------------

run_one () {
  local name="$1"; shift
  local out="${OUT_ROOT}/run_${name}"
  if [[ -d "$out" && -f "$out/summary.json" ]]; then
    echo ">>> ${name}: 이미 결과가 있습니다 — 건너뜁니다 ($out)"
    echo "    다시 돌리려면 그 폴더를 지우십시오."
    return 0
  fi
  mkdir -p "$out"
  echo ""
  echo "==================================================================="
  echo ">>> ${name}"
  echo "    추가 인자: $*"
  echo "    출력      : $out"
  echo "==================================================================="
  # 특징점 캐시는 이미지가 같으면 재사용되므로 2회차부터 크게 빨라진다.
  "$PY" "$PIPE" \
      --image-dir "$IMAGE_DIR" \
      --output-dir "$out" \
      "${BASE_ARGS[@]}" "$@" 2>&1 | tee "$out/run.log"
  echo ">>> ${name} 완료"
}

stage="${1:-}"
case "$stage" in
  0|base)
    run_one "0_base"
    ;;
  1|match)
    # 반복 격자 오매칭 제거. 지금 실행에는 안 들어가 있었다(summary: null).
    run_one "1_matchcheck" --rtk-match-check
    ;;
  2|anchor)
    # patch_sayou.py P2 가 적용돼 있어야 의미가 있다. 게이트 각도만 바꾼다.
    SAYOU_ANCHOR_ANGLE_DEG=30 run_one "2_anchor30" --rtk-match-check
    ;;
  3|nosmooth)
    # --smooth-weak-attitude 를 빼고 돌린다 (BASE_ARGS 에서 제거).
    local_args=()
    for a in "${BASE_ARGS[@]}"; do
      [[ "$a" == "--smooth-weak-attitude" ]] || local_args+=("$a")
    done
    BASE_ARGS=("${local_args[@]}")
    run_one "3_nosmooth" --rtk-match-check
    ;;
  4|minobs)
    run_one "4_minobs60" --rtk-match-check --min-frame-obs 60
    ;;
  5|layer)
    run_one "5_layer" --rtk-match-check --layer-surface --layer-cell 0.5
    ;;
  6|seam)
    run_one "6_seam" --rtk-match-check --seam-panel-penalty 0.5
    ;;
  all)
    for s in 0 1 2 3 4 5 6; do "$0" "$s"; done
    ;;
  *)
    sed -n '2,22p' "$0"
    exit 2
    ;;
esac

echo ""
echo "비교:  $PY compare_runs.py ${OUT_ROOT}/run_*"

#!/usr/bin/env bash
# 시험 명령을 만들기 전에, piece 가 이미 넘기는 옵션과 파이프라인의 실제 옵션 이름을 확인하겠습니다. PIECE_ARGS 로 덧붙인 값이 기존 값을 덮어쓰는지가 여기서 갈립니다.
#
# 확인 결과 두 가지를 알아 두셔야 합니다.
#
# PIECE_ARGS 로 덧붙인 값은 뒤에 오므로 앞의 값을 덮어씁니다. --panel-unit 1 을 주면 piece 의 --panel-unit 2 대신 그것이 쓰입니다.
# 2층 높이맵은 piece 가 --no-two-layer 로 끄고 있고, 다시 켜는 옵션이 없습니다. 그래서 그 경우만 piece 를 거치지 않고 같은 옵션으로 직접 돌립니다.
#
# 비교 기준은 같은 설정에서 이 옵션들만 뺀 gal_base2 입니다. 2차 안내 매칭은 넣지 않습니다. 이음매 옵션의 효과만 따로 보기 위해서입니다. 매칭 코드가 같으니 gal_base2 의 캐시를 복사해 SIFT 를 건너뜁니다.

# 준비

export SOLAR=~/Development/sayouzone/solar-thermal/data/solar IMAGE_DIR=$SOLAR/갈평저수지 S=RGB
export G=$(python -c "import json;print('%.4f' % json.load(open('$IMAGE_DIR/gal_rgb_noexp/summary.json'))['gsd_m'])")
PLR=$(python -c "import json;g=json.load(open('$IMAGE_DIR/gal_rgb_noexp/summary.json'))['ground_plane'];print('%.9f,%.9f,%.4f,%.3f,%.3f' % (g['a'],g['b'],g['c'],*g['origin_xy']))")
unset SAYOU_PAIR_MODEL SAYOU_RTK_CHECK_PLANE PIECE_ARGS
source scripts/sess_funcs.sh
ls $IMAGE_DIR/gal_base2/mosaic_labels.tif && echo "기준 결과에 라벨 지도 있음"

seed() { mkdir -p "$IMAGE_DIR/$1" && cp -r "$IMAGE_DIR/gal_base2/.feature_cache" "$IMAGE_DIR/$1/"; }


# 1. 경계 위치 옵션 — piece 로

for RUN in "gal_sp05:--seam-panel-penalty 0.5" "gal_sp10:--seam-panel-penalty 1.0" \
           "gal_sc2:--seam-cost 2.0" "gal_pu1:--panel-unit 1" "gal_pu0:--panel-unit 0" \
           "gal_ls:--layer-surface"; do
  NAME=${RUN%%:*}; ARGS=${RUN#*:}
  seed $NAME
  PIECE_ARGS="$ARGS" piece $NAME RGB "$PLR"
done

#=== gal_sp05  (RGB)   GSD 0.0065   매칭 축소 끔   override 0.000128022,0.000378210,227.1414,275133.707,428576.925   추가 옵션 --seam-panel-penalty 0.5
#2026-10-02 17:37:10,035 INFO 비행선 밖 구간 검사: 비행 방향 1°, 비행선 간격 15.5 m — 0곳 0장
#2026-10-02 17:37:40,595 WARNING 기준면을 환경변수로 지정했습니다 — 추정을 건너뜁니다: a=+0.000128 b=+0.000378 c=227.141 경사 0.023°
#2026-10-02 17:44:56,618 INFO 모자이크 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/갈평저수지/gal_sp05/mosaic.tif (244장 합성, 충전율 92.3%)
#=== gal_sp10  (RGB)   GSD 0.0065   매칭 축소 끔   override 0.000128022,0.000378210,227.1414,275133.707,428576.925   추가 옵션 --seam-panel-penalty 1.0
#2026-10-02 17:46:29,767 INFO 비행선 밖 구간 검사: 비행 방향 1°, 비행선 간격 15.5 m — 0곳 0장
#2026-10-02 17:46:59,516 WARNING 기준면을 환경변수로 지정했습니다 — 추정을 건너뜁니다: a=+0.000128 b=+0.000378 c=227.141 경사 0.023°
#2026-10-02 17:55:16,036 INFO 모자이크 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/갈평저수지/gal_sp10/mosaic.tif (244장 합성, 충전율 92.3%)
#=== gal_sc2  (RGB)   GSD 0.0065   매칭 축소 끔   override 0.000128022,0.000378210,227.1414,275133.707,428576.925   추가 옵션 --seam-cost 2.0
#2026-10-02 17:59:15,166 INFO 비행선 밖 구간 검사: 비행 방향 1°, 비행선 간격 15.5 m — 0곳 0장
#2026-10-02 17:59:58,395 WARNING 기준면을 환경변수로 지정했습니다 — 추정을 건너뜁니다: a=+0.000128 b=+0.000378 c=227.141 경사 0.023°
#2026-10-02 18:05:45,761 INFO 모자이크 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/갈평저수지/gal_sc2/mosaic.tif (244장 합성, 충전율 92.3%)
#=== gal_pu1  (RGB)   GSD 0.0065   매칭 축소 끔   override 0.000128022,0.000378210,227.1414,275133.707,428576.925   추가 옵션 --panel-unit 1
#2026-10-02 18:09:08,546 INFO 비행선 밖 구간 검사: 비행 방향 1°, 비행선 간격 15.5 m — 0곳 0장
#2026-10-02 18:09:34,222 WARNING 기준면을 환경변수로 지정했습니다 — 추정을 건너뜁니다: a=+0.000128 b=+0.000378 c=227.141 경사 0.023°
#2026-10-02 18:12:55,815 INFO 모자이크 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/갈평저수지/gal_pu1/mosaic.tif (244장 합성, 충전율 92.3%)
#=== gal_pu0  (RGB)   GSD 0.0065   매칭 축소 끔   override 0.000128022,0.000378210,227.1414,275133.707,428576.925   추가 옵션 --panel-unit 0
#2026-10-02 18:13:48,057 INFO 비행선 밖 구간 검사: 비행 방향 1°, 비행선 간격 15.5 m — 0곳 0장
#2026-10-02 18:14:12,060 WARNING 기준면을 환경변수로 지정했습니다 — 추정을 건너뜁니다: a=+0.000128 b=+0.000378 c=227.141 경사 0.023°
#2026-10-02 18:17:24,884 INFO 모자이크 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/갈평저수지/gal_pu0/mosaic.tif (244장 합성, 충전율 92.3%)
#=== gal_ls  (RGB)   GSD 0.0065   매칭 축소 끔   override 0.000128022,0.000378210,227.1414,275133.707,428576.925   추가 옵션 --layer-surface
#2026-10-02 18:18:16,157 INFO 비행선 밖 구간 검사: 비행 방향 1°, 비행선 간격 15.5 m — 0곳 0장
#2026-10-02 18:18:40,634 WARNING 기준면을 환경변수로 지정했습니다 — 추정을 건너뜁니다: a=+0.000128 b=+0.000378 c=227.141 경사 0.023°
#2026-10-02 18:21:52,822 INFO 모자이크 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/갈평저수지/gal_ls/mosaic.tif (244장 합성, 충전율 92.3%)


# 2. 2층 높이맵 — 직접 실행

seed gal_2l
SAYOU_PLANE_OVERRIDE="$PLR" python scripts/homography_pipeline.py --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/gal_2l \
    --offnadir-frac 0.32 --smooth-weak-attitude --gsd "$G" --panel-unit 2 --no-exposure-comp 2>&1 \
  | tee $IMAGE_DIR/gal_2l.log | grep -E "환경변수로 지정|2층|모자이크 저장"

#2026-10-02 18:23:50,434 WARNING 기준면을 환경변수로 지정했습니다 — 추정을 건너뜁니다: a=+0.000128 b=+0.000378 c=227.141 경사 0.023°
#2026-10-02 18:23:50,444 INFO 2층 표면 자동 해제: 층 경계가 GSD 기준 56 px 밀려 있어 (층 높이차 1.04 m × k 0.35 ÷ GSD 0.0065 m) 톱니 손해가 이득을 넘어섭니다. 한계 10 px. GSD 가 거친 IR 에서는 켜집니다.
#2026-10-02 18:27:03,914 INFO 모자이크 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/갈평저수지/gal_2l/mosaic.tif (244장 합성, 충전율 92.3%)


# 3. 옵션이 실제로 적용됐는지
# 옵션을 줬는데 결과가 소수점까지 같으면, 먼저 적용 여부부터 의심해야 합니다(--panel-unit 때 겪은 일). summary.json 에 기록된 값을 봅니다.

for R in gal_base2 gal_sp05 gal_sp10 gal_sc2 gal_pu1 gal_pu0 gal_ls gal_2l; do
  python - <<EOF
import json; s = json.load(open('$IMAGE_DIR/$R/summary.json')); o = s['ortho']
print('%-10s 패널벌점 %-4s 시임강도 %-4s 패널단위 %-4s 2층 %-6s 층표면 %-6s 캐시 %s' % ('$R',
      o.get('seam_panel_penalty'), o.get('seam_cost', s.get('seam_cost', '?')), o.get('panel_unit_m'),
      'dsm' if o.get('dsm') else '-', s.get('layer_surface', o.get('layer_surface', '?')),
      '적중' if '캐시 적중' in open('$IMAGE_DIR/$R.log', encoding='utf-8', errors='replace').read() else '새로'))
EOF
done

#gal_base2  패널벌점 0.0  시임강도 ?    패널단위 2.0  2층 -      층표면 ?      캐시 새로
#gal_sp05   패널벌점 0.5  시임강도 ?    패널단위 2.0  2층 -      층표면 ?      캐시 적중
#gal_sp10   패널벌점 1.0  시임강도 ?    패널단위 2.0  2층 -      층표면 ?      캐시 적중
#gal_sc2    패널벌점 0.0  시임강도 ?    패널단위 2.0  2층 -      층표면 ?      캐시 적중
#gal_pu1    패널벌점 0.0  시임강도 ?    패널단위 1.0  2층 -      층표면 ?      캐시 적중
#gal_pu0    패널벌점 0.0  시임강도 ?    패널단위 0.0  2층 -      층표면 ?      캐시 적중
#gal_ls     패널벌점 0.0  시임강도 ?    패널단위 2.0  2층 -      층표면 ?      캐시 적중
#gal_2l     패널벌점 0.0  시임강도 ?    패널단위 2.0  2층 -      층표면 ?      캐시 적중


# 4. 판정

python scripts/session_tools.py seams --mosaics $IMAGE_DIR/gal_base2/mosaic.tif \
    $IMAGE_DIR/gal_sp05/mosaic.tif $IMAGE_DIR/gal_sp10/mosaic.tif $IMAGE_DIR/gal_sc2/mosaic.tif \
    $IMAGE_DIR/gal_pu1/mosaic.tif $IMAGE_DIR/gal_pu0/mosaic.tif $IMAGE_DIR/gal_ls/mosaic.tif $IMAGE_DIR/gal_2l/mosaic.tif
for R in gal_base2 gal_sp05 gal_sp10 gal_sc2 gal_pu1 gal_pu0 gal_ls gal_2l; do
  python -c "import json;q=json.load(open('$IMAGE_DIR/$R/summary.json'))['ortho']['quality'];print('%-10s 패널 어긋남 %.3f m' % ('$R', q['panel_misalign_median_m']))"
done
qc_pair gal_base2 gal_sp10

#결과                        단차 비     뚜렷한단차 │ 어긋남 중앙값 (95%)          >5cm (95%)               >2GSD    어긋남90     표본    끝 걸림
#gal_base2                 2.24     28.0% │ 0.030 (0.025~0.035)    41.5% (38.7~44.4%)    65.9%    0.229    995      9%
#gal_sp05                  2.26     27.8% │ 0.027 (0.021~0.036)    41.2% (38.0~44.5%)    63.9%    0.210    963     10%  ← 첫 결과와 구분 안 됨
#gal_sp10                  2.24     27.7% │ 0.027 (0.022~0.032)    40.2% (37.1~43.2%)    63.2%    0.221    945      8%  ← 첫 결과와 구분 안 됨
#gal_sc2                   2.14     26.6% │ 0.025 (0.022~0.033)    37.7% (35.1~40.9%)    63.8%    0.218    955      8%  ← 첫 결과와 구분 안 됨
#gal_pu1                   2.24     27.5% │ 0.020 (0.017~0.025)    36.4% (33.3~39.3%)    59.2%    0.208    984     10%  ← 첫 결과와 구분 안 됨
#gal_pu0                   2.15     27.1% │ 0.037 (0.030~0.044)    43.6% (40.3~46.7%)    66.5%    0.228   1085     10%  ← 첫 결과와 구분 안 됨
#gal_ls                    2.24     28.0% │ 0.030 (0.025~0.035)    41.5% (38.7~44.4%)    65.9%    0.229    995      9%  ← 첫 결과와 구분 안 됨
#gal_2l                    2.24     28.0% │ 0.030 (0.025~0.035)    41.5% (38.7~44.4%)    65.9%    0.229    995      9%  ← 첫 결과와 구분 안 됨

#끝 걸림: 탐색 상한에 상관이 걸린 표본(반복 줄 한 주기 옆) — 어긋남 계산에서 뺌
#어긋남: 이음매 양쪽 띠의 무늬가 경계를 따라 밀린 거리 m (중앙값 · 90%) 와 5 cm 넘는 비율 — 기하
#단차 비: 이음매를 가로지른 밝기 차 ÷ 이음매 아닌 곳의 같은 간격 밝기 차 (1 이면 이음매가 안 보임)
#뚜렷한 단차: 이음매 단차 중 '이음매 아닌 곳' 상위 10% 를 넘는 비율 (10% 이면 구분 안 됨)
#gal_base2  패널 어긋남 0.046 m
#gal_sp05   패널 어긋남 0.046 m
#gal_sp10   패널 어긋남 0.046 m
#gal_sc2    패널 어긋남 0.049 m
#gal_pu1    패널 어긋남 0.045 m
#gal_pu0    패널 어긋남 0.052 m
#gal_ls     패널 어긋남 0.046 m
#gal_2l     패널 어긋남 0.046 m
#GSD  gal_base2 0.0065  ·  gal_sp10 0.0065
#qc_tear 는 기존(gal_base2)과 common 을 비교하십시오.
#  run                    지점      유효    >0.5m           95% 구간    >1.0m     p99
#  ---------------------------------------------------------------------------
#  ※ 패널 격자가 영상 축에서 -2도 기울어져 있어 그만큼 회전해 쟀습니다.
#  gal_sp10_common      3171   36.9%    8.29%    7.50~ 9.21%    0.00%   0.67m
#  gal_base2            3156   36.8%    8.33%    7.38~ 9.38%    0.00%   0.67m

#  ★ gal_sp10_common 과 구간이 겹쳐 **구분되지 않는** 실행: gal_base2
#    이들 사이의 순위에 의미를 두지 마십시오.

#  ※ >0.5m 가 주지표입니다. p50 이 좋아져도 >0.5m 와 최대가
#    안 움직이면 보기에는 그대로입니다.


# 읽는 법

# 옵션	seams 에서 기대하는 변화
# --seam-panel-penalty	"패널 위" 비율 감소, 어긋남 감소 — 경계가 패널을 피해 물 위로 감
# --seam-cost 2.0	어긋남 감소. 경계 밀도는 늘 수 있음
# --panel-unit 1 · 0	경계 밀도 증가. 패널 단위가 작아질수록 패널 안 끊김이 늘 수 있음
# --layer-surface · 2층	패널 상면 높이를 따로 써서 어긋남 감소 — 물 위 부지라 땅(수면)과 패널 높이 차가 뚜렷함



# 1. 갈평에서 묶어 보기 — 좋은 쪽이던 세 옵션을 함께 씁니다.

export SOLAR=~/Development/sayouzone/solar-thermal/data/solar IMAGE_DIR=$SOLAR/갈평저수지 S=RGB
export G=$(python -c "import json;print('%.4f' % json.load(open('$IMAGE_DIR/gal_rgb_noexp/summary.json'))['gsd_m'])")
PLR=$(python -c "import json;g=json.load(open('$IMAGE_DIR/gal_rgb_noexp/summary.json'))['ground_plane'];print('%.9f,%.9f,%.4f,%.3f,%.3f' % (g['a'],g['b'],g['c'],*g['origin_xy']))")
source scripts/sess_funcs.sh
seed() { mkdir -p "$IMAGE_DIR/$1" && cp -r "$IMAGE_DIR/gal_base2/.feature_cache" "$IMAGE_DIR/$1/"; }
seed gal_combo
PIECE_ARGS="--panel-unit 1 --seam-cost 2.0 --seam-panel-penalty 1.0" piece gal_combo RGB "$PLR"
python scripts/session_tools.py seams --mosaics $IMAGE_DIR/gal_base2/mosaic.tif $IMAGE_DIR/gal_pu1/mosaic.tif $IMAGE_DIR/gal_combo/mosaic.tif

#sess_funcs: piece · sweep · ovr · qc_pair 를 정의했습니다 (IMAGE_DIR=/Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/갈평저수지  S=RGB  G=0.0065  SAYOU_MATCH_SCALE=)
#=== gal_combo  (RGB)   GSD 0.0065   매칭 축소 끔   override 0.000128022,0.000378210,227.1414,275133.707,428576.925   추가 옵션 --panel-unit 1 --seam-cost 2.0 --seam-panel-penalty 1.0
#2026-10-02 18:55:37,834 INFO 비행선 밖 구간 검사: 비행 방향 1°, 비행선 간격 15.5 m — 0곳 0장
#2026-10-02 18:56:02,489 WARNING 기준면을 환경변수로 지정했습니다 — 추정을 건너뜁니다: a=+0.000128 b=+0.000378 c=227.141 경사 0.023°
#2026-10-02 19:00:28,401 INFO 모자이크 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/갈평저수지/gal_combo/mosaic.tif (244장 합성, 충전율 92.3%)
#결과                        단차 비     뚜렷한단차 │ 어긋남 중앙값 (95%)          >5cm (95%)               >2GSD    어긋남90     표본    끝 걸림
#gal_base2                 2.24     28.0% │ 0.030 (0.025~0.035)    41.5% (38.7~44.4%)    65.9%    0.229    995      9%
#gal_pu1                   2.24     27.5% │ 0.020 (0.017~0.025)    36.4% (33.3~39.3%)    59.2%    0.208    984     10%  ← 첫 결과와 구분 안 됨
#gal_combo                 2.13     25.9% │ 0.034 (0.027~0.042)    43.8% (40.6~46.8%)    66.9%    0.232    971      8%  ← 첫 결과와 구분 안 됨
#
#끝 걸림: 탐색 상한에 상관이 걸린 표본(반복 줄 한 주기 옆) — 어긋남 계산에서 뺌
#어긋남: 이음매 양쪽 띠의 무늬가 경계를 따라 밀린 거리 m (중앙값 · 90%) 와 5 cm 넘는 비율 — 기하
#단차 비: 이음매를 가로지른 밝기 차 ÷ 이음매 아닌 곳의 같은 간격 밝기 차 (1 이면 이음매가 안 보임)
#뚜렷한 단차: 이음매 단차 중 '이음매 아닌 곳' 상위 10% 를 넘는 비율 (10% 이면 구분 안 됨)


# 2. 다른 부지에서 같은 비교 — Site-1 RGB 입니다. 광각이고 경사 부지라 갈평(줌, 물 위)과 조건이 다릅니다.

export IMAGE_DIR=$SOLAR/Site-1 S=RGB
export G=$(python -c "import json;print('%.4f' % json.load(open('$IMAGE_DIR/site1_noexp/summary.json'))['gsd_m'])")
P1=$(python -c "import json;g=json.load(open('$IMAGE_DIR/site1_noexp/summary.json'))['ground_plane'];print('%.9f,%.9f,%.4f,%.3f,%.3f' % (g['a'],g['b'],g['c'],*g['origin_xy']))")
piece s1_base RGB "$P1"
seed() { mkdir -p "$IMAGE_DIR/$1" && cp -r "$IMAGE_DIR/s1_base/.feature_cache" "$IMAGE_DIR/$1/"; }
seed s1_pu1;   PIECE_ARGS="--panel-unit 1" piece s1_pu1 RGB "$P1"
seed s1_combo; PIECE_ARGS="--panel-unit 1 --seam-cost 2.0 --seam-panel-penalty 1.0" piece s1_combo RGB "$P1"
python scripts/session_tools.py seams --mosaics $IMAGE_DIR/s1_base/mosaic.tif $IMAGE_DIR/s1_pu1/mosaic.tif $IMAGE_DIR/s1_combo/mosaic.tif
for R in s1_base s1_pu1 s1_combo; do python -c "import json;q=json.load(open('$IMAGE_DIR/$R/summary.json'))['ortho']['quality'];print('%-9s 패널 어긋남 %.3f m' % ('$R', q['panel_misalign_median_m']))"; done
qc_pair s1_base s1_pu1

# === s1_base  (RGB)   GSD 0.0200   매칭 축소 끔   override -0.116317000,0.200190000,92.2050,188450.167,279043.333   추가 옵션 없음
#2026-10-02 19:03:39,421 INFO 비행선 밖 구간 검사: 비행 방향 27°, 비행선 간격 7.9 m — 0곳 0장
#2026-10-02 19:08:47,546 WARNING 기준면을 환경변수로 지정했습니다 — 추정을 건너뜁니다: a=-0.116317 b=+0.200190 c=92.205 경사 13.036°
#2026-10-02 19:09:50,342 INFO 모자이크 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/Site-1/s1_base/mosaic.tif (186장 합성, 충전율 45.3%)
#=== s1_pu1  (RGB)   GSD 0.0200   매칭 축소 끔   override -0.116317000,0.200190000,92.2050,188450.167,279043.333   추가 옵션 --panel-unit 1
#2026-10-02 19:09:54,643 INFO 비행선 밖 구간 검사: 비행 방향 27°, 비행선 간격 7.9 m — 0곳 0장
#2026-10-02 19:11:06,397 WARNING 기준면을 환경변수로 지정했습니다 — 추정을 건너뜁니다: a=-0.116317 b=+0.200190 c=92.205 경사 13.036°
#026-10-02 19:12:09,632 INFO 모자이크 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/Site-1/s1_pu1/mosaic.tif (186장 합성, 충전율 45.3%)
#=== s1_combo  (RGB)   GSD 0.0200   매칭 축소 끔   override -0.116317000,0.200190000,92.2050,188450.167,279043.333   추가 옵션 --panel-unit 1 --seam-cost 2.0 --seam-panel-penalty 1.0
#2026-10-02 19:12:14,797 INFO 비행선 밖 구간 검사: 비행 방향 27°, 비행선 간격 7.9 m — 0곳 0장
#2026-10-02 19:13:19,281 WARNING 기준면을 환경변수로 지정했습니다 — 추정을 건너뜁니다: a=-0.116317 b=+0.200190 c=92.205 경사 13.036°
#2026-10-02 19:15:07,171 INFO 모자이크 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/Site-1/s1_combo/mosaic.tif (186장 합성, 충전율 45.3%)
#결과                        단차 비     뚜렷한단차 │ 어긋남 중앙값 (95%)          >5cm (95%)               >2GSD    어긋남90     표본    끝 걸림
#s1_base                   1.79     18.7% │ 0.154 (0.145~0.161)    81.8% (79.3~84.3%)    85.7%    0.259    813     19%
#s1_pu1                    1.79     18.7% │ 0.154 (0.145~0.161)    81.8% (79.3~84.3%)    85.7%    0.259    813     19%  ← 첫 결과와 구분 안 됨
#s1_combo                  1.76     18.1% │ 0.153 (0.144~0.162)    81.1% (78.3~83.5%)    84.9%    0.255    808     22%  ← 첫 결과와 구분 안 됨
#
#끝 걸림: 탐색 상한에 상관이 걸린 표본(반복 줄 한 주기 옆) — 어긋남 계산에서 뺌
#어긋남: 이음매 양쪽 띠의 무늬가 경계를 따라 밀린 거리 m (중앙값 · 90%) 와 5 cm 넘는 비율 — 기하
#단차 비: 이음매를 가로지른 밝기 차 ÷ 이음매 아닌 곳의 같은 간격 밝기 차 (1 이면 이음매가 안 보임)
#뚜렷한 단차: 이음매 단차 중 '이음매 아닌 곳' 상위 10% 를 넘는 비율 (10% 이면 구분 안 됨)
#s1_base   패널 어긋남 0.569 m
#s1_pu1    패널 어긋남 0.569 m
#s1_combo  패널 어긋남 0.580 m
#GSD  s1_base 0.0200  ·  s1_pu1 0.0200
#qc_tear 는 기존(s1_base)과 common 을 비교하십시오.
#  run                  지점      유효    >0.5m           95% 구간    >1.0m     p99
#  -------------------------------------------------------------------------
#  ※ 패널 격자가 영상 축에서 -27도 기울어져 있어 그만큼 회전해 쟀습니다.
#  s1_base            3970   34.9%   13.40%   12.29~14.43%    0.00%   0.76m
#  s1_pu1_common      3970   34.9%   13.40%   12.29~14.43%    0.00%   0.76m
#
#  ★ s1_base 과 구간이 겹쳐 **구분되지 않는** 실행: s1_pu1_common
#    이들 사이의 순위에 의미를 두지 마십시오.
#
#  ※ >0.5m 가 주지표입니다. p50 이 좋아져도 >0.5m 와 최대가
#    안 움직이면 보기에는 그대로입니다.
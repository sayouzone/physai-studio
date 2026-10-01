# 세 조각 처리용 셸 함수 — zsh · bash 공용
#   source scripts/sess_funcs.sh
# 먼저 export 해 둘 것:  IMAGE_DIR  S(사진 폴더 이름)  G(GSD)
# P10 을 쓰는 부지는      export SAYOU_MATCH_SCALE=0.25

_sess_check() {
  if [ -z "$IMAGE_DIR" ] || [ -z "$G" ]; then
    echo "IMAGE_DIR 와 G 를 먼저 export 하십시오 (예: export G=0.045)"; return 1
  fi
}

# piece <출력 이름> <사진 폴더 이름> [override]
#   세션·띠 조각 하나를 노출 보정 없이 만든다
piece() {
  _sess_check || return 1
  if [ -n "$3" ]; then export SAYOU_PLANE_OVERRIDE="$3"; fi
  # PIECE_ARGS — 이 실행에만 더할 파이프라인 옵션 (끝나면 지움). bash · zsh 모두 같은 방식으로 나눔
  local extra; eval "extra=(${PIECE_ARGS:-})"
  echo "=== $1  ($2)   GSD $G   매칭 축소 ${SAYOU_MATCH_SCALE:-끔}   override ${SAYOU_PLANE_OVERRIDE:-없음}   추가 옵션 ${PIECE_ARGS:-없음}"
  python scripts/homography_pipeline.py --image-dir "$IMAGE_DIR/$2" --output-dir "$IMAGE_DIR/$1" \
    --offnadir-frac 0.32 --smooth-weak-attitude --gsd "$G" --panel-unit 2 --no-two-layer \
    --no-exposure-comp "${extra[@]}" 2>&1 | tee "$IMAGE_DIR/$1.log" | \
    grep -E "환경변수로 지정|기준면 출처|매칭용 축소|모자이크 저장|비행선 밖 구간|순번 제외|장을 빼고"
  unset SAYOU_PLANE_OVERRIDE PIECE_ARGS
  [ -f "$IMAGE_DIR/$1/summary.json" ] || echo "★ $1/summary.json 이 없습니다 — $IMAGE_DIR/$1.log 끝을 확인하십시오"
}

# sweep <실행 이름> <사진 폴더 이름> <z-min> <z-max> [측정 시작 프레임 목록]
#   plane_sweep 전체 출력을 <실행 이름>.sweep.txt 로 남기고 요약만 보여 준다
#   다섯째 인자(쉼표 목록)를 주면 그 프레임들에서만 잰다 — 옥상 위 프레임만 고를 때
sweep() {
  _sess_check || return 1
  if [ ! -f "$IMAGE_DIR/$1/summary.json" ]; then
    echo "★ $IMAGE_DIR/$1/summary.json 이 없습니다 — 먼저 piece $1 $2"; return 1
  fi
  local N ST P
  N=$(ls "$IMAGE_DIR/$2" | grep -ciE "\.jpg$")      # 사진만 셈 (결과 파일이 섞여도 어긋나지 않게)
  if [ -n "$5" ]; then ST="$5"
  else ST=$(python -c "import numpy as np;print(','.join(str(int(i)) for i in np.linspace(15,$N-20,12)))"); fi
  P=$(python -c "import json;print(round(json.load(open('$IMAGE_DIR/$1/summary.json'))['ortho']['panel_unit_info']['unit_m'],2))")
  echo "=== sweep $1   $N장   피치 $P m   범위 $3 ~ $4 m"
  python scripts/plane_sweep.py --image-dir "$IMAGE_DIR/$2" --use-ba "$IMAGE_DIR/$1/cameras.npz" \
    --summary "$IMAGE_DIR/$1/summary.json" --pitch-m "$P" --count 8 --z-min "$3" --z-max "$4" \
    --starts "$ST" > "$IMAGE_DIR/$1.sweep.txt" 2>&1
  grep -E "X [0-9]+ +Y [0-9]+|오프셋 중앙값|개선입니다|범위 끝 지점이|적합 잔차" "$IMAGE_DIR/$1.sweep.txt"
}

# ovr <실행 이름>   — plane_sweep 결과로 override 후보 계산
ovr() {
  python scripts/session_tools.py override --sweep "$IMAGE_DIR/$1.sweep.txt" --summary "$IMAGE_DIR/$1/summary.json"
}


# qc_pair <기준 폴더> <비교 폴더>
#   카메라 궤적으로 창을 잡고, 기준이 채운 자리만 남긴 비교본으로 이음매를 비교
#   (채운 면적 차이로 판정이 뒤집히는 것을 막음 — 옥산 14.11 → 공통 면적 11.29%)
qc_pair() {
  if [ -z "$IMAGE_DIR" ]; then echo "IMAGE_DIR 를 먼저 export 하십시오"; return 1; fi
  for d in "$1" "$2"; do
    [ -f "$IMAGE_DIR/$d/mosaic.tif" ] || { echo "★ $IMAGE_DIR/$d/mosaic.tif 이 없습니다"; return 1; }
  done
  local X Y SW SH P
  read X Y SW SH P <<< "$(python -c "
import json,numpy as np
d='$IMAGE_DIR'; s=json.load(open(d+'/$1/summary.json')); g=s['gsd_m']
c=np.load(d+'/$1/cameras.npz',allow_pickle=True)['cams_opt']
print('%.0f %.0f %d %d %.2f' % (c[:,0].min()-5, c[:,1].max()+5, int((np.ptp(c[:,0])+10)/g), int((np.ptp(c[:,1])+10)/g), s['ortho']['panel_unit_info']['unit_m']))")"
  python -c "
import rasterio as r
a=abs(r.open('$IMAGE_DIR/$1/mosaic.tif').transform.a); b=abs(r.open('$IMAGE_DIR/$2/mosaic.tif').transform.a)
print('GSD  %s %.4f  ·  %s %.4f%s' % ('$1', a, '$2', b, '' if abs(a-b) < 1e-4 else '   ★ 다름 — 비교본은 기준 격자로 맞춰집니다'))"
  python scripts/session_tools.py common --ref "$IMAGE_DIR/$1/mosaic.tif" --new "$IMAGE_DIR/$2/mosaic.tif" \
      --out-common "$IMAGE_DIR/$2_common/mosaic.tif" | tail -1
  python scripts/qc_tear.py "$IMAGE_DIR/$1/mosaic.tif" "$IMAGE_DIR/$2_common/mosaic.tif" \
      --window --x "$X" --y "$Y" --size "$SW" "$SH" --block-m 12 --pitch-m "$P"
}

echo "sess_funcs: piece · sweep · ovr · qc_pair 를 정의했습니다 (IMAGE_DIR=$IMAGE_DIR  S=$S  G=$G  SAYOU_MATCH_SCALE=${SAYOU_MATCH_SCALE:-})"

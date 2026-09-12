IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/EWP-서오창IC-2

python scripts/qc_row_phase.py $IMAGE_DIR/ab_rgb/1_noautoplane/mosaic.tif \
    --mask-pct 85 --max-px 6000 --out $IMAGE_DIR/image/p1.png

python scripts/debug_single_frame.py \
    --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/chk_base \
    --use-ba $IMAGE_DIR/out_rgb/cameras.npz --measure --count 12

python scripts/debug_single_frame.py \
    --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/chk_1 \
    --use-ba $IMAGE_DIR/ab_rgb/1_noautoplane/cameras.npz --measure --count 12

python scripts/debug_single_frame.py \
    --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/chk_init \
    --measure --count 12

python scripts/debug_single_frame.py \
    --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/chk_ba \
    --use-ba $IMAGE_DIR/ab_rgb/1_noautoplane/cameras.npz --measure --count 12

python scripts/compare_runs.py $IMAGE_DIR/ab_rgb/* $IMAGE_DIR/out_rgb

python scripts/patch_sayou.py --root <위 경로의 sayou 폴더> --skip P1 --apply
python scripts/patch_sayou.py --root label_studio/sayou --skip P1 --apply

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB \
  --output-dir $IMAGE_DIR/ab_rgb/1_noautoplane \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --gsd 0.03 \
  --no-auto-plane



python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB \
  --output-dir $IMAGE_DIR/ab_rgb/2_matchcheck \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --gsd 0.03 \
  --no-auto-plane --rtk-match-check

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB \
  --output-dir $IMAGE_DIR/ab_rgb/3_minobs \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --gsd 0.03 \
  --no-auto-plane --rtk-match-check --min-frame-obs 60

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB \
  --output-dir $IMAGE_DIR/ab_rgb/4_k12 \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --gsd 0.03 \
  --no-auto-plane --rtk-match-check --min-frame-obs 60 \
  --k-neighbors 12

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB \
  --output-dir $IMAGE_DIR/ab_rgb/5_layer \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --gsd 0.03 \
  --no-auto-plane --rtk-match-check --min-frame-obs 60 \
  --layer-surface --layer-cell 0.5

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB \
  --output-dir $IMAGE_DIR/ab_rgb/6_offnadir \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --gsd 0.03 \
  --no-auto-plane --rtk-match-check --max-offnadir 0.20

python scripts/compare_runs.py $IMAGE_DIR/ab_rgb/* $IMAGE_DIR/ab_out_rgb

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB \
  --output-dir $IMAGE_DIR/ab_rgb/6b_offnadir \
  --panel-unit 2 --smooth-weak-attitude --gsd 0.03 \
  --no-auto-plane --rtk-match-check --max-offnadir 0.20

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB \
  --output-dir $IMAGE_DIR/ab_rgb/5b_paneltop \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.03 \
  --no-auto-plane --rtk-match-check --panel-top 1.0

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/ab_rgb/7_p2only \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.03

SAYOU_ANCHOR_ANGLE_DEG=999 python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/ab_rgb/7b_p2off \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.03

python scripts/compare_runs.py $IMAGE_DIR/ab_rgb/* $IMAGE_DIR/out_rgb


python scripts/patch_sayou.py --root label_studio/sayou --only P1b --revert

python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 --apply

# BA 이전 — RTK+짐벌 초기값
python scripts/debug_single_frame.py \
    --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/chk_init \
    --measure --count 12

# BA 이후
python scripts/debug_single_frame.py \
    --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/chk_ba \
    --use-ba $IMAGE_DIR/ab_rgb/7_p2only/cameras.npz --measure --count 12


python scripts/plane_sweep.py \
  --image-dir $IMAGE_DIR/RGB \
  --use-ba $IMAGE_DIR/ab_rgb/7_p2only/cameras.npz \
  --z-min 165 --z-max 175 --z-step 1.0 --count 8 \
  --plot $IMAGE_DIR/sweep.png

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/ab_rgb/8_layer2 \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.03 \
  --panel-top 1.0 --layer-surface --layer-cell 0.5 --layer-gap 1.0


python scripts/patch_sayou.py --root label_studio/sayou --skip P1 --apply

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
  --use-ba $IMAGE_DIR/ab_rgb/7_p2only/cameras.npz \
  --summary $IMAGE_DIR/ab_rgb/7_p2only/summary.json \
  --start 600 --count 8 --plot $IMAGE_DIR/sweep_600.png

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
  --use-ba $IMAGE_DIR/ab_rgb/7_p2only/cameras.npz \
  --summary $IMAGE_DIR/ab_rgb/7_p2only/summary.json \
  --start 300 --count 8 --plot $IMAGE_DIR/sweep_300.png

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/ab_rgb/9_p4 \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.03

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
  --use-ba $IMAGE_DIR/ab_rgb/9_p4/cameras.npz --summary $IMAGE_DIR/ab_rgb/9_p4/summary.json \
  --start 400 --count 8 --plot $IMAGE_DIR/sweep_ba.png

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
  --use-ba $IMAGE_DIR/ab_rgb/9_p4/cameras.npz --summary $IMAGE_DIR/ab_rgb/9_p4/summary.json \
  --smoothed --start 400 --count 8 --plot $IMAGE_DIR/sweep_sm.png

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
  --use-ba $IMAGE_DIR/ab_rgb/9_p4/cameras.npz --summary $IMAGE_DIR/ab_rgb/9_p4/summary.json \
  --smoothed --start 600 --count 8 --plot $IMAGE_DIR/sweep_600_sm.png

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
  --use-ba $IMAGE_DIR/ab_rgb/9_p4/cameras.npz --summary $IMAGE_DIR/ab_rgb/9_p4/summary.json \
  --start 500 --count 8 --plot $IMAGE_DIR/sweep_500.png

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
  --use-ba $IMAGE_DIR/ab_rgb/9_p4/cameras.npz --summary $IMAGE_DIR/ab_rgb/9_p4/summary.json \
  --smoothed --start 500 --count 8 --plot $IMAGE_DIR/sweep_500_sm.png


N=$IMAGE_DIR/ab_rgb/9_p4

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
  --use-ba $N/cameras.npz --summary $N/summary.json \
  --pitch-m 6.00 --start 600 --count 8 --plot $IMAGE_DIR/sw600.png

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
  --use-ba $N/cameras.npz --summary $N/summary.json \
  --pitch-m 6.00 --smoothed --start 600 --count 8 --plot $IMAGE_DIR/sw600_sm.png

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
  --use-ba $N/cameras.npz --summary $N/summary.json \
  --pitch-m 6.00 --starts 100,300,600,800 --count 8


python scripts/patch_sayou.py --root label_studio/sayou --skip P1 --apply

export SAYOU_PLANE_OVERRIDE="-0.023678,0.049080,171.713,237203.233,458883.066"

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/ab_rgb/10_plane \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.03


unset SAYOU_PLANE_OVERRIDE
python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
  --use-ba $IMAGE_DIR/ab_rgb/10_plane/cameras.npz \
  --summary $IMAGE_DIR/ab_rgb/10_plane/summary.json \
  --pitch-m 6.00 --starts 100,300,600,800 --count 8


export SAYOU_PLANE_OVERRIDE="-0.023678,0.049080,171.713,237203.233,458883.066"

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/ab_rgb/11_layer \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.03 \
  --layer-surface --layer-cell 0.5 \
  --no-two-layer-auto --two-layer-max-shift-px 30

export SAYOU_PLANE_OVERRIDE="-0.023678,0.049080,170.750,237203.233,458883.066"

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/ab_rgb/12_mid \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.03


python scripts/qc_row_phase.py $IMAGE_DIR/ab_rgb/11_layer/mosaic.tif \
  --mask-pct 85 --max-px 6000 --out $IMAGE_DIR/image/p11.png

python scripts/qc_row_phase.py $IMAGE_DIR/ab_rgb/12_mid/mosaic.tif \
  --mask-pct 85 --max-px 6000 --out $IMAGE_DIR/image/p12.png


python scripts/patch_sayou.py --root label_studio/sayou --skip P1 --apply

export SAYOU_PLANE_OVERRIDE="-0.023678,0.049080,171.713,237203.233,458883.066"

# 13: 2층 마스크 정상화 + 시임이 패널을 피하게
python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/ab_rgb/13_seam \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.03 \
  --layer-gap 1.91 --layer-surface --layer-cell 0.5 \
  --no-two-layer-auto --two-layer-max-shift-px 30 \
  --seam-panel-penalty 1.0

python scripts/qc_tear.py $IMAGE_DIR/ab_rgb/*/mosaic.tif --x 237150 --y 458960


python scripts/patch_sayou.py --root label_studio/sayou --skip P1 --apply

export SAYOU_PLANE_OVERRIDE="-0.023678,0.049080,171.713,237203.233,458883.066"

# 14: 13_seam 에서 시임 벌점만 뺀다 (2층 마스크 정상화는 유지)
python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/ab_rgb/14_layer_fixed \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.03 \
  --no-two-layer-auto --two-layer-max-shift-px 30

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/ab_rgb/15_seam03 \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.03 \
  --no-two-layer-auto --two-layer-max-shift-px 30 \
  --seam-panel-penalty 0.3

python scripts/qc_tear.py $IMAGE_DIR/ab_rgb/*/mosaic.tif --x 237150 --y 458960 --size 3000 2500
python scripts/qc_tear.py $IMAGE_DIR/ab_rgb/*/mosaic.tif --x 237220 --y 458870 --size 3000 2500

unset SAYOU_PLANE_OVERRIDE

for T in 1.5 2.0 2.5; do
  python scripts/homography_pipeline.py \
    --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/ab_rgb/16_top$T \
    --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.03 \
    --no-auto-plane --panel-top $T
done

for T in 0.5 1.0; do
  python scripts/homography_pipeline.py \
    --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/ab_rgb/17_top$T \
    --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.03 \
    --no-auto-plane --panel-top $T
done


python scripts/qc_tear.py $IMAGE_DIR/ab_rgb/*/mosaic.tif

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 --apply


python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/final \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --no-auto-plane --panel-top 1.0

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/f4 \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --no-auto-plane --panel-top 1.0 --focal-rounds 4

exiftool -DewarpData -DewarpFlag -CalibratedFocalLength -RelativeAltitude -LRFTargetDistance \
  $IMAGE_DIR/RGB/DJI_20250828103410_0001_W.JPG

exiftool -q -n -T -RelativeAltitude -LRFTargetDistance -GPSAltitude \
  $IMAGE_DIR/RGB/*.JPG > $IMAGE_DIR/alt.tsv
head -3 $IMAGE_DIR/alt.tsv; wc -l $IMAGE_DIR/alt.tsv

python - <<'EOF'
import numpy as np
image_dir = '/Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/EWP-서오창IC-2'
d=np.genfromtxt(f'{image_dir}/alt.tsv', dtype=float)
rel,lrf,alt = d[:,0], d[:,1], d[:,2]
ok = np.isfinite(lrf) & (lrf>10) & (lrf<100)
print('프레임 %d장, LRF 유효 %d장'%(len(d), ok.sum()))
for n,v in [('RelativeAltitude',rel),('LRF',lrf[ok] if ok.any() else lrf),('GPSAltitude',alt)]:
    print('  %-18s 중앙값 %8.3f  p10 %8.3f  p90 %8.3f'%(n,np.nanmedian(v),
          np.nanpercentile(v,10), np.nanpercentile(v,90)))
print()
print('  LRF − Relative  중앙값 %.3f m  (표준편차 %.3f)'%(
    np.nanmedian(lrf[ok]-rel[ok]), np.nanstd(lrf[ok]-rel[ok])))
EOF


python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P7 --apply

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/f7 \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --no-auto-plane --panel-top 1.0

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
  --use-ba $IMAGE_DIR/f7/cameras.npz --summary $IMAGE_DIR/f7/summary.json \
  --pitch-m 6.00 --starts 300,600,800 --count 8

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
  --use-ba $IMAGE_DIR/f7/cameras.npz --summary $IMAGE_DIR/f7/summary.json \
  --pitch-m 6.00 --starts 100,250,400,550,700,850 --count 8

for T in 0 0.5 1.0; do
  python scripts/homography_pipeline.py \
    --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/g_top$T \
    --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
    --gsd 0.03 --no-auto-plane --panel-top $T
done

python scripts/qc_tear.py $IMAGE_DIR/g_top*/mosaic.tif

python scripts/qc_tear.py $IMAGE_DIR/f7/mosaic.tif $IMAGE_DIR/final/mosaic.tif

python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P7 --apply

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/final2 \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --no-auto-plane --panel-top 1.0

python scripts/qc_tear.py $IMAGE_DIR/final2/mosaic.tif \
    --window --x 237150 --y 458960 --size 3000 2500

python scripts/qc_tear.py $IMAGE_DIR/final2/mosaic.tif \
    --window --x 237220 --y 458870 --size 3000 2500

python scripts/qc_tear.py $IMAGE_DIR/*/mosaic.tif

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
  --use-ba $IMAGE_DIR/final2/cameras.npz \
  --summary $IMAGE_DIR/final2/summary.json \
  --pitch-m 6.00 --starts 100,250,400,550,700,850 --count 8

python scripts/frame_conditioning.py $IMAGE_DIR/final2/cameras.npz \
  --out-dir $IMAGE_DIR/final2/cond

exiftool -q -n -T -RelativeAltitude -LRFTargetDistance -GPSAltitude \
  $IMAGE_DIR/RGB/*.JPG > $IMAGE_DIR/alt.tsv

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
  --use-ba $IMAGE_DIR/final3/cameras.npz \
  --summary $IMAGE_DIR/final3/summary.json \
  --pitch-m 6.00 --starts 100,250,400,550,700,850 --count 8

python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P7 --apply
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P7 --apply

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
  --use-ba $IMAGE_DIR/release/cameras.npz \
  --summary $IMAGE_DIR/release/summary.json \
  --pitch-m 6.00 --starts 100,250,400,550,700,850 --count 8


python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P1 P2 P3 P4 P5 P7 --apply

export SAYOU_PLANE_OVERRIDE="-0.005449504793124913,0.024921527837221107,173.462,237218.87966874262,458883.3983103598"

python scripts/plane_sweep.py --image-dir <RGB> \
  --use-ba <출력>/cameras.npz --summary <출력>/summary.json \
  --pitch-m <행 피치> --count 8 --starts 100,200,300,400,500,600,700,800

exiftool -q -n -T -RelativeAltitude -LRFTargetDistance -GPSAltitude \
  <RGB>/*.JPG > alt.tsv



python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P7 --apply

python scripts/frame_conditioning.py $IMAGE_DIR/rgb_release/cameras.npz \
  --out-dir $IMAGE_DIR/rgb_release/cond




# EWP-서오창IC-2

## RGB

IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/EWP-서오창IC-2

#### 패치

unset SAYOU_PLANE_OVERRIDE

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P7 --apply

#### 실행

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/rgb_release \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --no-auto-plane --panel-top 1.0

python scripts/qc_tear.py $IMAGE_DIR/rgb_release/mosaic.tif \
    --window --x 237150 --y 458960 --size 6000 5000 --block-m 12

python scripts/qc_tear.py $IMAGE_DIR/rgb_release/mosaic.tif \
    --window --x 237220 --y 458870 --size 6000 5000 --block-m 12

python scripts/frame_conditioning.py $IMAGE_DIR/rgb_release/cameras.npz \
  --out-dir $IMAGE_DIR/rgb_release/cond

## IR

#### 패치

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P1 P2 P3 P4 P5 P7 --apply

#### 실행

export SAYOU_PLANE_OVERRIDE="-0.005449504793124913,0.024921527837221107,173.462,237218.87966874262,458883.3983103598"

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/TM --output-dir $IMAGE_DIR/tm_release \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude

#### 검증 기대값

python scripts/qc_tear.py $IMAGE_DIR/tm_release/mosaic.tif \
    --window --x 237150 --y 458960 --size 4000 3500 --block-m 12 --pitch-m 6.0
# >0.5m 10.77%, >1.0m 3.75%, p99 1.81 m



# 옥산 1호

## RGB

IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/옥산_1호

#### 패치

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou \
    --only P2 P3 P4 P5 P7 P8 --apply

#### 실행

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/rgb_release \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --no-auto-plane --panel-top 1.0

# 시선각 45° 상한
# 문제가 더 증가되었음
export SAYOU_OFFNADIR_CEILING=1.0

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/rgb_ceil \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --no-auto-plane --panel-top 1.0

# A. 노출 보정 끄기 — 이득 비 2.00배가 원인인지
python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/a_noexp \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --no-auto-plane --panel-top 1.0 --no-exposure-comp

# B. 픽셀별 완화 끄기 — 반지름 제각각인 원이 원인인지
python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/b_nofallback \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --no-auto-plane --panel-top 1.0 --no-offnadir-fallback

export SAYOU_PLANE_OVERRIDE="-0.009230714,0.068037923,74.8236,230822.571,457761.429"

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/c_tilt \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --no-exposure-comp

export SAYOU_PLANE_OVERRIDE="-0.009230714,0.068037923,74.8236,230822.571,457761.429"

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/d_tilt \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --no-exposure-comp

# 후보 B — LRF 1024점 기반 (경사 2.187°, 지면 + 1.0 m)
export SAYOU_PLANE_OVERRIDE="0.001790837,0.038142657,73.6230,230784.323,457767.230"

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/e_lrf \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --no-exposure-comp

for K in 0.6 0.4; do
  SAYOU_OFFNADIR_CEILING=$K python scripts/homography_pipeline.py \
    --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/ceil$K \
    --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
    --no-auto-plane --panel-top 1.0 --no-exposure-comp
done

# f: 게이트 사실상 해제 — 점수(테이퍼)만으로 승자 결정
python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/f_nogate \
  --panel-unit 2 --smooth-weak-attitude \
  --no-auto-plane --panel-top 1.0 --no-exposure-comp \
  --max-offnadir 10 --no-offnadir-auto

unset SAYOU_OFFNADIR_CEILING

# f: 게이트 사실상 해제 — 점수(테이퍼)만으로 승자 결정
python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/f_nogate \
  --panel-unit 2 --smooth-weak-attitude \
  --no-auto-plane --panel-top 1.0 --no-exposure-comp \
  --max-offnadir 10 --no-offnadir-auto

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/g_gate1 \
  --panel-unit 2 --smooth-weak-attitude \
  --no-auto-plane --panel-top 1.0 --no-exposure-comp \
  --max-offnadir 1.0 --no-offnadir-auto

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P7 --apply

#### 검증

# X 230479~231044, Y 457617~457859
# 카메라 밴드 안쪽 (양호 기대)
python scripts/qc_tear.py \
    $IMAGE_DIR/rgb_release/mosaic.tif $IMAGE_DIR/rgb_ceil/mosaic.tif \
    --window --x 230700 --y 457800 --size 6000 5000 --block-m 12

# 카메라 밴드 바깥 (원형 무늬 기대)
python scripts/qc_tear.py \
    $IMAGE_DIR/rgb_release/mosaic.tif $IMAGE_DIR/rgb_ceil/mosaic.tif \
    --window --x 230700 --y 457859 --size 6000 4000 --block-m 12

python scripts/qc_tear.py $IMAGE_DIR/rgb_release/mosaic.tif \
    $IMAGE_DIR/a_noexp/mosaic.tif $IMAGE_DIR/b_nofallback/mosaic.tif \
    --window --x 230700 --y 457800 --size 6000 5000 --block-m 12

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
    --use-ba $IMAGE_DIR/rgb_release/cameras.npz \
    --summary $IMAGE_DIR/rgb_release/summary.json \
    --pitch-m 5.65 --count 8 --z-min -4 --z-max 16 \
    --starts 100,200,300,400,500,600,700,800,900,1000

python scripts/frame_conditioning.py $IMAGE_DIR/rgb_release/cameras.npz \
  --out-dir $IMAGE_DIR/rgb_release/cond

unset SAYOU_PLANE_OVERRIDE
for R in d_tilt e_lrf; do
  echo "=== $R ==="
  python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
      --use-ba $IMAGE_DIR/$R/cameras.npz --summary $IMAGE_DIR/$R/summary.json \
      --pitch-m 5.65 --count 8 --z-min -8 --z-max 8 \
      --starts 100,200,300,400,500,600,700,800,900,1000
done

python scripts/qc_tear.py \
    $IMAGE_DIR/rgb_release/mosaic.tif \
    $IMAGE_DIR/d_tilt/mosaic.tif $IMAGE_DIR/e_lrf/mosaic.tif \
    --window --x 230700 --y 457800 --size 6000 5000 --block-m 12

python scripts/qc_tear.py \
    $IMAGE_DIR/rgb_release/mosaic.tif $IMAGE_DIR/f_nogate/mosaic.tif $IMAGE_DIR/g_gate1/mosaic.tif \
    --window --x 230520 --y 457825 --size 20000 7000 --block-m 12

## IR

#### 패치

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P1 P2 P3 P4 P5 P7 --apply

#### 실행

export SAYOU_PLANE_OVERRIDE="-0.005449504793124913,0.024921527837221107,173.462,237218.87966874262,458883.3983103598"

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/TM --output-dir $IMAGE_DIR/tm_release \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude

#### 검증 기대값

python scripts/qc_tear.py $IMAGE_DIR/tm_release/mosaic.tif \
    --window --x 230700 --y 457800 --size 6000 5000 --block-m 12 --pitch-m 6.0
# >0.5m 10.77%, >1.0m 3.75%, p99 1.81 m

python scripts/frame_conditioning.py $IMAGE_DIR/tm_release/cameras.npz \
  --out-dir $IMAGE_DIR/tm_release/cond



python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P7 P8 P9 --apply


# 카메라 밴드 안쪽 (양호 기대)
python scripts/qc_tear.py \
    $IMAGE_DIR/rgb_release/mosaic.tif $IMAGE_DIR/rgb_ceil/mosaic.tif \
    --window --x 230700 --y 457800 --size 6000 5000 --block-m 12

# 카메라 밴드 바깥 (원형 무늬 기대)
python scripts/qc_tear.py \
    $IMAGE_DIR/rgb_release/mosaic.tif $IMAGE_DIR/rgb_ceil/mosaic.tif \
    --window --x 230700 --y 457859 --size 6000 4000 --block-m 12

unset SAYOU_OFFNADIR_CEILING
python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P7 --apply



# 에스엘에너지_사천시

# RGB

IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/에스엘에너지_사천시


exiftool -q -n -T -GPSLatitude -GPSLongitude -GPSAltitude \
    -RelativeAltitude -LRFTargetDistance $IMAGE_DIR/RGB/*.JPG > $IMAGE_DIR/exif.tsv

python scripts/site_triage.py $IMAGE_DIR/exif.tsv

#### 패치

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P7 --apply

#### 실행

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/base \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/k16 \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --k-neighbors 16 --rtk-match-check

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/k16_g15 \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --k-neighbors 16 --rtk-match-check --gsd 0.015

# 평면 오차 시각화
python scripts/qc_tear.py $IMAGE_DIR/rgb_release/mosaic.tif \
    --window --x 230140 --y 458250 --size 6000 5000 --block-m 12

python scripts/qc_tear.py $IMAGE_DIR/rgb_release/mosaic.tif \
    --window --x 230140 --y 458230 --size 6000 5000 --block-m 12

#### 검증 기대값

python scripts/qc_tear.py \
    $IMAGE_DIR/k16_g15/mosaic.tif $IMAGE_DIR/sacheon/mosaic.tif \
    --window --x 290760 --y 267000 --size 6000 5000 --block-m 12 --pitch-m 2.0

python scripts/frame_conditioning.py $IMAGE_DIR/rgb_release/cameras.npz \
  --out-dir $IMAGE_DIR/rgb_release/cond

gdal_translate -srcwin 0 30000 4000 992 \
  $IMAGE_DIR/k16/mosaic.tif $IMAGE_DIR/tmp/bottom.tif && gdalinfo -stats $IMAGE_DIR/tmp/bottom.tif | tail -5

## TM

export SAYOU_PLANE_OVERRIDE="<계산값>"
python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P1 P2 P3 P4 P5 P7 --apply

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/TM --output-dir $IMAGE_DIR/tm_plane \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude

# 갈평저수지

IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/갈평저수지

## RGB

exiftool -q -n -T -GPSLatitude -GPSLongitude -GPSAltitude \
    -RelativeAltitude -LRFTargetDistance $IMAGE_DIR/RGB/*.JPG > $IMAGE_DIR/exif.tsv

python scripts/site_triage.py $IMAGE_DIR/exif.tsv

#### 패치

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P7 --apply

#### 실행

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/base \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude \
  --rtk-match-check

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/g15 \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.015

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/release \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.015

#### 검증

gdalinfo $IMAGE_DIR/base/mosaic.tif | grep -A4 "Corner"

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
    --use-ba $IMAGE_DIR/g15/cameras.npz --summary $IMAGE_DIR/g15/summary.json \
    --pitch-m 2.0 --count 8 --z-min -2 --z-max 2 \
    --starts 20,45,70,95,120,145,170,195,220

python scripts/frame_conditioning.py $IMAGE_DIR/rgb_release/cameras.npz \
  --out-dir $IMAGE_DIR/rgb_release/cond

## TM

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P1 P2 P3 P4 P5 P7 --apply

#### 실행

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/TM --output-dir $IMAGE_DIR/tm_release \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude

python scripts/frame_conditioning.py $IMAGE_DIR/tm_release/cameras.npz \
  --out-dir $IMAGE_DIR/tm_release/cond

# 그린환경센터

IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/그린환경센터

exiftool -q -n -T -GPSLatitude -GPSLongitude -GPSAltitude \
    -RelativeAltitude -LRFTargetDistance $IMAGE_DIR/RGB/*.JPG > $IMAGE_DIR/exif.tsv

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P1 P2 P3 P4 P5 P7 --apply

python scripts/site_triage.py $IMAGE_DIR/exif.tsv

#### 실행

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/base \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.015

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/native \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude

export SAYOU_PLANE_OVERRIDE="0.037268,-0.011732,115.881,192860.976,235020.780"

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/tilt \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.015

export SAYOU_PLANE_OVERRIDE="0.037268,-0.011732,115.881,192860.976,235020.780"
python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/tilt_rtk \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.015 \
  --rtk-match-check

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P5 P7 --apply

export SAYOU_PLANE_OVERRIDE="0.037268,-0.011732,115.881,192860.976,235020.780"

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/release \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.015

#### 평가

python scripts/qc_tear.py $IMAGE_DIR/base/mosaic.tif \
    --window --x 192800 --y 235050 --size 6000 4000 --block-m 12 --pitch-m 1.99

python scripts/qc_tear.py $IMAGE_DIR/base/mosaic.tif $IMAGE_DIR/native/mosaic.tif \
    --window --x 192800 --y 235050 --size 6000 4000 --block-m 12 --pitch-m 1.99

python scripts/qc_tear.py $IMAGE_DIR/native/mosaic.tif \
    --window --x 192800 --y 235050 --size 14400 9600 --block-m 12 --pitch-m 1.99

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
    --use-ba $IMAGE_DIR/base/cameras.npz --summary $IMAGE_DIR/base/summary.json \
    --pitch-m 1.99 --count 8 --z-min -4 --z-max 4 \
    --starts 40,80,120,160,200,240,280,320,360

python scripts/qc_tear.py $IMAGE_DIR/tilt/mosaic.tif $IMAGE_DIR/tilt_rtk/mosaic.tif \
    --window --x 192800 --y 235050 --size 14400 9600 --block-m 12 --pitch-m 1.99

unset SAYOU_PLANE_OVERRIDE
python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
    --use-ba $IMAGE_DIR/tilt/cameras.npz --summary $IMAGE_DIR/tilt/summary.json \
    --pitch-m 1.99 --count 8 --z-min -3 --z-max 3 \
    --starts 40,80,120,160,200,240,280,320,360 2>&1 | tail -20

## TM

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/TM --output-dir $IMAGE_DIR/tm_release \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude

python scripts/frame_conditioning.py $IMAGE_DIR/tm_release/cameras.npz \
  --out-dir $IMAGE_DIR/tm_release/cond

# K_Demo

IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/K_Demo

exiftool -q -n -T -GPSLatitude -GPSLongitude -GPSAltitude \
    -RelativeAltitude -LRFTargetDistance $IMAGE_DIR/RGB/*.JPG > $IMAGE_DIR/exif.tsv

python scripts/site_triage.py $IMAGE_DIR/exif.tsv

exiftool -FocalLengthIn35mmFormat -ImageWidth $(ls $IMAGE_DIR/RGB/*.JPG | head -1)

python scripts/site_triage.py $IMAGE_DIR/exif.tsv --f35 47 --image-w 5184


## RGB

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P7 --apply

#### 실행

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/base \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.015

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P5 P7 --apply

export SAYOU_PLANE_OVERRIDE="0.001564698,0.010320351,34.9500,181017.583,492620.864"

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/tilt \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.015


#### 검증

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
    --use-ba $IMAGE_DIR/base/cameras.npz --summary $IMAGE_DIR/base/summary.json \
    --pitch-m <panel_unit_info.unit_m> --count 8 --z-min -3 --z-max 3 \
    --starts 50,100,150,200,250,300,350,400,450 2>&1 | tail -25

python -c "import json;print(json.load(open('$IMAGE_DIR/base/summary.json'))['ortho']['panel_unit_info']['unit_m'])"
2.5568095955590255

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
    --use-ba $IMAGE_DIR/base/cameras.npz --summary $IMAGE_DIR/base/summary.json \
    --pitch-m 2.56 --count 8 --z-min -3 --z-max 3 \
    --starts 50,100,150,200,250,300,350,400,450 2>&1 | tail -25

python scripts/qc_tear.py $IMAGE_DIR/base/mosaic.tif $IMAGE_DIR/tilt/mosaic.tif \
    --window --x 192800 --y 235050 --size 6000 4000 --block-m 12 --pitch-m 2.56

python scripts/qc_tear.py $IMAGE_DIR/base/mosaic.tif $IMAGE_DIR/tilt/mosaic.tif \
    --window --x 192800 --y 235050 --size 6000 4000 --block-m 12 --pitch-m 1.99

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
    --use-ba $IMAGE_DIR/base/cameras.npz --summary $IMAGE_DIR/base/summary.json \
    --pitch-m 2.56 --count 8 --z-min -8 --z-max 2 \
    --starts 50,100,150,200,250,300,350,400,450 2>&1 | tail -25

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
    --use-ba $IMAGE_DIR/base/cameras.npz --summary $IMAGE_DIR/base/summary.json \
    --pitch-m 2.56 --count 8 --z-min -8 --z-max 2 \
    --starts 30,80,130,180,230,280,330,380,430,480 2>&1 | tail -25

## TM

#### 실행₩

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/TM --output-dir $IMAGE_DIR/tm_release \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude

python scripts/frame_conditioning.py $IMAGE_DIR/tm_release/cameras.npz \
  --out-dir $IMAGE_DIR/tm_release/cond


# Site-1

IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/Site-1

exiftool -q -n -T -GPSLatitude -GPSLongitude -GPSAltitude \
    -RelativeAltitude -LRFTargetDistance $IMAGE_DIR/RGB/*.JPG > $IMAGE_DIR/exif.tsv

python scripts/site_triage.py $IMAGE_DIR/exif.tsv


python - <<'EOF'
import numpy as np
from scipy.spatial import ConvexHull
d = np.genfromtxt('/Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/Site-1/exif.tsv', dtype=float)
lat, lon, alt, lrf = d[:,0], d[:,1], d[:,2], d[:,4]
lat0 = np.median(lat)
x = np.radians(lon-np.median(lon))*6378137*np.cos(np.radians(lat0))
y = np.radians(lat-lat0)*6356752
p = np.column_stack([x, y])
h = ConvexHull(p)
print('볼록껍질 %.0f m2 / 바운딩박스 %.0f m2 = %.2f'
      % (h.volume, np.ptp(x)*np.ptp(y), h.volume/(np.ptp(x)*np.ptp(y))))
# 주성분 — 궤적이 얼마나 가는가
c = p - p.mean(0)
s = np.linalg.svd(c, compute_uv=False)/np.sqrt(len(p))
print('주축 %.1f m  vs  부축 %.1f m   비 %.3f' % (s[0], s[1], s[1]/s[0]))
gz = alt - lrf
print('LRF 지면 표고 %.1f ~ %.1f m (중앙값 %.1f)'
      % (gz.min(), gz.max(), np.median(gz)))
print('지면 표고 vs X 상관 %.2f,  vs Y 상관 %.2f'
      % (np.corrcoef(x, gz)[0,1], np.corrcoef(y, gz)[0,1]))
EOF

## RGB

#### 실행

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P7 --apply

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/base \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.015

export SAYOU_PLANE_OVERRIDE="-0.025403763,0.077167721,90.6108,188435.833,279041.333"

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/tilt \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.02

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P5 P7 --apply

export SAYOU_PLANE_OVERRIDE="-0.115181229,0.213647894,91.2100,188450.167,279043.333"

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/tilt \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.02

export SAYOU_PLANE_OVERRIDE="-0.116317,0.200190,92.2050,188450.167,279043.333"

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/tilt2 \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.02

#### 검증

PITCH=2.30
python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
    --use-ba $IMAGE_DIR/base/cameras.npz --summary $IMAGE_DIR/base/summary.json \
    --pitch-m $PITCH --count 8 --z-min -15 --z-max 15 \
    --starts 20,40,60,80,100,120,140,160,180 2>&1 | tail -25

#### 지점을 더 넓게

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
    --use-ba $IMAGE_DIR/base/cameras.npz --summary $IMAGE_DIR/base/summary.json \
    --pitch-m 2.30 --count 8 --z-min -15 --z-max 15 \
    --starts 10,30,50,70,90,110,130,150,170,185 2>&1 | tail -28

unset SAYOU_PLANE_OVERRIDE
python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
    --use-ba $IMAGE_DIR/tilt/cameras.npz --summary $IMAGE_DIR/tilt/summary.json \
    --pitch-m 2.30 --count 8 --z-min -6 --z-max 6 \
    --starts 10,30,50,70,90,110,130,150,170 2>&1 | tail -22

#### ODM 비교

python scripts/qc_tear.py $IMAGE_DIR/tilt2/mosaic.tif <ODM>/odm_orthophoto.tif \
    --window --x 188380 --y 279090 --size 5000 4000 --block-m 12 --pitch-m 2.30

python scripts/qc_tear.py $IMAGE_DIR/tilt/mosaic.tif <ODM>/odm_orthophoto.tif \
    --window --x 188370 --y 279100 --size 7000 5500 --block-m 12 --pitch-m 2.31

#### TM

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/TM --output-dir $IMAGE_DIR/tm_release \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude

python scripts/frame_conditioning.py $IMAGE_DIR/tm_release/cameras.npz \
  --out-dir $IMAGE_DIR/tm_release/cond


# Site-2-29719

IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/Site-2-29719

exiftool -q -n -T -GPSLatitude -GPSLongitude -GPSAltitude \
    -RelativeAltitude -LRFTargetDistance $IMAGE_DIR/RGB/*.JPG > $IMAGE_DIR/exif.tsv

python scripts/site_triage.py $IMAGE_DIR/exif.tsv

python - <<'EOF'
import numpy as np
d = np.genfromtxt('/Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/Site-2-29719/exif.tsv', dtype=float)
lat, lon, alt, lrf = d[:,0], d[:,1], d[:,2], d[:,4]
lat0 = np.median(lat)
x = np.radians(lon-np.median(lon))*6378137*np.cos(np.radians(lat0))
y = np.radians(lat-lat0)*6356752
gz = alt - lrf
h, e = np.histogram(gz, bins=30)
print('LRF 지면 표고 분포')
for cnt, lo in zip(h, e[:-1]):
    if cnt: print('  %6.1f m  %s %d' % (lo, '#'*max(1,cnt//2), cnt))
print()
print('표고 vs X 상관 %+.2f,  vs Y 상관 %+.2f'
      % (np.corrcoef(x,gz)[0,1], np.corrcoef(y,gz)[0,1]))
EOF

python -c "
import json;s=json.load(open('/Users/seongjungkim/Development/sayouzone/solar-thermal/data/solar/Site-2-29719/base/summary.json'))
print(s['plane_source'], s['ground_plane'])"

## RGB

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P5 P7 --apply

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/base \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.02

unset SAYOU_PLANE_OVERRIDE
python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P5 P7 --apply

unset SAYOU_PLANE_OVERRIDE
echo "override=[$SAYOU_PLANE_OVERRIDE]"     # 비어 있어야 함

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/base2 \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.02

export SAYOU_PLANE_OVERRIDE="-0.085127334,0.109440174,92.4730,188586.600,279155.400"

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/tilt \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.02

#### 검증

PITCH=$(python -c "import json;print(round(json.load(open('$IMAGE_DIR/base2/summary.json'))['ortho']['panel_unit_info']['unit_m'],2))")

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
    --use-ba $IMAGE_DIR/base2/cameras.npz --summary $IMAGE_DIR/base2/summary.json \
    --pitch-m $PITCH --count 8 --z-min -6 --z-max 6 \
    --starts 8,25,42,59,76,93,110,127,144,160 2>&1 | tail -28

unset SAYOU_PLANE_OVERRIDE
python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
    --use-ba $IMAGE_DIR/tilt/cameras.npz --summary $IMAGE_DIR/tilt/summary.json \
    --pitch-m $PITCH --count 8 --z-min -4 --z-max 4 \
    --starts 8,25,42,59,76,93,110,127,144,160 2>&1 | tail -24


python -c "
import json;s=json.load(open('$IMAGE_DIR/tilt/summary.json'));o=s['ortho']
print('평면', s['plane_source'], s['ground_plane'])
print('품질', o['quality'])
print('관측', s['observations_per_frame'])
print('매칭', s['match_diagnosis'])
print('연직 k<=%.3f fallback %.3f' % (o['max_offnadir_ratio'], o['offnadir_fallback_ratio']))
"

#### ODM 비교

python scripts/qc_tear.py $IMAGE_DIR/tilt/mosaic.tif <ODM>/odm_orthophoto.tif \
    --window --x 188530 --y 279200 --size 5500 4500 --block-m 12 --pitch-m $PITCH

## TM

#### 실행

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/TM --output-dir $IMAGE_DIR/tm_release \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude

python scripts/frame_conditioning.py $IMAGE_DIR/tm_release/cameras.npz \
  --out-dir $IMAGE_DIR/tm_release/cond


# Site-2-29696

IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/Site-2-29696

exiftool -q -n -T -GPSLatitude -GPSLongitude -GPSAltitude \
    -RelativeAltitude -LRFTargetDistance $IMAGE_DIR/RGB/*.JPG > $IMAGE_DIR/exif.tsv

python scripts/site_triage.py $IMAGE_DIR/exif.tsv

## RGB

#### 실행

python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou --only P2 P3 P4 P5 P7 --apply

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/base \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.02

#### 검증

PITCH=$(python -c "import json;print(round(json.load(open('$IMAGE_DIR/base/summary.json'))['ortho']['panel_unit_info']['unit_m'],2))")
echo "PITCH=$PITCH,  plane_source=$(python -c "import json;print(json.load(open('$IMAGE_DIR/base/summary.json'))['plane_source'])")"

python scripts/plane_sweep.py --image-dir $IMAGE_DIR/RGB \
    --use-ba $IMAGE_DIR/base/cameras.npz --summary $IMAGE_DIR/base/summary.json \
    --pitch-m $PITCH --count 8 --z-min -10 --z-max 10 \
    --starts 10,30,50,70,90,110,130,150,170,195 2>&1 | tail -28

## TM

#### 실행

python scripts/homography_pipeline.py \
  --image-dir $IMAGE_DIR/TM --output-dir $IMAGE_DIR/tm_release \
  --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude

python scripts/frame_conditioning.py $IMAGE_DIR/tm_release/cameras.npz \
  --out-dir $IMAGE_DIR/tm_release/cond

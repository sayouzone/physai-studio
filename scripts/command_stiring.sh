
IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/옥산_1호
IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/EWP-서오창IC-2
IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/에스엘에너지_사천시
IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/그린환경센터
IMAGE_DIR=/Users/seongjungkim/Downloads/극동대학교/극동대캠퍼스/IV커브안됨/H20T
IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/Site-2-29696
IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/Site-2-29719
IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/Site-1
IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/K_Demo
IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/갈평저수지


python scripts/panel_mask_probe.py \
    --cameras $IMAGE_DIR/rgb_release/cameras.npz \
    --image-dir $IMAGE_DIR/RGB \
    --pairs 12


for S in 1.0 0.5 0.25; do
  echo "=== scale $S ==="
  python scripts/panel_mask_probe.py \
      --cameras $IMAGE_DIR/rgb_release/cameras.npz \
      --image-dir $IMAGE_DIR/RGB --pairs 20 \
      --modes none --scale $S
done



python scripts/patch_sayou.py --root label_studio/sayou --revert
python scripts/patch_sayou.py --root label_studio/sayou \
    --only P2 P3 P4 P5 P7 P10 --apply



IMAGE_DIR=~/Development/sayouzone/solar-thermal/data/solar/에스엘에너지_사천시
for S in 1.0 0.5 0.25; do
  SAYOU_MATCH_SCALE=$S python scripts/homography_pipeline.py \
    --image-dir $IMAGE_DIR/RGB --output-dir $IMAGE_DIR/ms$S \
    --offnadir-frac 0.32 --panel-unit 2 --smooth-weak-attitude --gsd 0.015
done


python scripts/qc_tear.py \
    $IMAGE_DIR/ms1.0/mosaic.tif $IMAGE_DIR/ms0.5/mosaic.tif $IMAGE_DIR/ms0.25/mosaic.tif \
    --window --x 290730 --y 267040 --size 10000 11000 --block-m 12 --pitch-m 2.0


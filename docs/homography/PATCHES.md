# RTK 기반 호모그래피 — 통합 가이드 및 수정 내역

## 배치

```
src/solar_thermal/georeferencing/
├── homography/                ← 신규 (RTK 기반 호모그래피 패키지, 9개 파일)
├── utils.py                   ← 신규 (없어서 ImportError 나던 모듈)
├── pipeline.py                ← 교체
├── rtk.py                     ← 교체
├── __init__.py                ← 신규
├── sfm/
│   ├── bundle_adjustment.py   ← 교체
│   └── __init__.py            ← 신규
├── features/
│   ├── sift.py                ← 교체
│   ├── extract.py             ← import 2줄 수정 (아래 참조)
│   └── __init__.py            ← 신규
└── ortho/
    └── __init__.py            ← 신규
```

`homography/` 는 `..geometry`, `..crs`, `..sfm`, `..utils` 상대 import 를
쓰므로 위 위치에 그대로 두면 동작한다.

검증:

```bash
python -m solar_thermal.georeferencing.homography.selftest   # 13/13 통과
```

---

## 1. ★ 자세 변환이 틀려 정사영상이 동서 거울반전되고 있었습니다

`pipeline.py::_build_initial_state` 의 기존 식:

```python
omega = deg2rad(gimbal_roll)
phi   = deg2rad(gimbal_pitch + 90)   # nadir(-90) → 0 "보정"
kappa = deg2rad(gimbal_yaw)
```

nadir 정북(roll=0, pitch=−90, yaw=0)에서 `(ω,φ,κ)=(0,0,0)` → `R = I`.
그런데 `geometry.project_point` 규약에서 광축은 `R[2]` 이므로 광축이
**`(0,0,+1)` — 하늘을 향합니다.**

```
카메라 (200000, 400000, 160), 지상점 48 m 아래
→ 가시 분모 (R[2]·diff) = −48.00      (전방이어야 하는데 후방)
```

연쇄 효과:

- `triangulate_dlt` 는 cheirality 를 검사하지 않아 조용히 **드론 위쪽**에
  3D 점을 만듭니다.
- BA 는 그 거울상 기하를 최소화하므로 수렴은 하지만 해가 무의미합니다.
- `ortho.simple_orthophoto` 는 *또 다른* 부호 규약(`d_cam=[...,−1]` +
  `px = cx + f·…`)을 써서 이 오류를 **부분 상쇄**합니다.

두 오류가 겹친 최종 결과는 **동서(E-W)만 거울반전된 정사영상**입니다.
정확한 nadir 합성 프레임에 제1원리 진리값을 대조한 결과:

| yaw | 기존 파이프라인 최대오차 | 수정판 최대오차 |
|---|---|---|
| 0° | 52.73 m | 0.0 m |
| 33° | 52.73 m | 6.5e-11 m |
| 90° | 52.73 m | 0.0 m |
| −140° | 52.73 m | 6.5e-11 m |

오차가 오프셋의 정확히 2배(26.4 m → 52.7 m)라는 점이 거울반전의 지문입니다.
남북 방향은 정상이라 오차가 0으로 나옵니다.

**태양광 패널 격자는 좌우 대칭에 가까워 육안으로는 거의 식별되지 않습니다.**
좌표를 실측과 대조해야만 드러납니다. 이미 납품/검토된 결과물이 있다면
동서 반전 여부를 우선 확인하시길 권합니다.

수정판은 `homography.pose.opk_from_gimbal` 을 씁니다. `R` 의 행이
`[−right, −down, +forward]` 이고, `selftest` 가 `project_point` 규약 및
OpenCV 픽셀 규약과의 일치를 1e-8 px 이내로 검증합니다.

---

## 2. BA의 가속 코드가 전부 dead code 였습니다

`bundle_adjustment.py` 는 `_build_residuals_np`, `_build_residuals_gpu`,
`_build_jacobian_sparsity` 를 **정의만 하고 한 번도 호출하지 않습니다.**
실제로 쓰이는 것은 `rtk_constrained_bundle_adjustment` 안에 중첩 정의된
Python 루프 `residuals` 입니다. docstring 이 설명하는 가속은 미적용 상태였습니다.

추가로 발견한 것: **`loss="huber", f_scale=2.0` 단일 호출은 초기 오차가 몇 px 를
넘으면 최적화가 시작조차 못 합니다.** huber 는 `|r| > f_scale` 인 잔차의 기울기를
눌러버리므로, 삼각측량 초기값의 통상 오차(수십~수백 px)에서는 **모든 관측이
이상점으로 취급되어** 기울기가 소멸합니다.

카메라 24대 / 점 400개 / 관측 2,563개 벤치마크:

| | 시간 | 재투영 RMSE | 3D점 오차 중앙값 |
|---|---|---|---|
| 원본 | 82.2 s | 28.64 px | 2.279 m |
| 수정 | **0.3 s** | **0.49 px** | **0.027 m** |

수정 내용: ① 벡터화 residual 연결 ② 희소 jacobian 연결(COO 일괄 구성으로
재작성 — 원본의 `lil_matrix` fancy indexing 18회는 관측 수십만에서 그 자체로
수십 초) ③ **2단계 robust 스케줄**(1단계 `soft_l1` + 잔차분포 기반 f_scale,
2단계 `huber 2px`) ④ 좌표 중심화 ⑤ RMSE 계산 수정(원본은 RTK 잔차와 huber
감쇠가 섞인 값을 재투영 RMSE 로 보고).

---

## 3. RTK 실측 σ 가 BA에 반영되지 않고 있었습니다

`metadata.py` 는 XMP 에서 `RtkStdLat/Lon/Hgt` 를 파싱해 `meta.rtk_std` 에
넣지만, `rtk.compute_rtk_prior_weights` 는 이 값을 **한 번도 읽지 않고**
고정 상수 `gps_std_xy=0.10`, `gps_std_z=0.15` 만 씁니다.

프레임별 실제 측위 정확도가 가중치에 전혀 반영되지 않으므로, RTK 가 열화하는
구간(반환점, 수목 근처)의 프레임이 정상 프레임과 같은 발언권을 갖습니다.

주의할 점 하나 더: `rtk_std = [σ_lat, σ_lon, σ_hgt]` 인데 투영좌표는
(X=Easting, Y=Northing) 이라 **σ_lon→X, σ_lat→Y** 로 교차시켜야 합니다.
한국 위도에서 두 값은 보통 비슷해서 뒤집어도 그럴듯해 보이지만, 정확히
그 열화 구간에서만 틀립니다.

`estimate_ground_z` 의 폴백 `altitude - 100.0` 도 제거했습니다. 100 m 는 근거가
없는 값이고, 그대로 평면 고도가 되어 GSD 와 지상범위 전체를 틀리게 만듭니다.
수정판은 `None` 을 반환해 호출자가 그 사진을 건너뛰게 합니다.

---

## 4. import 가 깨져 패키지가 로드되지 않았습니다

| 증상 | 원인 | 조치 |
|---|---|---|
| `No module named '...georeferencing.utils'` | `fmt_elapsed` 를 3개 모듈이 import 하는데 파일이 없음 | `utils.py` 신규 |
| `No module named 'torch'` | `sift.py` 가 최상단에서 torch/kornia import → `features/__init__` 체인으로 파이프라인 전체가 torch 없이 로드 불가 | `sift.py` 지연 import |
| `NameError: _load_grayscale` | `extract.py` 가 `_load_grayscale`, `Literal`, `Optional` 미import | 아래 2줄 수정 |
| `NameError: run_pipeline_gcp_free` | `pipeline.main()` 이 존재하지 않는 함수 호출 | 수정 |
| `F821 Undefined name 'ImageMetadata'` | `geometry.compute_focal_px` 의 타입힌트 | `from __future__ import annotations` 덕에 런타임은 무사하나, 힌트 평가 시 죽음 |

`extract.py` 수정 (2줄):

```diff
-from typing import Any
+from typing import Any, Literal, Optional
-from .sift import FeatureResult, KorniaSIFTExtractor
+from .sift import FeatureResult, KorniaSIFTExtractor, _load_grayscale
```

---

## 5. 호모그래피 자체 (7단계)

지표면을 평면으로 가정하는 순간 지상↔이미지 대응은 3×3 행렬로 *정확히*
닫힙니다. `simple_orthophoto` 의 출력 픽셀당 광선교차 계산이
`cv2.warpPerspective` 한 번으로 대체되고, 좌표맵 두 장(5000×5000 기준 200 MB)도
사라집니다.

```
H = K_neg · [ (r1 + a·r3) | (r2 + b·r3) | (c·r3 − R·C) ]
```

경사항 `a, b` 는 **1·2 열**에서 `r3` 에 곱해집니다. 수평평면만 시험하면
이 자리를 틀려도 통과하므로 `selftest` 는 반드시 경사평면으로 검증합니다.

**국소 원점이 필수입니다.** EPSG:5186 절대좌표(200000, 400000)로 세우면
`R·C` 항만 10⁵ 스케일이라 열 간 균형이 깨집니다:

| 원점 | cond(H) | 왕복오차 |
|---|---|---|
| (0, 0) | 1.1e+05 | 1.8e-15 m |
| (200000, 400000) | **1.7e+13** | 역행렬 무의미 |

프레임마다 자기 카메라 위치를 국소 원점으로 삼고, 공개 API 는 절대좌표를
주고받습니다.

기타: 경사평면 지원(경사 3% 부지, 촬영폭 120 m면 양끝 ±1.8 m 편차),
DewarpData 왜곡보정, feather + 연직근접 가중 모자이크, `bbox_to_wgs84_polygon`
(호모그래피는 평행성을 보존하지 않으므로 bbox 4모서리를 각각 변환).

---

## 5b. 정확도 문구 수정

기존 docstring 의 "수평 2~5 cm / 수직 5~15 cm" 를 오차 예산 표로 교체했습니다.

- **"수평 2~5 cm" 는 BA 경로 중심부에만 해당** — `skip_sfm` 은 24~30 cm 로
  한 자릿수 차이입니다.
- **"수직 5~15 cm" 는 이 산출물에 적용되지 않습니다** — 평면 호모그래피는
  점별 높이를 만들지 않습니다. 수직량은 적합 평면 하나뿐이고 품질은
  `inlier_rmse_m` 으로 이미 보고됩니다.
- **빠져 있던 지배항: 평면 가정.** 평면을 지면에 맞추고 결함이 패널 상면에
  있으면 가장자리 72 cm — RTK 항의 70배입니다. 태양광 검사에서는 평면을
  패널 상면에 맞추는 것이 정답이고, BA 점군 경로는 자연히 그렇게 됩니다.

---

## 6. 사용법

```python
from solar_thermal.georeferencing import run_pipeline

# 전체 (SfM + BA + 모자이크)
run_pipeline(Path("./data/RGB"), Path("./out"), target_epsg=5186)

# 빠른 현장 확인 (RTK 자세만)
run_pipeline(Path("./data/RGB"), Path("./out"), skip_sfm=True)

# 프레임별 GeoTIFF
run_pipeline(Path("./data/RGB"), Path("./out"), mosaic=False)
```

`gsd_m` 기본값을 `None`(자동)으로 바꿨습니다. 기존 `0.05` 고정은 H20T 를
45 m 고도로 날릴 때 실제 GSD 약 1.2 cm 대비 4배 다운샘플입니다.

합성 DJI JPG 15장(XMP 포함) 전 경로 실행 결과: 15/15 georeferencing,
BA 재투영 RMSE 0.107 px, 지상평면 c=25.05 m(진리값 25.0 m), 모자이크 충전율 98.1%.

---

## 검증하지 못한 것

- **실제 H20T 데이터로는 못 돌려봤습니다.** 합성 데이터는 렌즈왜곡·지형기복·
  조명변화가 없으므로 BA RMSE 0.107 px 같은 수치는 낙관적입니다.
- `_GPU_MIN_OBS` 이상에서의 CuPy 경로는 이 환경에 GPU 가 없어 미실행입니다.
- `geoid_undulation_m` 은 상수 오프셋 훅만 넣었습니다. 정밀 정표고가 필요하면
  KNGeoid 그리드를 붙여야 합니다.
- 실데이터 적용 시 **첫 확인 지점**: BA 시작 로그의 `BA 시작 전 재투영 RMSE`.
  이 값이 수천 px 이면 자세 규약이 여전히 어긋난 것이고, 수십~수백 px 이면
  정상입니다(경고 임계값을 500 px 로 걸어 두었습니다).


---

## 7. 모자이크 고스팅 — blend_mode="select" (winner-take-all)

**증상**: `mosaic_frames` 산출물에서 패널 행이 여러 겹의 흐릿한 줄무늬로
보임 (`mosaic_1.tif` 실측).

**원인**: 겹치는 프레임을 픽셀별 가중평균으로 섞는 것(`acc += warped·weight`)
이 기본 동작이었다. 자세 오차, 평면 근사 오차, GSD 이산화 오차를 다 없애도
두 프레임의 같은 지상점은 보통 수 픽셀 어긋난다. 평균을 내면 각 프레임의
에지가 겹쳐 흐림/줄무늬가 된다. 반복 구조가 강한 태양광 패널 격자에서
특히 두드러진다.

**수정**: `MosaicConfig.blend_mode` 추가, 기본값을 `"select"` 로 바꿨다.
각 출력 픽셀에 대해 기존 feather+연직근접 점수가 가장 높은 프레임 **하나만**
채택하는 winner-take-all. 여러 소스가 섞이지 않으므로 정합 잔차가 있어도
흐려지지 않는다. 기존 가중평균은 `blend_mode="average"` 로 남겼지만 실측
데이터에서 ghosting 을 만들므로 권장하지 않는다.

합성 데이터 검증 (자세/위치 잔차를 `mosaic_1.tif` 수준으로 주입):

| | 라플라시안 분산 (선명도) |
|---|---|
| average (기존 기본값) | 82.1 |
| select (신규 기본값) | 236.5 |

트레이드오프: 프레임 경계에서 노출/화이트밸런스 차이로 인한 밝기 이음선이
하드 컷으로 보일 수 있다. 검사 목적에는 흐림보다 이음선이 낫다는 판단이다.
좌표 정확도에는 영향 없음 — 각 픽셀은 정확히 그 지점을 찍은 프레임의
값이다.


---

## 8. 패널이 시임에서 끊김 — 기준 평면 높이 (relief displacement)

**증상**: `blend_mode="select"` 로 바꾼 뒤 ghosting 은 사라졌지만, 패널 행이
시임에서 뚝 끊기고 그 자리에 지면이 드러남 (`mosaic_2.tif`).

**원인**: 블렌딩 문제가 아니다. **기준 평면이 지면에 잡혀 있다.**
패널 상면은 지면보다 1.5~2 m 높은데, 기준면이 지면이면 패널이 각 프레임에서
카메라 연직점 바깥으로 `Δh·tanθ` 만큼 밀린다. 프레임마다 밀림 방향이 다르고,
`select` 는 픽셀별로 한 프레임만 쓰므로 시임에서 불연속이 그대로 드러난다.
(`average` 에서는 이게 흐릿하게 섞여 ghosting 으로 보였던 것 — 같은 원인의
다른 증상이다.)

H20T 실측 조건 (고도 45.87 m, 지상범위 26.7×35.4 m → 최외곽 k=0.483,
패널 1.8 m):

| 화면 위치 | 밀림 | 인접 프레임 반대방향 |
|---|---|---|
| 중심 | 0 cm | 0 cm |
| 중간 | 43.5 cm | 87 cm |
| 가장자리 | 87 cm | **174 cm** |

174 cm 는 패널 한 장 폭과 맞먹는다 — 이미지에서 보이는 어긋남과 일치.

**수정**:

1. `plane.fit_plane_ransac` 에 `surface="upper"` (기본) 추가. 점군 잔차
   히스토그램에서 지배 층보다 위에 유의한 층(전체의 10% 이상, 0.6 m 이상
   위)이 있으면 기준면을 그쪽으로 올린다 = 패널 상면. 두 층 합성 점군에서
   1.80 m 오프셋을 정확히 검출 확인.
2. `estimate_ground_plane` / `_resolve_ground_plane` / `run_pipeline` 에
   `panel_top_offset_m`, `plane_surface` 연결. LRF 는 조준점(=지면) 실측
   이므로 이 경로에서는 오프셋을 명시해야 BA 경로와 기준면이 맞는다.
3. 기준면이 지면으로 잡히면 경고 로그.
4. CLI: `--panel-top 1.8`, `--plane-surface {upper,dominant}`.
5. (보조) `MosaicConfig.seam_blend_px` — 시임 좌우 좁은 띠만 부드럽게.
   노출 차이로 이음선이 거슬릴 때만 20~60. **패널 끊김은 이걸로 해결되지
   않는다** — 그건 평면 높이 문제다.

**사용법**: BA 경로는 `surface="upper"` 가 자동 처리하므로 그대로 두면 된다.
`--skip-sfm` 이나 LRF 폴백 경로면 `--panel-top` 에 실제 가대 높이를 준다.

**검증 한계**: 합성 3D 렌더러로 IoU 0.49 → 0.54 개선을 확인했지만, 렌더러가
relief 를 완전히 재현하지 못해 절대값이 낮다. 위 표의 밀림량은 기하 계산
결과이지 실측이 아니다. 실데이터에서 `--panel-top` 값을 바꿔가며 시임에서
패널이 이어지는 값을 찾는 것이 가장 확실하다.

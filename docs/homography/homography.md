# 호모그래피 파이프라인

## 개요

## 파이프라인 실행

```bash
python scripts/homography_pipeline.py \
    --image-dir ~/Development/sayouzone/solar-thermal/data/solar/그린환경센터/RGB \
    --output-dir ~/Development/sayouzone/solar-thermal/data/solar/그린환경센터/output
```

```bash
python scripts/homography_pipeline.py \
    --image-dir ~/Development/sayouzone/solar-thermal/data/solar/그린환경센터/RGB \
    --output-dir ~/Development/sayouzone/solar-thermal/data/solar/그린환경센터/output \
    --panel-top 1.8 \
    --plane-surface "upper"
```

```bash
python scripts/homography_pipeline.py \
    --image-dir ~/Development/sayouzone/solar-thermal/data/solar/그린환경센터/RGB \
    --output-dir ~/Development/sayouzone/solar-thermal/data/solar/그린환경센터/output
```

## 파이프라인 단계

1. 메타데이터 추출
2. RTK 품질 검증 + 좌표 변환
3. SIFT 특징점 추출 + 페이별 매칭 + RANSAC outlier 제거
4. track 빌드
5. triangulation
6. RTK 제약 Bundle Adjustment
7. 지상평면 + 호모그래피
8. 정사영상 합성

```text
단일 EPSG 페어에 대한 양방향 변환기
WGS84 ↔ 투영좌표계 변환
기본 타깃은 EPSG:5186 (Korea 2000 / Central Belt). 필요시 다른 EPSG 코드로 바꿔 사용 가능.
변환기는 stateless 하지만 pyproj.Transformer 객체 생성 비용이 약간 있으므로 (수십 ms) 파이프라인당 한 번만 만들어 재사용한다.
```
`CRSConverter`

## 1. 메타데이터

DJI JPG 한 장에서 표준 메타데이터를 추출
- GPS (XMP 우선, fallback EXIF)
- Flight (Yaw, Pitch, Roll)
- Orientation (Yaw, Pitch, Roll)
- RTK
- DewarpData (공장 캘러브레이션)
    drone-dji:DewarpFlag가 1이면 왜곡 보정 정보가 유효
- Velocity (m/s, body frame X/Y/Z)
- Camera
- Capture time (epoch seconds, local tz from XMP)

`extract_metadata`

정사영상 단계 크래시를 유발하는 메타데이터 결손을 사전 경고

`_diagnose_metadata`

내부 파라미터를 못 만드는 이미지는 제외 (내부 파라미터가 없는 이미지)<br/>
triangulation/BA 인덱스가 어긋난다.<br/>
메타데이터 → PinholeIntrinsics
  1. DewarpData (DJI 공장 캘리브레이션). fx/fy/cx/cy + [k1,k2,p1,p2,k3]. 
  2. FocalLengthIn35mmFilm → f_px = f35 / 36 · width (센서폭 불필요).
  3. FocalLength + 1″ 센서폭 13.2 mm 가정

`intrinsics_from_metadata`

## 2~3. 좌표 변환 + RTK  품질

```text
RTK 측정 품질 검증 및 BA prior 가중치 계산
DJI RtkFlag 의미 (DJI SDK 문서 기준)
------------------------------------
* 0  = None / GPS only
* 16 = RTK Float  (수십 cm 정확도)
* 34 = RTK Single (저정밀)
* 50 = RTK Fixed  (1~3 cm 정확도) ← 신뢰 가능
------------------------------------
RtkStdLat / RtkStdLon / RtkStdHgt
프레임별 실제 측위 정확도가 BA 가중치에 전혀 반영되지 않았다. RTK 가 열화하는 구간(비행 반환점, 수목 근처)에서 σ 는 3~5 배까지 벌어지는데,
같은 가중치를 주면 그 프레임이 블록 전체를 끌고 간다.
RTK Fixed 비율이 임계값 이상인지 검사 + 로깅.
```
`validate_rtk_quality`

```text
RTK + 짐벌 자세 → 초기 외부표정 + RTK prior 배열
nadir 정북에서 (-180°, 0°, -180°)가 나오는데, ω 가 ±180° 근처인 것은 이 규약의 정상 동작이다 (φ 가 0 근처라 짐벌락도 없다)
```
`_build_initial_state`

```text
RTK 실측 표준편차 기반 (N, 3) BA prior 가중치 (1/σ²).
σ 단위로 정규화되어 픽셀 단위 재투영 잔차와 같은 척도에서 비교된다
```
`compute_rtk_prior_weights`

# 

`intrinsics_from_metadata`

## 4. 인접쌍 + 매칭

```text
RTK 위치 기반 인접쌍 선택
N 장을 전수 매칭하면 N(N−1)/2 쌍이다. 165 장이면 13,530 쌍, 1,000 장이면 499,500 쌍.
그런데 드론 사진의 대부분은 서로 겹치지 않는다 - 겹치지 않는 쌍의 매칭 시도는 전부 낭비이고, 
게다가 태양광 패널처럼 반복 텍스처가 강한 장면에서는 겹치지 않는 쌍이 가짜 매칭을 만들어 낸다.
RTK 가 cm 급 위치를 이미 알려주므로, 겹칠 수 없는 쌍은 매칭 전에 잘라낸다.
매칭 단계는 평면 추정 전이므로, 여기서는 보수적 근사를 쓴다:
  r_i = (촬영고도) × tan(화각 반각) × sqrt(2)      # 외접원 반경
  겹침 가능 ⟺ |C_i − C_j| < (r_i + r_j) × margin
sqrt(2)는 직사각형 footprint 의 대각 반경, margin 은 자세 기울기와
지형 기복에 대한 여유. 이 판정은 실제 겹침의 상위집합이라 안전하다 (진짜 겹치는 쌍을 버리지 않는다).
nadir 가정 footprint 외접원 반경 (m).
(w/2, h/2) 픽셀의 화각 tangent 를 그대로 쓴다. 초점거리가 픽셀 단위라 센서 크기가 필요 없다.
```
`footprint_radius_m`

```text
겹칠 가능성이 있는 이미지 쌍 목록
```
`select_gps_neighbor_pairs`

```text
KD-Tree 기반 인접 페어 탐색 + tie-point 빌드
수천 장 처리 시 풀 매칭 O(N²) 을 피하기 위해 RTK 좌표로 KD-Tree 를 만들어
각 이미지의 최근접 k 개 이웃과만 매칭. RANSAC fundamental matrix 로 outlier 제거 후 inlier 만 보존.
RTK 좌표 KD-Tree 로 인접 페어 후보 생성.
O(N²) 를 매칭 대신 O(N log N) 으로 단축. 수천 장 처리 시 필수.
```

`find_neighbor_pairs`

```text
SIFT 특징점 추출 + 페이별 매칭 + RANSAC outlier 제거.
특징점이 너무 적은 이미지는 사전 제외 (전체 페어에서 빠짐)
페어에 등장하는 이미지만 SIFT 추출 (불필요한 추출/메모리 절약)
```

`build_tie_points`

## 5a. track

```text
페어별 매칭을 다중 시점 track 으로 연결
페어별 매칭들을 연결해 track 리스트를 만든다.
track = 동일 지상점에 대응하는 관측들의 집합
      = [(image_idx, keypoint_idx, px, py), ...]
GCE g2 같은 vCPU 환경 (Cascade Lake, 낮은 클럭 + 작은 L3) 에서는 M4 Pro 대비 3~5 배 느림.
```

`build_tracks`

## 5b. triangulation

```text
DLT(Direct Linear Transform) 다중시점 삼각측량
각 관측 (u, v) 와 카메라 P 에 대해 [u·w, v·w, w]ᵀ = P · [X,Y,Z,1]ᵀ 에서
u, v 를 소거하면 X = [X,Y,Z,1]ᵀ 에 대한 2 개의 동차 선형식이 나온다::
    u · (P[2]·X) - (P[0]·X) = 0
    v · (P[2]·X) - (P[1]·X) = 0
N 개 시점이면 2N × 4 행렬 A 가 되고, A · X = 0 의 최소제곱해는
AᵀA 의 최소 특이값에 대응하는 우특이벡터 (SVD 마지막 행).
track 들을 삼각측량해서 BA 입력(observations, initial_points) 생성
```

`trangulate_tracks`

## 6. RTK 제약 Bundle Adjustment

```text
GCP 가 없으므로 절대 좌표계 기준점은 RTK 측정값. 하지만 RTK 도 cm 급 오차가
있으므로 hard constraint 가 아닌 soft constraint (가중치 = 1/σ²) 로 잔차에 추가::
    residual = [reprojection_errors, sqrt(w) * (camera_pos - rtk_prior)]
RTK 좌표를 카메라 위치의 사전확률(prior)로 묶는 번들 조정
```

`rtk_constraint_bundle_adjustment`

## 7. 지상평면 + 호모그래피

```text
BA 점군 → LRF → 메타데이터 순으로 지상평면 결정
```

`_resolve_ground_plane`

```text
사진측량 기본 기하: ω-φ-κ 회전행렬, 공선조건, 카메라 투영행렬
회전행렬
    R(ω,φ,κ) = R_ω · R_φ · R_κ
    (ω: roll, φ: pitch, κ: yaw, radian)
공선조건 (project_point)
    x = cx - f_px · (R[0]·diff) / (R[2]·diff)
    y = cy - f_px · (R[1]·diff) / (R[2]·diff)
    diff = world_xyz - camera_xyz
사진측량 ω-φ-κ 회전행렬 (radian)
```

`_rotation_from_opk`

```text
평면유도 호모그래피 (plane-induced homography) - RTK 기반 정사보정의 핵심
출력 레스터의 모든 픽셀에 대해 ray-plane 교점을 계산한다 (meshgrid → diff → einsum → 나눗셈 → cv2.map).
하지만 지표면을 평면으로 가정하는 순간, 지상평면 ↔ 이미지평면 대응은 3x3 호모그래피로 하나로 정확히 닫힌다.
즉 수천만 번의 광선 교차 계산은 불필요하고, 행렬 하나와 cv2.warpPerspective 한 번이면 수학적으로 동일한 결과가 나온다.
카메라 행렬 (geometry.camera_projection_matrix와 동일 규약):
    P = K_neg · [R | -R·C],   K_neg = [[-f, 0, cx], [0, -f, cy], [0, 0, 1]]
지상평면을 Z = a·x + b·y + c 로 두면 평면 위의 점은::
    [X, Y, Z, 1]ᵀ = S · [x, y, 1]ᵀ,   S = [[1,0,0], [0,1,0], [a,b,c], [0,0,1]]
따라서 지상평면 → 이미지 호모그래피는 단순히::
    H_g2i = P · S = K_neg · [ (r1 + a·r3) | (r2 + b·r3) | (c·r3 − R·C) ]
(r1, r2, r3 은 R 의 열). a=b=0 이면 수평평면 Z=c 에 대한 익숙한 형태 K_neg·[r1 | r2 | c·r3 − R·C] 로 환원된다
H = K_neg · [(r1 + a·r3) | (r2 + b·r3) | (c·r3 − R·C)] (국소 좌표)
```

`build_frame_homography`

```text
프레임 집합에 대한 권장 출력 GSD
각 프레임 주점 GSD 의 분위수. 중앙값(기본)은 원해상도를 대체로 보존하면서
고도가 튄 프레임 하나 때문에 출력 레스터가 과대해지는 것을 막는다.
```

`recommend_gsd`

## 8. 정사영상
호모그래피 기반 정사영상 생성 및 모자이크
meshgrid → diff → einsum → 나눗셈 → cv2.remap 대신 cv2.warpPerspective 사용
출력 픽셀당 부동소수 연산이 사라지고, 좌표 앱 두 장 (out_h×out_w×float32 ×2) 을 만들지 않으므로
메모리도 크게 준다. 5000x5000 출력 기준 앱만 200 MB 였다.
* 경사평면 지원 (GroundPlane.a/b). 수평 가정은 특수한 경우일 뿐이다.
* 다중 프레임 가중 모자이크 - feather 블렌딩 + 연직 근접 가중.
* 프레임마다 전체 캔버스를 다루지 않고 자기 footprint 창(window) 안에서만 warp 한다.
100장짜리 현장에서 이 차이가 수십 배다.

```text
여러 프레임을 하나의 정사 모자이크로 합성.
프레임은 한 번에 한 장씩 읽어 창 안에 누산하고 즉시 버린다.
전체 이미지를 메모리에 들고 있지 않으므로 수백 장도 처리 가능하다.
누산기는 float32 이고 크기는 out_h × out_w × (bands+1) 이다.
(5000×5000, RGB 기준 약 400MB)
```

```text
여러 프레임의 footprint 를 감싸는 전체 범위
```

```text
프레임 집합에 대한 권장 출력 GSD
각 프레임 주점 GSD 의 분위수. 중앙값(기본)은 원해상도를 대체로 보존하면서
고도가 튄 프레임 하나 때문에 출력 레스터가 과대해지는 것을 막는다.
```

```text
프레임 하나를 자기 창 안에서 정사 warp.
```

```text
Return an Affine transformation given upper left and pixel sizes.
```

- `mosaic_frames`
  - `ground_bounds_from_homography`
  - `recommend_gsd`
  - `warp_frame`
  - `from_origin`

사진 한 장 → 정사영상 GeoTIFF.
- `orthorectify_frame`

  - `fh.footprint_bounds`
  - `fh.gsd_at_principal_point`
  - `fh.intr.undistort_maps`
  - `from_origin`


## 기술 개념

### CNN (합성곱 신경망, Convolutional Neural Network)

CNN은 이미지나 영상처럼 격자 구조를 가진 데이터를 처리하는 데 특화된 딥러닝 신경망 구조입니다.

#### 핵심 구성 요소

**1. 합성곱층 (Convolutional Layer)**
- 작은 필터(커널)를 이미지 위로 슬라이딩하면서 특징을 추출
- 필터는 엣지, 질감, 패턴 등을 감지하도록 학습됨
- 층이 깊어질수록 저수준 특징(선, 모서리) → 고수준 특징(눈, 얼굴 형태 등)으로 추상화

**2. 활성화 함수 (ReLU 등)**
- 비선형성을 부여해 복잡한 패턴을 학습할 수 있게 함

**3. 풀링층 (Pooling Layer)**
- 특징 맵의 크기를 줄여 연산량을 줄이고, 위치 변화에 대한 강건성(robustness)을 높임
- Max Pooling이 가장 흔히 사용됨

**4. 완전연결층 (Fully Connected Layer)**
- 마지막에 추출된 특징을 바탕으로 최종 분류나 회귀 수행

#### 왜 CNN이 이미지에 강한가

- **파라미터 공유**: 같은 필터를 이미지 전체에 재사용 → 파라미터 수 절감
- **지역 연결성**: 각 뉴런이 이미지의 일부(지역)만 보기 때문에 공간적 구조를 잘 반영
- **평행이동 불변성**: 물체가 이미지 어디에 있든 비슷하게 인식 가능

#### 대표적인 활용 분야

- 이미지 분류 (예: 고양이 vs 강아지)
- 객체 탐지 (YOLO, Faster R-CNN 등)
- 이미지 분할 (semantic segmentation)
- 의료 영상 분석, 위성/드론 영상 분석 등

드론 기반 태양광 패널 결함 탐지 같은 작업에서도 CNN 계열 모델(또는 이를 발전시킨 구조)이 결함 영역 인식에 흔히 쓰입니다.

혹시 특정 용도(예: 결함 탐지, 세그멘테이션)에 맞춘 CNN 구조나 최신 아키텍처(ResNet, EfficientNet, YOLO 등)에 대해 더 자세히 알고 싶으신가요?
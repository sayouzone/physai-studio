## End-to-End GCP-Free Homography 파이프라인 (RTK 기반 호모그래피).

워크플로우
----------
1. EXIF/XMP 메타데이터 추출 → 사진별 RTK 좌표 + 짐벌 자세
2. 좌표계 변환 (WGS84 → EPSG:5186), 선택적 타원체고 → 정표고
3. RTK 품질 검증 + BA 가중치 공분산 (실측 σ 기반)
4. SfM Tie Point — RTK 인접쌍만 매칭
5. 공선조건 · 회전행렬 (5a track, 5b triangulation)
6. RTK 제약 Bundle Adjustment
7. 지상평면 추정 + 평면유도 호모그래피
8. 정사영상 합성 (프레임별 GeoTIFF 또는 단일 모자이크)

★ 원본 pipeline.py 에서 고친 것 — 반드시 읽을 것
------------------------------------------------
**``_build_initial_state`` 의 자세 변환이 틀려 있었다.**

원본::

    omega = deg2rad(gimbal_roll)
    phi   = deg2rad(gimbal_pitch + 90)      # nadir(-90) → 0 "보정"
    kappa = deg2rad(gimbal_yaw)

이 식은 nadir 정북 촬영 (roll=0, pitch=−90, yaw=0) 에서
``(ω, φ, κ) = (0, 0, 0)`` → ``R = I`` 를 만든다. 그런데
``geometry.project_point`` 규약에서 광축은 ``R[2]`` 이므로 광축이
``(0, 0, +1)`` — **하늘을 향한다.**

결과::

    카메라 (200000, 400000, 160), 지상점 48 m 아래
    → 가시 분모 (R[2]·diff) = −48.00   (전방이어야 하는데 후방)

* ``triangulate_dlt`` 는 cheirality 를 검사하지 않으므로 조용히
  **드론 위쪽**에 3D 점을 만든다.
* BA 는 그 거울상 기하를 최소화하므로 수렴은 하지만 해가 무의미하다.
* ``ortho.simple_orthophoto`` 는 *또 다른* 부호 규약
  (``d_cam=[...,−1]`` + ``px = cx + f·…``) 을 써서 이 오류를 **부분적으로**
  상쇄한다. 두 오류가 겹친 최종 결과는 **동서(E-W)가 거울반전된 정사영상**
  이다. 실측 검증::

      진짜 지상점  : 동 +10.0 m, 북 +7.0 m
      현 파이프라인: 동 −10.0 m, 북 +7.0 m   → 오차 (−20.0, +0.0) m

  태양광 패널 격자는 좌우 대칭에 가까워 육안으로는 거의 식별되지 않는다.
  좌표를 실측과 대조해야만 드러난다.

수정판은 ``homography.pose.opk_from_gimbal`` 을 쓴다. 이 함수는
``selftest`` 에서 ``project_point`` 규약 및 OpenCV 픽셀 규약과의 일치를
수치로 검증한다 (편차 < 1e-8 px).

정사영상도 ``ortho.simple_orthophoto`` 대신 ``homography.orthorectify_frame``
/ ``mosaic_frames`` 를 쓴다. 같은 부호 규약을 공유하므로 BA 산출물을 그대로
소비할 수 있고, 출력 픽셀당 광선교차 계산이 ``warpPerspective`` 한 번으로
대체된다.

GCP-free 한계
-------------
* 수평 정확도: 2~5 cm 가능 (RTK Fixed 기준)
* 수직 정확도: 5~15 cm (안테나 위상 중심 오프셋, GNSS 다중경로 영향)
* 절대 정확도가 중요한 측량/검측용은 최소 1~2 점의 Check Point 권장.

GCP-free 정확도 — 오차 예산
---------------------------
"RTK Fixed 니까 2~5 cm" 는 **카메라 위치**의 정확도이지 정사영상 위 지상점의
정확도가 아니다. 지상점 오차는 다른 항이 지배한다.
 
한 점의 수평 오차를 off-nadir 비 ``k = r/h`` 로 쓰면::
 
    자세오차 δ  →  h·δ·(1 + k²)        ← 보통 지배항
    평면가정 Δh →  Δh·k                 ← 잘못 잡으면 최대 지배항
    카메라 고도 →  σ_z·k
    카메라 수평 →  σ_xy                 (1:1 전파)
    레버암 잔차 →  약 2 cm
 
H20T 실측 조건 (고도 45.87 m, 지상범위 26.7 × 35.4 m → 최외곽 k=0.483,
off-nadir 25.8°; RTK Fixed σ_xy≈1 cm, σ_z≈2.5 cm) 에서::
 
    경로                                        자세     평면    RSS 합
    ------------------------------------------------------------------
    직접 (skip_sfm), 자세 0.3°, 중심           24.0     0.0    24.1 cm
    직접 (skip_sfm), 자세 0.3°, 가장자리        29.6     4.8    30.1 cm
    BA 후, 자세 0.03°, 중심                     2.4     0.0     3.3 cm
    BA 후, 자세 0.03°, 가장자리                 3.0     4.8     6.2 cm
    BA 후, 평면을 '지면' 에 맞춤 (Δh=1.5 m)      3.0    72.5    72.6 cm
 
읽는 법:
 
* **``skip_sfm`` 은 dm 급이다.** 짐벌 자세 정확도에 직접 묶여 2~5 cm 가 안
  나온다. 현장에서 눈으로 확인하는 용도지 좌표를 쓰는 용도가 아니다.
* **BA 경로는 중심 3 cm / 가장자리 6 cm 수준**이 현실적인 기대치다.
* **평면을 어디에 맞추느냐가 가장 크다.** 태양광 단지에서 결함은 패널
  상면에 있다. 평면을 지면에 맞추면 가장자리에서 70 cm 가 넘는다. BA 점군은
  패널 상면이 대부분이라 ``estimate_ground_plane`` 이 자연히 상면 평면을
  잡는데, 이것이 **의도된 동작**이다. LRF/메타데이터 폴백 경로로 떨어지면
  지면 평면이 나오므로 ``panel_top_offset_m`` 로 보정할 것.
 
**수직 정확도는 이 산출물에 정의되지 않는다.** 평면 호모그래피는 점별 높이를
만들지 않는다. 존재하는 수직량은 적합된 평면 ``(a, b, c)`` 하나뿐이고, 그
품질은 ``summary["ground_plane"]["inlier_rmse_m"]`` 로 보고된다. 정사영상에서
표고를 읽으려 하지 말 것 — DSM 이 필요하면 이 파이프라인이 의도적으로 포기한
부분이다.
 
BA 후 자세 정밀도는 로그에서 역산할 수 있다: ``δ ≈ 재투영RMSE / f_px``
(0.5 px, f_px=4000 → 0.007°). 위 표의 0.03° 는 그보다 보수적인 가정이다.
 
위 수치는 **계산된 예산이지 측정값이 아니다.** 자세오차 0.3° / 0.03° 는
가정이고, 실제 값은 기체·짐벌 개체차와 비행 조건에 따라 달라진다. 절대
정확도가 중요한 측량/검측용은 최소 1~2 점의 Check Point 로 실측 검증할 것.

## WGS84 ↔ 투영좌표계 변환.

기본 타깃은 EPSG:5186 (Korea 2000 / Central Belt). 필요 시 다른 EPSG 코드로
바꿔 사용 가능. 변환기는 stateless 하지만 ``pyproj.Transformer`` 객체 생성
비용이 약간 있으므로 (수십 ms) 파이프라인당 한 번만 만들어 재사용한다.

## RTK 측정 품질 검증 및 BA prior 가중치 계산.

DJI RtkFlag 의미 (DJI SDK 문서 기준)
------------------------------------
* 0  = None / GPS only
* 16 = RTK Float  (수십 cm 정확도)
* 34 = RTK Single (저정밀)
* 50 = RTK Fixed  (1~3 cm 정확도) ← 신뢰 가능

★ 이 버전에서 고친 것
---------------------
1. **실측 σ 사용.** ``image.metadata.extract_metadata`` 는 XMP 에서
   ``RtkStdLat / RtkStdLon / RtkStdHgt`` 를 파싱해 ``meta.rtk_std`` 에
   넣는데, 원본 ``compute_rtk_prior_weights`` 는 이 값을 한 번도 읽지 않고
   고정 상수 ``gps_std_xy=0.10``, ``gps_std_z=0.15`` 만 썼다. 즉 프레임별
   실제 측위 정확도가 BA 가중치에 전혀 반영되지 않았다. RTK 가 열화하는
   구간(비행 반환점, 수목 근처)에서 σ 는 3~5 배까지 벌어지는데, 같은
   가중치를 주면 그 프레임이 블록 전체를 끌고 간다.

2. **σ 축 순서.** ``rtk_std = [σ_lat, σ_lon, σ_hgt]`` 인데 투영좌표는
   (X=Easting, Y=Northing) 이다. → **σ_lon 이 X, σ_lat 이 Y**. 뒤집으면
   한국 위도에서는 두 값이 비슷해 결과가 그럴듯해 보이지만, 정확히
   그 "열화 구간"에서만 틀린다.

3. **σ 하한.** RTK 가 σ=0 을 보고하는 프레임이 실제로 존재한다. ``1/σ²``
   이 발산해 그 프레임이 사실상 hard constraint 가 되어 BA 를 고정시킨다.

4. **``estimate_ground_z`` 폴백 상수 제거.** 원본은 정보가 없을 때
   ``altitude - 100.0`` 을 반환했다. 100 m 는 아무 근거가 없는 값이고,
   이 값이 그대로 정사영상 평면 고도가 되어 GSD 와 지상범위 전체를
   틀리게 만든다. 조용히 틀린 값을 내느니 ``None`` 을 반환해 호출자가
   그 사진을 건너뛰게 하는 편이 낫다.


## RTK 위치 기반 인접쌍 선택 (파이프라인 4단계).

문제
----
N 장을 전수 매칭하면 ``N(N−1)/2`` 쌍이다. 165 장이면 13,530 쌍, 1,000 장이면
499,500 쌍. 그런데 드론 사진의 대부분은 서로 **겹치지 않는다** — 겹치지 않는
쌍의 매칭 시도는 전부 낭비이고, 게다가 태양광 패널처럼 반복 텍스처가 강한
장면에서는 겹치지 않는 쌍이 **가짜 매칭을 만들어 낸다**.

RTK 가 cm 급 위치를 이미 알려주므로, 겹칠 수 없는 쌍은 매칭 전에 잘라낸다.

기하학적 판정
-------------
두 사진의 footprint 는 이미 계산할 수 있다 (``homography`` 모듈). 그러나
매칭 단계는 평면 추정 전이므로, 여기서는 **보수적 근사**를 쓴다:

    r_i = (촬영고도) × tan(화각 반각) × sqrt(2)      # 외접원 반경
    겹침 가능 ⟺ |C_i − C_j| < (r_i + r_j) × margin

``sqrt(2)`` 는 직사각형 footprint 의 대각 반경, ``margin`` 은 자세 기울기와
지형 기복에 대한 여유. 이 판정은 실제 겹침의 **상위집합**이라 안전하다
(진짜 겹치는 쌍을 버리지 않는다).

인접쌍 필터가 없으면 뭐가 잘못되나
----------------------------------
겹치지 않는 두 프레임이 매칭되면 그 track 은 전혀 다른 두 지상점을 하나로
묶는다. ``tracks.build_tracks`` 의 충돌 검사는 *같은 이미지의 두 keypoint* 만
잡아내지 이 경우는 통과시킨다. 결과적으로 BA 가 존재하지 않는 지상점을
만족시키려고 카메라 자세를 왜곡한다.


## 페어별 매칭을 다중 시점 track 으로 연결.

track 의 정의
-------------
``track`` = 동일 지상점에 대응하는 관측들의 집합
         = ``[(image_idx, keypoint_idx, px, py), ...]``

Union-Find 로 ``(image_idx, keypoint_idx)`` 노드들을 연결해, 페어별로 끊겨 있던
매칭들을 하나의 그룹으로 묶는다. 그 후 그룹별로 다음 조건을 검사:

* **충돌 검사**: 한 이미지에서 두 개 이상의 keypoint 가 같은 track 에 들어가면
  매칭 오류로 보고 track 전체를 폐기 (다른 지상점이 한 track 으로 섞이는 사고).
* **길이 필터**: ``min_track_len`` 미만은 삼각측량 불가, ``max_track_len`` 초과는
  보통 반복 텍스처 (태양광 패널 그리드 등) 의 잘못된 매칭.


## DLT(Direct Linear Transform) 다중시점 삼각측량.

원리
----
각 관측 (u, v) 와 카메라 P 에 대해 ``[u·w, v·w, w]ᵀ = P · [X,Y,Z,1]ᵀ`` 에서
u, v 를 소거하면 ``X = [X,Y,Z,1]ᵀ`` 에 대한 2 개의 동차 선형식이 나온다::

    u · (P[2]·X) - (P[0]·X) = 0
    v · (P[2]·X) - (P[1]·X) = 0

N 개 시점이면 ``2N × 4`` 행렬 A 가 되고, ``A · X = 0`` 의 최소제곱해는
``AᵀA`` 의 최소 특이값에 대응하는 우특이벡터 (SVD 마지막 행).

GPU 가속 전략
-------------
SVD 는 length-N 마다 행렬 크기가 달라 단순 ``(B, 2N, 4)`` 배치가 안 된다.
→ **동일 length 끼리 그룹핑** 후 그룹별로 ``cp.linalg.svd`` 배치 호출.
드론 사진의 track 길이 분포는 보통 2~6 에 집중돼 있어 그룹 수가 적다.

삼각측량 후 두 가지 품질 필터:
1. **시선각 (parallax)** — 가장 멀리 떨어진 두 카메라에서 점을 바라본
   방향벡터 사이 각도가 ``min_triangulation_angle_deg`` 미만이면 폐기
   (깊이가 불안정). nadir 드론은 베이스라인이 짧아 2도 정도로 완화.
2. **재투영 오차** — 복원된 X 를 다시 모든 카메라로 투영한 오차의 평균이
   ``max_reproj_err_px`` 초과면 폐기.


## RTK 제약 Bundle Adjustment.

핵심 아이디어
-------------
GCP 가 없으므로 절대 좌표계 기준점은 RTK 측정값. 하지만 RTK 도 cm 급 오차가
있으므로 hard constraint 가 아닌 **soft constraint (가중치 = 1/σ²)** 로 잔차에
추가::

    residual = [reprojection_errors, sqrt(w) * (camera_pos - rtk_prior)]

* reprojection error 가 일관된 internal geometry 를 보장
* RTK prior 가 절대 georeferencing 을 보장
* 둘이 가중 평균되어 outlier 에 강건한 해를 찾음

★ 이 버전에서 고친 것
---------------------
원본은 ``_build_residuals_np`` / ``_build_residuals_gpu`` /
``_build_jacobian_sparsity`` 를 **정의만 하고 한 번도 호출하지 않았다**.
``rtk_constrained_bundle_adjustment`` 안에 중첩 정의된 Python 루프
``residuals`` 가 실제로 쓰였다. 즉 docstring 이 설명하는 가속은 전부
미적용 상태였다. 이 파일은 세 함수를 실제 경로에 연결한다.

1. **관측 평탄화 + 벡터화 residual** — ``residuals(x)`` 호출당 Python 루프
   M 회 → 단일 ``einsum`` 한 번.
2. **희소 jacobian** — ``least_squares(jac_sparsity=...)``. 이게 없으면
   ``trf`` 가 finite-difference 로 **전체 파라미터 수 P 만큼** residual 을
   재평가한다 (P = 6·n_cam + 3·n_pts, 보통 수만~수십만). sparsity 를 주면
   그래프 컬러링으로 수십 회로 줄어든다. **GPU 없이 얻는 가장 큰 이득.**
3. **CuPy backend** — 관측 M ≥ 5000 일 때만. 적으면 PCIe 전송이 더 비싸다.
4. **2단계 robust 스케줄** — 원본의 ``loss="huber", f_scale=2.0`` 단일 호출은
   초기 재투영 오차가 몇 px 를 넘으면 전 관측이 이상점 취급되어 최적화가
   시작조차 못 한다 (실측: 84초 소요, 카메라 이동량 0.000 m). 1단계는 잔차
   분포에 맞춘 느슨한 ``soft_l1``, 2단계에서 ``huber 2px`` 로 조인다.

★ 좌표 중심화 (추가)
--------------------
EPSG:5186 절대좌표는 (200000, 400000) 근방이다. ``diff = P_o − C_o`` 에서
두 항 모두 10⁵ 스케일인데 차이는 10¹ 스케일 → 상쇄로 유효자릿수 약 4 자리
손실. 또 ``least_squares`` 의 finite-difference 스텝은 파라미터 크기에
비례하므로 10⁵ 파라미터에 대한 상대 스텝이 지나치게 커진다.
→ 블록 중심을 빼고 최적화한 뒤 되돌린다. 결과는 동일하고 수치만 좋아진다.

★ RMSE 계산 (수정)
------------------
원본의 ``rmse_px = sqrt(2·result.cost / len(observations))`` 는
(a) ``cost`` 에 RTK prior 잔차가 섞여 있고 (b) huber 감쇠가 적용된 값이라
재투영 RMSE 가 아니다. 해에서 재투영 잔차만 다시 계산한다.


## '거대한 평면(가상 지표면)' 추정 — 파이프라인 7단계의 입력.

DSM 을 만들지 않는 대신 대상 지역을 평면 하나로 근사한다. 그 평면을 얼마나
잘 잡느냐가 정사영상 정확도를 지배한다.

오차의 크기 (직관)
------------------
평면에서 ``Δh`` 만큼 떨어진 실제 지형점은, 연직에서 ``θ`` 만큼 기울어진
시선으로 보면 지상에서 ``Δh · tan θ`` 만큼 밀린다. 화각 반각이 20° 인
사진의 가장자리에서 ``Δh = 1 m`` 면 밀림은 약 0.36 m. 즉:

* 패널 상면과 지면을 혼동 (Δh ≈ 1.5 m) → 가장자리에서 0.5 m 급 오차
* 경사 3% 인 부지를 수평평면으로 근사, 촬영폭 120 m → 양끝 ±1.8 m 편차
  → 가장자리 오차 0.6 m 급

두 번째 항목 때문에 이 모듈은 **경사평면** (``Z = a·x + b·y + c``) 을 기본으로
지원한다. 호모그래피는 임의 평면에 대해 정확하므로 경사를 넣어도 비용이 0 이다.
수평평면만 쓰는 것은 근거 없는 제약이다.

추정 우선순위
-------------
1. **BA 3D 점군** — 가장 신뢰도 높음. RANSAC 평면적합으로 이상점 제거.
   태양광 단지는 패널 상면이 점군의 대부분을 차지하므로, 적합된 평면은
   자연히 '패널 상면 평면' 이 된다. 검출 결과를 패널 위치로 쓸 거라면
   이게 오히려 **원하는 기준면**이다 (지면이 아니라).
2. **LRF** — H20T 는 촬영 시점 조준점의 실측 절대고도를 준다.
   ``altitude − relative_height`` 보다 훨씬 정확.
3. **메타데이터 폴백** — ``절대고도 − 상대고도``. 이륙지점 기준이라
   지형 기복이 있으면 편향된다.

⚠ 절대고도 하드코딩 금지
------------------------
"이 사이트의 패널 상면은 114.87 m" 같은 절대값은 다른 사이트에서 즉시
깨진다 (지형 표고 종속). 이식 가능한 값은 **지면 대비 상대 높이**
(예: 패널 상면 1.48 m) 이므로, 폴백 경로에서는 ``panel_top_offset_m`` 으로
받는다.


## 평면유도 호모그래피 (plane-induced homography) — RTK 기반 정사보정의 핵심.

배경
----
``ortho.orthophoto.simple_orthophoto`` 는 출력 래스터의 **모든 픽셀**에 대해
ray-plane 교점을 계산한다 (meshgrid → diff → einsum → 나눗셈 → ``cv2.remap``).
하지만 지표면을 평면으로 가정하는 순간, 지상평면 ↔ 이미지평면 대응은
**3×3 호모그래피 하나로 정확히 닫힌다**. 즉 수천만 번의 광선 교차 계산은
불필요하고, 행렬 하나와 ``cv2.warpPerspective`` 한 번이면 수학적으로 동일한
결과가 나온다.

유도
----
카메라 행렬 (``geometry.camera_projection_matrix`` 와 동일 규약)::

    P = K_neg · [R | -R·C],   K_neg = [[-f, 0, cx], [0, -f, cy], [0, 0, 1]]

지상평면을 ``Z = a·x + b·y + c`` 로 두면 평면 위의 점은::

    [X, Y, Z, 1]ᵀ = S · [x, y, 1]ᵀ,   S = [[1,0,0], [0,1,0], [a,b,c], [0,0,1]]

따라서 지상평면 → 이미지 호모그래피는 단순히::

    H_g2i = P · S = K_neg · [ (r1 + a·r3) | (r2 + b·r3) | (c·r3 − R·C) ]

(``r1, r2, r3`` 은 R 의 **열**). ``a=b=0`` 이면 수평평면 ``Z=c`` 에 대한
익숙한 형태 ``K_neg·[r1 | r2 | c·r3 − R·C]`` 로 환원된다.

경사항 ``a, b`` 가 1·2 열에 들어간다는 점에 주의. 수평평면만 시험하면 이
자리를 틀려도 통과하므로, ``selftest`` 는 반드시 경사평면으로 검증한다.

★ 국소 원점 — 이걸 빠뜨리면 조용히 정밀도가 무너진다
----------------------------------------------------
위 식을 EPSG:5186 절대좌표로 그대로 세우면 안 된다. 한국 중부원점 기준
투영좌표는 (200000, 400000) 근방이라 ``R·C`` 항이 10⁵~10⁶ 스케일인 반면
``r1, r2`` 는 O(1) 이다. 열 간 스케일 차가 10⁶ 이 되어 조건수가 폭증한다.

실측 (H20T, 고도 160 m, f=4000 px)::

    원점 (0, 0)             기준 → cond(H) = 1.1e+05,  왕복오차 1.8e-15 m
    원점 (200000, 400000)   기준 → cond(H) = 1.7e+13,  역행렬 무의미

float64 의 유효자릿수가 ~16 자리이므로 조건수 1e13 은 결과에 3 자리밖에
남기지 않는다. 그래서 이 모듈은 **프레임마다 자기 카메라 위치를 국소 원점**
으로 삼아 호모그래피를 세우고, 공개 API 에서만 절대좌표로 환산한다.
사용자는 항상 절대 투영좌표를 주고받으며, 내부 이동은 신경 쓸 필요가 없다.

부호 규약 — **중요**
--------------------
이 모듈은 ``geometry.project_point`` / ``geometry.camera_projection_matrix``
와 **동일한** 규약을 따른다::

    px = cx − f·(R[0]·diff) / (R[2]·diff)
    py = cy − f·(R[1]·diff) / (R[2]·diff)
    가시 조건: (R[2]·diff) > 0        ← 카메라 전방

이 규약에서 ``R`` 의 행은 ``[-right, -down, forward]`` (``pose.py`` 참조) 이고
광학축은 카메라 좌표계 **+Z** 이다.

⚠ 기존 ``ortho.orthophoto`` 는 ``d_cam = [-(px-cx)/f, -(py-cy)/f, -1]`` 과
``px = cx + f·…`` 를 쓴다. 이 조합은 그 안에서는 왕복 일치하지만
``project_point`` 규약과 **광축 주변 180° 회전만큼 어긋난다**
(``project_point`` 규약의 광선은 ``[-(px-cx)/f, -(py-cy)/f, +1]`` 에 비례).
Bundle Adjustment / triangulation 은 ``project_point`` 규약을 쓰므로,
BA 로 최적화한 자세를 그대로 ``simple_orthophoto`` 에 넣으면 정사영상이
180° 뒤집힌다. 본 모듈은 BA 산출물을 그대로 소비하기 위해 BA 규약에 맞췄다.


## 호모그래피 기반 정사영상 생성 및 모자이크 (파이프라인 7~8단계).

``simple_orthophoto`` 대비 달라진 점
------------------------------------
* meshgrid → diff → einsum → 나눗셈 → ``cv2.remap`` 대신
  ``cv2.warpPerspective(..., WARP_INVERSE_MAP)`` **한 번**. 출력 픽셀당
  부동소수 연산이 사라지고, 좌표 맵 두 장 (out_h×out_w×float32 ×2) 을 만들지
  않으므로 메모리도 크게 준다. 5000×5000 출력 기준 맵만 200 MB 였다.
* 경사평면 지원 (``GroundPlane.a/b``). 수평 가정은 특수한 경우일 뿐이다.
* 다중 프레임 **가중 모자이크** — feather 블렌딩 + 연직 근접 가중.
* 프레임마다 전체 캔버스를 다루지 않고 자기 footprint 창(window) 안에서만
  warp 한다. 100장짜리 현장에서 이 차이가 수십 배다.

블렌딩 가중치
-------------
두 성분의 곱:

1. **feather** — 이미지 경계로부터의 거리. 인접 프레임 접합선에서 밝기
   불연속(계단)이 생기지 않게 한다.
2. **연직 근접도** — 주점에서 멀수록 시선이 기울어 지형기복 오차
   (``Δh · tan θ``) 가 커진다. 같은 지점이 두 프레임에 찍혔다면 **더 연직에
   가까운 쪽**을 신뢰해야 한다. 가장자리에 낮은 가중을 줘서 이를 구현.

이 가중은 픽셀값 평균이지 기하 보정이 아니다. 기복 오차 자체를 없애려면
DSM 이 필요하고, 그건 이 파이프라인이 의도적으로 포기한 것이다.


## 실행

#### 규약 자기검증 — 자세/호모그래피를 건드리면 먼저 여기를 돌릴 것

```bash
python -m label_studio.sayou.homography.homography.selftest
```

#### 전체 (SfM + RTK 제약 BA + 정사 모자이크)

```bash
python scripts/homography_pipeline.py \
    --image-dir ./data/solar/images/RGB \
    --output-dir ./workspace/output
````

```bash
python scripts/homography_pipeline.py \
    --image-dir ~/Development/sayouzone/solar-thermal/data/solar/그린환경센터/RGB \
    --output-dir ~/Development/sayouzone/solar-thermal/workspace/output
```

#### 빠른 현장 확인 (특징점 매칭/BA 생략, RTK 자세만)
```bash
python scripts/homography_pipeline.py --skip-sfm ...
```

#### 프레임별 GeoTIFF (모자이크 대신)
```bash
python scripts/homography_pipeline.py --no-mosaic ...
```

#### CLI 실행

```bash
python scripts/run_homography_pipeline.py --no-mosaic
```

```bash
2026-07-30 14:55:15,779 INFO ======================================================================
2026-07-30 14:55:15,779 INFO RTK 기반 호모그래피 파이프라인 (GCP-Free) — 가속: CPU only (MP=on)
2026-07-30 14:55:15,779 INFO ======================================================================
2026-07-30 14:55:20,326 INFO 이미지 380장 로딩
2026-07-30 14:55:20,327 WARNING Thermal/Telephoto 패턴 (_Z, _T) 파일명 380장 감지: ['DJI_20251217130200_0001_Z.JPG', 'DJI_20251217130204_0002_Z.JPG', 'DJI_20251217130206_0003_Z.JPG'] ... — RGB 만 처리하려면 디렉토리를 분리하거나 글로브 패턴을 변경.
2026-07-30 14:55:20,328 INFO RTK Fixed 비율: 380/380 (100.0%)
2026-07-30 14:55:20,328 INFO   RtkFlag 분포: Fixed=380
2026-07-30 14:55:20,352 INFO RTK σ: X 1.0~1.2 cm, Y 1.1~1.3 cm, Z 2.3~2.8 cm (중앙값 1.1 / 1.2 / 2.4 cm)
2026-07-30 14:55:20,414 INFO 인접쌍 선택: 1609쌍 (전수 72010쌍의 2.2%), 이미지당 평균 8.5개
2026-07-30 15:04:44,236 INFO - SIFT 추출: 9m 24s
2026-07-30 15:06:15,469 INFO 유효 매칭 페어: 1488/1609 (제거: desc없음 0, 매칭부족 5, RANSAC실패 1, inlier부족 115)
2026-07-30 15:06:15,469 INFO [stage] SfM 매칭: 10m 55s
2026-07-30 15:06:16,519 INFO track 생성: 269615개 (제거: 짧음 0, 김 7, 충돌 1482)
2026-07-30 15:06:16,519 INFO   build_tracks 시간: 평탄화 0.02s + CC 0.21s + unique/정렬 0.22s + 필터 0.60s (총 1.05s, 관측 1720092, 노드 914586, 그룹 271104)
2026-07-30 15:06:16,529 INFO   track 길이 분포: 평균 3.3, 최대 30, 2장짜리 147680개
2026-07-30 15:06:16,533 INFO [stage] track 빌드: 1.06s
2026-07-30 15:06:29,888 INFO 삼각측량: 32001개 3D점 복원, 65441개 관측 (제거: 시선각 1083, 재투영 236531, 퇴화 0)
2026-07-30 15:06:29,889 INFO   복원점 Z범위: 42.7 ~ 240.7 m (중앙값 114.1)
2026-07-30 15:06:29,889 INFO [stage] triangulation: 13.36s
2026-07-30 15:06:29,927 INFO BA 준비: 카메라 380, 점 32001, 관측 65441, 파라미터 98283, jacobian 밀도 9.09e-05 (numpy, 0.03s)
2026-07-30 15:06:29,932 INFO BA 시작 전 재투영 RMSE: 1.8 px
2026-07-30 15:09:44,913 INFO - BA 1단계 (soft_l1, f_scale=8.0 px): 3m 15s (nfev=300)
2026-07-30 15:09:45,594 INFO - BA 2단계 (huber, f_scale=2.0 px): 0.68s (nfev=30, status=4)
2026-07-30 15:09:45,597 INFO BA 완료. 재투영 RMSE 0.58 px (중앙값 0.25, 95% 1.20) | RTK prior 대비 카메라 이동: 중앙값 0.027 m, 최대 0.768 m
2026-07-30 15:09:45,598 INFO [stage] bundle adjustment: 3m 16s
2026-07-30 15:09:45,635 WARNING 평면 RANSAC inlier 비율 부족: 12.7% < 30.0% (지형이 평면과 거리가 멀거나 자세 오차가 큼)
2026-07-30 15:09:45,635 WARNING BA 점군 평면 적합 실패 — LRF/메타데이터로 폴백
2026-07-30 15:09:45,642 INFO 지상평면 적합: 경사 0.830°, 중심표고 112.71 m, RMSE 0.542 m, inlier 258/380 (67.9%)
2026-07-30 15:09:45,642 INFO LRF 기준 평면: 380점, 경사 0.830°
2026-07-30 15:09:45,656 INFO GSD 자동 결정: 0.0068 m/px (프레임 GSD 중앙값)
2026-07-30 15:09:45,656 INFO [stage] 호모그래피 구성: 0.06s (380 프레임, 평면 경사 0.830°)
2026-07-30 15:09:47,189 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130200_0001_Z_ortho.tif (4115×5332 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:09:47,965 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130204_0002_Z_ortho.tif (4161×5397 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:09:48,735 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130206_0003_Z_ortho.tif (4113×5344 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:09:49,487 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130209_0004_Z_ortho.tif (4095×5324 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:09:50,232 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130212_0005_Z_ortho.tif (4073×5299 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:09:50,960 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130214_0006_Z_ortho.tif (4070×5297 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:09:51,695 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130217_0007_Z_ortho.tif (4071×5302 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:09:52,423 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130220_0008_Z_ortho.tif (4049×5279 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:09:53,149 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130223_0009_Z_ortho.tif (4063×5299 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:09:53,897 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130225_0010_Z_ortho.tif (4077×5319 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:09:54,642 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130228_0011_Z_ortho.tif (4090×5335 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:09:55,391 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130231_0012_Z_ortho.tif (4099×5347 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:09:56,130 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130233_0013_Z_ortho.tif (4126×5385 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:09:56,891 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130236_0014_Z_ortho.tif (4142×5405 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:09:57,680 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130239_0015_Z_ortho.tif (4116×5390 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:09:58,457 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130242_0016_Z_ortho.tif (4159×5443 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:09:59,248 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130244_0017_Z_ortho.tif (4200×5492 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:00,016 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130247_0018_Z_ortho.tif (4144×5425 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:00,750 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130250_0019_Z_ortho.tif (4087×5347 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:01,486 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130252_0020_Z_ortho.tif (3975×5317 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:02,226 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130255_0021_Z_ortho.tif (3969×5277 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:02,965 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130258_0022_Z_ortho.tif (3961×5256 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:03,711 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130300_0023_Z_ortho.tif (3975×5297 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:04,467 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130303_0024_Z_ortho.tif (3982×5330 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:05,202 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130306_0025_Z_ortho.tif (3951×5305 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:05,934 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130308_0026_Z_ortho.tif (3955×5267 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:06,682 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130311_0027_Z_ortho.tif (3969×5324 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:07,388 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130314_0028_Z_ortho.tif (3984×5387 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:08,139 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130317_0029_Z_ortho.tif (4110×5397 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:09,287 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130319_0030_Z_ortho.tif (6319×5529 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:10,551 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130322_0031_Z_ortho.tif (6542×6009 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:11,318 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130325_0032_Z_ortho.tif (4201×5448 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:12,085 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130328_0033_Z_ortho.tif (4176×5435 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:13,018 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130331_0034_Z_ortho.tif (4184×5454 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:13,812 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130333_0035_Z_ortho.tif (4143×5424 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:14,670 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130336_0036_Z_ortho.tif (4352×5554 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:15,492 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130339_0037_Z_ortho.tif (4364×5562 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:16,336 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130341_0038_Z_ortho.tif (4350×5546 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:17,236 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130344_0039_Z_ortho.tif (4402×5580 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:18,084 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130347_0040_Z_ortho.tif (4431×5597 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:18,914 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130349_0041_Z_ortho.tif (4464×5635 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:19,762 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130352_0042_Z_ortho.tif (4411×5592 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:20,601 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130355_0043_Z_ortho.tif (4391×5588 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:21,408 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130357_0044_Z_ortho.tif (4134×5379 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:22,183 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130400_0045_Z_ortho.tif (4077×5335 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:22,944 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130403_0046_Z_ortho.tif (4003×5282 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:23,683 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130406_0047_Z_ortho.tif (4071×5327 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:24,430 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130408_0048_Z_ortho.tif (3985×5261 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:25,146 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130411_0049_Z_ortho.tif (3939×5221 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:25,938 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130413_0050_Z_ortho.tif (4233×5436 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:26,758 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130416_0051_Z_ortho.tif (4332×5505 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:27,526 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130419_0052_Z_ortho.tif (4023×5277 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:28,264 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130422_0053_Z_ortho.tif (3922×5202 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:29,096 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130424_0054_Z_ortho.tif (4287×5465 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:29,843 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130427_0055_Z_ortho.tif (4100×5329 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:30,570 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130430_0056_Z_ortho.tif (4127×5339 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:31,303 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130432_0057_Z_ortho.tif (4148×5364 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:32,028 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130435_0058_Z_ortho.tif (4178×5375 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:32,762 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130438_0059_Z_ortho.tif (4263×5429 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:33,850 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130440_0060_Z_ortho.tif (6097×5290 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:35,062 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130442_0061_Z_ortho.tif (6342×5778 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:35,772 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130445_0062_Z_ortho.tif (4020×5298 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:36,475 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130447_0063_Z_ortho.tif (4032×5309 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:37,226 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130450_0064_Z_ortho.tif (4210×5395 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:37,975 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130453_0065_Z_ortho.tif (4246×5431 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:38,762 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130456_0066_Z_ortho.tif (4392×5534 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:39,615 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130458_0067_Z_ortho.tif (4397×5542 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:40,436 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130501_0068_Z_ortho.tif (4346×5508 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:41,208 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130504_0069_Z_ortho.tif (4315×5481 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:41,969 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130506_0070_Z_ortho.tif (4271×5450 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:42,709 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130509_0071_Z_ortho.tif (4231×5430 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:43,446 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130512_0072_Z_ortho.tif (4199×5413 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:44,165 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130515_0073_Z_ortho.tif (4170×5397 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:44,875 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130517_0074_Z_ortho.tif (4142×5383 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:45,637 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130520_0075_Z_ortho.tif (4126×5380 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:46,347 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130523_0076_Z_ortho.tif (4106×5370 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:47,064 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130525_0077_Z_ortho.tif (4096×5372 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:47,752 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130528_0078_Z_ortho.tif (4054×5338 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:48,452 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130531_0079_Z_ortho.tif (4030×5323 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:49,162 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130534_0080_Z_ortho.tif (4022×5324 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:49,882 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130536_0081_Z_ortho.tif (3990×5306 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:50,621 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130539_0082_Z_ortho.tif (4008×5355 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:51,349 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130542_0083_Z_ortho.tif (4025×5401 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:52,052 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130544_0084_Z_ortho.tif (4083×5357 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:52,757 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130547_0085_Z_ortho.tif (4127×5384 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:53,448 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130550_0086_Z_ortho.tif (4128×5381 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:54,154 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130553_0087_Z_ortho.tif (4142×5393 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:54,888 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130555_0088_Z_ortho.tif (4183×5433 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:55,647 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130558_0089_Z_ortho.tif (4252×5504 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:56,586 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130600_0090_Z_ortho.tif (5905×4852 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:57,834 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130602_0091_Z_ortho.tif (6531×5868 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:58,543 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130605_0092_Z_ortho.tif (4259×5493 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:10:59,319 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130608_0093_Z_ortho.tif (4558×5696 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:00,079 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130610_0094_Z_ortho.tif (4396×5592 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:00,808 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130613_0095_Z_ortho.tif (4268×5493 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:01,552 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130616_0096_Z_ortho.tif (4268×5484 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:02,306 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130619_0097_Z_ortho.tif (4279×5493 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:03,180 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130621_0098_Z_ortho.tif (4776×5844 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:04,070 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130624_0099_Z_ortho.tif (4805×5855 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:04,826 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130627_0100_Z_ortho.tif (4162×5401 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:05,560 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130629_0101_Z_ortho.tif (4092×5347 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:06,290 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130632_0102_Z_ortho.tif (4056×5318 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:06,992 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130635_0103_Z_ortho.tif (4039×5306 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:07,685 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130638_0104_Z_ortho.tif (4027×5293 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:08,387 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130640_0105_Z_ortho.tif (4012×5281 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:09,139 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130643_0106_Z_ortho.tif (3954×5228 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:09,839 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130646_0107_Z_ortho.tif (3968×5244 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:10,556 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130649_0108_Z_ortho.tif (3925×5247 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:11,371 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130651_0109_Z_ortho.tif (3928×5310 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:12,091 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130654_0110_Z_ortho.tif (3915×5234 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:12,953 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130656_0111_Z_ortho.tif (4561×5703 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:13,803 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130659_0112_Z_ortho.tif (4407×5580 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:14,625 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130702_0113_Z_ortho.tif (4322×5513 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:15,407 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130705_0114_Z_ortho.tif (4290×5476 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:16,200 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130707_0115_Z_ortho.tif (4282×5460 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:17,013 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130710_0116_Z_ortho.tif (4289×5462 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:17,846 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130713_0117_Z_ortho.tif (4317×5484 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:18,632 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130715_0118_Z_ortho.tif (4302×5463 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:19,297 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130718_0119_Z_ortho.tif (3971×5221 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:20,470 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130722_0120_Z_ortho.tif (6275×5661 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:21,188 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130725_0121_Z_ortho.tif (4012×5249 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:21,873 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130727_0122_Z_ortho.tif (3893×5171 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:22,576 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130731_0123_Z_ortho.tif (3910×5187 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:23,266 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130733_0124_Z_ortho.tif (3916×5188 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:23,951 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130736_0125_Z_ortho.tif (3929×5192 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:24,655 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130739_0126_Z_ortho.tif (3940×5205 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:25,337 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130741_0127_Z_ortho.tif (3953×5216 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:26,033 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130744_0128_Z_ortho.tif (4010×5265 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:26,779 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130747_0129_Z_ortho.tif (4079×5319 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:27,548 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130749_0130_Z_ortho.tif (4053×5296 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:28,279 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130751_0131_Z_ortho.tif (4064×5307 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:28,995 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130753_0132_Z_ortho.tif (4066×5311 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:29,708 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130757_0133_Z_ortho.tif (4071×5320 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:30,502 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130759_0134_Z_ortho.tif (4083×5336 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:31,236 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130802_0135_Z_ortho.tif (4072×5323 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:32,001 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130805_0136_Z_ortho.tif (4112×5354 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:32,740 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130807_0137_Z_ortho.tif (4126×5377 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:33,511 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130810_0138_Z_ortho.tif (4161×5417 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:34,273 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130813_0139_Z_ortho.tif (4037×5347 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:35,035 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130816_0140_Z_ortho.tif (4002×5277 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:35,778 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130818_0141_Z_ortho.tif (4000×5276 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:36,580 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130821_0142_Z_ortho.tif (3988×5265 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:37,322 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130824_0143_Z_ortho.tif (3993×5272 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:38,044 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130827_0144_Z_ortho.tif (3983×5267 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:38,791 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130829_0145_Z_ortho.tif (3994×5282 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:39,523 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130832_0146_Z_ortho.tif (4028×5373 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:40,281 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130835_0147_Z_ortho.tif (4095×5505 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:41,520 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130837_0148_Z_ortho.tif (5853×6462 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:42,752 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130840_0149_Z_ortho.tif (6506×5987 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:43,463 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130844_0150_Z_ortho.tif (4145×5467 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:44,195 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130846_0151_Z_ortho.tif (4131×5429 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:44,929 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130849_0152_Z_ortho.tif (4137×5417 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:45,658 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130852_0153_Z_ortho.tif (4146×5411 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:46,422 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130854_0154_Z_ortho.tif (4162×5412 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:47,237 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130857_0155_Z_ortho.tif (4209×5437 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:48,184 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130900_0156_Z_ortho.tif (4850×5890 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:49,133 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130903_0157_Z_ortho.tif (4045×5313 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:49,891 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130905_0158_Z_ortho.tif (4040×5307 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:50,719 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130908_0159_Z_ortho.tif (4254×5505 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:51,511 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130911_0160_Z_ortho.tif (4239×5484 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:52,316 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130913_0161_Z_ortho.tif (4246×5466 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:53,073 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130917_0162_Z_ortho.tif (3999×5280 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:53,831 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130919_0163_Z_ortho.tif (4115×5353 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:54,560 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130922_0164_Z_ortho.tif (4097×5334 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:55,297 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130924_0165_Z_ortho.tif (4101×5337 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:56,060 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130927_0166_Z_ortho.tif (4075×5316 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:56,789 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130930_0167_Z_ortho.tif (4009×5267 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:57,549 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130932_0168_Z_ortho.tif (4094×5390 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:58,314 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130936_0169_Z_ortho.tif (4087×5387 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:59,057 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130939_0170_Z_ortho.tif (4085×5372 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:11:59,788 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130941_0171_Z_ortho.tif (3985×5240 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:00,535 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130944_0172_Z_ortho.tif (4106×5343 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:01,303 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130947_0173_Z_ortho.tif (4209×5443 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:02,089 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130950_0174_Z_ortho.tif (4204×5425 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:02,942 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130952_0175_Z_ortho.tif (4279×5479 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:03,748 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130955_0176_Z_ortho.tif (4276×5464 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:04,970 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217130957_0177_Z_ortho.tif (5683×6287 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:06,179 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131000_0178_Z_ortho.tif (6325×5817 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:06,840 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131004_0179_Z_ortho.tif (3847×5122 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:07,510 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131006_0180_Z_ortho.tif (3844×5162 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:08,201 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131009_0181_Z_ortho.tif (3894×5194 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:08,880 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131012_0182_Z_ortho.tif (3862×5128 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:09,559 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131014_0183_Z_ortho.tif (3879×5153 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:10,282 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131017_0184_Z_ortho.tif (3899×5203 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:11,010 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131020_0185_Z_ortho.tif (3945×5221 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:11,708 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131023_0186_Z_ortho.tif (3954×5227 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:12,418 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131025_0187_Z_ortho.tif (3941×5208 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:13,146 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131028_0188_Z_ortho.tif (3942×5208 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:13,857 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131031_0189_Z_ortho.tif (3932×5221 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:14,593 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131033_0190_Z_ortho.tif (4031×5293 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:15,301 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131036_0191_Z_ortho.tif (4022×5284 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:16,006 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131039_0192_Z_ortho.tif (4028×5284 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:16,702 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131042_0193_Z_ortho.tif (4038×5298 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:17,428 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131044_0194_Z_ortho.tif (4022×5285 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:18,165 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131047_0195_Z_ortho.tif (4006×5267 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:18,966 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131050_0196_Z_ortho.tif (4014×5280 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:19,731 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131052_0197_Z_ortho.tif (4022×5292 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:20,457 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131055_0198_Z_ortho.tif (4005×5284 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:21,184 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131058_0199_Z_ortho.tif (4011×5285 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:21,918 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131101_0200_Z_ortho.tif (4024×5297 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:22,690 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131103_0201_Z_ortho.tif (4021×5291 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:23,355 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131106_0202_Z_ortho.tif (4033×5302 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:24,010 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131109_0203_Z_ortho.tif (4045×5311 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:24,669 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131111_0204_Z_ortho.tif (4074×5339 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:25,375 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131114_0205_Z_ortho.tif (4126×5416 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:26,697 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131117_0206_Z_ortho.tif (5883×6509 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:27,969 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131120_0207_Z_ortho.tif (6460×5910 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:28,675 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131123_0208_Z_ortho.tif (4099×5383 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:29,380 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131126_0209_Z_ortho.tif (4075×5352 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:30,118 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131129_0210_Z_ortho.tif (4065×5336 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:30,843 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131132_0211_Z_ortho.tif (4067×5331 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:31,573 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131134_0212_Z_ortho.tif (4087×5344 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:32,285 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131137_0213_Z_ortho.tif (4107×5355 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:33,005 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131140_0214_Z_ortho.tif (4083×5356 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:33,776 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131142_0215_Z_ortho.tif (4079×5342 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:34,596 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131145_0216_Z_ortho.tif (4061×5317 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:35,355 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131147_0217_Z_ortho.tif (4108×5365 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:36,131 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131150_0218_Z_ortho.tif (4105×5356 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:36,881 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131153_0219_Z_ortho.tif (4090×5335 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:37,659 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131156_0220_Z_ortho.tif (4098×5338 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:38,447 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131158_0221_Z_ortho.tif (4176×5395 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:39,252 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131201_0222_Z_ortho.tif (4183×5397 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:40,011 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131204_0223_Z_ortho.tif (4165×5372 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:40,747 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131207_0224_Z_ortho.tif (4070×5306 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:41,460 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131209_0225_Z_ortho.tif (4053×5292 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:42,192 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131212_0226_Z_ortho.tif (4060×5296 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:42,933 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131215_0227_Z_ortho.tif (4119×5354 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:43,678 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131217_0228_Z_ortho.tif (4083×5323 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:44,412 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131220_0229_Z_ortho.tif (4107×5337 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:45,133 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131223_0230_Z_ortho.tif (4087×5309 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:45,858 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131226_0231_Z_ortho.tif (4095×5314 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:46,573 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131228_0232_Z_ortho.tif (4114×5325 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:47,383 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131231_0233_Z_ortho.tif (4424×5542 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:48,082 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131233_0234_Z_ortho.tif (3917×5170 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:49,177 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131236_0235_Z_ortho.tif (5543×6208 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:50,275 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131239_0236_Z_ortho.tif (6228×5590 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:50,945 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131242_0237_Z_ortho.tif (3866×5142 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:51,640 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131245_0238_Z_ortho.tif (3907×5159 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:52,481 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131248_0239_Z_ortho.tif (4750×5761 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:53,279 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131250_0240_Z_ortho.tif (4511×5639 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:54,011 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131253_0241_Z_ortho.tif (3965×5301 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:54,754 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131256_0242_Z_ortho.tif (3960×5278 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:55,529 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131258_0243_Z_ortho.tif (4117×5370 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:56,250 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131301_0244_Z_ortho.tif (3921×5165 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:57,079 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131304_0245_Z_ortho.tif (3943×5280 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:57,819 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131306_0246_Z_ortho.tif (3969×5236 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:58,556 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131310_0247_Z_ortho.tif (3963×5234 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:59,265 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131312_0248_Z_ortho.tif (3925×5211 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:12:59,994 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131315_0249_Z_ortho.tif (3934×5202 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:00,713 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131317_0250_Z_ortho.tif (3945×5223 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:01,498 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131320_0251_Z_ortho.tif (3941×5225 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:02,207 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131323_0252_Z_ortho.tif (3960×5273 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:02,927 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131326_0253_Z_ortho.tif (3954×5281 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:03,629 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131328_0254_Z_ortho.tif (3948×5225 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:04,378 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131331_0255_Z_ortho.tif (3953×5226 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:05,115 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131334_0256_Z_ortho.tif (3979×5251 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:05,834 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131336_0257_Z_ortho.tif (3963×5302 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:06,530 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131339_0258_Z_ortho.tif (3938×5239 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:07,222 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131342_0259_Z_ortho.tif (3937×5222 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:07,906 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131345_0260_Z_ortho.tif (3945×5234 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:08,598 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131347_0261_Z_ortho.tif (3964×5295 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:09,295 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131350_0262_Z_ortho.tif (3982×5349 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:10,011 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131352_0263_Z_ortho.tif (4028×5438 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:11,208 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131355_0264_Z_ortho.tif (5818×6447 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:12,482 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131358_0265_Z_ortho.tif (6624×6067 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:13,174 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131401_0266_Z_ortho.tif (4033×5343 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:13,827 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131404_0267_Z_ortho.tif (4021×5298 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:14,494 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131407_0268_Z_ortho.tif (3999×5362 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:15,164 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131409_0269_Z_ortho.tif (3979×5298 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:15,824 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131412_0270_Z_ortho.tif (3989×5260 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:16,488 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131415_0271_Z_ortho.tif (3983×5243 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:17,204 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131417_0272_Z_ortho.tif (3988×5272 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:17,960 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131420_0273_Z_ortho.tif (4012×5292 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:18,700 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131423_0274_Z_ortho.tif (4029×5299 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:19,450 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131426_0275_Z_ortho.tif (4110×5374 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:20,254 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131428_0276_Z_ortho.tif (4096×5349 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:21,064 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131431_0277_Z_ortho.tif (4184×5431 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:21,979 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131434_0278_Z_ortho.tif (4234×5467 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:22,799 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131436_0279_Z_ortho.tif (4200×5430 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:23,568 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131439_0280_Z_ortho.tif (4193×5415 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:24,356 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131442_0281_Z_ortho.tif (4191×5404 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:25,145 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131445_0282_Z_ortho.tif (4194×5417 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:25,905 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131447_0283_Z_ortho.tif (4197×5409 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:26,708 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131450_0284_Z_ortho.tif (4208×5413 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:27,524 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131453_0285_Z_ortho.tif (4157×5363 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:28,311 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131456_0286_Z_ortho.tif (4142×5355 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:29,092 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131459_0287_Z_ortho.tif (4228×5424 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:29,910 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131501_0288_Z_ortho.tif (4312×5489 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:30,687 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131504_0289_Z_ortho.tif (4320×5485 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:31,482 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131507_0290_Z_ortho.tif (4330×5484 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:32,300 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131509_0291_Z_ortho.tif (4329×5475 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:33,125 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131512_0292_Z_ortho.tif (4423×5535 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:34,304 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131514_0293_Z_ortho.tif (5780×6294 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:35,382 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131517_0294_Z_ortho.tif (6189×5526 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:36,078 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131520_0295_Z_ortho.tif (3913×5160 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:36,799 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131522_0296_Z_ortho.tif (4062×5269 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:37,507 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131525_0297_Z_ortho.tif (3953×5197 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:38,214 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131528_0298_Z_ortho.tif (3941×5189 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:38,924 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131531_0299_Z_ortho.tif (3940×5192 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:39,602 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131533_0300_Z_ortho.tif (3917×5173 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:40,301 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131536_0301_Z_ortho.tif (3923×5189 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:40,978 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131539_0302_Z_ortho.tif (3875×5148 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:41,658 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131541_0303_Z_ortho.tif (3867×5132 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:42,347 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131544_0304_Z_ortho.tif (3870×5140 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:43,045 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131547_0305_Z_ortho.tif (3872×5144 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:43,760 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131550_0306_Z_ortho.tif (3901×5170 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:44,487 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131552_0307_Z_ortho.tif (3917×5196 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:45,229 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131555_0308_Z_ortho.tif (3957×5224 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:45,973 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131558_0309_Z_ortho.tif (4012×5266 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:46,771 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131601_0310_Z_ortho.tif (4009×5259 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:47,504 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131603_0311_Z_ortho.tif (4027×5267 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:48,232 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131606_0312_Z_ortho.tif (4060×5291 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:48,997 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131609_0313_Z_ortho.tif (4069×5312 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:49,718 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131611_0314_Z_ortho.tif (4077×5317 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:50,487 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131614_0315_Z_ortho.tif (4084×5325 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:51,158 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131617_0316_Z_ortho.tif (4082×5319 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:51,938 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131619_0317_Z_ortho.tif (4086×5320 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:52,603 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131622_0318_Z_ortho.tif (4109×5347 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:53,318 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131625_0319_Z_ortho.tif (4148×5389 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:54,027 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131628_0320_Z_ortho.tif (4188×5434 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:54,766 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131630_0321_Z_ortho.tif (4241×5524 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:55,949 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131632_0322_Z_ortho.tif (5813×6414 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:57,118 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131635_0323_Z_ortho.tif (6527×5936 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:57,767 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131638_0324_Z_ortho.tif (3974×5378 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:58,405 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131641_0325_Z_ortho.tif (3970×5361 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:59,068 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131644_0326_Z_ortho.tif (3981×5379 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:13:59,744 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131647_0327_Z_ortho.tif (3975×5355 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:00,418 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131649_0328_Z_ortho.tif (3988×5371 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:01,076 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131652_0329_Z_ortho.tif (3971×5307 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:01,736 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131953_0001_Z_ortho.tif (3936×5241 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:02,405 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131956_0002_Z_ortho.tif (3930×5196 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:03,087 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217131959_0003_Z_ortho.tif (3939×5207 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:03,801 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132002_0004_Z_ortho.tif (3954×5230 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:04,505 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132004_0005_Z_ortho.tif (3948×5287 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:05,210 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132007_0006_Z_ortho.tif (3912×5181 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:05,982 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132010_0007_Z_ortho.tif (4143×5367 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:06,715 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132013_0008_Z_ortho.tif (4149×5361 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:07,574 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132016_0009_Z_ortho.tif (4853×5852 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:08,881 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132018_0010_Z_ortho.tif (6615×6258 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:10,118 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132021_0011_Z_ortho.tif (6478×6118 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:10,853 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132024_0012_Z_ortho.tif (4008×5287 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:11,549 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132026_0013_Z_ortho.tif (3988×5243 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:12,253 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132029_0014_Z_ortho.tif (3980×5227 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:12,965 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132032_0015_Z_ortho.tif (3948×5223 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:13,695 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132034_0016_Z_ortho.tif (3963×5229 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:14,407 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132037_0017_Z_ortho.tif (3938×5188 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:15,182 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132040_0018_Z_ortho.tif (4137×5374 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:15,957 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132043_0019_Z_ortho.tif (4150×5375 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:16,745 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132046_0020_Z_ortho.tif (4159×5367 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:17,521 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132048_0021_Z_ortho.tif (4163×5356 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:18,737 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132051_0022_Z_ortho.tif (5700×6248 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:19,946 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132054_0023_Z_ortho.tif (6286×5822 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:20,751 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132057_0024_Z_ortho.tif (3992×5337 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:21,489 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132100_0025_Z_ortho.tif (3967×5254 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:22,235 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132102_0026_Z_ortho.tif (3952×5201 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:22,961 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132105_0027_Z_ortho.tif (3939×5176 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:23,669 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132108_0028_Z_ortho.tif (3939×5165 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:24,366 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132110_0029_Z_ortho.tif (3963×5184 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:25,092 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132113_0030_Z_ortho.tif (3999×5214 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:25,831 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132116_0031_Z_ortho.tif (4045×5256 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:26,706 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132118_0032_Z_ortho.tif (4102×5329 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:27,817 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132121_0033_Z_ortho.tif (4411×5542 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:28,664 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132124_0034_Z_ortho.tif (4368×5536 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:29,482 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132127_0035_Z_ortho.tif (4334×5496 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:30,322 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132129_0036_Z_ortho.tif (4288×5456 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:31,124 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132132_0037_Z_ortho.tif (4272×5440 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:31,907 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132135_0038_Z_ortho.tif (4249×5414 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:32,703 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132138_0039_Z_ortho.tif (4246×5420 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:33,470 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132140_0040_Z_ortho.tif (4217×5393 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:34,224 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132143_0041_Z_ortho.tif (4200×5383 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:35,184 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132146_0042_Z_ortho.tif (4179×5371 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:35,998 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132148_0043_Z_ortho.tif (4183×5382 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:36,735 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132151_0044_Z_ortho.tif (4186×5391 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:37,493 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132154_0045_Z_ortho.tif (4187×5397 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:38,245 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132156_0046_Z_ortho.tif (4195×5415 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:39,015 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132159_0047_Z_ortho.tif (4212×5445 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:39,756 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132202_0048_Z_ortho.tif (4224×5468 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:40,485 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132205_0049_Z_ortho.tif (4239×5490 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:41,273 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132208_0050_Z_ortho.tif (4266×5531 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:42,087 INFO 정사영상 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/DJI_20251217132210_0051_Z_ortho.tif (4294×5572 px, GSD=0.0068 m, 평면경사 0.83°)
2026-07-30 15:14:42,090 INFO [stage] 정사영상: 4m 56s
{
  "images": 380,
  "frames_georeferenced": 380,
  "epsg": 5186,
  "gsd_m": 0.006759876570501243,
  "ba_reprojection_rmse_px": 0.5805754311377027,
  "ground_plane": {
    "a": 0.009994722968409888,
    "b": 0.010479201754635747,
    "c": 112.71317703401026,
    "origin_xy": [
      192857.58655561222,
      235022.9616495924
    ],
    "slope_deg": 0.8296595911996921,
    "inlier_rmse_m": 0.5422522181420664,
    "n_inliers": 258
  },
  "ortho": {
    "frames_ok": 380,
    "frames_failed": 0,
    "gsd_m": 0.006759876570501243
  }
}
Elapsed: 0:19:26
```

```bash
python -m sayou.homography.pipeline
```

```bash
<frozen runpy>:128: RuntimeWarning: 'sayou.homography.pipeline' found in sys.modules after import of package 'sayou.homography', but prior to execution of 'sayou.homography.pipeline'; this may result in unpredictable behaviour
2026-07-29 19:09:01,527 INFO ======================================================================
2026-07-29 19:09:01,527 INFO RTK 기반 호모그래피 파이프라인 (GCP-Free) — 가속: CPU only (MP=on)
2026-07-29 19:09:01,527 INFO ======================================================================
2026-07-29 19:09:05,754 INFO 이미지 380장 로딩
2026-07-29 19:09:05,756 WARNING Thermal/Telephoto 패턴 (_Z, _T) 파일명 380장 감지: ['DJI_20251217130200_0001_Z.JPG', 'DJI_20251217130204_0002_Z.JPG', 'DJI_20251217130206_0003_Z.JPG'] ... — RGB 만 처리하려면 디렉토리를 분리하거나 글로브 패턴을 변경.
2026-07-29 19:09:05,756 INFO RTK Fixed 비율: 380/380 (100.0%)
2026-07-29 19:09:05,756 INFO   RtkFlag 분포: Fixed=380
2026-07-29 19:09:05,778 INFO RTK σ: X 1.0~1.2 cm, Y 1.1~1.3 cm, Z 2.3~2.8 cm (중앙값 1.1 / 1.2 / 2.4 cm)
2026-07-29 19:09:05,829 INFO 인접쌍 선택: 1609쌍 (전수 72010쌍의 2.2%), 이미지당 평균 8.5개
2026-07-29 19:17:59,285 INFO - SIFT 추출: 8m 53s
2026-07-29 19:19:26,011 INFO 유효 매칭 페어: 1488/1609 (제거: desc없음 0, 매칭부족 5, RANSAC실패 1, inlier부족 115)
2026-07-29 19:19:26,011 INFO [stage] SfM 매칭: 10m 20s
2026-07-29 19:19:27,130 INFO track 생성: 269615개 (제거: 짧음 0, 김 7, 충돌 1482)
2026-07-29 19:19:27,131 INFO   build_tracks 시간: 평탄화 0.04s + CC 0.24s + unique/정렬 0.21s + 필터 0.63s (총 1.12s, 관측 1720092, 노드 914586, 그룹 271104)
2026-07-29 19:19:27,140 INFO   track 길이 분포: 평균 3.3, 최대 30, 2장짜리 147680개
2026-07-29 19:19:27,146 INFO [stage] track 빌드: 1.14s
2026-07-29 19:19:40,334 INFO 삼각측량: 32001개 3D점 복원, 65441개 관측 (제거: 시선각 1083, 재투영 236531, 퇴화 0)
2026-07-29 19:19:40,334 INFO   복원점 Z범위: 42.7 ~ 240.7 m (중앙값 114.1)
2026-07-29 19:19:40,334 INFO [stage] triangulation: 13.19s
2026-07-29 19:19:40,408 INFO BA 준비: 카메라 380, 점 32001, 관측 65441, 파라미터 98283, jacobian 밀도 9.09e-05 (numpy, 0.06s)
2026-07-29 19:19:40,412 INFO BA 시작 전 재투영 RMSE: 1.8 px
2026-07-29 19:22:50,291 INFO - BA 1단계 (soft_l1, f_scale=8.0 px): 3m 10s (nfev=300)
2026-07-29 19:22:50,979 INFO - BA 2단계 (huber, f_scale=2.0 px): 0.69s (nfev=30, status=4)
2026-07-29 19:22:50,982 INFO BA 완료. 재투영 RMSE 0.58 px (중앙값 0.25, 95% 1.20) | RTK prior 대비 카메라 이동: 중앙값 0.027 m, 최대 0.768 m
2026-07-29 19:22:50,982 INFO [stage] bundle adjustment: 3m 11s
2026-07-29 19:22:51,017 WARNING 평면 RANSAC inlier 비율 부족: 12.7% < 30.0% (지형이 평면과 거리가 멀거나 자세 오차가 큼)
2026-07-29 19:22:51,017 WARNING BA 점군 평면 적합 실패 — LRF/메타데이터로 폴백
2026-07-29 19:22:51,024 INFO 지상평면 적합: 경사 0.830°, 중심표고 112.71 m, RMSE 0.542 m, inlier 258/380 (67.9%)
2026-07-29 19:22:51,024 INFO LRF 기준 평면: 380점, 경사 0.830°
2026-07-29 19:22:51,037 INFO GSD 자동 결정: 0.0068 m/px (프레임 GSD 중앙값)
2026-07-29 19:22:51,037 INFO [stage] 호모그래피 구성: 0.05s (380 프레임, 평면 경사 0.830°)
2026-07-29 19:22:51,247 INFO 모자이크 캔버스: 24503×15647 px, GSD=0.0068 m, 범위 165.6×105.8 m, 누산기 6134 MB
2026-07-29 19:25:08,280 INFO 모자이크 저장: /Users/seongjungkim/Development/sayouzone/solar-thermal/workspace/output/mosaic.tif (380장 합성, 0장 실패, 충전율 87.9%)
2026-07-29 19:25:08,543 INFO [stage] 정사영상: 2m 18s
Elapsed: 0:16:08
```

```bash
python -m label_studio.sayou.homography.homography.selftest
```

```bash
==================================================================
RTK 호모그래피 규약 자기검증
==================================================================
  geometry 규약 일치 확인
  회전행렬 정규직교 오차 1.11e-15, det 오차 1.11e-15
  (ω,φ,κ) 왕복 오차 6.66e-16
  OpenCV 픽셀 규약 일치, 최대 편차 8.12e-09 px
  가시 조건 (R[2]·diff > 0) 확인
  호모그래피 forward 6.37e-12 px, inverse 0.00e+00 m
  P·S 등가성 오차 5.82e-11
  GSD 1.200 cm/px, footprint 58.37 × 43.78 m (고도차 48.0 m)
  방위 확인: 동→+u(우) 100.0px, 북→−v(위) 100.0px
  yaw 회전 반영 확인
  카메라 후방 점 NaN 처리 확인
호모그래피 특이 (cond=inf) — 카메라가 지상평면 위에 있거나 자세/고도가 비정상. 프레임 스킵.
  축퇴 판정 확인 (평면 위 카메라만 None, 저각 촬영은 유효)
  절대좌표 조건수 최대 1.11e+05, 왕복오차 3.55e-15 m
==================================================================
13/13 통과
```
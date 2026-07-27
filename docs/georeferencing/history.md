```text
Annotation Detection 데이터를 GeoJson으로 변환했는데, 상단의 데이터는 Google Maps의 위치와 비슷하게 매핑되지만 하단으로 가면서 점차 간격이 넓어지면서 오차가 크게 발생하는데 원인과 해결 방안에 대해 조사해줘
```

## 결론부터 — `fx ≠ fy` 이방성 초점거리 버그입니다

`annotation_georeferencer._pixel_to_camera_ray()` 99~100행:

```python
fx_px = (metadata.width  / 2.0) / tan(hfov/2)   # → 6768.0 px
fy_px = (metadata.height / 2.0) / tan(vfov/2)   # → 7614.0 px
```

정사각 픽셀 카메라에서 `fx ≠ fy`는 물리적으로 불가능합니다. **비율이 1.125, 즉 12.5% 어긋나 있습니다.**

원인은 `metadata.py`의

```python
SENSOR_35MM_W = 36.0
SENSOR_35MM_H = 24.0
hfov = 2*atan(36/(2*f35))   # 41.91°
vfov = 2*atan(24/(2*f35))   # 28.65°
```

35mm 환산 초점거리(`FocalLengthIn35mmFilm`)는 **대각선 기준으로 정의**된 값입니다. 그런데 위 코드는 이를 가로/세로에 각각 독립적으로 적용하면서, 4:3 센서인 H20T 줌 카메라(5184×3888)를 3:2 프레임(36×24mm)에 억지로 끼워 맞추고 있습니다.

### 숫자 검증 (업로드하신 GeoJSON 실측과 일치)

| | fx / GSD_x | fy / GSD_y | footprint |
|---|---|---|---|
| 현재 코드 | 6768 px / **6.632 mm/px** | 7614 px / **5.895 mm/px** | 34.38 × 22.92 m |
| 정석(대각선 기준) | 7039 px / 6.376 mm/px | 7039 px / 6.376 mm/px | 33.06 × 24.79 m |

GeoJSON의 pv 검출 #19(`8Jr0NXQ4pO`, 픽셀 404.6×1945.8)를 실제로 역산하면 x변 2.683 m → **6.632 mm/px**, y변 11.470 m → **5.895 mm/px** 로 위 표와 소수점까지 일치합니다.

→ **x축(짐벌 yaw −88.7°이므로 북쪽 방향) +4.0%, y축(동쪽 방향) −7.6% 스케일 오차.**

### 증상이 정확히 설명됩니다

- 패널 열 간격이 이미지에서 664 px → 현재 **4.401 m**, 정상 **4.231 m**. **열당 17 cm씩 벌어짐**, 8열 누적 **1.36 m** ≈ 패널 폭의 절반. 스크린샷의 하단 어긋남과 크기가 맞습니다.
- 오차는 이미지 중심(드론 nadir 발끝)에서 0이고 바깥으로 방사형 증가합니다. 눈으로 상단을 기준 삼아 보시면 반대쪽 끝에 오차가 전부 몰려 보입니다 — "상단은 맞는데 하단만 벌어진다"는 인상은 여기서 나옵니다.
- 결정적 증거: anomaly 박스는 픽셀 종횡비가 **2.002 : 1**(163.1×326.5 px)인데 현재 출력은 1.925 × 1.082 m = **1.78 : 1**로 찌그러져 나옵니다. 실제 PV 모듈 종횡비(2278×1134 또는 1956×992, 약 1.97~2.01)와 맞지 않습니다. 등방 f로 고치면 2.082 × 1.040 m가 되어 실물과 일치합니다.

### 덤: RGB-IR 3.65 m 오프셋이 "affine 보정이 필요했던" 이유

이 버그의 왜곡률은 해상도 종횡비에 의존합니다.

- RGB 5184×3888 (4:3) → fy/fx = **1.1250**
- IR 640×512 (5:4) → fy/fx = **1.2000**

두 센서가 **서로 다른 이방성 왜곡**을 받으므로, 정합 잔차가 순수 평행이동이 아니라 축별 스케일이 다른 affine으로 나타납니다. 특허 준비 중인 co-registration 쪽도 이걸 고치면 잔차 구조가 크게 단순해질 가능성이 높습니다.

---

## 수정 방안

### 1) 즉시 패치 — 등방 초점거리 (필수)

`metadata.py`:

```python
SENSOR_35MM_DIAG = math.hypot(SENSOR_35MM_W, SENSOR_35MM_H)  # 43.2666

@property
def focal_px(self) -> float:
    """35mm 환산 초점거리는 '대각선' 기준 정의 → 대각선으로 환산해야 등방 f가 나온다."""
    diag_px = math.hypot(self.width, self.height)
    return self.focal_length_in_35mm * diag_px / SENSOR_35MM_DIAG

@property
def hfov_deg(self) -> float:
    return 2 * math.degrees(math.atan(self.width  / (2 * self.focal_px)))

@property
def vfov_deg(self) -> float:
    return 2 * math.degrees(math.atan(self.height / (2 * self.focal_px)))
```

`_pixel_to_camera_ray`는 `fx_px = fy_px = metadata.focal_px` 로 통일. `estimate_intrinsics_from_metadata`의 `f_pixels = (f35/36)*width`도 같은 버그이니 함께 교체하세요 (여긴 fx=fy라 이방성은 없지만 절대 스케일이 4% 큽니다).

### 2) 권장 — XMP `DewarpData` 사용 (가장 정확)

H20T XMP에는 공장 캘리브레이션 값이 들어있습니다. `_parse_xmp_attrs`가 이미 전부 읽고 있으니 파싱만 추가하면 됩니다.

```python
# drone-dji:DewarpData = "2025-12-17;fx,fy,cx,cy,k1,k2,p1,p2,k3"
raw = xmp.get("drone-dji:DewarpData", "")
if ";" in raw:
    vals = [float(v) for v in raw.split(";", 1)[1].split(",")]
    fx, fy, cx_off, cy_off, *dist = vals
    K = np.array([[fx, 0, width/2 + cx_off],
                  [0, fy, height/2 + cy_off],
                  [0,  0, 1.0]])
    D = np.array(dist)   # k1,k2,p1,p2,k3
```

주의: DewarpData의 cx,cy는 **이미지 중심으로부터의 오프셋**이지 절대 좌표가 아닙니다. 이 경로를 쓰면 왜곡 보정(현재 `D=zeros`)과 주점 이동까지 한 번에 해결됩니다. `dewarp_flag` / `pre_calib_param` 필드를 이미 만들어 두셨으니 그 자리에 넣으시면 됩니다.

### 3) 함께 고쳐야 할 것들

**(a) `georeferencer.py`의 LRF 판정 버그 — IR 경로가 조용히 깨집니다**

```python
elif metadata.lrf[3] is not None:      # _to_float 기본값이 0.0이라 절대 None이 안 됨
    self.ground_altitude = metadata.lrf[3]   # → LRF 없으면 지면고도 0 m
```

LRF 미탑재/미측정 사진에서 `ground_altitude = 0.0`, `ground_up_in_enu = −158.273 m`가 되어 스케일이 3.5배 튑니다. 이미 만들어 두신 프로퍼티를 쓰세요:

```python
elif metadata.has_valid_lrf:
    self.ground_altitude = metadata.lrf_target_abs_alt
```

`validate_with_lrf()`의 `if self.metadata.lrf[1] is None:` 도 같은 버그입니다. 그리고 두 번째 클래스 `DJIGeoreferencer`는 `metadata.relative_altitude_m`, `metadata.latitude` 등 `ImageMetadata`에 없는 필드를 참조합니다 — 죽은 코드라면 지우는 게 안전합니다.

**(b) 측지 상수** — `M_PER_DEG_LAT = 111_320.0`은 위도 34.71°에서 실제 **110,935 m/deg**입니다(+0.35% 남북 스케일 오차). 33 m 구간에서 12 cm. 자오선 곡률반경으로 바꾸면 공짜로 개선됩니다.

**(c) 절대 스케일 = AGL 불확실성** — 이방성을 고쳐도 남는 가장 큰 요인입니다. `relative_height`(44.885 m)는 **이륙 지점 기준**인데, 패널 상면은 지면보다 1~2.5 m 높습니다. 실제 AGL이 42~43 m라면 그 자체로 4~6% 스케일 오차 — **지금과 똑같은 증상**이 남습니다. RGB 경로는 `rel_alt`, IR 경로는 LRF를 쓰는 현재 구조를 LRF 우선으로 통일하고, 없으면 "패널면 높이" 파라미터나 DSM을 받도록 하세요.

---

## 수정 후 검증 순서

1. **anomaly 박스 종횡비**가 2.00 근처로 나오는지 (즉시 확인 가능한 회귀 테스트)
2. **`validate_with_lrf()`** — 중심 픽셀 계산치 vs LRF 실측 위치. 수평 오차가 GSD 몇 배 수준인지로 절대 스케일을 직접 검증
3. **중첩 프레임 교차검증** — 같은 패널이 인접 2장에서 얼마나 벌어지는지. 잔차 벡터가 프레임 중심에서 **방사형**이면 초점거리/고도 오차, **평행**이면 위치 오프셋, **회전형**이면 짐벌 yaw 오차입니다. 원인 분리에 가장 확실한 방법이고, f를 미지수로 두고 중첩 검출 재투영 잔차를 최소화하면 GCP 없이도 f를 0.5% 이내로 역산할 수 있습니다.
4. 그래도 남는 잔차는 이미 구축하신 **PostGIS 격자 적합 / `GeoAlignmentCorrection`** 계층에서 프레임별 similarity 보정으로 흡수 — 다만 그건 물리 모델을 고친 **다음** 단계여야 합니다. 지금은 12.5% 이방성을 affine 보정이 가려주고 있는 상태라 보정 파라미터가 프레임마다 불안정할 겁니다.

패널 모서리 2~4점을 GCP로 잡을 수 있다면 호모그래피 한 번으로 초점거리·yaw·평면고도 오차를 모두 흡수할 수 있지만, GCP-free 설계 방향에는 위 1~3번이 맞습니다.


```text
metadata.py & georeferencer.py & annotation_georeferencer.py 전체 파일을 제공해 줘
```

metadata.py 전체를 수정 반영해서 만들겠습니다. 핵심은 (1) 등방 초점거리(대각선 기준), (2) H20T XMP `DewarpData` 공장 캘리브레이션 파싱, (3) 측지 상수를 위도별 자오선 곡률반경으로 개선하는 것입니다. 나머지 구조·주석·필드는 그대로 유지했습니다.Import과 문법이 깨지지 않았는지, 그리고 수정 전/후 스케일 차이가 의도대로 나오는지 검증하겠습니다.전부 정상 동작합니다.수정 반영한 전체 파일입니다. 검증 결과 요약:

**바뀐 부분 (원본 대비)**

1. **등방 초점거리** — `SENSOR_35MM_DIAG` 추가, `hfov_deg`/`vfov_deg`가 새 `focal_px` 프로퍼티(대각선 기준)를 쓰도록 교체. 검증에서 `fy/fx = 1.0`, footprint 33.06 × 24.79 m로 정상 복원됐습니다. 원본은 fy/fx = 1.125였습니다.

2. **DewarpData 파싱** — `_parse_dewarp_data()` 추가, `extract_metadata()`가 `drone-dji:DewarpData`/`DewarpFlag`를 읽어 `dewarp_fx/fy/cx/cy/dist` 필드를 채웁니다. `estimate_intrinsics_from_metadata()`는 캘리브레이션이 있으면 실측 K·D를, 없으면 등방 fallback을 반환합니다. 주점 오프셋은 "중심 + offset"으로 올바르게 처리했고, 깨진 문자열엔 `{}`를 돌려주는 견고성도 확인했습니다.

3. **측지 상수** — `meters_per_deg_lat/lon()`을 곡률반경 기반으로 추가하고 `m_per_deg_lat` 프로퍼티를 교체(위도 34.71°에서 110,935 m/deg). 기존 `M_PER_DEG_LAT = 111_320.0` 상수는 하위 호환용으로 남겨뒀습니다.

**주의 — metadata.py만으로는 끝나지 않습니다**

`annotation_georeferencer.py`의 `_pixel_to_camera_ray()`는 여전히 `fx_px`/`fy_px`를 hfov/vfov에서 각각 계산합니다. 이제 hfov/vfov가 등방 f에서 나오므로 자동으로 fx=fy가 되지만, 왜곡 보정(`D`)은 그 경로에서 아예 안 쓰이므로 DewarpData의 이점을 살리려면 `_pixel_to_camera_ray`를 `metadata.focal_px` + `cv2.undistortPoints` 기반으로 바꾸는 게 좋습니다. 그리고 지난번 지적한 `georeferencer.py`의 LRF `is None` 버그(`_to_float` 기본값 0.0 탓에 절대 None이 안 됨)는 이 파일 범위 밖이라 그대로 남아 있으니 별도로 고쳐야 합니다.

원하시면 `annotation_georeferencer.py`와 `georeferencer.py`도 이 새 프로퍼티에 맞춰 수정한 버전을 만들어 드리겠습니다.


georeferencer.py를 수정하겠습니다. 반영할 것: LRF `is None` 버그, `cv2` 미임포트(왜곡보정 경로가 이제 실제로 실행됨), `image_width`/`image_height` 오타, 그리고 새 metadata 프로퍼티(`has_valid_lrf`, 등방 K) 연동입니다. 두 번째 클래스 `DJIGeoreferencer`는 존재하지 않는 필드를 참조하는 죽은 코드라 처리 방침을 정하고 반영합니다.문법 검증하겠습니다 (sayou 패키지 의존성은 스텁으로 대체).컴파일 통과, 로직 검증 완료.수정 반영한 전체 파일입니다. 바뀐 부분:

**1. LRF 지면고도 판정 버그 (가장 중요)**
`metadata.lrf[3] is not None`은 `_to_float` 기본값 0.0 때문에 항상 참이었습니다. `has_valid_lrf` / `lrf_target_abs_alt`로 교체했습니다. 검증에서 LRF 미측정 사진의 `up_in_enu`가 예전 −158.3 m(3.5배 스케일 폭주) → 정상 −44.885 m로 복원됩니다. 우선순위는 `명시값 > LRF 실측 > AGL 폴백`입니다.

**2. `cv2` 미임포트**
`pixel_to_camera_ray`의 왜곡보정 분기가 `cv2`를 import 없이 참조했습니다. 지금까지는 `D=zeros`라 이 경로가 안 타서 숨어 있었지만, 새 metadata.py가 DewarpData의 왜곡계수를 채우면 실제로 실행됩니다. 폐쇄망/최소 환경 로딩을 고려해 분기 안에서 지연 import 하도록 했습니다.

**3. `image_width`/`image_height` 오타**
`yolo_bbox_to_geodetic`이 `ImageMetadata`에 없는 `metadata.image_width`를 참조 → `width`/`height`로 수정.

**4. `validate_with_lrf` 동일 버그 + 견고화**
`lrf[1] is None` 판정을 `has_valid_lrf` 기반으로 바꾸고, 절대고도만 있고 위경도가 없는 경우(`no_lrf_position`)를 분리했습니다. 이제 이 함수가 절대 스케일 검증용으로 실제 작동합니다.

**5. 레거시 `DJIGeoreferencer` 클래스 제거**
`metadata.relative_altitude_m`, `metadata.latitude`, `metadata.absolute_altitude_m`처럼 `ImageMetadata`에 없는 필드를 참조해 인스턴스화 즉시 `AttributeError`가 나는 실행 불가 죽은 코드였습니다(현재 파이프라인은 `DJIImageGeoreferencer`만 사용). 파일 상단 주석에 제거 사유를 남겨뒀습니다. 만약 이 클래스를 다른 곳에서 실제로 쓰고 계시다면 `CameraIntrinsics` + 새 metadata API 기준으로 복원해 드리겠습니다 — 알려주세요.

**아직 남은 것 (이 파일 밖)**
`annotation_georeferencer.py`의 `_pixel_to_camera_ray`는 여전히 hfov/vfov에서 `fx_px`/`fy_px`를 뽑습니다. metadata.py를 고쳐서 이제 등방이 되긴 하지만, DewarpData의 왜곡보정(`D`)은 그 경로에서 안 쓰입니다. RGB 검출까지 캘리브레이션 이점을 살리려면 그 함수를 `metadata.focal_px` + `undistortPoints` 기반으로 통일하는 게 좋습니다. 그 파일 수정본도 원하시면 만들어 드리겠습니다.


annotation_georeferencer.py를 수정하겠습니다. 핵심은 `_pixel_to_camera_ray`를 등방 초점거리(`metadata.focal_px`) + DewarpData 왜곡보정 기반으로 바꾸고, `_enu_to_lonlat`을 곡률반경 기반 프로퍼티로 통일하는 것입니다. 나머지 구조·주석은 유지합니다.컴파일 통과, 회귀 검증 완료 — anomaly 박스 종횡비가 픽셀 종횡비(2.002)와 정확히 일치하도록 복원됩니다.수정 반영한 전체 파일입니다. 바뀐 부분:

**1. `_pixel_to_camera_ray` — 이방성 초점거리 제거 (핵심)**
`fx_px`(hfov 기반) ≠ `fy_px`(vfov 기반)로 나뉘던 것을 `metadata.focal_px`(등방) 하나로 통일했습니다. DewarpData가 있으면 실측 fx/fy와 왜곡계수(k1,k2,p1,p2,k3)를 `cv2.undistortPoints`로 보정하고, 없으면 등방 pinhole로 계산합니다. cv2는 지연 import입니다. 검증에서 anomaly 박스 종횡비가 기존 1.779 → 수정 2.002로 픽셀 종횡비와 일치합니다.

**2. `_enu_to_lonlat` — 곡률반경 기반 변환**
고정 상수 `M_PER_DEG_LAT`(111320) 대신 `metadata.m_per_deg_lat` / `m_per_deg_lon` 프로퍼티를 사용합니다. 위도 34.7°에서 남북 +0.35% 오차 제거.

**3. `_pixel_to_ground_enu` — LRF 우선**
지면고도를 `has_valid_lrf`면 LRF 실측 절대고도 기반 AGL로, 없으면 `-relative_height`로 폴백. RGB 경로도 이제 LRF 정밀도를 활용합니다 (기존엔 IR만 LRF를 썼음).

**4. `_rgb_detection_to_feature` — width_m/height_m 계산 개선**
기존 `max(es)-min(es)` 방식(축 정렬 bounding extent)을 ENU 코너 변 길이 평균(대각 회전에 강건)으로 바꿨습니다. 짐벌 yaw −88.7°처럼 이미지가 회전돼 있어도 실제 변 길이가 나옵니다. 디버그용 `print` 두 줄도 제거했습니다.

**세 파일 함께 적용 시 정합성**
`metadata.py`(focal_px·DewarpData·곡률반경) → `georeferencer.py`(IR 경로·LRF·cv2) → `annotation_georeferencer.py`(RGB 경로·등방·LRF)가 이제 동일한 카메라 모델을 공유합니다. 두 센서 경로가 같은 `focal_px`/주점 정의를 쓰므로, 이전에 지적한 RGB-IR affine 잔차의 상당 부분(이방성 왜곡률 차이 1.125 vs 1.200)이 사라질 것입니다.

**남은 확인 사항**
`metadata.R_cam_to_enu`는 `get_geojson`에서 세팅되지만, 외부에서 `annotation_to_geojson`을 직접 호출하는 경로가 있다면 `R_cam_to_enu`가 None일 수 있습니다. 그 진입점이 있으면 방어 코드를 넣는 게 안전합니다. 실제 이미지로 돌려보시고 footprint 크기(현재 대략 33×25 m로 나와야 정상)와 `validate_with_lrf()` 수평오차를 확인해 보시면 절대 스케일까지 검증됩니다.


#### 수정파일

`label/physai-studio/label_studio/sayou/image/metadata_2.py`<br/>
`label/physai-studio/label_studio/sayou/georeferencing/dji/georeferencer_2.py`<br/>
`label/physai-studio/label_studio/sayou/annotation_georeferencer_2.py`<br/>

```text
3개 파일을 적용했지만, GeoJson의 오차는 거의 변화가 없는데 원인과 해결책을 찾아줘
```

3개 파일을 적용했는데 오차가 "거의 그대로"라면, 가장 먼저 의심할 것은 **새 코드가 실제로 실행되지 않고 있다**는 쪽입니다. 이유는 수치로 명확합니다.

## 먼저: 수정이 반영됐는지 판별하는 결정적 지표

수정 전후로 **반드시 바뀌어야 하는 값**들이 있습니다. 새로 생성한 GeoJSON에서 이것부터 확인하세요.

| 필드 | 수정 전 (기존 출력) | 수정 후 (반드시 이 값) |
|---|---|---|
| `metadata.hfov_deg` | **41.9116** | **40.4301** |
| `metadata.vfov_deg` | **28.6454** | **30.8771** |
| footprint 크기 | 34.38 × 22.92 m | 33.06 × 24.79 m |
| det#0 `width_m` / `height_m` | 1.949 / 1.126 | 약 **2.08 / 1.04** |

**`hfov_deg`가 여전히 41.9116이면 새 코드가 100% 안 돌고 있는 것입니다.** 이 값은 `metadata.focal_px`에서 직접 나오므로 우회 경로가 없습니다.

흔한 원인 순서대로:

1. **GeoJSON이 캐시/DB에 저장되어 있음** — 스크린샷의 모달이 이전에 계산해 저장해 둔 결과를 읽고 있을 가능성. Task/Annotation 필드나 PostGIS 테이블에 이미 들어간 값이면 코드를 고쳐도 화면은 안 바뀝니다. 재생성 트리거가 필요합니다.
2. **Celery 워커 미재시작** — Django dev 서버는 auto-reload되지만 워커는 안 됩니다. georeferencing이 비동기 태스크면 워커를 죽였다 켜야 합니다.
3. **`metadata.py`를 잘못된 위치에 배치** — 프로젝트에 metadata.py가 **두 개**입니다. `sayou/image/metadata.py`(`ImageMetadata`, 제가 수정한 것)와 `sayou/georeferencing/dji/metadata.py`(`CameraIntrinsics`, `DJIMetadata`, `parse_dji_metadata`). 후자에 덮어썼다면 ImportError가 났을 테니 아마 아니겠지만 확인해 두세요.
4. `__pycache__` 잔존.

## 진단 스크립트

문제 이미지 한 장으로 바로 돌려보세요. 어디서 끊기는지 한 번에 나옵니다.

```python
from sayou.image.metadata import extract_metadata
import math, inspect, sayou.image.metadata as MM

p = "/Users/seongjungkim/Development/label-studio/media/upload/3/d1c50983-DJI_20251217131012_0182_Z.JPG"
m = extract_metadata(p)

print("모듈 실제 경로 :", MM.__file__)              # ← 어느 파일이 로드됐나
print("focal_px 존재  :", hasattr(m, "focal_px"))   # ← False면 구 버전
print(f"focal_px      = {m.focal_px:.1f}  (기대 7039.1)")
print(f"hfov/vfov     = {m.hfov_deg:.4f} / {m.vfov_deg:.4f}  (기대 40.4301 / 30.8771)")
print("has_dewarp    =", m.has_dewarp, "| fx,fy =", m.dewarp_fx, m.dewarp_fy)
print("dewarp_dist   =", m.dewarp_dist)
print(f"주점          = {m.principal_point_px}")
print("lrf           =", m.lrf, "| has_valid_lrf =", m.has_valid_lrf)
print(f"rel_height    = {m.relative_height}  abs_alt = {m.gps.altitude}")
print(f"GSD           = {m.relative_height/m.focal_px*1000:.3f} mm/px")
print(f"footprint     = {m.width*m.relative_height/m.focal_px:.2f} x {m.height*m.relative_height/m.focal_px:.2f} m")
```

**여기서 특히 주의할 함정**: `has_dewarp = True`이고 `dewarp_fx`가 7039와 5% 이상 차이 나면, DJI가 DewarpData를 **네이티브가 아닌 다른 해상도 기준**으로 기록한 경우입니다(Mavic 계열에서 흔함). 이때는 `fx × (5184 / 캘리브레이션_폭)`으로 스케일해야 하며, 그냥 쓰면 제 수정이 오히려 새 오차를 만듭니다. 값이 이상하면 일단 `has_dewarp` 경로를 끄고 fallback(7039)으로 검증하세요.

## 새 코드가 돌고 있는데도 남는 오차라면 — 투영 평면 문제

검출 데이터로 역산해 보면 남은 오차의 정체가 보입니다.

anomaly 박스 41개 평균이 **160.4 × 312.7 px**(종횡비 1.950)입니다. 수정된 등방 GSD 6.376 mm/px를 적용하면 **1.022 × 1.994 m**. 이건 992×1956 mm 72셀 모듈(종횡비 1.972)과 거의 일치합니다 — 즉 **등방 수정 후 절대 스케일은 이미 2~3% 이내로 들어왔습니다.**

역산되는 AGL은 43.5~44.0 m인데 코드는 44.885 m를 씁니다. 차이 약 1.1 m = **패널면이 이륙지점 지면보다 그만큼 높다**는 뜻입니다. `relative_height`는 이륙지점 기준이고, 검출 대상은 지면이 아니라 **패널 상면**이니 당연한 결과입니다.

투영 평면을 패널면으로 올렸을 때:

| 패널면 높이 | AGL | GSD | PV 행 피치 | footprint |
|---|---|---|---|---|
| +0.0 m (현재) | 44.89 | 6.376 | 4.197 m | 33.06 × 24.79 |
| +1.0 m | 43.88 | 6.234 | 4.104 m | 32.32 × 24.24 |
| +1.5 m | 43.38 | 6.163 | 4.057 m | 31.95 × 23.96 |
| +2.0 m | 42.88 | 6.092 | 4.010 m | 31.58 × 23.69 |

그리고 더 중요한 게 **기복변위(relief displacement)** 입니다. 높이 h인 구조물을 지면 평면으로 투영하면 nadir에서 반경 r만큼 떨어진 지점이 `d = h·r/H` 만큼 바깥으로 밀립니다:

- 패널높이 1.5 m 기준 → r=8 m에서 27 cm, r=16.5 m(프레임 끝)에서 **55 cm**
- 패널높이 2.0 m 기준 → 프레임 끝에서 **74 cm**

**중심에서 0이고 바깥으로 갈수록 선형 증가** — 사용자가 상단을 기준으로 보면 정확히 "아래로 갈수록 간격이 벌어지는" 모습이 됩니다. 이건 초점거리 수정으로는 해결되지 않고, 투영 평면 자체를 올려야 사라집니다.

## 조치

**1) `annotation_to_geojson` / `_pixel_to_ground_enu`에 패널면 높이 파라미터 노출**

이미 `ground_z_below_drone` 인자가 있으니 호출부에서 넘기기만 하면 됩니다:

```python
panel_plane_height_m = 1.5   # 사이트별 설정값 (구조물 도면 or 실측)
ground_z = -(metadata.relative_height - panel_plane_height_m)
```

프로젝트 설정이나 Task 메타로 빼서 사이트별로 관리하는 게 맞습니다.

**2) LRF를 절대 스케일 기준으로 사용**

`m.lrf`가 0이 아니면 `lrf_target_distance`가 곧 중심 픽셀까지의 실측 슬랜트 거리입니다. nadir이므로 이 값이 사실상 패널면까지의 AGL입니다. `relative_height`(44.885)와 비교해서 차이를 보세요 — 그게 바로 패널면 높이 + 이륙지점 표고차입니다. `validate_with_lrf()`가 이제 제대로 동작하니 수평 오차도 직접 나옵니다.

**3) 모듈 규격으로 최종 캘리브레이션**

현장 모듈 실제 치수를 아신다면 한 줄로 AGL이 확정됩니다:

```
AGL_true = 모듈_실제폭_m × focal_px / 모듈_평균_픽셀폭
         = 모듈_실제폭_m × 7039.1 / 160.4
```

992 mm면 43.5 m, 1046 mm면 45.9 m, 1134 mm면 49.8 m입니다. 이 한 값만 확정하면 남은 오차의 대부분이 정리됩니다.

---

우선 **1번(hfov 41.91 vs 40.43)만 확인해서 알려주시면**, 캐시 문제인지 물리 모델 문제인지 갈리므로 다음 단계를 정확히 좁힐 수 있습니다.


```text
듈 실제 경로 : /Users/seongjungkim/Development/sayouzone/label/label-studio/label_studio/sayou/image/metadata.py
focal_px 존재  : True
focal_px      = 7039.1  (기대 7039.1)
hfov/vfov     = 40.4301 / 30.8771  (기대 40.4301 / 30.8771)
has_dewarp    = False | fx,fy = None None
dewarp_dist   = []
주점          = (2592.0, 1944.0)
lrf           = [45.375, 34.7109337, 126.9224548, 113.0] | has_valid_lrf = True
rel_height    = 45.016  abs_alt = 158.404
GSD           = 6.395 mm/px
footprint     = 33.15 x 24.86 m
```

## 코드는 정상 동작합니다 — 그런데 "안 바뀐 것처럼 보이는" 이유가 수치로 설명됩니다

`focal_px = 7039.1`, `hfov = 40.4301` — 등방 수정이 제대로 적용됐고, `has_dewarp = False`라 DewarpData 해상도 함정도 없습니다. 그런데 **`has_valid_lrf = True`** 라는 새 정보가 결정적입니다.

제 수정으로 지면 평면이 `rel_height`(45.016) 기준에서 **LRF 기준(158.404 − 113.0 = 45.404 m)** 으로 바뀌었습니다. 즉 초점거리를 줄이는 방향으로 고쳤는데, LRF가 동시에 고도를 0.39 m 올려서 **서로 상쇄**됐습니다.

| 방향 | 구 코드 | 신 코드 | 변화 |
|---|---|---|---|
| x축 = **남북 = 행 피치 방향** | 6.651 mm/px | 6.450 | **−3.0%** |
| y축 = 동서 = 행 길이 방향 | 5.912 mm/px | 6.450 | +9.1% |

**+9.1% 개선은 행 길이 방향(동서)에 몰렸는데, 그쪽은 행이 길고 균일해서 눈에 안 보입니다.** 정작 사용자가 정렬을 판단하는 행 간격(남북) 방향은 3%밖에 안 줄었습니다. 33 m 프레임에서 1 m 남짓 — "거의 변화 없다"는 체감이 정확합니다.

*(참고: 진단을 돌린 사진은 GeoJSON을 만든 `_0182`와 다른 프레임입니다. rel_height 45.016 vs 44.885. 비교엔 영향 없는 수준입니다.)*

## 남은 오차의 정체: 투영 평면이 패널면이 아니라 행간 지면

`lrf_distance = 45.375`, LRF 기준 수직 강하 45.404 — 3 cm 차이니 **빔은 거의 정확히 수직**입니다. 그리고 LRF 조준점 고도 113.0은 `abs_alt − rel_height` = 113.388보다 0.39 m **낮습니다**. 패널 상면이라면 지면보다 높아야 하는데 오히려 낮으니, **LRF 빔이 패널 사이 행간 지면에 떨어진 것**입니다.

검출 데이터로 역산하면 이게 맞습니다. anomaly 박스 41개 평균 160.4 × 312.7 px, **종횡비 1.950**. 현재 GSD로는 1.035 × 2.017 m가 나옵니다.

| 실제 모듈 규격 | 필요 AGL | 지면고도 | LRF(113.0) 대비 패널면 | 현재 대비 |
|---|---|---|---|---|
| **992 × 1956** (72셀, 종횡비 1.972) | 43.78 m | 114.62 | **+1.62 m** | −3.6% |
| 1046 × 2062 (1.971) | 46.16 m | 112.24 | −0.76 m | +1.7% |
| 1096 × 2172 (1.982) | 48.50 m | 109.91 | −3.09 m | +6.8% |
| 1134 × 2278 (2.009) | 50.52 m | 107.88 | −5.12 m | +11.3% |

측정된 종횡비 1.950은 **992×1956 (1.972)** 과 가장 가깝고, 이 경우에만 패널면 높이가 물리적으로 말이 되는 양수(+1.62 m)로 나옵니다. 나머지는 패널이 지면보다 낮다는 뜻이 되어 불가능합니다.

패널면을 +1.9 m로 올리면:

```
AGL 43.50 → GSD 6.180 mm/px → 행 피치 4.068 m, 모듈 0.991 × 1.933 m
```

모듈 실측 992 × 1956과 거의 일치합니다.

## 추가로 중요한 문제: LRF 기반 지면고도는 프레임마다 흔들립니다

nadir 촬영에서 LRF 조준점은 프레임에 따라 **행간 지면에 맞기도 하고 패널 상면에 맞기도** 합니다.

- 지면 명중 → AGL 45.40 → GSD 6.450
- 패널 명중 → AGL 43.50 → GSD 6.180

**프레임 간 스케일 편차 4.2%.** 인접 사진끼리 같은 패널이 어긋나는 원인이 되고, 모자이크·격자 적합 단계에서 보정 파라미터가 프레임마다 튀게 만듭니다. 제가 지난번에 LRF 우선을 권한 건 이 사이트 조건에선 오히려 불안정 요인이었습니다 — 정정합니다.

## 조치

**LRF를 프레임별 지면고도로 직접 쓰지 말고, 사이트 단위의 고정 패널면 평면을 쓰세요.**

```python
# 사이트 설정값 (프로젝트/Task 메타로 관리)
SITE_PANEL_PLANE_ALT = 114.6      # 절대고도(m). 아래 방법으로 1회 캘리브레이션

def _pixel_to_ground_enu(self, px, py, metadata, ground_z_below_drone=None):
    if ground_z_below_drone is None:
        if SITE_PANEL_PLANE_ALT is not None:
            ground_z_below_drone = SITE_PANEL_PLANE_ALT - metadata.gps.altitude
        elif metadata.has_valid_lrf:
            ground_z_below_drone = metadata.lrf_target_abs_alt - metadata.gps.altitude
        else:
            ground_z_below_drone = -metadata.relative_height
    ...
```

`georeferencer.py`의 `DJIImageGeoreferencer`도 `ground_altitude` 인자를 이미 받으니 동일 값을 넘기면 RGB/IR 경로가 같은 평면을 공유합니다. 이게 두 센서 정합 잔차를 줄이는 데도 직접 기여합니다.

**캘리브레이션 (1회):**

```
AGL_true   = 모듈_실제폭_m × 7039.1 / 모듈_평균픽셀폭
평면_절대고도 = metadata.gps.altitude - AGL_true
```

모듈 폭만 확정되면 끝납니다. 992 mm면 114.62, 1046 mm면 112.24입니다.

**부수 효과 — 기복변위 제거:** 평면을 패널 상면에 두면 패널 높이로 인한 방사형 변위(높이 1.9 m 기준 프레임 끝에서 약 70 cm)도 함께 사라집니다. 이게 중심에서 0이고 바깥으로 선형 증가하는 성분이라, 지금 "아래로 갈수록 벌어지는" 잔여분의 상당 부분입니다.

---

**현장 모듈 실제 규격(폭 × 길이)만 알려주시면** 평면 고도를 확정해 드리겠습니다. 모르신다면 위성영상에서 패널 행 피치를 재보세요 — 현재 산출값 4.246 m와 비교하면 같은 배율을 역산할 수 있습니다(실측 4.07 m면 992 mm 모듈 가설이 확정됩니다).


```text
georeferencer.py 전체 소스를 제공해 줘
```

패널면 평면을 사이트 단위로 고정하고, 지면 평면의 출처를 추적할 수 있도록 반영하겠습니다.컴파일 통과, 우선순위·캘리브레이션 로직 검증 완료.수정 반영한 전체 파일입니다. 1차 수정(LRF `is None` 버그, cv2, 오타, 레거시 클래스 제거)은 그대로 유지하고 투영 평면 정책을 바꿨습니다.

**1. `SITE_PANEL_PLANE_ALT` — 사이트 단위 고정 패널면**

모듈 상단 상수로 노출했고, 평면 선택 우선순위는 `ground_altitude`(명시) → `panel_plane_altitude`/`SITE_PANEL_PLANE_ALT` → `prefer_lrf=True`일 때만 LRF → `relative_height` 폴백입니다.

**`prefer_lrf` 기본값을 `False`로 뒀습니다.** LRF는 프레임마다 행간 지면(113.0)과 패널 상면(약 114.9)을 번갈아 맞아 GSD가 6.450 ↔ 6.185로 4% 흔들립니다. 인접 사진 간 정합이 깨지는 원인이라, 단일 프레임 절대정확도를 볼 때만 켜는 옵션으로 남겼습니다.

**2. `ground_plane_source` — 평면 출처 추적**

어떤 경로로 평면이 결정됐는지 문자열로 남습니다. `GeoreferencingResult`와 `validate_with_lrf()` 반환값에 포함되니, **RGB 경로와 IR 경로가 같은 평면을 쓰고 있는지** 바로 확인할 수 있습니다. 두 센서 정합 잔차를 볼 때 필수입니다.

**3. `calibrate_plane_altitude()` — 1회 캘리브레이션 헬퍼**

```python
m = extract_metadata(path)
calibrate_plane_altitude(m, known_size_m=0.992, measured_size_px=160.4)
# → {'agl_m': 43.53, 'plane_altitude_m': 114.87, 'gsd_m_per_px': 0.006185,
#    'panel_height_over_lrf_m': 1.87}
```

`plane_altitude_m` 값을 `SITE_PANEL_PLANE_ALT`에 넣으면 끝입니다. 992 mm 가설에서 패널면이 LRF 지면보다 +1.87 m — 물리적으로 타당한 값입니다. 1046 mm(−0.50 m), 1134 mm(−4.36 m)는 패널이 지면보다 낮다는 뜻이라 기각됩니다.

**4. 진단 보조**

- `agl_m`, `nominal_gsd_m` 프로퍼티
- `relief_displacement_m(h, r)` — 평면을 지면에 둘 때 남는 방사형 오차. 패널높이 1.62 m 기준 프레임 끝에서 59~74 cm. 평면을 패널면에 맞추면 0이 됩니다.
- `validate_with_lrf()`에 `error_in_pixels`, `vertical_gap_m` 추가. `vertical_gap_m`이 +1~2 m면 LRF가 행간 지면을, 0 근처면 패널 상면을 맞은 것입니다.
- 평면이 드론보다 높으면 `__init__`에서 `ValueError`로 즉시 차단.

**적용 순서**

1. 모듈 실제 폭을 확인하고 `calibrate_plane_altitude()` 실행
2. 나온 `plane_altitude_m`을 `SITE_PANEL_PLANE_ALT`에 설정
3. `annotation_georeferencer.py`의 `_pixel_to_ground_enu`도 같은 값을 쓰도록 연결 — 안 하면 RGB 경로만 예전 평면을 계속 씁니다. 이 부분 수정본도 필요하시면 말씀해 주세요.

#### 수정파일

`label/physai-studio/label_studio/sayou/georeferencing/dji/georeferencer_3.py`

```text
적용 순서에 대해 구체적으로 표시해 주거나 아니면 해당 파일의 전체 소스를 제공해 줘
```

`annotation_georeferencer.py`가 마지막 연결 고리입니다. 평면 상수를 두 곳에 중복 정의하면 반드시 어긋나므로, `georeferencer.py`의 값을 **런타임에 참조**하도록 연결하고 RGB/IR 두 경로가 동일한 평면을 쓰도록 만들겠습니다.전체 소스입니다. 이어서 적용 순서를 구체적으로 정리합니다.

## 이 파일에서 바뀐 핵심

**평면 상수를 중복 정의하지 않았습니다.** `from ... import SITE_PANEL_PLANE_ALT`로 값을 가져오면 import 시점 스냅샷이 박혀 런타임 변경이 반영되지 않으므로, 모듈을 import해서 `_geo.SITE_PANEL_PLANE_ALT`로 참조합니다.

**RGB/IR이 같은 평면을 씁니다.** `_resolve_plane_altitude()` 하나가 평면을 정하고, IR 경로는 `DJIImageGeoreferencer(metadata, ground_altitude=plane_alt)`로 그 값을 명시 주입합니다. 기존에는 RGB가 `-relative_height`, IR이 내부 로직을 각각 써서 두 센서가 구조적으로 어긋났습니다.

`R_cam_to_enu`가 없을 때 `annotation_to_geojson`이 직접 채우도록 방어 코드도 넣었습니다(지난번 지적 사항).

---

## 적용 순서

**1단계 — 파일 배치**

| 파일 | 배치 경로 |
|---|---|
| `metadata.py` | `label_studio/sayou/image/metadata.py` |
| `georeferencer.py` | `label_studio/sayou/georeferencing/dji/georeferencer.py` |
| `annotation_georeferencer.py` | 현재 쓰시던 경로 그대로 |

`sayou/georeferencing/dji/metadata.py`(다른 파일)는 건드리지 마세요.

**2단계 — 캘리브레이션값 산출**

```python
from sayou.image.metadata import extract_metadata
from sayou.georeferencing.dji.georeferencer import calibrate_plane_altitude

m = extract_metadata("...DJI_20251217131012_0182_Z.JPG")
print(calibrate_plane_altitude(m, known_size_m=0.992, measured_size_px=160.4))
```

`known_size_m`에 **모듈 실제 폭**, `measured_size_px`에 anomaly 박스 평균 픽셀 폭을 넣습니다. 후자는 새로 추가한 `measure_mean_box_px(detections, 5184, 3888, class_id=1)`로 뽑을 수 있습니다.

**3단계 — 값 설정**

2단계의 `plane_altitude_m`을 `georeferencer.py` 상단에 넣습니다.

```python
SITE_PANEL_PLANE_ALT: Optional[float] = 114.87
```

사이트가 여러 개면 상수 대신 생성자로 넘기세요 — 그쪽이 우선합니다.

```python
AnnotationGeoreferencer(image_path, annotation, panel_plane_altitude=114.87)
```

**4단계 — 캐시 무효화**

`__pycache__` 삭제, **Celery 워커 재시작**(dev 서버와 달리 auto-reload 안 됨), 저장된 GeoJSON 재생성.

**5단계 — 검증**

새 GeoJSON의 `metadata.ground_plane`을 확인합니다.

```json
"ground_plane": {
  "altitude_m": 114.87, "agl_m": 43.534, "gsd_m_per_px": 0.00619,
  "source": "site_panel_plane",
  "lrf_abs_alt_m": 113.0, "vertical_gap_to_lrf_m": 1.87
}
```

`source`가 `"rel_height"`면 3단계가 반영 안 된 것입니다. RGB 이미지와 IR 이미지의 `altitude_m`이 같은 값인지도 확인하세요 — 두 센서 정합의 전제입니다.

검출 크기 변화:

| 평면 | 모듈 크기 | 행 피치 |
|---|---|---|
| 폴백(rel_height) | 1.026 × 2.000 m | 4.209 m |
| LRF | 1.035 × 2.017 m | 4.246 m |
| **패널면 114.87** | **0.992 × 1.934 m** | **4.071 m** |

그리고 기복변위가 사라집니다 — 패널높이 1.87 m 기준 프레임 끝에서 68~85 cm였던 방사형 오차가 0이 됩니다. **이게 "하단으로 갈수록 벌어지던" 성분의 실체**이므로, 위 표의 스케일 변화(약 4%)보다 체감 개선이 클 것입니다.

---

한 가지 짚어둘 점은, 위 수치가 **모듈 폭 992 mm 가설**에 기대고 있다는 것입니다. 종횡비(측정 1.950 vs 992×1956의 1.972)와 패널면 높이가 양수(+1.87 m)로 나온다는 두 근거는 일관되지만, 실제 규격이 다르면 `SITE_PANEL_PLANE_ALT`만 다시 계산하면 됩니다. 현장 모듈 스펙을 확인하실 수 있으면 확정하고 가는 게 좋습니다.


```text
'ground_plane': {'altitude_m': 114.87, 'agl_m': 43.403, 'gsd_m_per_px': 0.00617, 'source': 'site_panel_plane', 'lrf_abs_alt_m': 113.0, 'vertical_gap_to_lrf_m': 1.87}
```

정상 반영됐습니다. 값이 모두 자기일관적입니다 — 158.273 − 114.87 = 43.403, 43.403 ÷ 7039.1 = 0.00617. `source: site_panel_plane`이니 RGB 경로가 사이트 평면을 타고 있습니다.

`vertical_gap_to_lrf_m: 1.87`이 물리적 근거를 확인해 줍니다. LRF가 맞은 행간 지면(113.0)보다 패널 상면이 1.87 m 높다는 뜻이고, 지상 가대형 PV 배열로 타당한 값입니다.

## 원본 GeoJSON 대비 변화

| | 원본 | 현재 | 변화 |
|---|---|---|---|
| x축 (남북 = **행 피치 방향**) | 6.632 mm/px | 6.166 | **−7.0%** |
| y축 (동서 = 행 길이 방향) | 5.895 mm/px | 6.166 | +4.6% |
| footprint | 34.38 × 22.92 m | 31.96 × 23.97 m | 남북 2.42 m 축소 |

지난번 등방 수정만 했을 때 행 피치 방향이 −3.0%밖에 안 줄어 체감이 없었는데, 이제 **−7.0%** 입니다. 프레임 끝단 기준 1.21 m 이동입니다.

여기에 기복변위 제거가 더해집니다. 평면이 지면에 있을 때 패널 높이 1.87 m가 만들던 방사형 변위가 사라집니다 — r=8 m에서 33 cm, r=16.5 m에서 68 cm, 프레임 모서리에서 85 cm. **중심에서 0이고 바깥으로 선형 증가하던 성분**이라, 스케일 변화보다 이쪽 체감이 클 것입니다.

## 예상 검출값

- 모듈: **0.989 × 1.928 m** (가정한 규격 0.992 × 1.956 대비 −0.3% / −1.4%)
- 행 피치: **4.058 m**

세로변이 1.4% 작게 나오는 건 라벨 박스가 모듈 경계보다 살짝 안쪽으로 그어졌을 때 자연스럽게 생기는 차이입니다. 문제 수준은 아니지만, 실제 규격이 992×1956이 아니라면 `SITE_PANEL_PLANE_ALT`만 다시 계산하면 됩니다.

## 한 가지 주의 — `width_m` / `height_m` 의미가 바뀌었습니다

det#0(163.1 × 326.5 px) 기준:

| | 원본 | 현재 |
|---|---|---|
| `width_m` | 1.949 (E-W extent) | **1.006** (박스 x변) |
| `height_m` | 1.126 (N-S extent) | **2.013** (박스 y변) |

원본은 축정렬 bounding extent였고, 지금은 **박스 자체의 변 길이**(회전 불변)입니다. 패널 치수로는 후자가 맞지만, CSV export나 작업지시서 등 다운스트림에서 "width = 동서 폭"으로 가정한 코드가 있으면 조용히 어긋납니다. 한 번 확인해 보세요.

## 다음 검증

1. **위성 정합 육안 확인** — 하단 드리프트가 실제로 줄었는지
2. **IR 이미지 한 장** 돌려서 `ground_plane.altitude_m`이 동일하게 114.87인지 (두 센서 정합의 전제)
3. **인접 프레임 2장의 중첩 영역** — 같은 패널이 얼마나 벌어지는지. 잔차 벡터가 방사형이면 아직 평면/초점거리, 평행이면 위치, 회전형이면 짐벌 yaw 문제입니다.

3번까지 통과하면 물리 모델은 정리된 것이고, 남는 잔차는 PostGIS 격자 적합 계층에서 흡수하면 됩니다. 이제는 그 보정 파라미터가 프레임마다 안정적으로 나올 조건이 갖춰졌습니다.


```text
[2026-07-24 03:27:00,909] [core.utils.common::custom_exception_handler::97] [ERROR] 423a6344-ee13-46d5-a2df-77943ee8164f 투영 평면이 드론보다 높습니다 (plane=114.87, drone=73.892, source=explicit_ground_altitude). 평면 고도 설정을 확인하세요.
Traceback (most recent call last):
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/rest_framework/views.py", line 506, in dispatch
    response = handler(request, *args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/sayouzone/.venv/lib/python3.11/site-packages/django/utils/decorators.py", line 48, in _wrapper
    return bound_method(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/sayouzone/label/physai-studio/label_studio/tasks/api.py", line 750, in get
    metadata, geojson = georeferencer.get_geojson()
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/sayouzone/label/physai-studio/label_studio/sayou/annotation_georeferencer.py", line 620, in get_geojson
    geojson = self.annotation_to_geojson(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/sayouzone/label/physai-studio/label_studio/sayou/annotation_georeferencer.py", line 672, in annotation_to_geojson
    DJIImageGeoreferencer(metadata, ground_altitude=plane_alt)
  File "/Users/seongjungkim/Development/sayouzone/label/physai-studio/label_studio/sayou/georeferencing/dji/georeferencer.py", line 248, in __init__
    raise ValueError(
ValueError: 투영 평면이 드론보다 높습니다 (plane=114.87, drone=73.892, source=explicit_ground_altitude). 평면 고도 설정을 확인하세요.
```

`drone=73.892` — 이 이미지는 절대고도가 완전히 다릅니다. 원래 사이트는 158.273이었으니 **다른 사이트(또는 다른 표고의 비행)** 입니다.

원인은 제 설계 실수입니다. `SITE_PANEL_PLANE_ALT`를 **절대고도**로 잡았는데, 절대고도는 지형 표고에 종속되므로 사이트가 바뀌면 무조건 깨집니다. 전역 상수로 둘 값이 아니었습니다.

**이식 가능한 값은 절대고도가 아니라 "지면 대비 패널 상면 높이"입니다.** 그 사이트에서 역산하면:

```
114.87 − (158.273 − 44.885) = 1.482 m
```

다른 프레임(158.404 / 45.016)으로 검산해도 지면 추정이 113.388로 동일하게 나옵니다. 이 **1.48 m는 구조물 기하**라 표고와 무관하게 어느 사이트든 적용됩니다.

두 파일을 수정하겠습니다. 이번엔 평면 결정 로직을 `georeferencer.py`의 함수 하나로 모아서 두 파일이 갈라질 여지를 없앱니다 — 방금 겪은 게 정확히 그 실패 패턴입니다.두 파일입니다. 핵심 변경은 세 가지입니다.

**1. 기본 정책을 절대고도 → 지면 대비 패널 높이로 전환**

```python
# 구조물 기하 — 지형 표고와 무관, 사이트 이식 가능
SITE_PANEL_HEIGHT_ABOVE_GROUND: Optional[float] = 1.48

# 실측 표고가 있을 때만. 사이트별 주입 권장
SITE_PANEL_PLANE_ALT: Optional[float] = None

SITE_PLANE_SANITY_RANGE_M: float = 30.0
```

평면 = `(abs_alt − relative_height) + 1.48`로 매 이미지마다 계산합니다.

| | plane | AGL | GSD | source |
|---|---|---|---|---|
| 사이트 A (abs 158.273) | 114.868 | 43.405 | 0.00617 | panel_height |
| 사이트 B (abs 73.892, rel 45) | 30.372 | 43.520 | 0.00618 | panel_height |

사이트 A는 이전 결과(114.87 / 43.403 / 0.00617)와 동일하고, 에러가 나던 B도 정상 처리됩니다.

**2. 절대고도 지정 시 sanity check — 크래시 대신 폴백**

`|지정값 − (abs_alt − rel_height)|`가 30 m를 넘으면 다른 사이트로 판단해 **경고 로그 후 무시**합니다. 사이트 B에서 114.87을 넣으면 차이 86.0 m로 기각되고 패널높이 경로로 넘어갑니다. 같은 사이트에서는 차이 1.48 m라 그대로 채택됩니다. 다중 사이트 플랫폼에서 설정 하나가 엔드포인트 전체를 죽이면 안 되니까요.

**3. 평면 결정 로직을 `resolve_ground_plane()` 하나로 통합**

`annotation_georeferencer._resolve_plane_altitude()`는 이제 이 함수를 호출만 합니다. 두 파일에 같은 정책을 복제해 둔 게 이번 문제를 두 곳에서 고쳐야 하게 만든 원인이라, 구조적으로 갈라질 수 없게 했습니다.

---

**제 판단 착오였습니다.** 절대고도를 전역 상수로 제안한 건 단일 사이트만 가정한 설계였고, 다중 사이트 플랫폼에는 맞지 않았습니다.

**적용:** 파일 교체 → `__pycache__` 삭제 → 워커 재시작. 사이트 A는 결과가 동일하니 재검증 불필요하고, 다른 사이트 이미지가 정상 처리되는지만 확인하시면 됩니다.

**확인 사항:** GeoJSON `metadata.ground_plane`에 `panel_height_above_ground_m`과 `takeoff_ground_alt_m`을 추가했습니다. 새 사이트에서 이 값이 1.48로 나오는데 위성 정합이 안 맞으면, 그 사이트는 가대 높이가 다르거나 **이륙 지점 표고가 패널 부지와 다른** 경우입니다. 후자라면 `panel_height_above_ground`를 사이트별로 주입하세요 — `relative_height`가 이륙지점 기준이라 그 차이가 섞여 들어갑니다.

### 수정파일

`label/physai-studio/label_studio/sayou/georeferencing/dji/georeferencer.py`<br/>
`label/physai-studio/label_studio/sayou/annotation_georeferencer.py`
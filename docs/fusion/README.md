# rgb_ir_fusion

RGB · 열화상(IR) 이미지를 **동일 좌표 패널 단위로 정합**한 뒤,
**Late Fusion(결정 수준 융합)** 으로 태양광 모듈 결함을 판정하는 파이프라인.

```
RGB ─┐                         ┌─ YOLOv11(RGB 학습)  ─┐
     ├─ register_pair ─ ROI ───┤                      ├─ LateFusionEngine ─ 리포트
IR  ─┘                         └─ YOLOv11(IR 학습)   ─┘
                                  + 물리 특징 → 신뢰도
```

분기는 **YOLOv11 모델 / 규칙 기반 / 둘의 하이브리드** 중에서 고를 수 있고,
셋 다 같은 `analyze(image, roi, array_ref) -> BranchEvidence` 인터페이스를 쓴다.
융합 계층은 어느 쪽을 쓰든 바뀌지 않는다.

핵심은 두 분기가 **서로를 보지 않고** 각자 결론을 낸 뒤, 결정 단계에서만
신뢰도를 가중해 결합한다는 점이다. 그래서

- IR 프레임이 없거나 정합이 깨져도 RGB 단독 판정으로 **열화(degrade)만 되고 멈추지 않는다**
- 각 분기를 독립적으로 교체·재학습할 수 있다 (규칙 → CNN)
- 정반사 포화, 저일사량 같은 **분기별 신뢰 불능 조건을 가중치로 명시**할 수 있다

---

## 설치

```bash
pip install numpy opencv-python scipy
pip install ultralytics          # YOLOv11 분기를 쓸 때만
```

## 빠른 확인

```bash
python examples/demo_synthetic.py   # 규칙 기반 경로
python examples/demo_yolo.py        # YOLOv11 통합 경로 (가중치 없이 동작)
```

`demo_yolo.py` 는 `YoloRuntime` 을 주입 가능하게 만들어 둔 덕에 **.pt 파일 없이도
통합 로직 전체**를 검증한다 — 타일 추론과 좌표 복원, 타일 경계 NMS, 검출→패널
귀속, conf 보정, 집합값 초점원소, IEC ΔT 게이트, 하이브리드 경로까지.

## CLI

```bash
python -m rgb_ir_fusion.pipeline \
    --rgb DJI_0001_W.JPG --ir DJI_0001_T.tif \
    --rgb-hfov 84 --ir-hfov 61 \
    --plane-distance 42 --baseline 0.035 \
    --out ./out
```

`--ir` 는 radiometric 16-bit TIFF 를 권장한다. R-JPEG 는 DJI TSDK 로
온도 배열을 먼저 추출해야 한다. 비-radiometric 8-bit 만 있으면
`--no-radiometric` 을 쓰되, ΔT 가 상대 단위가 되므로 IEC 임계값은 무의미해지고
신뢰도가 자동으로 하향된다.

## 라이브러리 사용

```python
import cv2, numpy as np
from rgb_ir_fusion import (RGBIRLateFusionPipeline, PipelineConfig,
                           CameraIntrinsics, StereoExtrinsics, render_overlay)

cfg = PipelineConfig(
    rgb_cam=CameraIntrinsics.from_fov(4000, 3000, 84.0),
    ir_cam=CameraIntrinsics.from_fov(640, 512, 61.0),
    extrinsics=StereoExtrinsics(t=np.array([0.035, 0.012, 0.0])),
    plane_normal=(0.0, 0.0, 1.0),
    plane_distance_m=42.0,        # LRF 또는 지면 평면 거리
    emissivity=0.85,
    fusion_method="ds",
)

pipe = RGBIRLateFusionPipeline(cfg)
res = pipe.run(rgb, thermal_dn)
print(res.to_json())
```

정사영상(ortho) 쌍이 이미 지오레퍼런싱 되어 있으면 `gt_rgb` / `gt_ir` 에
GeoTransform 을 넣는 것이 가장 정확하다. 이때 정합 오차의 하한은
두 ortho 의 지오레퍼런싱 정확도 자체가 된다.

---

## 모듈 구성

| 파일 | 역할 |
|---|---|
| `yolo_backend.py` | 런타임 추상화, 타일 추론·NMS, 열화상 정규화, 클래스 매핑, conf 보정 적용, 패널 seg 검출 |
| `yolo_branch.py` | YOLO 분기 분석기(RGB/IR), 검출→패널 귀속, YOLO+규칙 하이브리드 |
| `registration.py` | 평면 유도 호모그래피 → gradient 도메인 ECC → MI 폴백, 잔차·NCC 기반 신뢰도 |
| `panels.py` | FFT 격자각, 배경 차분 기반 패널 검출, 테이블 분할, Label Studio 로더, ROI 사상 |
| `rgb_branch.py` | 오염·그림자·크랙·변색·박리·유리파손 (어레이 상대 L*/b* 기준) |
| `ir_branch.py` | 셀 핫스팟·서브스트링·모듈 전체·정션박스·PID (IEC TS 62446-3 ΔT 기준) |
| `fusion.py` | Dempster-Shafer / 로그 오피니언 풀 / Noisy-OR / 학습형 + 교차 중재 규칙 |
| `pipeline.py` | 오케스트레이션, preflight, 배치 추론, 오버레이 렌더, JSON 리포트, CLI |

---

## YOLOv11 분기 (추론)

이 패키지는 **학습된 모델을 받아 추론하는 쪽**만 담당한다. 학습과 신뢰도 보정은
별도 파이프라인에서 하고, 여기서는 그 산출물을 읽어 쓴다.

### 모델과 함께 받아야 하는 것

| 아티팩트 | 필수 | 내용 |
|---|:--:|---|
| `best.pt` (RGB / IR / panel-seg) | ○ | 모달리티별로 **따로** 학습된 가중치 |
| `normalizer.json` | IR 모델이면 ○ | 학습 때 쓴 열화상 → 8bit 정규화 설정 |
| `class_map.json` | △ | 모델 클래스명 → 파이프라인 택소노미 매핑 |
| `calib_*.json` | △ | 클래스별 conf 보정 파라미터 `{"hotspot": [T, b]}` |

`normalizer.json` 이 학습 때와 다르면 같은 ΔT 가 다른 화소값이 되어, **가중치는
멀쩡한데 성능만 조용히 무너진다.** IR 모델 배포에서 가장 흔한 사고다.
`class_map.json` 이 없으면 `DEFAULT_RGB_CLASS_MAP` / `DEFAULT_IR_CLASS_MAP` 이
쓰이는데, 팀마다 클래스 이름이 달라서 대개 한 번은 맞춰줘야 한다.

### 단일 프레임

```bash
python -m rgb_ir_fusion.pipeline \
    --rgb DJI_0001_W.JPG --ir DJI_0001_T.tif \
    --rgb-weights   models/rgb/best.pt \
    --ir-weights    models/ir/best.pt \
    --panel-weights models/panel_seg/best.pt \
    --thermal-norm-json models/ir/normalizer.json \
    --ir-class-map  models/ir/class_map.json \
    --rgb-calib models/rgb/calib.json --ir-calib models/ir/calib.json \
    --device cuda:0 --tile 1280 \
    --rgb-hfov 84 --ir-hfov 61 --plane-distance 42 --out ./out
```

가중치를 주지 않은 분기는 자동으로 규칙 기반으로 동작한다. IR 모델만 먼저
나왔다면 `--ir-weights` 만 줘도 된다.

### 배치 (사이트 단위)

```bash
python -m rgb_ir_fusion.pipeline \
    --rgb-dir /data/siteA/rgb --ir-dir /data/siteA/ir \
    --rgb-token _W --ir-token _T \
    --rgb-weights models/rgb/best.pt --ir-weights models/ir/best.pt \
    --thermal-norm-json models/ir/normalizer.json \
    --device cuda:0 --plane-distance 42 --out ./siteA_out
```

DJI 다중 센서 기체는 한 셔터에 `..._W.JPG` 와 `..._T.JPG` 를 같은 시퀀스
번호로 떨구므로 토큰 치환으로 짝을 맞춘다. 모델은 한 번만 로드되어 전 프레임에
재사용되고, 프레임 하나가 실패해도 배치는 계속된다.

산출물은 프레임별 `*.json` / `*_overlay.jpg` 와 사이트 집계 `site_summary.json`
이다. 집계에는 정합 잔차의 중앙값·p90 과 `n_low_registration` 이 들어간다.
정합이 나쁜 프레임의 융합 결과는 믿지 말고 재처리 대상으로 본다.

### 라이브러리

```python
from rgb_ir_fusion import (RGBIRLateFusionPipeline, PipelineConfig,
                           ThermalNormalizer, find_pairs, run_batch)

cfg = PipelineConfig(
    rgb_weights="models/rgb/best.pt",
    ir_weights="models/ir/best.pt",
    panel_weights="models/panel_seg/best.pt",
    thermal_normalizer_json="models/ir/normalizer.json",
    rgb_calibration_json="models/rgb/calib.json",
    ir_calibration_json="models/ir/calib.json",
    device="cuda:0", tile=1280, tile_overlap=0.2,
    hybrid_w_yolo=None,        # 0.7 을 주면 YOLO+규칙 하이브리드
)

pipe = RGBIRLateFusionPipeline(cfg)
print(pipe.preflight())        # 배치 전 자기 점검

summary = run_batch(pipe, find_pairs("/data/siteA/rgb", "/data/siteA/ir"), "./out")
```

### preflight — 조용한 실패를 먼저 잡는다

`pipe.preflight()` 는 모델이 실제로 가진 클래스명과 매핑 테이블을 대조하고
보정·정규화 설정을 점검한다. 이 세 가지는 **에러 없이 성능만 깎는** 종류라
런타임 로그로는 안 잡힌다.

```
[경고] ir: 매핑 안 된 모델 클래스 ['hot_cell'] - 이 클래스의 검출은 전부 버려진다
[경고] ir: conf 보정 미적용 - YOLO 분기가 과신한 채 융합된다
[경고] IR 모델을 쓰면서 normalizer 설정을 주지 않았다 - 학습 때와 같은 정규화인지 확인할 것
```

`--preflight-only` 로 추론 없이 점검만 할 수도 있다.

### 추론에서 실제로 문제가 되는 것들

**1. YOLO 의 conf 는 확률이 아니다.**
objectness × cls 에 NMS 가 얹힌 값이라 보통 과신한다. 보정 없이
Dempster-Shafer 에 넣으면 그 분기가 거의 항상 이겨서 융합이 무의미해진다.
`calib_*.json` 을 반드시 함께 받아 `--rgb-calib` / `--ir-calib` 로 물린다.

**2. "검출 없음"은 "정상"이 아니라 "이 모델이 못 봤음"이다.**
그래서 **클래스 확률은 YOLO 가, 신뢰도는 물리 특징이** 만든다. 정반사 포화,
저일사량, GSD 부족은 확률로 표현하면 과신이 되고 가중치로 표현해야 맞다.
`YoloRGBAnalyzer` 가 `RGBPanelAnalyzer` 의 특징 추출기를 그대로 재사용하므로
`glare_suppress` 같은 교차 규칙이 YOLO 분기에서도 동작한다.

**3. 관측 가능 클래스는 모델이 아니라 모달리티가 정한다.**
RGB 모델이 우연히 `hotspot` 클래스를 갖고 있어도 RGB 는 셀 발열을 못 본다.
마스크는 `RGB_MASK` / `IR_MASK` 로 고정되며 모델 클래스 목록과 무관하다.

**4. IEC ΔT 게이트.**
열화상 모델은 반사·주변 구조물을 핫스팟으로 오검출하기 쉽고, 그 오류는
**온도값으로 직접 반증할 수 있다.** 모델이 `hotspot` 이라 했는데 ΔT_intra 가
임계의 45% 에도 못 미치면 증거를 0.3배로 감쇄한다.
`PipelineConfig(iec_severity_check=False)` 로 끌 수 있지만 권장하지 않는다.

**5. 오소모자이크는 반드시 타일로 썬다.**
한 변 2만 픽셀짜리를 imgsz=960 에 리사이즈해 넣으면 셀 핫스팟이 1픽셀 미만이
되어 사라진다. `--tile` / `tile_overlap` 으로 썰고 타일 경계 NMS 로 정리한다.

### 진단

요약에 `rgb_orphan_detections` / `ir_orphan_detections` 가 실린다. 어떤 패널에도
귀속되지 않은 검출로, 많으면 보통 패널 검출이 모듈을 놓쳤거나 정합이 어긋난
것이다. 둘 다 조용히 재현율을 깎으므로 항상 노출시킨다.
`*_unmapped_classes` 는 택소노미에 매핑되지 않은 모델 클래스명이다.

### 하이브리드 (모델 도입 초기)

```python
cfg = PipelineConfig(..., hybrid_w_yolo=0.7)
```

같은 모달리티 안에서 YOLO와 규칙을 신뢰도 가중 기하평균으로 먼저 합친 뒤
분기 간 late fusion 으로 넘긴다. 학습 데이터가 적은 클래스는 규칙이 받쳐주고,
모델이 안정되면 `hybrid_w_yolo` 를 1.0 으로 올려 규칙을 걷어낸다.

---

## 설계상 중요한 세 가지

### 1. 분기의 '정상'은 그 분기가 볼 수 있는 범위 안에서만 정상이다

가장 흔한 late fusion 버그가 여기서 난다. RGB 는 셀 핫스팟이나 바이패스
다이오드 이상을 **원리적으로 볼 수 없다.** 그런데 RGB 의 `normal` 확률을
단일톤 `{normal}` 에 주면, RGB 가 "정상"이라고 말하는 순간 IR 이 찾은
핫스팟과 정면 충돌하고, Dempster 결합에서 질량이 큰 쪽이 이겨
**열 결함이 통째로 지워진다.** 조용히 재현율만 깎아먹기 때문에 발견이 늦다.

그래서 분기 `b` 의 초점원소를 이렇게 둔다.

```
m_b({c})   = r_b · P_b(c)                     # b 가 판별 가능한 결함 c
m_b(A_b)   = r_b · P_b(normal),
             A_b = {normal} ∪ {b 가 관측 불가능한 클래스}
m_b(Theta) = 1 - r_b                          # 신뢰 불능분
```

`A_rgb ∩ {hotspot_cell} = {hotspot_cell}` 이므로 IR 의 발견이 살아남고,
`A_rgb ∩ A_ir = {normal}` 이므로 **두 분기가 모두 정상일 때만** 정상으로 수렴한다.

### 2. 두 분기 모두 절대값이 아니라 어레이 상대값으로 판정한다

시간대·노출·화이트밸런스는 절대 밝기를 통째로 바꾸고, 기온·풍속은 절대
온도를 통째로 바꾼다. 그래서 RGB 는 어레이 L*/b* 중앙값 대비 ΔL*·Δb* 를,
IR 은 어레이 온도 중앙값 대비 ΔT_inter 를 근거로 쓴다.
`array_id` 를 부여해 스트링 단위로 기준선을 나누면 정확도가 더 올라간다.

### 3. 정합이 나쁘면 융합하지 않는 것도 정상 동작이다

`RegistrationResult.reliability()` 가 낮으면 IR 분기 가중치를 깎고,
`rule_min_registration`(기본 0.45) 미만이면 교차 중재 규칙을 아예 끈다.
어긋난 대응 위에서의 융합은 융합이 아니라 오염이다.

---

## 교차 중재 규칙

`fusion.default_rules()` 에 들어 있는 규칙들이 현장 오탐의 상당 부분을 잡는다.

| 규칙 | 내용 |
|---|---|
| `glare_suppress` | 정반사 포화 구간에서 RGB 외관 결함(오염 포함) 판정을 감쇄 |
| `shadow_explains_heat` | 그림자로 설명되는 발열은 결함에서 제외 |
| `crack_plus_hotspot` | 외관 크랙 + 셀 핫스팟 동시 검출 시 확증 강화 |
| `soiling_confirmed` | 외관 오염 + 광범위 저강도 발열 = 오염 확증(세정 대상) |
| `clean_surface_vs_heat` | 외관 정상(고신뢰) + 모듈 전체 발열 = 전기적 결함 의심 강화 |

`Rule(name, apply, description)` 로 직접 추가할 수 있다. `apply` 는
`(probs, rgb_ev, ir_ev)` 를 받아 수정된 확률 또는 `None` 을 돌려준다.

---

## 임계값 튜닝

IR 기본값은 **일사량 > 600 W/m², 풍속 < 4 m/s** 조건을 가정한다.

| 파라미터 | 기본 | 의미 |
|---|---|---|
| `IRPanelAnalyzer.dt_cell` | 10.0 K | 단일 셀 핫스팟 ΔT |
| `IRPanelAnalyzer.dt_substring` | 5.0 K | 서브스트링/다이오드 ΔT |
| `IRPanelAnalyzer.dt_module` | 4.0 K | 모듈 전체 ΔT_inter |
| `IRPanelAnalyzer.netd_k` | 0.08 K | 센서 잡음 바닥 (신뢰도 산정용) |
| `RGBPanelAnalyzer.dL_soiling` | 8.0 | 어레이 대비 L* 상승 |
| `RGBPanelAnalyzer.glare_frac_warn` | 0.05 | 이 비율 넘는 포화부터 신뢰도 감쇄 |
| `LateFusionEngine.abstain_threshold` | 0.30 | 유보 기준 (2위와의 격차도 함께 봄) |

## 분기 교체

YOLOv11 외에 다른 모델을 쓰려면 `analyze()` 시그니처만 지키면 된다.
프레임 단위 추론이 필요하면 `prepare(image, rois)` 를 구현하면 파이프라인이
ROI 루프 전에 한 번 호출해 준다.

```python
class MyRGBModel:
    def prepare(self, image, rois): self._out = self.net(image)
    def array_reference(self, image, rois): return {}
    def analyze(self, image, roi, array_ref=None) -> BranchEvidence:
        probs = ...                              # RGB_OBSERVABLE 위의 분포
        return BranchEvidence("rgb", probs, RGB_MASK, reliability=..., features={...})

pipe = RGBIRLateFusionPipeline(cfg, rgb_analyzer=MyRGBModel())
```

추론 백엔드만 바꾸려면(TensorRT/ONNX) `YoloRuntime` 을 구현해 주입한다.

```python
pipe = RGBIRLateFusionPipeline(cfg, ir_runtime=MyTensorRTRuntime("ir.engine"))
```

융합기 자체도 라벨이 쌓이면 교체할 수 있다. (오프라인에서 fit 한 뒤 주입)

```python
X = np.array([LearnedFusion.featurize([v.rgb, v.ir]) for v in verdicts])
y = np.array([CLASS_INDEX[label] for label in ground_truth])
engine = LateFusionEngine(fusion=LearnedFusion().fit(X, y))
```

`reliability` 는 학습 모델에서도 반드시 채워야 한다. 정반사·저일사량처럼
**입력이 판정을 지지하지 못하는 조건**을 확률로 표현하면 과신이 되고,
가중치로 표현해야 late fusion 이 제대로 작동한다.

---

## 알려진 한계

- **정반사(specular) 차이는 기하 오차가 아니다.** 패널 단위 밝기 스텝은
  상호상관 기반 지표로 잡히지 않는다. 그래서 여기서는 검출 대상이 아니라
  RGB 분기의 *신뢰도 감쇄 요인*으로만 다룬다.
- 기본 `detect_panels`(규칙)는 "주변보다 어두운 직사각형"을 찾는다. 완전 포화된
  글레어 패널은 밝아지므로 놓친다. `panel_weights` 로 YOLOv11-seg 를 주면
  이 경로가 대체되고, seg 가 아무것도 못 찾으면 규칙으로 폴백한다.
- YOLO 분기는 학습 분포 밖에서 조용히 틀린다. 사이트·계절·기체가 바뀌면
  `*_orphan_detections` 와 유보(`n_abstain`) 비율을 먼저 본다. 급증하면
  재학습 또는 재보정 신호다.
- IR 단독으로는 오염과 그림자를 구분할 수 없다. 이 모호성은 설계상 남겨두고
  RGB 분기가 해소한다 — late fusion 이 실제로 값을 만드는 지점이다.
- 경사 지형에서 평면 유도 호모그래피는 단일 평면 가정을 쓴다. 기복이 크면
  DSM 기반 사상으로 대체해야 하고, 프레임 단위 정합 잔차가 커질 수 있다.
- `dn_to_celsius` 의 기본 스케일(0.04 K/DN)은 FLIR 계열 관례다.
  **카메라 메타데이터로 반드시 확인할 것.** 틀리면 모든 ΔT 임계값이 어긋난다.

"""
YOLOv11 통합 데모 / 스모크 테스트.

가중치(.pt) 없이도 **통합 로직 전체**를 검증한다. `YoloRuntime` 을 주입 가능하게
만들어 두었기 때문에, 학습된 모델 대신 정답을 알고 있는 가짜 런타임을 꽂아
다음을 확인할 수 있다.

  - 타일 추론 후 좌표 복원과 타일 경계 NMS
  - 검출 -> 패널 ROI 귀속 (작은 결함은 중심 포함, 큰 결함은 IoU)
  - YOLO conf 보정 -> BranchEvidence 확률
  - RGB 가 '정상'이어도 IR 전용 결함이 살아남는지 (집합값 초점원소)
  - 정반사 포화 시 RGB 신뢰도 하락 + 교차 중재 규칙
  - IEC ΔT 게이트: 모델이 핫스팟이라 해도 온도가 안 따라오면 감쇄
  - 하이브리드(YOLO + 규칙) 경로
  - preflight 자기 점검 (클래스 매핑 누락 / 보정 미적용 경고)

실제 가중치로 돌릴 때:

    python -m rgb_ir_fusion.pipeline --rgb a_W.JPG --ir a_T.tif \\
        --rgb-weights runs/pv/rgb/weights/best.pt \\
        --ir-weights  runs/pv/ir/weights/best.pt \\
        --panel-weights runs/pv/panel_seg/weights/best.pt \\
        --device cuda:0 --plane-distance 42 --rgb-hfov 84 --ir-hfov 61

실행:  python examples/demo_yolo.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence

import cv2
import numpy as np
from pathlib import Path

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "label_studio"))

import demo_synthetic as syn
from sayou.fusion import (CameraIntrinsics, ConfidenceCalibrator,
                           HybridBranchAnalyzer, PanelROI, PipelineConfig,
                           RGBIRLateFusionPipeline, StereoExtrinsics,
                           ThermalNormalizer, YoloRuntime,
                           plane_induced_homography)


# --------------------------------------------------------------------------- #
# 가짜 런타임: 좌표를 알고 있는 '완벽하지 않은' 모델을 흉내낸다
# --------------------------------------------------------------------------- #

class ScriptedRuntime(YoloRuntime):
    """
    미리 정해둔 전역 좌표의 검출을 돌려준다. 타일 좌표계로 변환해서 내보내므로
    `YoloDetector` 의 타일 복원 로직이 실제로 검증된다.

    boxes: [(cls, conf, x0, y0, x1, y1), ...]  - 원본 영상 좌표
    """

    def __init__(self, boxes: Sequence, class_names: Sequence[str],
                 first_tile_only: bool = False):
        self.boxes = list(boxes)
        self._names = list(class_names)
        self.first_tile_only = first_tile_only   # 타일 복원 검증용
        self.calls = 0

    @property
    def names(self):
        return self._names

    def predict(self, images):
        # YoloDetector 는 타일을 잘라서 넘긴다. 타일 원점을 모르므로,
        # 여기서는 '타일 크기와 같은 이미지가 오면 전체 이미지'로 간주한다.
        # (데모 해상도에서는 타일이 1장이다 — 타일 복원은 아래 test_tiling 에서 검증)
        out = []
        for img in images:
            self.calls += 1
            if self.first_tile_only and self.calls > 1:
                out.append([])
                continue
            h, w = img.shape[:2]
            dets = []
            for cls, conf, x0, y0, x1, y1 in self.boxes:
                if x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h:
                    dets.append({"cls": cls, "conf": float(conf),
                                 "xyxy": np.array([x0, y0, x1, y1], float),
                                 "polygon": None})
            out.append(dets)
        return out


class PanelSegRuntime(YoloRuntime):
    """정답 패널 폴리곤을 돌려주는 가짜 seg 모델."""

    def __init__(self, polygons: Sequence[np.ndarray]):
        self.polygons = [np.asarray(p, float) for p in polygons]

    @property
    def names(self):
        return ["panel"]

    def predict(self, images):
        out = []
        for img in images:
            h, w = img.shape[:2]
            dets = []
            for p in self.polygons:
                x0, y0 = p[:, 0].min(), p[:, 1].min()
                x1, y1 = p[:, 0].max(), p[:, 1].max()
                if x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h:
                    dets.append({"cls": "panel", "conf": 0.93,
                                 "xyxy": np.array([x0, y0, x1, y1], float),
                                 "polygon": p.copy()})
            out.append(dets)
        return out


# --------------------------------------------------------------------------- #

def build_scene():
    rgb = syn.make_rgb()
    rgb_cam = CameraIntrinsics.from_fov(syn.W, syn.H, 84.0)
    ir_cam = CameraIntrinsics.from_fov(640, 512, 61.0)
    ext = StereoExtrinsics(R=cv2.Rodrigues(np.array([0.0, 0.0, np.deg2rad(1.3)]))[0],
                           t=np.array([0.035, 0.012, 0.0]))
    H = plane_induced_homography(rgb_cam.K, ir_cam.K, ext,
                                 np.array([0.0, 0.0, 1.0]), 42.0)
    thermal = syn.make_thermal(H, (512, 640)).astype(np.float32)
    return rgb, thermal, rgb_cam, ir_cam, ext, H


def panel_polys():
    out = {}
    for r in range(syn.ROWS):
        for c in range(syn.COLS):
            x, y, w, h = syn.panel_rect(r, c)
            out[f"R{r}C{c}"] = np.array([[x, y], [x + w, y],
                                         [x + w, y + h], [x, y + h]], float)
    return out


def main():
    rgb, thermal, rgb_cam, ir_cam, ext, H = build_scene()
    polys = panel_polys()

    # --- RGB 모델이 낼 법한 검출 (오염 1건, 글레어를 크랙으로 오검출 1건) ------ #
    x, y, w, h = syn.panel_rect(0, 3)
    x2, y2, w2, h2 = syn.panel_rect(2, 3)
    rgb_boxes = [
        ("soiling", 0.81, x + 8, y + 6, x + w - 8, y + h - 6),
        ("crack", 0.64, x2 + 30, y2 + 20, x2 + 95, y2 + 60),   # 정반사 오검출
    ]
    rgb_rt = ScriptedRuntime(rgb_boxes, ["soiling", "crack", "shadow", "panel"])

    # --- IR 모델이 낼 법한 검출 ------------------------------------------- #
    def to_ir(pt):
        p = cv2.perspectiveTransform(np.array([[pt]], float), H)[0, 0]
        return float(p[0]), float(p[1])

    xa, ya, wa, ha = syn.panel_rect(1, 1)
    hs = to_ir((xa + w * 0.62, ya + ha * 0.5))
    xb, yb, wb, hb = syn.panel_rect(2, 0)
    s0 = to_ir((xb + wb * 2 / 3, yb))
    s1 = to_ir((xb + wb, yb + hb))
    xc, yc, wc, hc = syn.panel_rect(0, 0)
    f0 = to_ir((xc + 10, yc + 10))
    f1 = to_ir((xc + 40, yc + 40))
    ir_boxes = [
        ("hotspot", 0.88, hs[0] - 9, hs[1] - 9, hs[0] + 9, hs[1] + 9),
        ("substring", 0.72, min(s0[0], s1[0]), min(s0[1], s1[1]),
         max(s0[0], s1[0]), max(s0[1], s1[1])),
        # 반사에 의한 가짜 핫스팟: ΔT 가 따라오지 않으므로 IEC 게이트가 잡아야 한다
        ("hotspot", 0.69, min(f0[0], f1[0]), min(f0[1], f1[1]),
         max(f0[0], f1[0]), max(f0[1], f1[1])),
    ]
    ir_rt = ScriptedRuntime(ir_boxes, ["hotspot", "substring", "module_hot", "panel"])

    panel_rt = PanelSegRuntime(list(polys.values()))

    cfg = PipelineConfig(
        rgb_cam=rgb_cam, ir_cam=ir_cam, extrinsics=ext,
        plane_normal=(0.0, 0.0, 1.0), plane_distance_m=42.0,
        radiometric=True, emissivity=1.0,
        rgb_weights="rgb.pt", ir_weights="ir.pt", panel_weights="panel-seg.pt",
        thermal_normalizer=ThermalNormalizer(mode="relative", span=24.0),
        tile=2048, fusion_method="ds",
    )
    pipe = RGBIRLateFusionPipeline(cfg, rgb_runtime=rgb_rt, ir_runtime=ir_rt,
                                   panel_runtime=panel_rt)

    # 신뢰도 보정: 운영에서는 모델과 함께 배포된 calib JSON 을 읽는다
    #   PipelineConfig(rgb_calibration_json="calib_rgb.json", ...)
    pipe.rgb_analyzer.detector.calibrator = ConfidenceCalibrator(
        {"soiling": (1.4, 0.3), "crack": (1.6, 0.5)})
    pipe.ir_analyzer.detector.calibrator = ConfidenceCalibrator(
        {"hotspot": (1.3, 0.2), "substring": (1.3, 0.2)})

    # 추론 전 자기 점검: 클래스 매핑 누락 / 보정 미적용을 여기서 잡는다
    pf = pipe.preflight()
    print("preflight:", {k: pf[k] for k in ("rgb_branch", "ir_branch", "panel_source")})
    for w in pf["warnings"]:
        print("  [경고]", w)

    res = pipe.run(rgb, thermal)

    print("=" * 96)
    print(f"정합 residual={res.registration.residual_px:.2f}px  "
          f"reliability={res.registration.reliability():.3f}   "
          f"패널 {len(res.rois)}장 (YOLO-seg)")
    print("=" * 96)
    print(f"{'panel':8}{'fused':20}{'conf':>6}{'sev':>4}   {'RGB':22}{'IR':24} rules")
    for v in sorted(res.verdicts, key=lambda v: v.panel_id):
        rt = f"{v.rgb.top()[0]}({v.rgb.reliability:.2f})" if v.rgb else "-"
        it = f"{v.ir.top()[0]}({v.ir.reliability:.2f})" if v.ir else "-"
        print(f"{v.panel_id:8}{v.label:20}{v.confidence:6.2f}{v.severity:4d}   "
              f"{rt:22}{it:24} {','.join(v.rules_fired)}")
    print("-" * 96)
    print(res.summary)

    by = {v.panel_id: v for v in res.verdicts}
    # 패널 ID 는 YOLO-seg 검출 순서라 좌표로 되짚는다
    def at(r, c):
        x, y, w, h = syn.panel_rect(r, c)
        cx, cy = x + w / 2, y + h / 2
        best, bd = None, 1e18
        for roi in res.rois:
            d = np.linalg.norm(roi.polygon_rgb.mean(axis=0) - [cx, cy])
            if d < bd:
                best, bd = roi.panel_id, d
        return by[best]

    print("\n--- 검증 ---")
    v = at(1, 1)
    print(f"R1C1  fused={v.label} (RGB={v.rgb.top()[0]}, IR={v.ir.top()[0]})")
    assert v.label == "hotspot_cell", v.label
    assert v.rgb.top()[0] == "normal", "RGB 는 셀 핫스팟을 못 본다"

    v = at(2, 0)
    print(f"R2C0  fused={v.label}")
    assert v.label == "hotspot_substring", v.label

    v = at(0, 3)
    print(f"R0C3  fused={v.label}  (rules={v.rules_fired})")
    assert v.label == "soiling", v.label

    v = at(0, 0)
    gated = [n for n in v.ir.notes if "ΔT 근거 부족" in n]
    print(f"R0C0  fused={v.label}  IEC 게이트: {gated}")
    assert v.label == "normal", v.label
    assert gated, "온도 근거 없는 YOLO 핫스팟이 감쇄되지 않았다"

    v = at(2, 3)
    print(f"R2C3  fused={v.label}  RGB rel={v.rgb.reliability:.2f} rules={v.rules_fired}")
    assert v.rgb.reliability < 0.5, "정반사 패널의 RGB 신뢰도가 안 떨어짐"
    assert v.label == "normal", v.label

    # --- 타일 복원 검증 --------------------------------------------------- #
    from sayou.fusion import YoloDetector
    big = np.zeros((3000, 4000, 3), np.uint8)
    # 타일 3번째에서만 검출을 내도록 해서, 전역 좌표 복원이 맞는지 본다
    rt = ScriptedRuntime([("soiling", 0.9, 10, 10, 60, 60)], ["soiling"])
    det = YoloDetector(rt, class_map={"soiling": "soiling"}, tile=1024, overlap=0.25)
    tiles = det._tiles(big.shape[0], big.shape[1])
    dets = det.detect(big)
    print(f"\n타일 추론: {len(tiles)} 타일 / 런타임 호출 {rt.calls}회 / 검출 {len(dets)}건")
    assert rt.calls == len(tiles) > 1, "타일 분할이 일어나지 않았다"
    expected = {(round(x0 + 10), round(y0 + 10)) for (x0, y0, _, _) in tiles}
    got = {(round(float(d.xyxy[0])), round(float(d.xyxy[1]))) for d in dets}
    assert got <= expected and len(got) == len(tiles), \
        "타일 오프셋이 원본 좌표로 복원되지 않았다"
    print(f"  전역 좌표 복원 OK: {sorted(got)[:4]} ...")

    # NMS: 겹친 타일에서 같은 결함이 두 번 잡히면 하나로 합쳐져야 한다
    from sayou.fusion import Detection, nms_per_class
    dup = [Detection("hotspot", "hotspot_cell", c, c,
                     np.array([100.0 + dx, 100.0, 160.0 + dx, 160.0]))
           for c, dx in ((0.90, 0.0), (0.71, 6.0), (0.66, 9.0))]
    other = Detection("hotspot", "hotspot_cell", 0.8, 0.8,
                      np.array([400.0, 400.0, 460.0, 460.0]))
    merged = nms_per_class(dup + [other], iou_thresh=0.55)
    print(f"  중복 제거: 4건 -> {len(merged)}건 "
          f"(남은 conf {[round(d.conf, 2) for d in merged]})")
    assert len(merged) == 2

    # --- preflight 가 조용한 실패를 잡는가 --------------------------------- #
    cfg_bad = PipelineConfig(**{**cfg.__dict__,
                                "ir_class_map": {"hotsopt": "hotspot_cell"}})  # 오타
    bad = RGBIRLateFusionPipeline(cfg_bad, rgb_runtime=rgb_rt, ir_runtime=ir_rt,
                                  panel_runtime=panel_rt)
    warns = bad.preflight()["warnings"]
    print(f"\n오타 매핑 preflight 경고 {len(warns)}건: {warns[0][:60]}...")
    assert any("매핑 안 된" in w for w in warns), "클래스명 오타를 preflight 가 놓쳤다"

    # --- 하이브리드 경로 --------------------------------------------------- #
    cfg2 = PipelineConfig(**{**cfg.__dict__, "hybrid_w_yolo": 0.65})
    pipe2 = RGBIRLateFusionPipeline(cfg2, rgb_runtime=rgb_rt, ir_runtime=ir_rt,
                                    panel_runtime=panel_rt)
    res2 = pipe2.run(rgb, thermal)
    assert isinstance(pipe2.rgb_analyzer, HybridBranchAnalyzer)
    print(f"하이브리드(w_yolo=0.65): {res2.summary['by_label']}")

    print("\n스모크 테스트 통과")


if __name__ == "__main__":
    main()

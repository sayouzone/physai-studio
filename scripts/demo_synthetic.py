"""
합성 데이터 엔드투엔드 데모 / 스모크 테스트.

실제 M3T 데이터 없이 파이프라인 전체(정합 -> ROI -> 분기 -> late fusion)를
검증한다. 합성 장면에 아래 결함을 심는다.

    panel[1][1] : IR 셀 핫스팟 (+18K) - RGB 는 정상
    panel[0][3] : RGB 오염 + IR 광범위 약한 발열 (+2.5K)
    panel[2][0] : IR 서브스트링 발열 (+7K, 우측 1/3)
    panel[2][3] : RGB 정반사 포화 (late fusion 이 RGB 신뢰도를 낮춰야 함)

실행:  python examples/demo_synthetic.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from pathlib import Path

#sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
#.rgb_ir_fusion
# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "label_studio"))

from sayou.fusion import (CameraIntrinsics, PipelineConfig, PanelROI,
                           RGBIRLateFusionPipeline, StereoExtrinsics,
                           render_overlay)

RNG = np.random.default_rng(7)

ROWS, COLS = 3, 4
PW, PH = 150, 88          # 패널 크기 (RGB px)
GAP_X, GAP_Y = 34, 40
# RGB 는 84도, 열화상은 61도 화각이라 IR 은 RGB 프레임의 중앙 ~65% 만 덮는다.
# 12장 전부가 IR 대응을 갖도록 어레이를 중앙에 배치한다.
W, H = 1200, 900
GRID_W = COLS * PW + (COLS - 1) * GAP_X
GRID_H = ROWS * PH + (ROWS - 1) * GAP_Y
ORG = ((W - GRID_W) // 2, (H - GRID_H) // 2)


def panel_rect(r, c):
    x = ORG[0] + c * (PW + GAP_X)
    y = ORG[1] + r * (PH + GAP_Y)
    return x, y, PW, PH


def make_rgb():
    img = np.full((H, W, 3), 118, np.uint8)          # 지면
    img = cv2.add(img, RNG.integers(-12, 12, (H, W, 3), dtype=np.int16).astype(np.uint8))
    for r in range(ROWS):
        for c in range(COLS):
            x, y, w, h = panel_rect(r, c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (52, 44, 40), -1)
            # 셀 격자 (6x3)
            for i in range(1, 6):
                cx = x + int(i * w / 6)
                cv2.line(img, (cx, y + 2), (cx, y + h - 2), (86, 78, 72), 1)
            for j in range(1, 3):
                cy = y + int(j * h / 3)
                cv2.line(img, (x + 2, cy), (x + w - 2, cy), (86, 78, 72), 1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (150, 150, 150), 2)

    # 오염: panel[0][3]
    x, y, w, h = panel_rect(0, 3)
    soil = np.zeros((h, w, 3), np.float32)
    for _ in range(90):
        cx, cy = RNG.integers(0, w), RNG.integers(0, h)
        cv2.circle(soil, (int(cx), int(cy)), int(RNG.integers(4, 13)),
                   (28, 42, 58), -1)
    soil = cv2.GaussianBlur(soil, (0, 0), 6)
    img[y:y + h, x:x + w] = np.clip(img[y:y + h, x:x + w] + soil, 0, 255).astype(np.uint8)

    # 정반사 포화: panel[2][3]
    x, y, w, h = panel_rect(2, 3)
    glare = np.zeros((h, w), np.float32)
    cv2.ellipse(glare, (w // 2, h // 2), (int(w * 0.34), int(h * 0.42)), 18, 0, 360, 1.0, -1)
    glare = cv2.GaussianBlur(glare, (0, 0), 9)
    patch = img[y:y + h, x:x + w].astype(np.float32)
    patch += glare[..., None] * 230.0
    img[y:y + h, x:x + w] = np.clip(patch, 0, 255).astype(np.uint8)

    return img


def make_thermal(H_rgb_to_ir, ir_shape):
    """섭씨 float 열화상. RGB 좌표계에서 그린 뒤 호모그래피로 IR 좌표계에 사상."""
    t = np.full((H, W), 28.0, np.float32)            # 지면
    for r in range(ROWS):
        for c in range(COLS):
            x, y, w, h = panel_rect(r, c)
            t[y:y + h, x:x + w] = 45.0

    # 셀 핫스팟: panel[1][1]
    x, y, w, h = panel_rect(1, 1)
    cv2.circle(t, (x + int(w * 0.62), y + int(h * 0.5)), 9, 45.0 + 18.0, -1)

    # 오염 패널 광범위 약한 발열: panel[0][3]
    x, y, w, h = panel_rect(0, 3)
    t[y:y + h, x:x + w] += 2.5

    # 서브스트링: panel[2][0] 우측 1/3
    x, y, w, h = panel_rect(2, 0)
    t[y:y + h, x + int(w * 2 / 3):x + w] += 7.0

    t = cv2.GaussianBlur(t, (0, 0), 1.6)
    t += RNG.normal(0, 0.25, t.shape).astype(np.float32)

    ih, iw = ir_shape
    return cv2.warpPerspective(t, H_rgb_to_ir, (iw, ih), flags=cv2.INTER_LINEAR,
                               borderValue=28.0)


def main():
    rgb = make_rgb()

    # IR 카메라: 더 좁은 FOV, 낮은 해상도, 약간의 회전/평행이동 (실제 M3T 유사)
    ir_w, ir_h = 640, 512
    rgb_cam = CameraIntrinsics.from_fov(W, H, 84.0)
    ir_cam = CameraIntrinsics.from_fov(ir_w, ir_h, 61.0)
    ext = StereoExtrinsics(R=cv2.Rodrigues(np.array([0.0, 0.0, np.deg2rad(1.3)]))[0],
                           t=np.array([0.035, 0.012, 0.0]))
    from sayou.fusion import plane_induced_homography
    H_true = plane_induced_homography(rgb_cam.K, ir_cam.K, ext,
                                      np.array([0.0, 0.0, 1.0]), 42.0)

    thermal = make_thermal(H_true, (ir_h, ir_w))

    # ROI 는 정답 좌표로 제공 (실제 운용에서는 detect_panels 또는 Label Studio)
    rois = []
    for r in range(ROWS):
        for c in range(COLS):
            x, y, w, h = panel_rect(r, c)
            rois.append(PanelROI(
                panel_id=f"R{r}C{c}",
                polygon_rgb=np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                                     dtype=np.float64),
                array_id=f"A{r}"))

    cfg = PipelineConfig(
        rgb_cam=rgb_cam, ir_cam=ir_cam, extrinsics=ext,
        plane_normal=(0.0, 0.0, 1.0), plane_distance_m=42.0,
        radiometric=True, emissivity=1.0,     # 합성값은 이미 물체 온도
        fusion_method="ds",
    )
    pipe = RGBIRLateFusionPipeline(cfg)
    res = pipe.run(rgb, thermal.astype(np.float32), rois=rois)

    print("=" * 74)
    print(f"정합: method={res.registration.method}  "
          f"residual={res.registration.residual_px:.2f} px  "
          f"ncc={res.registration.ncc:.3f}  "
          f"reliability={res.registration.reliability():.3f}")
    print("=" * 74)
    print(f"{'panel':8}{'fused':20}{'conf':>6}{'sev':>4}   "
          f"{'RGB':18}{'IR':18} rules")
    for v in res.verdicts:
        rt = f"{v.rgb.top()[0]}({v.rgb.reliability:.2f})" if v.rgb else "-"
        it = f"{v.ir.top()[0]}({v.ir.reliability:.2f})" if v.ir else "-"
        print(f"{v.panel_id:8}{v.label:20}{v.confidence:6.2f}{v.severity:4d}   "
              f"{rt:18}{it:18} {','.join(v.rules_fired)}")
    print("-" * 74)
    print(res.summary)

    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    cv2.imwrite(os.path.join(out, "overlay.png"),
                render_overlay(rgb, thermal, res.registration, res.verdicts, rois))
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        f.write(res.to_json())
    print(f"\n출력: {out}/overlay.png, {out}/report.json")

    # --- 스모크 검증 ---------------------------------------------------- #
    by = {v.panel_id: v for v in res.verdicts}
    assert res.registration.residual_px < 3.0, "정합 잔차 과다"

    # 심어둔 결함이 정확히 그 패널에서만 나와야 한다
    assert by["R1C1"].label == "hotspot_cell", by["R1C1"].label
    assert by["R2C0"].label == "hotspot_substring", by["R2C0"].label
    assert by["R0C3"].label == "soiling", by["R0C3"].label
    assert by["R2C3"].rgb.reliability < 0.5, "정반사 패널의 RGB 신뢰도가 낮아지지 않음"
    assert "glare_suppress" in by["R2C3"].rules_fired
    clean = [k for k in by if k not in ("R1C1", "R2C0", "R0C3")]
    assert all(by[k].label == "normal" for k in clean), \
        [(k, by[k].label) for k in clean if by[k].label != "normal"]

    # 핵심 회귀 테스트: RGB 가 '정상'이라고 해도 IR 전용 결함을 지워서는 안 된다.
    assert by["R1C1"].rgb.top()[0] == "normal"

    # --- 열화 내성(graceful degradation): IR 없이도 동작해야 한다 --------- #
    from sayou.fusion.types import BranchEvidence
    eng = pipe.engine
    rgb_only = eng.decide("solo", by["R0C3"].rgb, None, registration_reliability=0.0)
    assert rgb_only.label in ("soiling", "uncertain"), rgb_only.label

    # --- 정합 실패 시 교차 규칙이 꺼져야 한다 ----------------------------- #
    bad = eng.decide("bad", by["R2C3"].rgb, by["R2C3"].ir, registration_reliability=0.1)
    assert "cross_modal_rules_disabled(low_registration)" in bad.rules_fired

    print("스모크 테스트 통과")


if __name__ == "__main__":
    main()

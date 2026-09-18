"""
엔드투엔드 파이프라인 + CLI.

    RGB ─┐
         ├─ register_pair ─ project_rois_to_ir ─┬─ RGB분기 ─┐
    IR  ─┘                                      └─ IR분기 ──┴─ LateFusionEngine ─ 리포트
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .fusion import (DempsterShaferFusion, LateFusionEngine, LogOpinionPoolFusion,
                     NoisyOrFusion)
from .ir_branch import IRPanelAnalyzer, apply_emissivity_correction, dn_to_celsius
from .panels import (detect_panels, load_panels_from_labelstudio,
                     project_rois_to_ir)
from .registration import register_pair, warp_rgb_to_ir, warp_points
from .rgb_branch import RGBPanelAnalyzer
from .types import (CameraIntrinsics, GeoTransform, PanelROI, PanelVerdict,
                    RegistrationResult, StereoExtrinsics)
from .yolo_backend import (DEFAULT_IR_CLASS_MAP, DEFAULT_RGB_CLASS_MAP,
                           ConfidenceCalibrator, ThermalNormalizer,
                           UltralyticsRuntime, YoloDetector, YoloPanelDetector,
                           YoloRuntime, load_class_map)
from .yolo_branch import HybridBranchAnalyzer, YoloIRAnalyzer, YoloRGBAnalyzer


# --------------------------------------------------------------------------- #
# 설정
# --------------------------------------------------------------------------- #

@dataclass
class PipelineConfig:
    # 정합
    rgb_cam: Optional[CameraIntrinsics] = None
    ir_cam: Optional[CameraIntrinsics] = None
    extrinsics: Optional[StereoExtrinsics] = None
    plane_normal: Optional[Tuple[float, float, float]] = None
    plane_distance_m: Optional[float] = None
    H_init: Optional[np.ndarray] = None
    gt_rgb: Optional[GeoTransform] = None
    gt_ir: Optional[GeoTransform] = None
    residual_tol_px: float = 3.0

    # 열화상
    radiometric: bool = True
    dn_scale: float = 0.04
    dn_offset: float = -273.15
    emissivity: float = 0.85
    reflected_temp_c: float = 20.0
    irradiance_ok: bool = True

    # 패널
    array_grouping_px: Optional[float] = 260.0
    labelstudio_json: Optional[str] = None

    # YOLOv11 (None 이면 해당 분기는 규칙 기반으로 동작)
    rgb_weights: Optional[str] = None       # RGB 학습 가중치 (.pt)
    ir_weights: Optional[str] = None        # 열화상 학습 가중치 (.pt)
    panel_weights: Optional[str] = None     # 모듈 세그멘테이션 가중치 (-seg 권장)
    device: str = "cpu"                     # "cuda:0" 등
    imgsz: int = 960
    tile: int = 1280
    tile_overlap: float = 0.2
    yolo_conf: float = 0.15
    yolo_iou: float = 0.5
    rgb_class_map: Optional[Dict[str, str]] = None
    ir_class_map: Optional[Dict[str, str]] = None
    rgb_class_map_json: Optional[str] = None
    ir_class_map_json: Optional[str] = None
    rgb_calibration_json: Optional[str] = None
    ir_calibration_json: Optional[str] = None
    thermal_normalizer: Optional[ThermalNormalizer] = None
    thermal_normalizer_json: Optional[str] = None   # 학습 시 저장한 normalizer.json
    hybrid_w_yolo: Optional[float] = None   # 설정 시 YOLO+규칙 하이브리드로 동작
    iec_severity_check: bool = True

    # 융합
    fusion_method: str = "ds"           # ds | pool | noisyor
    abstain_threshold: float = 0.35

    # 출력
    save_overlay: bool = True


@dataclass
class PipelineResult:
    registration: RegistrationResult
    verdicts: List[PanelVerdict] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    rois: List[PanelROI] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({
            "registration": {
                "method": self.registration.method,
                "residual_px": None if not np.isfinite(self.registration.residual_px)
                else round(float(self.registration.residual_px), 3),
                "ncc": None if not np.isfinite(self.registration.ncc)
                else round(float(self.registration.ncc), 3),
                "overlap_ratio": round(float(self.registration.overlap_ratio), 3),
                "reliability": round(self.registration.reliability(), 3),
                "stages": self.registration.stages,
            },
            "summary": self.summary,
            "panels": [v.to_dict() for v in self.verdicts],
        }, ensure_ascii=False, indent=indent)


# --------------------------------------------------------------------------- #
# 파이프라인
# --------------------------------------------------------------------------- #

class RGBIRLateFusionPipeline:

    def __init__(self, config: Optional[PipelineConfig] = None,
                 rgb_analyzer=None,
                 ir_analyzer=None,
                 panel_detector=None,
                 rgb_runtime: Optional[YoloRuntime] = None,
                 ir_runtime: Optional[YoloRuntime] = None,
                 panel_runtime: Optional[YoloRuntime] = None):
        """
        분석기를 직접 주입하면 그것이 우선한다. 주입이 없고 설정에 가중치 경로가
        있으면 YOLOv11 분기를, 둘 다 없으면 규칙 기반 분기를 쓴다.

        `*_runtime` 은 ultralytics 대신 다른 백엔드(TensorRT/ONNX)를 꽂거나
        테스트용 가짜 런타임을 넣기 위한 통로다.
        """
        self.cfg = config or PipelineConfig()
        cfg = self.cfg

        self.thermal_normalizer = (cfg.thermal_normalizer
                                   or ThermalNormalizer.load(cfg.thermal_normalizer_json))

        self.rgb_analyzer = rgb_analyzer or self._build_rgb_branch(rgb_runtime)
        self.ir_analyzer = ir_analyzer or self._build_ir_branch(ir_runtime)
        self.panel_detector = panel_detector or self._build_panel_detector(panel_runtime)

        fusion = {"ds": DempsterShaferFusion,
                  "pool": LogOpinionPoolFusion,
                  "noisyor": NoisyOrFusion}[cfg.fusion_method]()
        self.engine = LateFusionEngine(fusion=fusion,
                                       abstain_threshold=cfg.abstain_threshold)

    # -- 분기 구성 --------------------------------------------------------- #
    def _runtime(self, weights: str, injected: Optional[YoloRuntime]) -> YoloRuntime:
        if injected is not None:
            return injected
        return UltralyticsRuntime(weights, device=self.cfg.device,
                                  imgsz=self.cfg.imgsz, conf=self.cfg.yolo_conf,
                                  iou=self.cfg.yolo_iou)

    def _detector(self, weights: Optional[str], injected: Optional[YoloRuntime],
                  class_map: Dict[str, str], calib_json: Optional[str],
                  source: str) -> Optional[YoloDetector]:
        if weights is None and injected is None:
            return None
        return YoloDetector(runtime=self._runtime(weights or "", injected),
                            class_map=class_map,
                            calibrator=ConfidenceCalibrator.load(calib_json),
                            tile=self.cfg.tile, overlap=self.cfg.tile_overlap,
                            min_conf=self.cfg.yolo_conf, source=source)

    def _build_rgb_branch(self, injected: Optional[YoloRuntime]):
        rules = RGBPanelAnalyzer()
        det = self._detector(
            self.cfg.rgb_weights, injected,
            self.cfg.rgb_class_map or load_class_map(self.cfg.rgb_class_map_json,
                                                     DEFAULT_RGB_CLASS_MAP),
            self.cfg.rgb_calibration_json, "rgb")
        if det is None:
            return rules
        yolo = YoloRGBAnalyzer(det, feature_extractor=rules)
        if self.cfg.hybrid_w_yolo is not None:
            return HybridBranchAnalyzer(yolo, rules, self.cfg.hybrid_w_yolo)
        return yolo

    def _build_ir_branch(self, injected: Optional[YoloRuntime]):
        rules = IRPanelAnalyzer(radiometric=self.cfg.radiometric,
                                min_irradiance_ok=self.cfg.irradiance_ok)
        det = self._detector(
            self.cfg.ir_weights, injected,
            self.cfg.ir_class_map or load_class_map(self.cfg.ir_class_map_json,
                                                    DEFAULT_IR_CLASS_MAP),
            self.cfg.ir_calibration_json, "ir")
        if det is None:
            return rules
        yolo = YoloIRAnalyzer(det, normalizer=self.thermal_normalizer,
                              radiometric=self.cfg.radiometric,
                              feature_extractor=rules,
                              iec_severity_check=self.cfg.iec_severity_check)
        if self.cfg.hybrid_w_yolo is not None:
            return HybridBranchAnalyzer(yolo, rules, self.cfg.hybrid_w_yolo)
        return yolo

    def _build_panel_detector(self, injected: Optional[YoloRuntime]):
        det = self._detector(self.cfg.panel_weights, injected,
                             {"panel": "__panel__", "module": "__panel__"},
                             None, "panel")
        if det is None:
            return None
        return YoloPanelDetector(det, array_grouping_px=self.cfg.array_grouping_px)

    def preflight(self) -> Dict:
        """
        추론 시작 전 자기 점검. 배치 작업 전에 한 번 호출하는 것을 권장한다.

        조용한 실패를 잡는 것이 목적이다.
          - 클래스명 오타로 매핑이 통째로 빠지면 파이프라인은 에러 없이 '정상'만 뱉는다
          - 보정 JSON 을 안 주면 YOLO 분기가 과신한 채로 융합된다
          - IR normalizer 설정이 학습과 다르면 가중치는 멀쩡한데 성능만 무너진다
        """
        info: Dict = {
            "rgb_branch": type(self.rgb_analyzer).__name__,
            "ir_branch": type(self.ir_analyzer).__name__,
            "panel_source": "yolo-seg" if self.panel_detector is not None
            else ("labelstudio" if self.cfg.labelstudio_json else "rule-based"),
            "thermal_normalizer": self.thermal_normalizer.to_dict(),
            "warnings": [],
        }
        for name, br in (("rgb", self.rgb_analyzer), ("ir", self.ir_analyzer)):
            det = getattr(getattr(br, "yolo", br), "detector", None)
            if det is None:
                continue
            pf = det.preflight()
            info[f"{name}_model"] = pf
            if pf.get("unmapped"):
                info["warnings"].append(
                    f"{name}: 매핑 안 된 모델 클래스 {pf['unmapped']} - "
                    f"이 클래스의 검출은 전부 버려진다")
            if pf.get("mapped") == {}:
                info["warnings"].append(f"{name}: 매핑된 클래스가 하나도 없다")
            if not pf.get("calibrated", False):
                info["warnings"].append(
                    f"{name}: conf 보정 미적용 - YOLO 분기가 과신한 채 융합된다")
        if self.cfg.ir_weights and not (self.cfg.thermal_normalizer
                                        or self.cfg.thermal_normalizer_json):
            info["warnings"].append(
                "IR 모델을 쓰면서 normalizer 설정을 주지 않았다 - "
                "학습 때와 같은 정규화인지 확인할 것")
        return info

    # -- 열화상 준비 ------------------------------------------------------ #
    def prepare_thermal(self, thermal_raw: np.ndarray) -> np.ndarray:
        if not self.cfg.radiometric:
            return thermal_raw.astype(np.float32)
        t = thermal_raw
        if t.dtype != np.float32 and t.dtype != np.float64:
            t = dn_to_celsius(t, self.cfg.dn_scale, self.cfg.dn_offset)
        return apply_emissivity_correction(np.asarray(t, np.float32),
                                           self.cfg.emissivity,
                                           self.cfg.reflected_temp_c)

    # -- 실행 -------------------------------------------------------------- #
    def run(self,
            rgb: np.ndarray,
            thermal_raw: np.ndarray,
            ir_preview: Optional[np.ndarray] = None,
            rois: Optional[List[PanelROI]] = None) -> PipelineResult:
        """
        rgb          : BGR uint8
        thermal_raw  : 16bit DN 또는 섭씨 float (radiometric=True 일 때)
        ir_preview   : 정합용 8bit 열화상 시각화 영상. 없으면 thermal_raw 에서 생성.
        rois         : 미리 정의된 패널 ROI. 없으면 RGB 에서 자동 검출.
        """
        thermal = self.prepare_thermal(thermal_raw)
        if ir_preview is None:
            ir_preview = _to_u8(thermal)

        # 1) 정합
        H_init = self.cfg.H_init
        if H_init is None and self.cfg.gt_rgb is not None and self.cfg.gt_ir is not None:
            from .registration import homography_from_geotransform
            H_init = homography_from_geotransform(self.cfg.gt_rgb, self.cfg.gt_ir)

        reg = register_pair(
            rgb, ir_preview,
            H_init=H_init,
            rgb_cam=self.cfg.rgb_cam, ir_cam=self.cfg.ir_cam,
            extrinsics=self.cfg.extrinsics,
            plane_normal=None if self.cfg.plane_normal is None
            else np.array(self.cfg.plane_normal, dtype=float),
            plane_distance=self.cfg.plane_distance_m,
            residual_tol_px=self.cfg.residual_tol_px,
        )
        reg_rel = reg.reliability(tol_px=self.cfg.residual_tol_px)

        # 2) 패널 ROI
        if rois is None:
            if self.cfg.labelstudio_json:
                rois = load_panels_from_labelstudio(
                    self.cfg.labelstudio_json, rgb.shape[1], rgb.shape[0])
            elif self.panel_detector is not None:
                rois = self.panel_detector(rgb)
                if not rois:                       # 세그 모델이 아무것도 못 찾으면 폴백
                    rois = detect_panels(rgb, array_grouping_px=self.cfg.array_grouping_px)
            else:
                rois = detect_panels(rgb, array_grouping_px=self.cfg.array_grouping_px)
        rois = project_rois_to_ir(rois, reg.H_rgb_to_ir, thermal.shape)

        # 3) 프레임 단위 추론 (YOLO 분기는 여기서 1회만 돌고 ROI 별로 나눠 쓴다)
        if hasattr(self.rgb_analyzer, "prepare"):
            self.rgb_analyzer.prepare(rgb, rois)
        if hasattr(self.ir_analyzer, "prepare"):
            self.ir_analyzer.prepare(thermal, rois)

        # 4) 어레이 기준값 (두 분기 모두 '주변 모듈 대비'로 판정한다)
        t_ref = self.ir_analyzer.array_reference(thermal, rois)
        c_ref = self.rgb_analyzer.array_reference(rgb, rois)

        # 5) 분기별 독립 분석 + late fusion
        verdicts: List[PanelVerdict] = []
        for roi in rois:
            rgb_ev = self.rgb_analyzer.analyze(
                rgb, roi, c_ref.get(roi.array_id, c_ref.get(None)))
            ir_ev = self.ir_analyzer.analyze(
                thermal, roi, t_ref.get(roi.array_id, t_ref.get(None)))
            verdicts.append(self.engine.decide(roi.panel_id, rgb_ev, ir_ev, reg_rel))

        result = PipelineResult(registration=reg, verdicts=verdicts,
                                summary=_summarize(verdicts, reg, reg_rel))
        result.summary.update(self._detector_diagnostics())
        result.rois = rois
        return result

    def _detector_diagnostics(self) -> Dict:
        """
        어떤 패널에도 귀속되지 않은 검출(orphan)과 미매핑 클래스를 보고한다.

        orphan 이 많다는 것은 보통 두 가지 중 하나다.
          - 패널 검출이 모듈을 놓쳤다 (결함은 봤는데 그 모듈이 ROI 에 없음)
          - 정합이 어긋나 IR 검출이 엉뚱한 곳에 떨어졌다
        둘 다 조용히 재현율을 깎으므로 요약에 항상 띄운다.
        """
        d: Dict = {}
        for name, br in (("rgb", self.rgb_analyzer), ("ir", self.ir_analyzer)):
            inner = getattr(br, "yolo", br)
            orphans = getattr(inner, "_orphans", None)
            if orphans is None:
                continue
            d[f"{name}_orphan_detections"] = len(orphans)
            det = getattr(inner, "detector", None)
            if det is not None:
                um = det.unmapped_classes(getattr(inner, "_dets", []))
                if um:
                    d[f"{name}_unmapped_classes"] = um
        return d


# --------------------------------------------------------------------------- #
# 보조
# --------------------------------------------------------------------------- #

def _to_u8(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    lo, hi = np.percentile(x[np.isfinite(x)], [1, 99]) if np.isfinite(x).any() else (0, 1)
    if hi - lo < 1e-9:
        return np.zeros(x.shape, np.uint8)
    return np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _summarize(verdicts: Sequence[PanelVerdict], reg: RegistrationResult,
               reg_rel: float) -> Dict:
    from collections import Counter
    c = Counter(v.label for v in verdicts)
    sev = Counter(v.severity for v in verdicts)
    return {
        "n_panels": len(verdicts),
        "registration_reliability": round(float(reg_rel), 3),
        "registration_usable": bool(reg.is_usable),
        "by_label": dict(c),
        "by_severity": {str(k): v for k, v in sorted(sev.items())},
        "n_ir_missing": sum(1 for v in verdicts
                            if v.ir is not None and v.ir.reliability == 0.0),
        "n_abstain": sum(1 for v in verdicts if v.label == "uncertain"),
        "mean_confidence": round(float(np.mean([v.confidence for v in verdicts])), 3)
        if verdicts else 0.0,
    }


SEVERITY_COLOR = {0: (90, 200, 90), 1: (60, 200, 240), 2: (40, 120, 255), 3: (40, 40, 235)}


def render_overlay(rgb: np.ndarray, thermal: np.ndarray, reg: RegistrationResult,
                   verdicts: Sequence[PanelVerdict], rois: Sequence[PanelROI],
                   alpha: float = 0.45) -> np.ndarray:
    """RGB 위에 열화상 블렌딩 + 패널별 판정 오버레이 (검수용)."""
    ih, iw = thermal.shape[:2]
    H_inv = np.linalg.inv(reg.H_rgb_to_ir)
    t8 = _to_u8(thermal)
    t_color = cv2.applyColorMap(t8, cv2.COLORMAP_INFERNO)
    t_on_rgb = cv2.warpPerspective(t_color, H_inv, (rgb.shape[1], rgb.shape[0]))
    valid = cv2.warpPerspective(np.ones((ih, iw), np.uint8), H_inv,
                                (rgb.shape[1], rgb.shape[0])) > 0

    out = rgb.copy()
    out[valid] = cv2.addWeighted(rgb, 1 - alpha, t_on_rgb, alpha, 0)[valid]

    by_id = {v.panel_id: v for v in verdicts}
    for roi in rois:
        v = by_id.get(roi.panel_id)
        if v is None:
            continue
        color = SEVERITY_COLOR.get(v.severity, (200, 200, 200))
        pts = np.round(roi.polygon_rgb).astype(np.int32)
        cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
        if v.label != "normal":
            org = tuple(pts.min(axis=0) + np.array([2, -4]))
            cv2.putText(out, f"{v.label} {v.confidence:.2f}", org,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _load_thermal(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# --------------------------------------------------------------------------- #
# 배치 추론
# --------------------------------------------------------------------------- #

_RGB_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def find_pairs(rgb_dir: str, ir_dir: Optional[str] = None,
               rgb_token: str = "_W", ir_token: str = "_T") -> List[Tuple[str, str]]:
    """
    RGB/IR 프레임 쌍을 찾는다.

    DJI 다중 센서 기체는 한 셔터에 `..._W.JPG`(광각) 와 `..._T.JPG`(열화상) 를
    같은 시퀀스 번호로 떨군다. 그래서 토큰 치환으로 짝을 맞추는 것이 가장 확실하다.
    토큰이 없으면 파일명 stem 이 같은 것끼리 맞춘다.
    """
    ir_dir = ir_dir or rgb_dir
    ir_index: Dict[str, str] = {}
    for f in sorted(os.listdir(ir_dir)):
        if f.lower().endswith(_RGB_EXT):
            ir_index[os.path.splitext(f)[0]] = os.path.join(ir_dir, f)

    pairs: List[Tuple[str, str]] = []
    for f in sorted(os.listdir(rgb_dir)):
        if not f.lower().endswith(_RGB_EXT):
            continue
        stem = os.path.splitext(f)[0]
        if rgb_token and rgb_token in stem:
            key = stem.replace(rgb_token, ir_token)
        elif stem in ir_index and os.path.join(rgb_dir, f) != ir_index[stem]:
            key = stem
        else:
            continue
        if key in ir_index:
            pairs.append((os.path.join(rgb_dir, f), ir_index[key]))
    return pairs


def run_batch(pipe: "RGBIRLateFusionPipeline",
              pairs: Sequence[Tuple[str, str]],
              out_dir: str,
              save_overlay: bool = True,
              on_error: str = "skip") -> Dict:
    """
    프레임 쌍 목록에 대해 추론을 돌리고 프레임별 리포트 + 사이트 집계를 쓴다.

    모델은 한 번만 로드되어 전 프레임에 재사용된다. 프레임 하나가 실패해도
    배치 전체가 멈추지 않는 것이 기본 동작이다(`on_error="raise"` 로 변경 가능).
    """
    from collections import Counter

    os.makedirs(out_dir, exist_ok=True)
    labels, sev = Counter(), Counter()
    frames: List[Dict] = []
    failures: List[Dict] = []
    reg_res: List[float] = []

    for rgb_path, ir_path in pairs:
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        try:
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(rgb_path)
            res = pipe.run(rgb, _load_thermal(ir_path))
        except Exception as e:
            if on_error == "raise":
                raise
            failures.append({"frame": stem, "error": f"{type(e).__name__}: {e}"})
            continue

        with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
            f.write(res.to_json())
        if save_overlay and res.rois:
            cv2.imwrite(os.path.join(out_dir, f"{stem}_overlay.jpg"),
                        render_overlay(rgb, pipe.prepare_thermal(_load_thermal(ir_path)),
                                       res.registration, res.verdicts, res.rois),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])

        labels.update(v.label for v in res.verdicts)
        sev.update(v.severity for v in res.verdicts)
        if np.isfinite(res.registration.residual_px):
            reg_res.append(float(res.registration.residual_px))
        frames.append({"frame": stem,
                       "n_panels": len(res.verdicts),
                       "registration_reliability":
                           res.summary.get("registration_reliability"),
                       "by_label": dict(Counter(v.label for v in res.verdicts)),
                       "defects": [v.to_dict() for v in res.verdicts
                                   if v.label not in ("normal",)]})

    summary = {
        "n_frames": len(frames),
        "n_failed": len(failures),
        "n_panels": int(sum(f["n_panels"] for f in frames)),
        "by_label": dict(labels),
        "by_severity": {str(k): v for k, v in sorted(sev.items())},
        "registration_residual_px": {
            "median": round(float(np.median(reg_res)), 3) if reg_res else None,
            "p90": round(float(np.percentile(reg_res, 90)), 3) if reg_res else None,
        },
        # 정합이 나쁜 프레임은 융합 결과를 믿지 말고 재비행/재처리 대상으로 본다
        "n_low_registration": sum(
            1 for f in frames if (f["registration_reliability"] or 0) < 0.45),
        "failures": failures,
        "frames": frames,
    }
    with open(os.path.join(out_dir, "site_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="RGB-IR 패널 정합 기반 Late Fusion 결함 분석")
    ap.add_argument("--rgb", help="RGB 파일 1장")
    ap.add_argument("--ir", help="열화상 1장 (16bit radiometric TIFF 권장)")
    ap.add_argument("--rgb-dir", help="배치: RGB 디렉터리")
    ap.add_argument("--ir-dir", help="배치: 열화상 디렉터리 (생략 시 --rgb-dir 과 동일)")
    ap.add_argument("--rgb-token", default="_W", help="배치 파일명 매칭 토큰")
    ap.add_argument("--ir-token", default="_T")
    ap.add_argument("--out", default="./out")
    ap.add_argument("--labels", default=None, help="Label Studio JSON (선택)")
    ap.add_argument("--fusion", default="ds", choices=["ds", "pool", "noisyor"])
    ap.add_argument("--no-radiometric", action="store_true")
    ap.add_argument("--dn-scale", type=float, default=0.04)
    ap.add_argument("--dn-offset", type=float, default=-273.15)
    ap.add_argument("--emissivity", type=float, default=0.85)
    ap.add_argument("--plane-distance", type=float, default=None,
                    help="LRF 또는 지면평면 거리 [m]")
    ap.add_argument("--rgb-hfov", type=float, default=None)
    ap.add_argument("--ir-hfov", type=float, default=None)
    ap.add_argument("--baseline", type=float, default=0.0,
                    help="RGB->IR 광축 간 수평 베이스라인 [m]")
    ap.add_argument("--rgb-weights", default=None, help="YOLOv11 RGB 가중치 (.pt)")
    ap.add_argument("--ir-weights", default=None, help="YOLOv11 열화상 가중치 (.pt)")
    ap.add_argument("--panel-weights", default=None, help="YOLOv11-seg 모듈 가중치 (.pt)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda:0 ...")
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--yolo-conf", type=float, default=0.15)
    ap.add_argument("--rgb-calib", default=None, help="RGB 신뢰도 보정 JSON")
    ap.add_argument("--ir-calib", default=None, help="IR 신뢰도 보정 JSON")
    ap.add_argument("--hybrid-w", type=float, default=None,
                    help="지정 시 YOLO+규칙 하이브리드. YOLO 가중치 0~1")
    ap.add_argument("--thermal-norm-json", default=None,
                    help="학습 시 저장한 normalizer.json (IR 모델과 함께 배포)")
    ap.add_argument("--thermal-norm", default="relative",
                    choices=["relative", "fixed", "percentile"])
    ap.add_argument("--thermal-vmin", type=float, default=10.0)
    ap.add_argument("--thermal-vmax", type=float, default=80.0)
    ap.add_argument("--rgb-class-map", default=None, help="RGB 클래스 매핑 JSON")
    ap.add_argument("--ir-class-map", default=None, help="IR 클래스 매핑 JSON")
    ap.add_argument("--preflight-only", action="store_true",
                    help="모델/매핑/보정 설정만 점검하고 종료")
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args(argv)

    batch = bool(args.rgb_dir)
    if not batch and not (args.rgb and args.ir):
        ap.error("--rgb/--ir (단일) 또는 --rgb-dir (배치) 중 하나는 필요합니다")

    cfg = PipelineConfig(
        radiometric=not args.no_radiometric,
        dn_scale=args.dn_scale, dn_offset=args.dn_offset,
        emissivity=args.emissivity,
        plane_distance_m=args.plane_distance,
        plane_normal=(0.0, 0.0, 1.0) if args.plane_distance else None,
        fusion_method=args.fusion,
        labelstudio_json=args.labels,
        save_overlay=not args.no_overlay,
        rgb_weights=args.rgb_weights,
        ir_weights=args.ir_weights,
        panel_weights=args.panel_weights,
        device=args.device,
        tile=args.tile,
        yolo_conf=args.yolo_conf,
        rgb_calibration_json=args.rgb_calib,
        ir_calibration_json=args.ir_calib,
        hybrid_w_yolo=args.hybrid_w,
        thermal_normalizer_json=args.thermal_norm_json,
        thermal_normalizer=None if args.thermal_norm_json else ThermalNormalizer(
            mode=args.thermal_norm, vmin=args.thermal_vmin, vmax=args.thermal_vmax),
        rgb_class_map_json=args.rgb_class_map,
        ir_class_map_json=args.ir_class_map,
    )
    first_rgb = args.rgb
    first_ir = args.ir
    pairs: List[Tuple[str, str]] = []
    if batch:
        pairs = find_pairs(args.rgb_dir, args.ir_dir, args.rgb_token, args.ir_token)
        if not pairs:
            print("짝이 맞는 RGB/IR 프레임을 찾지 못했습니다. "
                  "--rgb-token/--ir-token 을 확인하세요.")
            return 2
        first_rgb, first_ir = pairs[0]

    probe_rgb = cv2.imread(first_rgb, cv2.IMREAD_COLOR)
    if probe_rgb is None:
        raise FileNotFoundError(first_rgb)
    probe_ir = _load_thermal(first_ir)

    if args.rgb_hfov and args.ir_hfov:
        cfg.rgb_cam = CameraIntrinsics.from_fov(probe_rgb.shape[1], probe_rgb.shape[0],
                                                args.rgb_hfov)
        cfg.ir_cam = CameraIntrinsics.from_fov(probe_ir.shape[1], probe_ir.shape[0],
                                               args.ir_hfov)
        cfg.extrinsics = StereoExtrinsics(R=np.eye(3),
                                          t=np.array([args.baseline, 0.0, 0.0]))

    pipe = RGBIRLateFusionPipeline(cfg)

    pf = pipe.preflight()
    print(json.dumps(pf, ensure_ascii=False, indent=2))
    for w in pf.get("warnings", []):
        print(f"  [경고] {w}")
    if args.preflight_only:
        return 0

    os.makedirs(args.out, exist_ok=True)

    if batch:
        print(f"\n{len(pairs)} 쌍 추론 시작...")
        summary = run_batch(pipe, pairs, args.out, save_overlay=cfg.save_overlay)
        print(json.dumps({k: v for k, v in summary.items()
                          if k not in ("frames", "failures")},
                         ensure_ascii=False, indent=2))
        if summary["failures"]:
            print(f"실패 {len(summary['failures'])}건 - site_summary.json 참고")
        print(f"결과: {args.out}/site_summary.json")
        return 0

    rgb, thermal_raw = probe_rgb, probe_ir
    res = pipe.run(rgb, thermal_raw)

    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        f.write(res.to_json())

    if cfg.save_overlay and res.rois:
        cv2.imwrite(os.path.join(args.out, "overlay.png"),
                    render_overlay(rgb, pipe.prepare_thermal(thermal_raw),
                                   res.registration, res.verdicts, res.rois))

    print(json.dumps(res.summary, ensure_ascii=False, indent=2))
    print(f"\n[정합] method={res.registration.method} "
          f"residual={res.registration.residual_px:.2f}px "
          f"ncc={res.registration.ncc:.3f} "
          f"reliability={res.registration.reliability():.3f}")
    print(f"리포트: {os.path.join(args.out, 'report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""RGB-IR 동일 좌표 패널 정합 기반 Late Fusion 결함 분석."""

from .types import (CameraIntrinsics, GeoTransform, PanelROI, PanelVerdict,
                    BranchEvidence, RegistrationResult, StereoExtrinsics,
                    DEFECT_CLASSES, SEVERITY_BY_CLASS)
from .registration import (register_pair, plane_induced_homography,
                           homography_from_geotransform, warp_points,
                           warp_rgb_to_ir, crossmodal_feature)
from .panels import (detect_panels, project_rois_to_ir, dominant_grid_angle,
                     load_panels_from_labelstudio, crop_polygon)
from .rgb_branch import RGBPanelAnalyzer
from .ir_branch import IRPanelAnalyzer, dn_to_celsius, apply_emissivity_correction
from .fusion import (LateFusionEngine, DempsterShaferFusion, LogOpinionPoolFusion,
                     NoisyOrFusion, LearnedFusion, Rule, default_rules)
from .yolo_backend import (YoloDetector, YoloRuntime, UltralyticsRuntime,
                           YoloPanelDetector, ThermalNormalizer, Detection,
                           ConfidenceCalibrator, DEFAULT_RGB_CLASS_MAP,
                           DEFAULT_IR_CLASS_MAP, nms_per_class, load_class_map)
from .yolo_branch import (YoloRGBAnalyzer, YoloIRAnalyzer, HybridBranchAnalyzer,
                          assign_detections, detections_to_scores)
from .pipeline import (RGBIRLateFusionPipeline, PipelineConfig, PipelineResult,
                       render_overlay, run_batch, find_pairs)

__version__ = "0.1.0"

__all__ = [
    "CameraIntrinsics", "GeoTransform", "PanelROI", "PanelVerdict",
    "BranchEvidence", "RegistrationResult", "StereoExtrinsics",
    "DEFECT_CLASSES", "SEVERITY_BY_CLASS",
    "register_pair", "plane_induced_homography", "homography_from_geotransform",
    "warp_points", "warp_rgb_to_ir", "crossmodal_feature",
    "detect_panels", "project_rois_to_ir", "dominant_grid_angle",
    "load_panels_from_labelstudio", "crop_polygon",
    "RGBPanelAnalyzer", "IRPanelAnalyzer", "dn_to_celsius",
    "apply_emissivity_correction",
    "LateFusionEngine", "DempsterShaferFusion", "LogOpinionPoolFusion",
    "NoisyOrFusion", "LearnedFusion", "Rule", "default_rules",
    "RGBIRLateFusionPipeline", "PipelineConfig", "PipelineResult",
    "render_overlay", "run_batch", "find_pairs",
    # YOLOv11 (추론)
    "YoloDetector", "YoloRuntime", "UltralyticsRuntime", "YoloPanelDetector",
    "ThermalNormalizer", "Detection", "ConfidenceCalibrator",
    "DEFAULT_RGB_CLASS_MAP", "DEFAULT_IR_CLASS_MAP", "nms_per_class",
    "load_class_map",
    "YoloRGBAnalyzer", "YoloIRAnalyzer", "HybridBranchAnalyzer",
    "assign_detections", "detections_to_scores",
]

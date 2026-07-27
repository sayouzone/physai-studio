"""
dji_metadata_extractor.py

metadata.py(extract_metadata / ImageMetadata)를 **그대로 사용**하는 확장 레이어.
metadata.py는 수정하지 않고 import하며, 그 위에 다음을 추가한다:

  1. extract()      — extract_metadata() 호출 + 보강(augment):
                       · modality 판별 (_W/_Z/_T 파일명 -> ImageSource -> 해상도)
                       · 요소(element) 표기 XMP 재스캔으로 metadata.py가 놓친 GPS·자세 채움
                       · focal_length_in_35mm 부재 시 페이로드 스펙 폴백 주입
                         (metadata.py의 hfov/intrinsics ZeroDivision 방지)
                       · 경고는 meta.extractor_warnings (동적 속성)에 기록
  2. validate()     — georeferencing 투입 전 필수 필드 점검
  3. pair_rgb_ir()  — RGB(_W) / IR(_T) 페어링 + 시각·GPS 검증
  4. iter_unpaired()— 페어 실패 파일 열거 (import 시 unpaired 플래그용)
  5. to_georeferencing_dict() — Django GeoreferencingData 필드 매핑
  6. CLI            — 단건/디렉토리/페어링 점검

사용:
    from dji_metadata_extractor import extract, pair_rgb_ir, validate

    meta = extract('DJI_20260620103012_0042_W.JPG')   # ImageMetadata 반환
    problems = validate(meta)
    pairs = pair_rgb_ir('/data/flight_0620/')

CLI:
    python dji_metadata_extractor.py <file_or_dir> [--json] [--pairs]

의존성: metadata.py (같은 패키지/디렉토리), Pillow, numpy
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .image.metadata import (
    ImageMetadata,
    RTK_FIXED,
    M_PER_DEG_LAT,
    extract_metadata,
    estimate_intrinsics_from_metadata,
    _extract_xmp_block,   # 보강 재스캔에 재사용 (metadata.py 내부 헬퍼)
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 상수
# -----------------------------------------------------------------------------

_SUFFIX_MODALITY = {"W": "wide", "Z": "zoom", "T": "ir", "S": "screen"}

_FILENAME_RE = re.compile(
    r"DJI_(?P<ts>\d{14})_(?P<index>\d{4})_(?P<suffix>[WZTS])", re.IGNORECASE)

# FocalLengthIn35mmFilm이 EXIF에 없을 때의 폴백 (35mm 환산, 수평 FOV 기준 역산).
# ※ IR 값은 스펙의 40.6°를 수평 FOV로 해석한 값. DFOV(대각) 기준이면 ≈62가 맞으므로
#   실기체 EXIF 확인 또는 자체 캘리브레이션(camera_pose.py) 값으로 교체할 것.
FALLBACK_35MM_EQUIV = {"wide": 20, "zoom": 31, "ir": 49}

# RGB/IR 페어링 허용 오차
# metadata.py의 capture_time은 epoch '초' 단위이므로 시각 비교 정밀도가 1s다.
# 동시 셔터(H20T) 특성상 같은 초 또는 ±1초까지 정상으로 본다.
PAIR_TIME_TOLERANCE_S = 1
PAIR_POS_TOLERANCE_M = 1.0

# 요소(element) 표기 XMP: <drone-dji:X>+1.23</drone-dji:X>
_ELEM_RE = re.compile(r"<([\w\-]+:[\w\-]+)>([^<]*)</\1>")


# -----------------------------------------------------------------------------
# 보강(augmentation) 레이어
# -----------------------------------------------------------------------------

def _element_style_xmp(path: Path) -> dict[str, str]:
    """metadata.py의 속성 파서가 놓치는 요소 표기 필드를 재스캔한다."""
    try:
        text = _extract_xmp_block(path)
    except OSError:
        return {}
    if not text:
        return {}
    return {k: v for k, v in _ELEM_RE.findall(text)}


def _f(value: Optional[str], default: float = 0.0) -> float:
    if not value:
        return default
    try:
        return float(value.lstrip("+"))
    except ValueError:
        return default


def _detect_modality(meta: ImageMetadata, path: Path) -> str:
    # 1) 파일명 접미사 (가장 신뢰)
    m = _FILENAME_RE.search(path.name)
    if m:
        return _SUFFIX_MODALITY.get(m.group("suffix").upper(), "")
    # 2) ImageSource — metadata.py가 camera_model 접미사로 보존함
    src = (meta.payload_model or meta.camera_model or "").lower()
    if "infrared" in src:
        return "ir"
    if "wide" in src:
        return "wide"
    if "zoom" in src:
        return "zoom"
    # 3) 해상도 (H20T IR = 640x512)
    if meta.width == 640 and meta.height == 512:
        return "ir"
    return ""


def _augment(meta: ImageMetadata, path: Path) -> ImageMetadata:
    """metadata.py 결과를 비파괴적으로 보강한다.

    보강 내용은 meta의 기존 필드를 채우는 방식이며, 경고/판별 결과는
    동적 속성(extractor_warnings, modality_detected)으로 붙인다 —
    ImageMetadata 스키마(JSON 출력)는 변경하지 않는다.
    """
    warnings: list[str] = []

    # ---- modality ----
    modality = _detect_modality(meta, path)
    if not modality:
        warnings.append("modality 판별 실패 (파일명/ImageSource/해상도 모두 불명)")

    # ---- 오탈자·요소 표기 XMP 재스캔으로 결손 채움 ----
    needs_gps = meta.gps.lat == 0.0 or meta.gps.lng == 0.0
    needs_ori = meta.orientation == [0.0, 0.0, 0.0]
    needs_time = meta.capture_time == 0
    capture_time_source = "metadata" if not needs_time else ""
    if needs_gps or needs_ori or needs_time:
        elem = _element_style_xmp(path)
        if elem:
            if needs_gps:
                lat = _f(elem.get("drone-dji:GpsLatitude")
                         or elem.get("drone-dji:GpsLattitude"))
                lng = _f(elem.get("drone-dji:GpsLongitude")
                         or elem.get("drone-dji:GpsLongitude"))
                if lat and lng:
                    meta.gps.lat, meta.gps.lng = lat, lng
                    meta.position[0], meta.position[1] = lat, lng
                    meta.pos_info.pos[0], meta.pos_info.pos[1] = lat, lng
                    warnings.append("GPS를 요소표기/오탈자 XMP 재스캔으로 복구")
            if needs_ori:
                ypr = [
                    _f(elem.get("drone-dji:GimbalYawDegree")),
                    _f(elem.get("drone-dji:GimbalPitchDegree")),
                    _f(elem.get("drone-dji:GimbalRollDegree")),
                ]
                if any(ypr):
                    meta.orientation[:] = ypr
                    meta.pos_info.orientation[:] = ypr
                    warnings.append("짐벌 자세를 요소표기 XMP 재스캔으로 복구")
            if meta.relative_height == 0.0:
                rh = _f(elem.get("drone-dji:RelativeAltitude"))
                if rh:
                    meta.relative_height = rh
            if meta.rtk_flag == 0:
                rf = elem.get("drone-dji:RtkFlag")
                if rf:
                    meta.rtk_flag = int(_f(rf))
                    meta.rtk_active = meta.rtk_flag in (16, RTK_FIXED)
            if needs_time:
                cd = (elem.get("xmp:CreateDate")
                      or elem.get("xmp:ModifyDate"))
                if cd:
                    try:
                        meta.capture_time = int(
                            datetime.fromisoformat(cd).timestamp())
                        capture_time_source = "xmp_rescan"
                        warnings.append("촬영시각을 요소표기 XMP 재스캔으로 복구")
                    except ValueError:
                        warnings.append(f"XMP 시각 파싱 실패: {cd}")

    # ---- 35mm 환산 초점거리 폴백 주입 ----
    # metadata.py의 hfov_deg / estimate_intrinsics는 이 값이 0이면
    # ZeroDivisionError가 나므로, 여기서 미리 채워 안전하게 만든다.
    if meta.focal_length_in_35mm <= 0:
        fallback = FALLBACK_35MM_EQUIV.get(modality, 0)
        if fallback > 0:
            meta.focal_length_in_35mm = fallback
            warnings.append(
                f"FocalLengthIn35mmFilm 부재 — 페이로드 스펙 폴백 {fallback}mm 주입")
        else:
            warnings.append(
                "FocalLengthIn35mmFilm 부재 + 폴백 불가 — FOV/intrinsics 계산 불가")

    # ---- 촬영시각 파일명 폴백 ----
    # 주의: 파일명 시각은 naive(드론 로컬 시각)라 타임존 포함 XMP epoch과
    # 직접 비교할 수 없다. 출처를 기록해 페어링에서 구분한다.
    if meta.capture_time == 0:
        m = _FILENAME_RE.search(path.name)
        if m:
            dt = datetime.strptime(m.group("ts"), "%Y%m%d%H%M%S")
            meta.capture_time = int(dt.timestamp())
            capture_time_source = "filename"
            warnings.append("촬영시각을 파일명에서 추출 (초 단위, naive 시각)")

    # ---- 동적 속성 부착 (스키마 비변경) ----
    meta.modality_detected = modality          # type: ignore[attr-defined]
    meta.capture_time_source = capture_time_source  # type: ignore[attr-defined]
    meta.extractor_warnings = warnings         # type: ignore[attr-defined]
    return meta


# -----------------------------------------------------------------------------
# 공개 API
# -----------------------------------------------------------------------------

def extract(image_path: str | Path,
            origin_path: str | None = None) -> ImageMetadata:
    """metadata.extract_metadata() + 보강. 반환 타입은 동일한 ImageMetadata."""
    path = Path(image_path)
    meta = extract_metadata(path, origin_path=origin_path)
    return _augment(meta, path)


def modality_of(meta: ImageMetadata) -> str:
    """extract()로 얻은 meta의 판별 modality. (없으면 빈 문자열)"""
    return getattr(meta, "modality_detected", "")


def warnings_of(meta: ImageMetadata) -> list[str]:
    return list(getattr(meta, "extractor_warnings", []))


def validate(meta: ImageMetadata) -> list[str]:
    """georeferencing 투입 전 필수 필드 점검. 문제 목록 반환(빈 리스트=OK)."""
    problems: list[str] = []
    if meta.gps.lat == 0.0 or meta.gps.lng == 0.0:
        problems.append("GPS 좌표 없음")
    if meta.orientation == [0.0, 0.0, 0.0]:
        problems.append("짐벌 자세 없음 (XMP 파싱 실패 가능성)")
    if (meta.relative_height == 0.0 and meta.gps.altitude == 0.0
            and not meta.has_valid_lrf):
        problems.append("고도 정보 없음 (relative/absolute/LRF 모두 부재)")
    if meta.rtk_flag != RTK_FIXED:
        problems.append(f"RTK 미고정 (flag={meta.rtk_flag}) — 위치 정확도 저하")
    if meta.capture_time == 0:
        problems.append("촬영시각 없음 — RGB/IR 페어링 불가")
    if meta.focal_length_in_35mm <= 0:
        problems.append("초점거리(35mm 환산) 없음 — FOV/intrinsics 계산 불가")
    return problems


def safe_intrinsics(meta: ImageMetadata):
    """estimate_intrinsics_from_metadata의 안전 래퍼.

    보강 레이어가 폴백을 주입했어도 남는 예외 상황(width=0 등)을
    ZeroDivisionError 대신 명시적 ValueError로 바꾼다.
    """
    if meta.focal_length_in_35mm <= 0 or meta.width <= 0:
        raise ValueError(
            f"intrinsics 추정 불가: f35={meta.focal_length_in_35mm}, "
            f"width={meta.width} ({meta.origin_path})")
    return estimate_intrinsics_from_metadata(meta)


# -----------------------------------------------------------------------------
# RGB / IR 페어링
# -----------------------------------------------------------------------------

def _approx_dist_m(a: ImageMetadata, b: ImageMetadata) -> Optional[float]:
    if 0.0 in (a.gps.lat, a.gps.lng, b.gps.lat, b.gps.lng):
        return None
    return math.hypot(
        (a.gps.lat - b.gps.lat) * M_PER_DEG_LAT,
        (a.gps.lng - b.gps.lng) * a.m_per_deg_lon,
    )


def _index_of(meta: ImageMetadata) -> Optional[str]:
    m = _FILENAME_RE.search(Path(meta.origin_path).name)
    return m.group("index") if m else None


def pair_rgb_ir(
    directory: str | Path,
    rgb_modality: str = "wide",
) -> list[tuple[ImageMetadata, ImageMetadata, list[str]]]:
    """폴더 내 RGB(_W)와 IR(_T)을 매칭한다.

    1차 키: 파일명 인덱스(DJI_..._NNNN_) — H20T는 동시 셔터라 인덱스가 같음
    2차 검증: 촬영시각 차 <= 1s (capture_time이 초 단위), GPS 거리 <= 1m

    Returns:
        [(rgb_meta, ir_meta, issues), ...]  issues 빈 리스트 = 정상 페어
    """
    directory = Path(directory)
    metas = [extract(p) for p in sorted(directory.iterdir())
             if p.suffix.lower() in (".jpg", ".jpeg")]

    rgb = {_index_of(m): m for m in metas if modality_of(m) == rgb_modality}
    ir = {_index_of(m): m for m in metas if modality_of(m) == "ir"}

    pairs = []
    for idx, r in rgb.items():
        t = ir.get(idx)
        if t is None:
            continue
        issues: list[str] = []
        r_src = getattr(r, "capture_time_source", "")
        t_src = getattr(t, "capture_time_source", "")
        comparable = (r.capture_time and t.capture_time
                      and ("filename" not in (r_src, t_src)
                           or r_src == t_src))
        if comparable:
            dt = abs(r.capture_time - t.capture_time)
            if dt > PAIR_TIME_TOLERANCE_S:
                issues.append(f"촬영시각 차 {dt}s > {PAIR_TIME_TOLERANCE_S}s")
        # 출처가 섞이면(한쪽만 파일명 naive) epoch 비교 불가 — GPS 검증에 의존
        dist = _approx_dist_m(r, t)
        if dist is not None and dist > PAIR_POS_TOLERANCE_M:
            issues.append(f"GPS 거리 {dist:.2f}m > {PAIR_POS_TOLERANCE_M}m")
        pairs.append((r, t, issues))
    return pairs


def iter_unpaired(directory: str | Path) -> Iterator[ImageMetadata]:
    """페어를 찾지 못한 파일들 (import 시 'unpaired' 플래그 처리용)."""
    directory = Path(directory)
    paired: set[str] = set()
    for r, t, _ in pair_rgb_ir(directory):
        paired.add(Path(r.origin_path).name)
        paired.add(Path(t.origin_path).name)
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        if p.name not in paired:
            yield extract(p)


# -----------------------------------------------------------------------------
# Django GeoreferencingData 매핑
# -----------------------------------------------------------------------------

def to_georeferencing_dict(meta: ImageMetadata) -> dict:
    """extract_exif_and_populate async job에서 GeoreferencingData 필드로
    바로 넣을 수 있는 평탄화 dict.

    LRF가 유효하면 ground_alt로 LRF 실측 절대고도를 제공한다 —
    (altitude - relative_height)보다 정확한 지면고도 소스.
    """
    ground_alt = (meta.lrf_target_abs_alt
                  if meta.has_valid_lrf
                  else meta.gps.altitude - meta.relative_height)
    return {
        "rtk_lon": meta.gps.lng,
        "rtk_lat": meta.gps.lat,
        "rtk_alt": meta.gps.altitude,
        "relative_height": meta.relative_height,
        "ground_alt": ground_alt,
        "ground_alt_source": "lrf" if meta.has_valid_lrf else "relative",
        "yaw": meta.gimbal_yaw_deg,
        "pitch": meta.gimbal_pitch_deg,
        "roll": meta.gimbal_roll_deg,
        "flight_yaw": meta.flight[0] if len(meta.flight) >= 3 else 0.0,
        "rtk_flag": meta.rtk_flag,
        "rtk_std_lat": meta.rtk_std[0] if len(meta.rtk_std) >= 3 else None,
        "rtk_std_lon": meta.rtk_std[1] if len(meta.rtk_std) >= 3 else None,
        "rtk_std_hgt": meta.rtk_std[2] if len(meta.rtk_std) >= 3 else None,
        "captured_at": (datetime.fromtimestamp(meta.capture_time)
                        if meta.capture_time else None),
        "camera_model": meta.camera_model,
        "focal_length_mm": meta.focal_length,
        "focal_length_35mm": meta.focal_length_in_35mm,
        "image_width": meta.width,
        "image_height": meta.height,
        "modality": modality_of(meta),
        "image_sha1": meta.id,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _main() -> int:
    ap = argparse.ArgumentParser(description="DJI 메타데이터 추출 (metadata.py 기반)")
    ap.add_argument("target", help="이미지 파일 또는 디렉토리")
    ap.add_argument("--json", action="store_true", help="JSON 출력")
    ap.add_argument("--pairs", action="store_true",
                    help="디렉토리에서 RGB/IR 페어링 결과 출력")
    args = ap.parse_args()

    target = Path(args.target)

    if args.pairs and target.is_dir():
        pairs = pair_rgb_ir(target)
        for r, t, issues in pairs:
            flag = " ⚠ " + "; ".join(issues) if issues else " ✓"
            print(f"{Path(r.origin_path).name}  <->  "
                  f"{Path(t.origin_path).name}{flag}")
        print(f"-- {len(pairs)} pairs")
        return 0

    files = ([target] if target.is_file()
             else sorted(p for p in target.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg")))
    for f in files:
        meta = extract(f)
        if args.json:
            d = meta.to_dict()
            d.pop("R_cam_to_enu", None)      # ndarray 직렬화 방지
            d["modality"] = modality_of(meta)
            d["extractor_warnings"] = warnings_of(meta)
            print(json.dumps(d, ensure_ascii=False, indent=2, default=str))
        else:
            problems = validate(meta)
            status = "OK" if not problems else "; ".join(problems)
            warn = warnings_of(meta)
            warn_s = f"  [warn: {'; '.join(warn)}]" if warn else ""
            print(f"{Path(meta.origin_path).name} [{modality_of(meta)}] "
                  f"({meta.gps.lat}, {meta.gps.lng}) "
                  f"rel_h={meta.relative_height} ypr={meta.orientation} "
                  f"rtk={meta.rtk_flag} lrf_alt={meta.lrf_target_abs_alt} "
                  f"-> {status}{warn_s}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

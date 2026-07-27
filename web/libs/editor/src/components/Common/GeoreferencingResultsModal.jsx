// web/libs/editor/src/components/Common/GeoreferencingResultsModal.jsx
//
// 폴리곤 커버리지 검색 결과 모달.
// 항목 클릭 시 해당 Task를 새 탭으로 열되, "원본 region의 GPS 폴리곤"을
// URL에 실어 보내 새 탭 이미지 위에 참조 사각형으로 그려지게 한다.

import { useMemo } from "react";
import { Modal } from "apps/labelstudio/src/components/Modal/ModalPopup";

import { buildTaskUrlWithGpsRef } from "../ImageView/gpsRefUrl";

const CARD = {
  display: "flex",
  gap: 12,
  alignItems: "center",
  padding: "10px 12px",
  marginTop: 8,
  border: "1px solid #e4e7ec",
  borderRadius: 10,
  cursor: "pointer",
  background: "#fff",
};

// 참조 사각형 색 — 열리는 이미지의 modality에 맞춰 선택한다.
//
// IR(열화상)은 DJI ironbow 팔레트라 화면이 보라(#200050)·주황(#F07000)·
// 노랑(#F0F050)·백색고온부로 채워지고, Region 색으로 이미 파랑(Panel)과
// 녹색(Hotspot)을 쓰고 있다. 참조색은 이 6가지 모두와 구분돼야 한다.
//
// 실측 근거 (이 프로젝트 IR 샘플 기준, CIE Lab ΔE):
//   빨강 #F04438 : 주황 패널과 ΔE 13.6 → 묻힘
//   시안 #00E5FF : Panel(파랑)과 ΔE 35.5, 색맹 시 24.9 → 혼동
//   마젠타 #FF00FF: 색맹 시 Panel과 ΔE 33.1 → 혼동
//   흰색 #FFFFFF : 백색 고온부와 ΔE 31.8 → 묻힘
//   연핑크 #FFB2D2: 최소 ΔE 57.7 (모든 배경/Region/색맹 통과) → 채택
const REF_COLOR_IR = "#FFB2D2"; // 연핑크 — IR 팔레트·Panel·Hotspot 모두와 구분
const REF_COLOR_RGB = "#F04438"; // 빨강 — RGB 이미지용 (패널이 짙은 남색이라 잘 보임)

/** 열리는 이미지의 modality에 맞는 참조 사각형 색 */
function refColorFor(modality) {
  return String(modality ?? "").toLowerCase() === "ir" ? REF_COLOR_IR : REF_COLOR_RGB;
}

function coverageColor(ratio) {
  if (ratio >= 0.999) return { bg: "#ecfdf3", fg: "#067647" };
  if (ratio >= 0.5) return { bg: "#fffaeb", fg: "#b54708" };
  return { bg: "#fef3f2", fg: "#b42318" };
}

function Badge({ children, bg, fg, title }) {
  return (
    <span
      title={title}
      style={{
        fontSize: 12,
        fontWeight: 600,
        padding: "2px 8px",
        borderRadius: 999,
        background: bg,
        color: fg,
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </span>
  );
}

/**
 * @param {object}  result            검색 API 응답
 * @param {string}  [error]
 * @param {number}  [currentTaskId]   현재 task (결과에서 제외)
 * @param {Array}   [sourceCoordinates] 원본 region GPS 폴리곤 [[lng,lat],...]
 *                                     — 없으면 result에서 추출 시도
 * @param {string}  [sourceLabel]     참조 사각형 라벨 (예: "hot_spot #117")
 * @param {Function} onHide
 */
export function GeoreferencingResultsModal({
  result,
  error,
  currentTaskId,
  sourceCoordinates,
  sourceLabel,
  onHide,
}) {
  console.log('sourceCoordinates', sourceCoordinates);
  console.log('sourceLabel', sourceLabel);
  // 결과 배열 흡수 (래핑 형태 무관) + 현재 task 제외 + 커버리지 내림차순
  const images = useMemo(() => {
    if (!result) return [];
    const arr = Array.isArray(result)
      ? result
      : (result.results ?? result.images ?? result.hits ?? []);
    return [...arr]
      .filter((h) => h.task_id == null || h.task_id !== currentTaskId)
      .sort((a, b) => (b.coverage_ratio ?? 0) - (a.coverage_ratio ?? 0));
  }, [result, currentTaskId]);

  // 참조로 그릴 GPS 폴리곤: props 우선, 없으면 응답에서 탐색
  const refCoords = useMemo(() => {
    if (Array.isArray(sourceCoordinates) && sourceCoordinates.length >= 3) {
      return sourceCoordinates;
    }
    const r = result?.region ?? result?.source_region;
    if (Array.isArray(r?.coordinates) && r.coordinates.length >= 3) return r.coordinates;
    // GeoJSON Polygon 형태 { type:'Polygon', coordinates:[[[lng,lat],...]] }
    if (Array.isArray(r?.coordinates?.[0])) return r.coordinates[0];
    return null;
  }, [sourceCoordinates, result]);

  const refLabel = useMemo(() => {
    if (sourceLabel) return sourceLabel;
    const cls = result?.region?.defect_class;
    return cls ? `${cls} (Task #${currentTaskId ?? "?"})` : "참조 영역";
  }, [sourceLabel, result, currentTaskId]);

  const openTask = (img) => {
    if (!img.task_id) return;
    const url = buildTaskUrlWithGpsRef({
      taskId: img.task_id,
      projectId: img.project_id,
      coordinates: refCoords,   // null이면 참조 없이 그냥 task만 열림
      label: refLabel,
      color: refColorFor(img.modality),
    });
    window.open(url, "_blank", "noopener");
  };

  const subtitle = !error
    ? `${images.length}개${currentTaskId ? ` · 현재 Task #${currentTaskId} 제외` : ""}`
    : null;

  return (
    <Modal
      visible
      onHide={onHide}
      title="이 영역을 포함하는 이미지"
      style={{ width: "min(720px, 92vw)", maxHeight: "84vh", overflow: "auto" }}
      closeOnClickOutside
      allowClose
    >
      {subtitle && (
        <div style={{ marginTop: -8, marginBottom: 8, fontSize: 13, color: "#667085" }}>
          {subtitle}
          {refCoords && (
            <span style={{ marginLeft: 8, color: "#b42318" }}>
              · 열린 이미지에 참조 영역이 표시됩니다
            </span>
          )}
        </div>
      )}

      {error && (
        <div style={{ padding: "10px 12px", borderRadius: 8, background: "#fef3f2", color: "#b42318" }}>
          {error}
        </div>
      )}

      {!error && images.length === 0 && (
        <p style={{ color: "#667085" }}>
          이 영역을 커버하는 다른 이미지가 없습니다. 검색 임계(min_coverage)를 낮추거나
          반경 검색을 시도해 보세요.
        </p>
      )}

      {images.map((img) => {
        const ratio = img.coverage_ratio ?? 0;
        const cov = coverageColor(ratio);
        return (
          <div
            key={img.id ?? `${img.task_id}-${img.filename}`}
            role="button"
            tabIndex={0}
            onClick={() => openTask(img)}
            onKeyDown={(e) => e.key === "Enter" && openTask(img)}
            style={CARD}
          >
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span
                  style={{
                    fontWeight: 600,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                  title={img.filename}
                >
                  {img.filename ?? `이미지 ${img.id?.slice(0, 8) ?? ""}`}
                </span>
                {img.modality && (
                  <Badge bg="#eff8ff" fg="#175cd3">
                    {img.modality.toUpperCase()}
                  </Badge>
                )}
                {refCoords && (
                  <span
                    title={`이 이미지에서는 참조 영역이 ${refColorFor(img.modality)} 색으로 표시됩니다`}
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: 3,
                      background: refColorFor(img.modality),
                      border: "1px solid rgba(0,0,0,0.15)",
                      flexShrink: 0,
                    }}
                  />
                )}
              </div>

              <div
                style={{
                  marginTop: 4,
                  fontSize: 12.5,
                  color: "#475467",
                  display: "flex",
                  gap: 10,
                  flexWrap: "wrap",
                }}
              >
                {img.task_id != null && <span>Task #{img.task_id}</span>}
                {img.rtk_flag != null && (
                  <span style={{ color: img.rtk_flag === 50 ? "#067647" : "#b54708" }}>
                    {img.rtk_flag === 50 ? "RTK Fixed" : `RTK flag ${img.rtk_flag}`}
                  </span>
                )}
                {img.low_confidence && (
                  <span title="footprint 신뢰 낮음 (near-horizon 등)" style={{ color: "#b54708" }}>
                    ⚠ 저신뢰
                  </span>
                )}
              </div>
            </div>

            <Badge bg={cov.bg} fg={cov.fg} title={`이 영역의 ${(ratio * 100).toFixed(1)}%가 이 이미지에 포함됨`}>
              {img.fully_covered ? "전체 포함" : `${Math.round(ratio * 100)}% 포함`}
            </Badge>

            <span style={{ color: "#98a2b3", fontSize: 18 }}>›</span>
          </div>
        );
      })}

      {!error && images.length > 0 && (
        <p style={{ marginTop: 14, fontSize: 12, color: "#98a2b3" }}>
          항목을 클릭하면 해당 Task가 새 탭에서 열리고, 이미지 위에 이 영역이
          빨간 참조 사각형으로 표시됩니다.
        </p>
      )}
    </Modal>
  );
}

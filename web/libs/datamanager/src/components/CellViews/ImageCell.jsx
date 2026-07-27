import { getRoot } from "mobx-state-tree";
import { AnnotationPreview } from "../Common/AnnotationPreview/AnnotationPreview";

const imgDefaultProps = { crossOrigin: "anonymous" };

export const ImageCell = (column) => {
  const {
    original,
    value,
    column: { alias },
  } = column;
  const root = getRoot(original);

  const renderImagePreview = original.total_annotations === 0 || !root.showPreviews;
  const imgSrc = Array.isArray(value) ? value[0] : value;

  // ✅ 제외 여부 확인
  const isExcluded = original.exclude_from_export === true;

  // ✅ 파일 경로 추출 (URL이면 경로만, 로컬이면 그대로)
  const getFilePath = (src) => {
    if (!src) return "";
    try {
      const url = new URL(src);
      return decodeURIComponent(url.pathname);  // URL이면 pathname만
    } catch {
      return src.replace("/data/upload/", "");  // 로컬 경로면 그대로
    }
  };

  const tooltipText = getFilePath(imgSrc);

  if (!imgSrc) return null;

  const imageStyle = {
    maxHeight: "100%",
    maxWidth: "100px",
    objectFit: "contain",
    borderRadius: 3,
    // ✅ 제외된 이미지는 흐리게 + 채도 낮춤
    filter: isExcluded ? "blur(1.5px) grayscale(60%) brightness(0.75)" : "none",
    transition: "filter 0.15s ease",
  };

  return (
    <div title={tooltipText} style={{ position: "relative", display: "inline-block" }}>
      {renderImagePreview ? (
        <img
          {...imgDefaultProps}
          key={imgSrc}
          src={imgSrc}
          alt="Data"
          loading="lazy"
          style={imageStyle}
        />
      ) : (
        <AnnotationPreview
          task={original}
          annotation={original.annotations[0]}
          config={getRoot(original).SDK}
          name={alias}
          variant="120x120"
          fallbackImage={value}
          style={imageStyle}
        />
      )}

      {/* ✅ 제외 표시 오버레이 — 오른쪽 상단 배지 */}
      {isExcluded && (
        <div
          style={{
            position: "absolute",
            top: 2,
            right: 2,
            width: 18,
            height: 18,
            borderRadius: "50%",
            background: "rgba(220, 38, 38, 0.9)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 1px 3px rgba(0,0,0,0.4)",
            pointerEvents: "none",
          }}
        >
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="4">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </div>
      )}
    </div>
  );
};

// web/libs/editor/src/components/ImageView/GpsOverlayLayer.jsx
//
// GPS 좌표(폴리곤/점)를 현재 이미지의 픽셀로 역변환해 LSF Konva Stage 위에
// 오버레이로 그린다. result JSON에 쓰지 않는 순수 표시(참조)용 레이어.
//
// LSF 정합 원리 (ImageView.jsx의 Layer 패턴과 동일):
//   <Layer scale={{ x: item.stageZoom, y: item.stageZoom }}> 안에서는
//   원본 이미지 픽셀 좌표를 그대로 쓰면 Stage의 zoom/pan에 자동 정합된다.
//
// 삽입 위치:
//   StageContent(ImageView.jsx)의 Layer 나열에서 <Selection ...> 앞에 추가.
//
// 2026-07 수정
// ────────────
// 1) 훅 순서 버그: `if (!isAlive(item)) return null;` 가 useFootprint/useMemo
//    앞에 있었다. item이 죽는 렌더에서 훅 개수가 달라져
//    "Rendered fewer hooks than expected"로 크래시할 수 있다.
//    → 모든 훅을 무조건 호출하고, 렌더 단계에서만 분기한다.
//
// 2) modality 교차 폴백 제거: RGB footprint가 없을 때 IR footprint로
//    폴백하고 있었다. 두 카메라는 FOV·광축이 달라(H20T zoom 40.4° vs
//    IR 광각) 결과가 조용히 크게 틀린다. "아무것도 안 보이는 것보다 낫다"가
//    성립하지 않는 기능이다 — 참조 위치가 틀리면 잘못된 패널을 보게 된다.
//    → 기본 비활성. 필요하면 allowModalityFallback prop으로 명시 opt-in.
//
// 3) 역변환 실패/프레임 밖 사유를 onStatus로 노출하고, 프레임 밖일 때는
//    가장자리에 방향 인디케이터를 그린다(그냥 사라지면 원인을 알 수 없다).
//
// 4) footprint 유효성 검사(퇴화·(0,0) 코너·면적)와 이미지 크기 정합 확인.
//
// 정확도 주의:
//   이 레이어의 정확도는 백엔드 footprint에 100% 종속된다. footprint가
//   구버전(이방성 초점거리 / 패널면이 아닌 지면 평면) 파이프라인 산출물이면
//   여기서 아무리 정확히 역변환해도 계통 오차가 남는다.
//   → 백엔드 georeferencing 재생성이 선행되어야 한다.

import { observer } from "mobx-react";
import { getRoot, isAlive } from "mobx-state-tree";
import { useEffect, useMemo, useState } from "react";
import { Circle, Group, Layer, Line, Text } from "react-konva";

import { checkFootprintScale, projectGpsPolygon } from "./gpsToPixel";

/** decodeURIComponent 실패해도 원본 반환 */
function safeDecode(s) {
  try {
    return decodeURIComponent(s);
  } catch {
    return s;
  }
}

/**
 * 현재 표시 중인 이미지의 modality를 판별한다.
 * DJI 파일명 접미사: _W(wide) / _Z(zoom) / _T(thermal=IR)
 * 판별 실패 시 null (호출부에서 폴백 처리).
 */
function detectModality(item) {
  const src =
    item?.currentImageEntity?.src ??
    item?.currentSrc ??
    item?.src ??
    "";
  if (typeof src !== "string") return null;

  // 파일명 추출.
  // 주의: PhysAI Studio local-files 서빙은 파일 경로가 쿼리에 있다
  //   /data/local-files/?d=solar%2FDJI_0101_T.JPG
  let name = "";
  const dMatch = /[?&]d=([^&]+)/.exec(src);
  if (dMatch) {
    name = safeDecode(dMatch[1]);
  } else {
    name = safeDecode(src).split("?")[0];
  }
  name = name.split("/").pop() ?? "";

  const m = /_([WZTS])\.(jpe?g|png|tiff?)$/i.exec(name);
  if (m) {
    const s = m[1].toUpperCase();
    if (s === "T") return "ir";
    if (s === "W" || s === "Z") return "rgb";
    return null; // S(screenshot) 등
  }

  // 파일명으로 판별 안 되면 해상도로 추정 (H20T IR = 640x512)
  const w = item?.currentImageEntity?.naturalWidth;
  const h = item?.currentImageEntity?.naturalHeight;
  if (w === 640 && h === 512) return "ir";

  return null;
}

/** #rrggbb + 알파(0~1) -> #rrggbbaa (Konva fill이 8자리 hex 알파 지원) */
function withAlpha(hex, alpha) {
  const a = Math.round(Math.max(0, Math.min(1, alpha)) * 255)
    .toString(16)
    .padStart(2, "0");
  return `${hex}${a}`;
}

/**
 * 현재 task의 georeferencing footprint를 백엔드에서 가져온다.
 *
 * 응답에서 쓰는 필드:
 *   footprint_rgb / footprint_ir : GeoJSON Polygon
 *   pixel_corners_rgb / _ir      : (선택) footprint 코너에 대응하는 픽셀 좌표.
 *                                  없으면 (0,0),(W,0),(W,H),(0,H) 가정.
 *   image_size_rgb / _ir         : (선택) [W, H]. 프론트 naturalWidth와 대조용.
 *   ground_plane                 : (선택) {altitude_m, source, ...} 진단용.
 *
 * @param {number} taskId
 * @param {"rgb"|"ir"} modality
 * @param {boolean} allowFallback  다른 modality footprint로 대체할지 (기본 false)
 */
function useFootprint(taskId, modality, allowFallback) {
  const [state, setState] = useState({
    ring: null,
    pixelCorners: null,
    imageSize: null,
    plane: null,
    usedModality: null,
    error: null,
  });

  useEffect(() => {
    if (!taskId) {
      setState({
        ring: null, pixelCorners: null, imageSize: null,
        plane: null, usedModality: null, error: "task id 없음",
      });
      return;
    }

    const controller = new AbortController();

    fetch(`/api/tasks/${taskId}/footprint/`, {
      signal: controller.signal,
      headers: {
        "X-CSRFToken": document.cookie.match(/csrftoken=([^;]+)/)?.[1] ?? "",
      },
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!data) {
          setState({
            ring: null, pixelCorners: null, imageSize: null,
            plane: null, usedModality: null,
            error: "footprint 응답 없음 (404/권한?)",
          });
          return;
        }

        const primary = modality === "ir" ? data.footprint_ir : data.footprint_rgb;
        const other = modality === "ir" ? data.footprint_rgb : data.footprint_ir;

        let fp = primary;
        let used = modality;

        if (!primary) {
          if (allowFallback && other) {
            // RGB와 IR은 FOV/광축이 다르다. 위치가 눈에 띄게 어긋나므로
            // 명시적으로 켠 경우에만 허용하고 경고를 남긴다.
            fp = other;
            used = modality === "ir" ? "rgb" : "ir";
            console.warn(
              "[gps-overlay] %s footprint 없어 %s로 폴백 — FOV가 달라 위치가 어긋납니다. task:",
              modality, used, taskId,
            );
          } else {
            setState({
              ring: null, pixelCorners: null, imageSize: null,
              plane: data.ground_plane ?? null, usedModality: null,
              error: `${modality} footprint 없음`,
            });
            return;
          }
        }

        const ring = fp?.coordinates?.[0];
        if (!ring || ring.length < 4) {
          setState({
            ring: null, pixelCorners: null, imageSize: null,
            plane: data.ground_plane ?? null, usedModality: null,
            error: "footprint 폴리곤이 비어있음",
          });
          return;
        }

        const cornersKey = used === "ir" ? "pixel_corners_ir" : "pixel_corners_rgb";
        const sizeKey = used === "ir" ? "image_size_ir" : "image_size_rgb";

        setState({
          ring: ring.slice(0, 4),
          pixelCorners: Array.isArray(data[cornersKey]) ? data[cornersKey] : null,
          imageSize: Array.isArray(data[sizeKey]) ? data[sizeKey] : null,
          plane: data.ground_plane ?? null,
          usedModality: used,
          error: null,
        });
      })
      .catch((e) => {
        if (e.name === "AbortError") return;
        setState({
          ring: null, pixelCorners: null, imageSize: null,
          plane: null, usedModality: null, error: String(e),
        });
      });

    return () => controller.abort();
  }, [taskId, modality, allowFallback]);

  return state;
}

/**
 * @param {object} item      LSF image item (store)
 * @param {Array}  overlays  [{ id, coordinates:[[lng,lat],...], label?, color? }]
 * @param {(id:string)=>void} [onSelect]
 * @param {"rgb"|"ir"} [modality]
 * @param {boolean} [allowModalityFallback=false]
 *        RGB/IR footprint 교차 사용 허용. 위치가 어긋나므로 기본 false.
 * @param {boolean} [showOffscreenIndicator=true]
 *        참조 영역이 프레임 밖일 때 가장자리에 방향 표시.
 * @param {(status:object)=>void} [onStatus]
 *        {overlayId, state:"drawn"|"offscreen"|"failed", reason?} 통지.
 */
export const GpsOverlayLayer = observer(({
  item,
  overlays = [],
  onSelect,
  modality,
  allowModalityFallback = false,
  showOffscreenIndicator = true,
  onStatus,
}) => {
  // ── 훅은 무조건, 항상 같은 순서로 호출한다 (조건부 return 금지) ──
  const alive = isAlive(item);
  const taskId = alive ? getRoot(item)?.task?.id : null;
  const effectiveModality =
    (alive ? (modality ?? detectModality(item)) : null) ?? "rgb";

  const fp = useFootprint(taskId, effectiveModality, allowModalityFallback);

  const naturalWidth = alive ? item.currentImageEntity?.naturalWidth : null;
  const naturalHeight = alive ? item.currentImageEntity?.naturalHeight : null;

  // 백엔드가 계산에 쓴 이미지 크기와 브라우저가 로드한 크기가 다르면
  // 호모그래피 dst가 어긋난다. 백엔드 값이 있으면 그쪽을 신뢰한다.
  const [imgW, imgH] = useMemo(() => {
    if (fp.imageSize && fp.imageSize.length >= 2) {
      const [bw, bh] = fp.imageSize;
      if (
        naturalWidth && naturalHeight &&
        (bw !== naturalWidth || bh !== naturalHeight)
      ) {
        console.warn(
          "[gps-overlay] 이미지 크기 불일치 — 백엔드 %dx%d vs 로드된 %dx%d. 백엔드 값 사용.",
          bw, bh, naturalWidth, naturalHeight,
        );
      }
      return [bw, bh];
    }
    return [naturalWidth, naturalHeight];
  }, [fp.imageSize, naturalWidth, naturalHeight]);

  // footprint가 최신 파이프라인 산출물인지 교차검증.
  // 위치 오차의 거의 전부가 여기서 발생한다 — 호모그래피는 주어진 4코너를
  // 정확히 재현하므로 낡은 footprint를 스스로 감지하지 못한다.
  const staleCheck = useMemo(() => {
    if (!fp.ring || !imgW || !imgH) return null;
    return checkFootprintScale(
      fp.ring, imgW, imgH, fp.plane?.gsd_m_per_px, 1.5,
    );
  }, [fp.ring, fp.plane, imgW, imgH]);

  const { shapes, offscreen } = useMemo(() => {
    const drawn = [];
    const off = [];
    if (!fp.ring || !imgW || !imgH) return { shapes: drawn, offscreen: off };

    // 낡은 footprint로 그리면 "그럴듯하지만 틀린 위치"가 나온다.
    // 잘못된 패널을 가리키는 것보다 안 그리는 편이 낫다 (modality 폴백과 같은 판단).
    if (staleCheck && staleCheck.stale) {
      console.error("[gps-overlay]", staleCheck.reason);
      return { shapes: drawn, offscreen: off };
    }

    const opts = fp.pixelCorners ? { pixelCorners: fp.pixelCorners } : undefined;

    for (const ov of overlays) {
      const res = projectGpsPolygon(ov.coordinates, fp.ring, imgW, imgH, opts);

      if (!res.ok) {
        console.warn("[gps-overlay] 역변환 실패:", ov.id, res.reason);
        off.push({ id: ov.id, label: ov.label, color: ov.color ?? "#12b76a", failed: true, reason: res.reason });
        continue;
      }

      if (!res.anyInside) {
        // 프레임 밖 — 중심을 프레임 가장자리로 클램프해 방향만 표시
        const cx = res.points.reduce((s, p) => s + p[0], 0) / res.points.length;
        const cy = res.points.reduce((s, p) => s + p[1], 0) / res.points.length;
        const margin = Math.min(imgW, imgH) * 0.02;
        off.push({
          id: ov.id,
          label: ov.label,
          color: ov.color ?? "#12b76a",
          at: [
            Math.max(margin, Math.min(imgW - margin, cx)),
            Math.max(margin, Math.min(imgH - margin, cy)),
          ],
          failed: false,
        });
        continue;
      }

      drawn.push({
        id: ov.id,
        label: ov.label,
        color: ov.color ?? "#12b76a",
        points: res.points.flat(), // Konva Line: [x0,y0,x1,y1,...]
        labelAt: res.points[0],
        clipped: !res.allInside,
      });
    }
    return { shapes: drawn, offscreen: off };
  }, [fp.ring, fp.pixelCorners, imgW, imgH, overlays, staleCheck]);

  // 상태 통지 (렌더 중 부모 setState 방지를 위해 effect에서)
  useEffect(() => {
    if (!onStatus) return;
    for (const s of shapes) {
      onStatus({ overlayId: s.id, state: "drawn", clipped: s.clipped });
    }
    for (const o of offscreen) {
      onStatus({
        overlayId: o.id,
        state: o.failed ? "failed" : "offscreen",
        reason: o.reason,
      });
    }
    if (fp.error) onStatus({ state: "failed", reason: fp.error });
    if (staleCheck?.stale) onStatus({ state: "stale_footprint", reason: staleCheck.reason });
  }, [shapes, offscreen, fp.error, staleCheck, onStatus]);

  // ── 여기서부터 렌더 분기 ──
  if (!alive) return null;
  if (shapes.length === 0 && offscreen.length === 0) return null;

  const zoom = item.stageZoom || 1;

  // 이 레이어의 좌표계 = 원본 이미지 픽셀 (스케일만 stageZoom 적용).
  // strokeScaleEnabled=false로 줌과 무관하게 선 두께 일정.
  return (
    <Layer name="gps-overlay" scale={{ x: zoom, y: zoom }} listening={!!onSelect}>
      {shapes.map((s) => (
        <Group key={s.id}>
          <Line
            points={s.points}
            closed
            stroke={s.color}
            strokeWidth={2}
            strokeScaleEnabled={false}
            // 일부만 보이면 점선으로 잘렸음을 표시
            dash={s.clipped ? [8, 6] : undefined}
            dashEnabled={!!s.clipped}
            fill={withAlpha(s.color, 0.12)}
            listening={!!onSelect}
            onClick={onSelect ? () => onSelect(s.id) : undefined}
            onTap={onSelect ? () => onSelect(s.id) : undefined}
            onMouseEnter={(e) => {
              if (onSelect) e.target.getStage().container().style.cursor = "pointer";
            }}
            onMouseLeave={(e) => {
              if (onSelect) e.target.getStage().container().style.cursor = "default";
            }}
            perfectDrawEnabled={false}
          />
          {s.label && (
            // 라벨은 화면상 크기 고정: 레이어 스케일(zoom)을 상쇄
            <Text
              x={s.labelAt[0]}
              y={s.labelAt[1] - 18 / zoom}
              text={s.label}
              fontSize={14}
              fontStyle="bold"
              fill={s.color}
              stroke="#ffffff"
              strokeWidth={0.4}
              scaleX={1 / zoom}
              scaleY={1 / zoom}
              listening={false}
            />
          )}
        </Group>
      ))}

      {showOffscreenIndicator &&
        offscreen
          .filter((o) => !o.failed && o.at)
          .map((o) => (
            <Group key={`off-${o.id}`}>
              <Circle
                x={o.at[0]}
                y={o.at[1]}
                radius={7 / zoom}
                stroke={o.color}
                strokeWidth={2}
                strokeScaleEnabled={false}
                fill={withAlpha(o.color, 0.25)}
                listening={false}
              />
              <Text
                x={o.at[0] + 10 / zoom}
                y={o.at[1] - 7 / zoom}
                text={`${o.label ?? "참조"} (화면 밖)`}
                fontSize={12}
                fill={o.color}
                stroke="#ffffff"
                strokeWidth={0.4}
                scaleX={1 / zoom}
                scaleY={1 / zoom}
                listening={false}
              />
            </Group>
          ))}
    </Layer>
  );
});

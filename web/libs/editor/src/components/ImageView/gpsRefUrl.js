// web/libs/editor/src/components/ImageView/gpsRefUrl.js
//
// GPS 참조 폴리곤을 URL 쿼리로 주고받는 인코딩/디코딩.
//
// 용도:
//   GeoreferencingResultsModal에서 검색 결과를 클릭해 다른 Task를 새 탭으로 열 때,
//   "원본 region의 GPS 폴리곤"을 URL에 실어 보낸다. 새 탭의 ImageView가 이를
//   읽어 참조 사각형으로 오버레이한다.
//
// 왜 URL인가:
//   - 새로고침/북마크/공유에도 유지 (localStorage는 탭 간 경합·잔존 문제)
//   - 백엔드 왕복 불필요
//   - Data Manager의 기존 쿼리(?task=)와 공존
//
// 포맷 (compact):
//   gps_ref=<lng>,<lat>;<lng>,<lat>;...
//   좌표는 소수 7자리로 반올림 (약 1cm 정밀도 — 드론 RTK 정확도보다 세밀)
//   추가 파라미터: gps_ref_label, gps_ref_color
//
// 길이:
//   5점 폴리곤 기준 약 130자. 브라우저/서버 URL 한계(2000자+)에 여유.

export const GPS_REF_PARAM = "gps_ref";
export const GPS_REF_LABEL_PARAM = "gps_ref_label";
export const GPS_REF_COLOR_PARAM = "gps_ref_color";

const COORD_PRECISION = 7; // 소수 7자리 ≈ 1.1cm

/**
 * GPS 폴리곤 -> URL 파라미터 문자열.
 * @param {Array<[number,number]>} coordinates [[lng,lat],...]
 * @returns {string} "126.9224788,34.7106465;126.9224787,34.7106558;..."
 */
export function encodeGpsRef(coordinates) {
  if (!Array.isArray(coordinates) || coordinates.length < 3) return "";
  return coordinates
    .map(([lng, lat]) => `${round(lng)},${round(lat)}`)
    .join(";");
}

function round(v) {
  // toFixed 후 불필요한 0 제거 (URL 길이 절약)
  return Number.parseFloat(Number(v).toFixed(COORD_PRECISION)).toString();
}

/**
 * URL 파라미터 문자열 -> GPS 폴리곤.
 * @param {string} raw
 * @returns {Array<[number,number]> | null} 유효하지 않으면 null
 */
export function decodeGpsRef(raw) {
  if (!raw || typeof raw !== "string") return null;
  const pts = [];
  for (const pair of raw.split(";")) {
    const [lngStr, latStr] = pair.split(",");
    const lng = Number.parseFloat(lngStr);
    const lat = Number.parseFloat(latStr);
    if (!Number.isFinite(lng) || !Number.isFinite(lat)) return null;
    if (lng < -180 || lng > 180 || lat < -90 || lat > 90) return null;
    pts.push([lng, lat]);
  }
  return pts.length >= 3 ? pts : null;
}

/**
 * Task 링크 URL 생성 (GPS 참조 폴리곤 포함).
 *
 * @param {object} opts
 * @param {number} opts.taskId
 * @param {number} [opts.projectId]
 * @param {Array<[number,number]>} [opts.coordinates] 참조로 그릴 GPS 폴리곤
 * @param {string} [opts.label]
 * @param {string} [opts.color]
 * @returns {string}
 */
export function buildTaskUrlWithGpsRef({ taskId, projectId, coordinates, label, color }) {
  const base = projectId ? `/projects/${projectId}/data` : "/projects/data";
  const params = new URLSearchParams({ task: String(taskId) });

  const encoded = coordinates ? encodeGpsRef(coordinates) : "";
  if (encoded) {
    params.set(GPS_REF_PARAM, encoded);
    if (label) params.set(GPS_REF_LABEL_PARAM, label);
    if (color) params.set(GPS_REF_COLOR_PARAM, color);
  }
  return `${base}?${params.toString()}`;
}

/**
 * 현재 URL(window.location.search)에서 GPS 참조 오버레이를 읽는다.
 * @returns {{ id, coordinates, label, color } | null}
 */
export function readGpsRefFromUrl(search = window.location.search) {
  const params = new URLSearchParams(search);
  const coordinates = decodeGpsRef(params.get(GPS_REF_PARAM));
  if (!coordinates) return null;
  return {
    id: "gps-ref-from-url",
    coordinates,
    label: params.get(GPS_REF_LABEL_PARAM) || "참조 영역",
    color: params.get(GPS_REF_COLOR_PARAM) || "#f04438",
  };
}

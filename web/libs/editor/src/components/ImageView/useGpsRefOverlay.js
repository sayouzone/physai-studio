// web/libs/editor/src/components/ImageView/useGpsRefOverlay.js
//
// 새 탭에서 URL의 gps_ref 파라미터를 읽어 참조 오버레이 배열을 만든다.
// GeoreferencingResultsModal이 실어 보낸 원본 region 폴리곤이 여기로 들어온다.
//
// 해제(dismiss):
//   참조 사각형은 "어디를 봐야 하는지" 알려주는 일회성 힌트다.
//   사용자가 캔버스를 클릭해 작업을 시작하면 자동으로 사라진다.
//   store로 주입한 오버레이(extraOverlays)는 해제 대상이 아니다.
//
// 사용 (ImageView.jsx의 StageContent 안 — early return보다 앞):
//
//   const gpsOverlays = useGpsRefOverlay(store.gpsOverlays ?? []);
//   ...
//   {gpsOverlays.length > 0 && (
//     <GpsOverlayLayer item={item} overlays={gpsOverlays} />
//   )}

import { useCallback, useEffect, useMemo, useState } from "react";

import { GPS_REF_COLOR_PARAM, GPS_REF_LABEL_PARAM, GPS_REF_PARAM, readGpsRefFromUrl } from "./gpsRefUrl";

/** URL에서 gps_ref 관련 파라미터만 제거 (뒤로가기 이력 남기지 않음) */
function stripGpsRefFromUrl() {
  try {
    const url = new URL(window.location.href);
    let changed = false;
    for (const p of [GPS_REF_PARAM, GPS_REF_LABEL_PARAM, GPS_REF_COLOR_PARAM]) {
      if (url.searchParams.has(p)) {
        url.searchParams.delete(p);
        changed = true;
      }
    }
    if (changed) {
      window.history.replaceState(window.history.state, "", url.toString());
    }
  } catch {
    // URL 조작 실패는 무시 — 오버레이 해제 자체에는 영향 없음
  }
}

/**
 * URL에서 참조 오버레이를 읽고, 클릭 시 해제한다.
 *
 * @param {Array} [extraOverlays]  URL 외에 추가로 표시할 오버레이 (해제되지 않음)
 * @param {object} [options]
 * @param {boolean} [options.dismissOnClick=true]  캔버스 클릭 시 참조 해제
 * @param {boolean} [options.clearUrlOnDismiss=true] 해제 시 URL 파라미터도 정리
 *        (false면 새로고침 시 참조가 다시 나타남)
 * @returns {Array} GpsOverlayLayer의 overlays prop 형식
 */
export function useGpsRefOverlay(extraOverlays = [], options = {}) {
  const { dismissOnClick = true, clearUrlOnDismiss = true } = options;

  // URL 파싱은 마운트 시 1회 (탭 수명 동안 URL은 고정)
  const urlRef = useMemo(() => readGpsRefFromUrl(), []);
  const [dismissed, setDismissed] = useState(false);

  const dismiss = useCallback(() => {
    setDismissed(true);
    if (clearUrlOnDismiss) stripGpsRefFromUrl();
  }, [clearUrlOnDismiss]);

  useEffect(() => {
    if (!urlRef || dismissed || !dismissOnClick) return;

    // pointerdown: 클릭/터치/펜 모두 커버. capture 단계로 받아
    // LSF 캔버스가 이벤트를 소비하더라도 확실히 감지.
    const onPointerDown = () => dismiss();

    window.addEventListener("pointerdown", onPointerDown, {
      capture: true,
      once: true,
    });
    return () => {
      window.removeEventListener("pointerdown", onPointerDown, { capture: true });
    };
  }, [urlRef, dismissed, dismissOnClick, dismiss]);

  return useMemo(() => {
    if (!urlRef || dismissed) return extraOverlays;
    return [urlRef, ...extraOverlays];
  }, [urlRef, dismissed, extraOverlays]);
}
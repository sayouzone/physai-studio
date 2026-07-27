// web/libs/editor/src/components/ImageView/gpsToPixel.js
//
// GPS(경위도) 좌표 -> 이미지 픽셀 좌표 역변환.
//
// 원리:
//   georeferencing 순변환은 이미지 4코너 픽셀 -> 지면 GPS(footprint)였다.
//   평면을 보는 무왜곡 핀홀 카메라에서 지면<->이미지 대응은 **정확히**
//   호모그래피이고, 4점 대응이면 유일하게 결정된다. 따라서 footprint
//   4코너로 세운 H는 근사가 아니라 정확한 역변환이다.
//
// 이 파일의 수학은 옳다. 정확도를 좌우하는 건 전적으로 **입력 footprint**다:
//   - footprint가 구버전(이방성 초점거리 / 지면 평면) 파이프라인 산출물이면
//     H도 그 좌표계를 그대로 재현한다. 백엔드 재생성이 선행되어야 한다.
//   - footprint가 다른 modality(RGB<->IR)의 것이면 FOV가 달라 완전히 틀린다.
//   - DewarpData 렌즈 왜곡계수가 도입되면 더 이상 순수 호모그래피가 아니다.
//     그때는 백엔드 역투영(sayou/georeferencing/gps_ref.py)으로 옮겨야 한다.
//
// 좌표계 주의:
//   footprint 코너 순서가 이미지 픽셀 코너와 1:1 대응해야 한다.
//   백엔드 pixel_bbox_to_polygon((0,0,W,H))는 (0,0),(W,0),(W,H),(0,H)
//   = 좌상,우상,우하,좌하 순으로 생성한다. 기본 dst를 여기에 맞춘다.
//   compute_image_corners()는 (W-1,H-1)을 쓰므로 규약이 다르다 —
//   백엔드가 pixel_corners를 내려주면 그 값을 쓰는 것이 가장 안전하다.
//   (두 규약 차이는 약 0.2~1.1 px 로 작지만 공짜로 없앨 수 있다)
//
// m/deg 상수에 대해:
//   toLocal의 mPerDegLat/mPerDegLng 값이 부정확해도 **결과는 바뀌지 않는다**.
//   대각 선형변환 D에 대해 toLocal' = D·toLocal 이면 H' = H·D⁻¹ 이 동일한
//   대응을 정확히 만족하므로 호모그래피가 흡수한다. (111320 -> 110935 로
//   바꿔도 재투영 결과 차이 < 1e-12 px 임을 확인했다.)
//   따라서 여기서는 상수를 '고치지 않는다'. 스케일 정규화 목적일 뿐이다.

// -------------------------------------------------------------------------
// 3x3 선형계 풀이 (8x8 -> homography 8DOF)
// -------------------------------------------------------------------------

/** 8x8 가우스 소거로 Ax=b 풀이. 실패 시 null. */
function solve8(A, b) {
  const n = 8;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    // 부분 피벗
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    }
    if (Math.abs(M[piv][col]) < 1e-15) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    // 소거
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

/**
 * 4점 대응 (src -> dst)으로 homography 계수 [h0..h7] 계산.
 * dst = H * src (h8 = 1 고정).
 */
function computeHomography(src, dst) {
  const A = [];
  const b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }
  return solve8(A, b); // [h0..h7] or null
}

/**
 * homography 적용: (x,y) -> (u,v)
 *
 * 분모 d <= 0 이면 그 점은 **카메라 뒤 / 지평선 너머**다.
 * 이때 나오는 좌표는 부호가 뒤집힌 허상이며, 경우에 따라 프레임 안쪽
 * 좌표로 나와 "엉뚱한 위치에 참조 박스가 그려지는" 원인이 된다.
 * nadir 촬영에서는 h6=h7≈0 이라 d≡1 이지만, oblique 촬영에서는 실제로
 * 뒤집힌다(재현 확인). 반드시 걸러낸다.
 *
 * @returns {[number,number] | null}
 */
function applyH(h, x, y) {
  const d = h[6] * x + h[7] * y + 1;
  if (!(d > 1e-9)) return null; // NaN도 여기서 걸러짐
  const u = (h[0] * x + h[1] * y + h[2]) / d;
  const v = (h[3] * x + h[4] * y + h[5]) / d;
  if (!Number.isFinite(u) || !Number.isFinite(v)) return null;
  return [u, v];
}

// -------------------------------------------------------------------------
// footprint 검증
// -------------------------------------------------------------------------

/** 다각형 부호付 면적 (shoelace) */
function signedArea(ring) {
  let a = 0;
  for (let i = 0; i < ring.length; i++) {
    const [x1, y1] = ring[i];
    const [x2, y2] = ring[(i + 1) % ring.length];
    a += x1 * y2 - x2 * y1;
  }
  return a / 2;
}

/**
 * footprint가 호모그래피를 세울 수 있는 형태인지 검사.
 *
 * @returns {{ ok: boolean, reason?: string }}
 */
export function validateFootprint(footprint) {
  if (!Array.isArray(footprint) || footprint.length < 4) {
    return { ok: false, reason: "footprint에 코너가 4개 미만" };
  }
  const ring = footprint.slice(0, 4);
  for (const c of ring) {
    if (
      !Array.isArray(c) ||
      c.length < 2 ||
      !Number.isFinite(c[0]) ||
      !Number.isFinite(c[1])
    ) {
      return { ok: false, reason: "footprint 좌표에 비수치 값" };
    }
    if (c[0] < -180 || c[0] > 180 || c[1] < -90 || c[1] > 90) {
      return { ok: false, reason: "footprint 좌표 범위 이탈 (lng,lat 순서 확인)" };
    }
    // 백엔드가 georeferencing 실패 시 GeodeticPoint(0,0,0)을 채우는 경로가 있다.
    if (c[0] === 0 && c[1] === 0) {
      return { ok: false, reason: "footprint에 (0,0) 코너 — 백엔드 계산 실패" };
    }
  }
  // 로컬 미터로 환산해 면적/자기교차 확인
  const lat0 = ring.reduce((s, c) => s + c[1], 0) / 4;
  const lng0 = ring.reduce((s, c) => s + c[0], 0) / 4;
  const mLat = 111320;
  const mLng = 111320 * Math.cos((lat0 * Math.PI) / 180);
  const local = ring.map((c) => [(c[0] - lng0) * mLng, (c[1] - lat0) * mLat]);

  const area = Math.abs(signedArea(local));
  if (!(area > 1)) {
    return { ok: false, reason: `footprint 면적이 비정상 (${area.toFixed(2)} m²)` };
  }
  return { ok: true };
}

/**
 * footprint가 **현재(수정된) 파이프라인 산출물인지** 검사한다.
 *
 * 이 레이어에서 위치 오차의 거의 전부는 footprint가 낡았을 때 발생한다.
 * 레거시 footprint(이방성 FOV + 지면 평면)는 수정본과
 *   가로 x1.076, 세로 x0.956
 * 만큼 다르고, 프레임 가장자리에서 같은 GPS 점이 약 1.2~1.3 m 어긋난다.
 * 호모그래피는 주어진 4코너를 정확히 재현하므로 이 오차를 스스로 못 잡는다.
 *
 * 백엔드가 ground_plane.gsd_m_per_px 를 내려주면 물리 치수로 교차검증할 수 있다.
 *   기대 폭 = gsd * imageW,  기대 높이 = gsd * imageH
 *
 * @param {Array<[number,number]>} footprint
 * @param {number} imageW
 * @param {number} imageH
 * @param {number} expectedGsd  m/px (백엔드 ground_plane.gsd_m_per_px)
 * @param {number} [tolerancePct=1.5]
 * @returns {{ ok: boolean, stale?: boolean, reason?: string,
 *             storedM?: [number,number], expectedM?: [number,number],
 *             ratio?: [number,number] }}
 */
export function checkFootprintScale(
  footprint,
  imageW,
  imageH,
  expectedGsd,
  tolerancePct = 1.5,
) {
  if (!Number.isFinite(expectedGsd) || expectedGsd <= 0) {
    // 백엔드가 gsd를 안 내려주면 검사 불가 — 통과시키되 검사 안 했음을 알린다.
    return { ok: true, stale: undefined, reason: "expectedGsd 없음 (검사 생략)" };
  }
  const v = validateFootprint(footprint);
  if (!v.ok) return { ok: false, reason: v.reason };

  const ring = footprint.slice(0, 4);
  const lat0 = ring.reduce((s, c) => s + c[1], 0) / 4;
  // 여기서는 절대 거리를 재야 하므로 곡률반경 기반 상수를 쓴다.
  // (호모그래피와 달리 스케일이 흡수되지 않는다)
  const mLat = 110574 + 1175 * Math.cos((2 * lat0 * Math.PI) / 180);
  const mLng = 111320 * Math.cos((lat0 * Math.PI) / 180);
  const dist = (a, b) =>
    Math.hypot((b[0] - a[0]) * mLng, (b[1] - a[1]) * mLat);

  const storedW = (dist(ring[0], ring[1]) + dist(ring[3], ring[2])) / 2;
  const storedH = (dist(ring[0], ring[3]) + dist(ring[1], ring[2])) / 2;
  const expW = expectedGsd * imageW;
  const expH = expectedGsd * imageH;

  const rx = storedW / expW;
  const ry = storedH / expH;
  const stale =
    Math.abs(rx - 1) * 100 > tolerancePct || Math.abs(ry - 1) * 100 > tolerancePct;

  return {
    ok: !stale,
    stale,
    reason: stale
      ? `footprint 치수 불일치 — 저장 ${storedW.toFixed(2)}x${storedH.toFixed(2)} m, ` +
        `기대 ${expW.toFixed(2)}x${expH.toFixed(2)} m (비율 ${rx.toFixed(3)}/${ry.toFixed(3)}). ` +
        `백엔드 georeferencing 재생성 필요.`
      : undefined,
    storedM: [storedW, storedH],
    expectedM: [expW, expH],
    ratio: [rx, ry],
  };
}

// -------------------------------------------------------------------------
// 공개: GPS -> 픽셀 변환기 생성
// -------------------------------------------------------------------------

/**
 * footprint(4 GPS 코너)와 이미지 크기로 GPS->픽셀 변환 함수를 만든다.
 *
 * @param {Array<[number,number]>} footprint  [[lng,lat] x4] 좌상,우상,우하,좌하
 * @param {number} imageW  원본 이미지 픽셀 폭
 * @param {number} imageH  원본 이미지 픽셀 높이
 * @param {object} [options]
 * @param {Array<[number,number]>} [options.pixelCorners]
 *        footprint 코너에 대응하는 픽셀 좌표. 백엔드가 내려주면 그 값을 쓴다.
 *        생략 시 (0,0),(W,0),(W,H),(0,H).
 * @returns {{ ok: boolean, reason?: string, toPixel: (lng, lat) => [px,py] | null }}
 */
export function makeGpsToPixel(footprint, imageW, imageH, options = {}) {
  if (!imageW || !imageH) {
    return { ok: false, reason: "이미지 크기 없음", toPixel: () => null };
  }

  const v = validateFootprint(footprint);
  if (!v.ok) return { ok: false, reason: v.reason, toPixel: () => null };

  const ring = footprint.slice(0, 4);
  const lat0 = ring.reduce((s, c) => s + c[1], 0) / 4;
  const lng0 = ring.reduce((s, c) => s + c[0], 0) / 4;

  // 스케일 정규화용 상수. 값이 정확하지 않아도 호모그래피가 흡수한다(상단 주석).
  const mPerDegLat = 111320;
  const mPerDegLng = 111320 * Math.cos((lat0 * Math.PI) / 180);

  const toLocal = (lng, lat) => [
    (lng - lng0) * mPerDegLng,
    (lat - lat0) * mPerDegLat,
  ];

  const src = ring.map((c) => toLocal(c[0], c[1]));
  const dst =
    Array.isArray(options.pixelCorners) && options.pixelCorners.length >= 4
      ? options.pixelCorners.slice(0, 4)
      : [
          [0, 0],
          [imageW, 0],
          [imageW, imageH],
          [0, imageH],
        ];

  const H = computeHomography(src, dst);
  if (!H) {
    return { ok: false, reason: "호모그래피 해 없음 (코너 퇴화)", toPixel: () => null };
  }

  return {
    ok: true,
    toPixel: (lng, lat) => {
      if (!Number.isFinite(lng) || !Number.isFinite(lat)) return null;
      const [e, n] = toLocal(lng, lat);
      return applyH(H, e, n);
    },
  };
}

/**
 * GPS 폴리곤([[lng,lat],...])을 픽셀 폴리곤([[px,py],...])으로 변환.
 *
 * 한 점이라도 역변환 불가(카메라 뒤/지평선 너머)면 도형 전체가 의미를
 * 잃으므로 null을 반환한다. 부분 결과를 그리면 실제와 다른 모양이 된다.
 *
 * @returns {Array<[number,number]> | null}
 */
export function gpsPolygonToPixels(coordinates, footprint, imageW, imageH, options) {
  if (!Array.isArray(coordinates) || coordinates.length < 3) return null;
  const conv = makeGpsToPixel(footprint, imageW, imageH, options);
  if (!conv.ok) return null;

  const out = [];
  for (const c of coordinates) {
    const p = conv.toPixel(c[0], c[1]);
    if (!p) return null; // 하나라도 실패하면 도형 폐기
    out.push(p);
  }
  return out;
}

/**
 * 변환 결과와 진단 정보를 함께 반환하는 버전.
 * 오버레이 레이어처럼 "왜 안 보이는지"를 알려줘야 하는 곳에서 쓴다.
 *
 * @returns {{ points: Array|null, ok: boolean, reason?: string,
 *             anyInside: boolean, allInside: boolean }}
 */
export function projectGpsPolygon(coordinates, footprint, imageW, imageH, options) {
  const conv = makeGpsToPixel(footprint, imageW, imageH, options);
  if (!conv.ok) {
    return { points: null, ok: false, reason: conv.reason, anyInside: false, allInside: false };
  }
  if (!Array.isArray(coordinates) || coordinates.length < 3) {
    return { points: null, ok: false, reason: "폴리곤 점이 3개 미만", anyInside: false, allInside: false };
  }

  const points = [];
  for (const c of coordinates) {
    const p = conv.toPixel(c[0], c[1]);
    if (!p) {
      return {
        points: null,
        ok: false,
        reason: "일부 점이 카메라 뒤/지평선 너머",
        anyInside: false,
        allInside: false,
      };
    }
    points.push(p);
  }

  const inside = points.map(
    ([x, y]) => x >= 0 && x <= imageW && y >= 0 && y <= imageH,
  );
  return {
    points,
    ok: true,
    anyInside: inside.some(Boolean),
    allInside: inside.every(Boolean),
  };
}

/**
 * GPS 폴리곤을 축정렬 bbox(픽셀)로 변환.
 * PhysAI Studio rectanglelabels는 회전 없는 bbox이므로 외접 사각형을 반환한다.
 *
 * 주의: 짐벌 yaw만큼 회전한 검출 박스를 축정렬로 만들면 실제보다 커진다.
 * 표시용 오버레이는 gpsPolygonToPixels로 회전을 살려 그릴 것.
 *
 * @returns {{ x, y, width, height } | null}  픽셀 단위
 */
export function gpsPolygonToPixelBBox(coordinates, footprint, imageW, imageH, options) {
  const px = gpsPolygonToPixels(coordinates, footprint, imageW, imageH, options);
  if (!px) return null;
  const xs = px.map((p) => p[0]);
  const ys = px.map((p) => p[1]);
  const minX = Math.max(0, Math.min(...xs));
  const minY = Math.max(0, Math.min(...ys));
  const maxX = Math.min(imageW, Math.max(...xs));
  const maxY = Math.min(imageH, Math.max(...ys));
  if (maxX <= minX || maxY <= minY) return null;
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

/**
 * PhysAI Studio result 좌표(%)로 변환.
 * LS rectanglelabels의 x,y,width,height는 원본 대비 백분율(0~100)이다.
 */
export function pixelBBoxToLSPercent(bbox, imageW, imageH) {
  if (!bbox) return null;
  return {
    x: (bbox.x / imageW) * 100,
    y: (bbox.y / imageH) * 100,
    width: (bbox.width / imageW) * 100,
    height: (bbox.height / imageH) * 100,
    rotation: 0,
  };
}

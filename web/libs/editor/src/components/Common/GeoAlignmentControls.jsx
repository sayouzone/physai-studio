/**
 * GeoAlignmentControls.jsx
 *
 * Georeferencing 지도 모달("Georeferencing 결과 — Task #N")에 삽입하는
 * 정합 보정 컨트롤. 폴리곤 전체를 화살표 키/버튼으로 미세 이동·회전시키고
 * 백엔드(api/tasks/<id>/geo-alignment/)에 저장한다.
 *
 * 조작:
 *   ← → ↑ ↓        : 0.25 m 이동
 *   Shift + 방향키  : 1.0 m 이동
 *   Q / E           : 0.1° 회전 (Shift: 0.5°)
 *   R               : 초기화
 *
 * 사용 예 (기존 지도 모달 내부):
 *
 *   const [correction, setCorrection] = useState(null);
 *
 *   // GeoJSON feature -> google.maps.Polygon 경로 생성 시:
 *   const path = ring.map(([lng, lat]) =>
 *     applyCorrectionToLatLng(lng, lat, correction));
 *
 *   <GeoAlignmentControls
 *     taskId={task.id}
 *     anchor={{ lng: centerLng, lat: centerLat }}   // 녹색 점 좌표
 *     correction={correction}
 *     onChange={setCorrection}                      // 폴리곤 다시 그리기
 *     apiBase="/api"
 *   />
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';

/* ------------------------------------------------------------------ */
/* 좌표 변환 (백엔드 geo_alignment_transform.py와 동일한 수학)          */
/* ------------------------------------------------------------------ */

const WGS84_A = 6378137.0;
const WGS84_E2 = 6.69437999014e-3;

function metersPerDegree(latDeg) {
  const lat = (latDeg * Math.PI) / 180;
  const s = Math.sin(lat);
  const denom = Math.sqrt(1 - WGS84_E2 * s * s);
  const rM = (WGS84_A * (1 - WGS84_E2)) / (denom * denom * denom);
  const rN = WGS84_A / denom;
  const deg = Math.PI / 180;
  return { mLon: deg * rN * Math.cos(lat), mLat: deg * rM };
}

/**
 * 단일 (lng, lat)에 보정을 적용해 google.maps.LatLngLiteral을 반환.
 * correction이 null이면 원본 그대로.
 */
export function applyCorrectionToLatLng(lng, lat, correction) {
  if (!correction) return { lat, lng };
  const { anchorLng, anchorLat, dE, dN, rotDeg, scale } = correction;
  const { mLon, mLat } = metersPerDegree(anchorLat);

  const e = (lng - anchorLng) * mLon;
  const n = (lat - anchorLat) * mLat;
  const th = (rotDeg * Math.PI) / 180;
  const c = Math.cos(th);
  const s = Math.sin(th);
  const e2 = scale * (c * e - s * n) + dE;
  const n2 = scale * (s * e + c * n) + dN;

  return { lng: anchorLng + e2 / mLon, lat: anchorLat + n2 / mLat };
}

const IDENTITY = (anchor) => ({
  anchorLng: anchor.lng,
  anchorLat: anchor.lat,
  dE: 0,
  dN: 0,
  rotDeg: 0,
  scale: 1,
});

/* ------------------------------------------------------------------ */
/* 컴포넌트                                                            */
/* ------------------------------------------------------------------ */

const NUDGE_M = 0.25;
const NUDGE_BIG_M = 1.0;
const ROT_STEP = 0.1;
const ROT_BIG = 0.5;

export default function GeoAlignmentControls({
  taskId,
  anchor,
  correction,
  onChange,
  apiBase = '/api',
}) {
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState(null); // 'saved' | 'error' | null
  const corr = correction ?? IDENTITY(anchor);

  /* 서버에 저장된 보정 불러오기 */
  useEffect(() => {
    let cancelled = false;
    fetch(`${apiBase}/tasks/${taskId}/geo-alignment/`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (cancelled || !data) return;
        onChange({
          anchorLng: data.anchor_lon,
          anchorLat: data.anchor_lat,
          dE: data.offset_east_m,
          dN: data.offset_north_m,
          rotDeg: data.rotation_deg,
          scale: data.scale,
        });
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskId]);

  const update = useCallback(
    (patch) => {
      setStatus(null);
      onChange({ ...corr, ...patch });
    },
    [corr, onChange],
  );

  const nudge = useCallback(
    (dE, dN) => update({ dE: corr.dE + dE, dN: corr.dN + dN }),
    [corr, update],
  );

  const rotate = useCallback(
    (deg) => update({ rotDeg: +(corr.rotDeg + deg).toFixed(3) }),
    [corr, update],
  );

  const reset = useCallback(() => {
    setStatus(null);
    onChange(IDENTITY(anchor));
  }, [anchor, onChange]);

  /* 키보드 단축키 */
  useEffect(() => {
    const handler = (ev) => {
      // 입력 필드에 포커스가 있으면 무시
      const tag = document.activeElement?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;

      const step = ev.shiftKey ? NUDGE_BIG_M : NUDGE_M;
      const rot = ev.shiftKey ? ROT_BIG : ROT_STEP;
      let handled = true;

      switch (ev.key) {
        case 'ArrowUp':
          nudge(0, step);
          break;
        case 'ArrowDown':
          nudge(0, -step);
          break;
        case 'ArrowLeft':
          nudge(-step, 0);
          break;
        case 'ArrowRight':
          nudge(step, 0);
          break;
        case 'q':
        case 'Q':
          rotate(rot);
          break;
        case 'e':
        case 'E':
          rotate(-rot);
          break;
        case 'r':
        case 'R':
          reset();
          break;
        default:
          handled = false;
      }
      if (handled) ev.preventDefault();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [nudge, rotate, reset]);

  /* 저장 */
  const save = useCallback(async () => {
    setSaving(true);
    setStatus(null);
    try {
      const resp = await fetch(`${apiBase}/tasks/${taskId}/geo-alignment/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          anchor_lon: corr.anchorLng,
          anchor_lat: corr.anchorLat,
          offset_east_m: corr.dE,
          offset_north_m: corr.dN,
          rotation_deg: corr.rotDeg,
          scale: corr.scale,
          method: 'manual',
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      setStatus('saved');
    } catch (err) {
      console.error('GeoAlignment save failed', err);
      setStatus('error');
    } finally {
      setSaving(false);
    }
  }, [apiBase, taskId, corr]);

  /* 자동 정합 요청 */
  const autoAlign = useCallback(
    async (geojson) => {
      setSaving(true);
      setStatus(null);
      try {
        const resp = await fetch(
          `${apiBase}/tasks/${taskId}/geo-alignment/auto/`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              geojson,
              center: [anchor.lng, anchor.lat],
            }),
          },
        );
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        onChange({
          anchorLng: data.anchor_lon,
          anchorLat: data.anchor_lat,
          dE: data.offset_east_m,
          dN: data.offset_north_m,
          rotDeg: data.rotation_deg,
          scale: data.scale,
        });
        setStatus('saved');
      } catch (err) {
        console.error('Auto align failed', err);
        setStatus('error');
      } finally {
        setSaving(false);
      }
    },
    [apiBase, taskId, anchor, onChange],
  );

  const magnitude = useMemo(
    () => Math.hypot(corr.dE, corr.dN),
    [corr.dE, corr.dN],
  );

  /* ---------------------------------------------------------------- */

  const btn = {
    padding: '4px 10px',
    border: '1px solid #d0d5dd',
    borderRadius: 6,
    background: '#fff',
    cursor: 'pointer',
    fontSize: 13,
    lineHeight: 1.4,
  };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '8px 12px',
        border: '1px solid #e4e7ec',
        borderRadius: 8,
        background: '#fafafa',
        fontSize: 13,
        flexWrap: 'wrap',
      }}
    >
      <strong style={{ marginRight: 4 }}>지도 정합 보정</strong>

      {/* 방향 버튼 */}
      <span style={{ display: 'inline-flex', gap: 4 }}>
        <button style={btn} onClick={() => nudge(-NUDGE_M, 0)} title="서쪽 0.25m (←)">←</button>
        <button style={btn} onClick={() => nudge(0, NUDGE_M)} title="북쪽 0.25m (↑)">↑</button>
        <button style={btn} onClick={() => nudge(0, -NUDGE_M)} title="남쪽 0.25m (↓)">↓</button>
        <button style={btn} onClick={() => nudge(NUDGE_M, 0)} title="동쪽 0.25m (→)">→</button>
        <button style={btn} onClick={() => rotate(ROT_STEP)} title="반시계 0.1° (Q)">⟲</button>
        <button style={btn} onClick={() => rotate(-ROT_STEP)} title="시계 0.1° (E)">⟳</button>
      </span>

      {/* 현재 보정값 */}
      <span style={{ fontVariantNumeric: 'tabular-nums', color: '#475467' }}>
        ΔE {corr.dE.toFixed(2)}m · ΔN {corr.dN.toFixed(2)}m · Δθ{' '}
        {corr.rotDeg.toFixed(2)}° · |Δ| {magnitude.toFixed(2)}m
      </span>

      <span style={{ flex: 1 }} />

      <button style={btn} onClick={reset} disabled={saving}>
        초기화 (R)
      </button>
      <button
        style={{ ...btn, background: '#1570ef', color: '#fff', borderColor: '#1570ef' }}
        onClick={save}
        disabled={saving}
      >
        {saving ? '저장 중…' : '보정 저장'}
      </button>

      {status === 'saved' && (
        <span style={{ color: '#067647' }}>저장됨 — export에도 적용됩니다</span>
      )}
      {status === 'error' && (
        <span style={{ color: '#d92d20' }}>저장 실패 — 다시 시도하세요</span>
      )}

      <span style={{ width: '100%', color: '#98a2b3', fontSize: 12 }}>
        방향키 0.25m 이동 (Shift 1m) · Q/E 회전 · 위성 이미지 자체 오차(1~5m)를
        보정하는 표시용 변환이며 원본 RTK 좌표는 보존됩니다
      </span>
    </div>
  );
}

/* eslint-disable-next-line no-unused-vars */
export function useAutoAlign() {
  /* GeoAlignmentControls의 autoAlign을 외부에서 쓰고 싶을 때를 위한 자리 */
}

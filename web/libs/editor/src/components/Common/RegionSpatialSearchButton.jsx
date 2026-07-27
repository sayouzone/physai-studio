/**
 * RegionSpatialSearchButton.jsx
 *
 * GeoreferencingButton 대체 컴포넌트.
 * Info 패널에서 region이 선택된 상태로 클릭하면, 그 region의 지리 좌표를
 * 포함하는(footprint가 커버하는) 다른 이미지들을 검색해 모달로 보여준다.
 *
 * 백엔드: GET /api/regions/<region_id>/find-images/?with_regions=1
 *
 * 교체 방법 — 기존 GeoreferencingButton이 렌더되던 자리에서:
 *
 *   - <GeoreferencingButton task={task} selectedRegion={region} />
 *   + <RegionSpatialSearchButton
 *   +     selectedRegion={region}          // { id, ... } PhysAI Studio region
 *   +     currentProjectId={project.id}
 *   +     apiBase="/api"
 *   + />
 *
 * 결과 항목 클릭 시 해당 Task 라벨링 화면으로 이동한다.
 */

import React, { useCallback, useEffect, useState } from 'react';

/* ------------------------------------------------------------------ */
/* 아이콘 (지구본 -> 이미지 검색으로 교체)                              */
/* ------------------------------------------------------------------ */

const SearchImagesIcon = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round"
    strokeLinejoin="round" aria-hidden="true">
    <rect x="3" y="3" width="12" height="12" rx="2" />
    <circle cx="8" cy="8" r="1.6" />
    <path d="M15 11l-3.5 4H5" />
    <circle cx="17" cy="17" r="4" />
    <path d="M20 20l2.5 2.5" />
  </svg>
);

/* ------------------------------------------------------------------ */
/* 메인 버튼                                                           */
/* ------------------------------------------------------------------ */

export default function RegionSpatialSearchButton({
  selectedRegion,
  currentProjectId,
  apiBase = '/api',
}) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);   // API 응답
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(false);

  const disabled = !selectedRegion?.id;

  const search = useCallback(async () => {
    if (disabled || loading) return;
    setLoading(true);
    setError(null);
    try {
      const url = `/api/regions/${encodeURIComponent(
        selectedRegion.id)}/find-images/?with_regions=1`;
      const resp = await fetch(url);
      const body = await resp.json().catch(() => null);

      if (resp.status === 409) {
        // DetectionRegion 미동기화 — 지리좌표 없음
        setError(body?.detail
          ?? '이 region의 지리 좌표가 아직 계산되지 않았습니다.');
        setOpen(true);
        return;
      }
      if (!resp.ok) {
        throw new Error(body?.detail ?? `HTTP ${resp.status}`);
      }
      setResult(body);
      setOpen(true);
    } catch (err) {
      console.error('Region image search failed', err);
      setError(String(err.message ?? err));
      setOpen(true);
    } finally {
      setLoading(false);
    }
  }, [apiBase, selectedRegion, disabled, loading]);

  return (
    <>
      <button
        type="button"
        onClick={search}
        disabled={disabled || loading}
        title={disabled
          ? 'region을 먼저 선택하세요'
          : '이 영역을 포함하는 다른 이미지 검색'}
        aria-label="이 영역을 포함하는 이미지 검색"
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: 28,
          height: 28,
          border: 'none',
          borderRadius: 6,
          background: 'transparent',
          color: disabled ? '#c2c8d0' : '#5a6472',
          cursor: disabled ? 'default' : 'pointer',
        }}
      >
        {loading
          ? <span style={{ fontSize: 11 }}>…</span>
          : <SearchImagesIcon />}
      </button>

      {open && (
        <ResultsModal
          result={result}
          error={error}
          currentProjectId={currentProjectId}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}

/* ------------------------------------------------------------------ */
/* 결과 모달                                                           */
/* ------------------------------------------------------------------ */

function ResultsModal({ result, error, currentProjectId, onClose }) {
  /* ESC로 닫기 */
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const region = result?.region;
  const images = result?.images ?? [];

  const openTask = (img) => {
    const projectId = img.project_id ?? currentProjectId;
    // PhysAI Studio Data Manager 라벨링 딥링크
    window.location.href =
      `/projects/${projectId}/data?task=${img.task_id}`;
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(16,24,40,0.45)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(760px, 92vw)',
          maxHeight: '84vh',
          overflow: 'auto',
          background: '#fff',
          borderRadius: 12,
          padding: '20px 24px',
          boxShadow: '0 20px 50px rgba(16,24,40,0.25)',
          fontSize: 14,
        }}
      >
        {/* 헤더 */}
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
          <h3 style={{ margin: 0, fontSize: 17 }}>
            이 영역을 포함하는 이미지
          </h3>
          {region && (
            <span style={{ color: '#667085', fontSize: 13 }}>
              {region.defect_class ?? ''} · Task #{region.task_id}
              {region.flight_date ? ` · ${region.flight_date}` : ''}
            </span>
          )}
          <span style={{ flex: 1 }} />
          <button onClick={onClose} aria-label="닫기"
            style={{
              border: 'none', background: 'none',
              fontSize: 18, cursor: 'pointer', color: '#667085'
            }}>
            ×
          </button>
        </div>

        {/* 오류 */}
        {error && (
          <div style={{
            marginTop: 14, padding: '10px 12px', borderRadius: 8,
            background: '#fef3f2', color: '#b42318',
          }}>
            {error}
          </div>
        )}

        {/* 빈 결과 */}
        {!error && images.length === 0 && (
          <p style={{ color: '#667085', marginTop: 16 }}>
            이 위치를 커버하는 다른 이미지가 없습니다.
            근접 검색이 필요하면 radius_m 파라미터를 늘려보세요.
          </p>
        )}

        {/* 결과 목록 */}
        {images.map((img) => (
          <div
            key={img.task_id}
            onClick={() => openTask(img)}
            onKeyDown={(e) => e.key === 'Enter' && openTask(img)}
            role="button"
            tabIndex={0}
            style={{
              display: 'flex', gap: 14, alignItems: 'center',
              padding: '10px 8px', marginTop: 10,
              border: '1px solid #e4e7ec', borderRadius: 10,
              cursor: 'pointer',
            }}
          >
            {/* 썸네일 */}
            <div style={{
              width: 96, height: 64, flexShrink: 0,
              borderRadius: 6, overflow: 'hidden', background: '#f2f4f7',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              {img.resolved_url ? (
                <img
                  src={img.resolved_url}
                  alt={`Task ${img.task_id}`}
                  loading="lazy"
                  style={{
                    width: '100%', height: '100%',
                    objectFit: 'cover'
                  }}
                />
              ) : (
                <span style={{ fontSize: 11, color: '#98a2b3' }}>
                  no preview
                </span>
              )}
            </div>

            {/* 정보 */}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontWeight: 600 }}>
                Task #{img.task_id}
                <span style={{
                  marginLeft: 8, fontWeight: 400,
                  color: '#667085', fontSize: 13,
                }}>
                  {img.flight_date || '촬영일 미상'}
                  {' · '}
                  {img.distance_m === 0
                    ? '영역 포함'
                    : `중심에서 ${img.distance_m}m`}
                </span>
              </div>

              {/* 같은 위치의 region 매칭 (시계열 비교 핵심 정보) */}
              {Array.isArray(img.matched_regions)
                && img.matched_regions.length > 0 && (
                  <div style={{
                    marginTop: 4, fontSize: 12.5,
                    color: '#475467'
                  }}>
                    같은 위치 검출:{' '}
                    {img.matched_regions.map((m, i) => (
                      <span key={m.region_id}>
                        {i > 0 && ', '}
                        <strong>{m.defect_class}</strong>
                        {typeof m.confidence === 'number'
                          ? ` (${m.confidence.toFixed(2)})` : ''}
                      </span>
                    ))}
                  </div>
                )}
              {Array.isArray(img.matched_regions)
                && img.matched_regions.length === 0 && (
                  <div style={{
                    marginTop: 4, fontSize: 12.5,
                    color: '#98a2b3'
                  }}>
                    이 이미지에는 같은 위치 검출 없음
                  </div>
                )}
            </div>

            <span style={{ color: '#98a2b3', fontSize: 18 }}>›</span>
          </div>
        ))}
      </div>
    </div>
  );
}

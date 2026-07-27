// web/libs/editor/src/components/Common/GeoMapView.jsx
import { useEffect, useRef } from "react";

// ⚠️ 실제 키로 교체 (GCP Console → Maps JavaScript API)
const GOOGLE_MAPS_API_KEY = "AIzaSyBUyoyBl2APXFI59pQar4WsSUUpjZs0E3c";

const CLASS_STYLES = {
    pv: { strokeColor: "#4285F4", fillColor: "#4285F4", fillOpacity: 0.25 },
    Panel: { strokeColor: "#4285F4", fillColor: "#4285F4", fillOpacity: 0.25 },
    anomaly: { strokeColor: "#EA4335", fillColor: "#EA4335", fillOpacity: 0.40 },
    Hotspot: { strokeColor: "#EA4335", fillColor: "#EA4335", fillOpacity: 0.40 },
};

// Google Maps 스크립트는 전역에서 1회만 로드
let mapsScriptPromise = null;
const loadGoogleMaps = () => {
    if (window.google?.maps) return Promise.resolve();
    if (mapsScriptPromise) return mapsScriptPromise;

    mapsScriptPromise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = `https://maps.googleapis.com/maps/api/js?key=${GOOGLE_MAPS_API_KEY}`;
        script.async = true;
        script.onload = resolve;
        script.onerror = () => {
            mapsScriptPromise = null;
            reject(new Error("Google Maps 스크립트 로드 실패"));
        };
        document.head.appendChild(script);
    });
    return mapsScriptPromise;
};

export const GeoMapView = ({ geojson }) => {
    const mapRef = useRef(null);

    useEffect(() => {
        if (!geojson || !mapRef.current) return;
        let cancelled = false;

        loadGoogleMaps()
            .then(() => {
                if (cancelled) return;
                const g = window.google;

                const map = new g.maps.Map(mapRef.current, {
                    zoom: 18,
                    center: { lat: 37.26, lng: 127.02 },  // fitBounds로 자동 이동됨
                    mapTypeId: "satellite",
                    tilt: 0,
                    streetViewControl: false,
                    fullscreenControl: true,
                });

                map.data.addGeoJson(geojson);

                // ── Feature별 스타일 ──────────────────────────────
                map.data.setStyle((feature) => {
                    const name = feature.getProperty("name");
                    const className = feature.getProperty("class_name");

                    if (name === "image_coverage") {
                        return {
                            strokeColor: "#FBBC04", strokeWeight: 2,
                            fillColor: "#FBBC04", fillOpacity: 0.08, zIndex: 1,
                        };
                    }
                    if (name === "drone_position") {
                        return {
                            icon: {
                                path: g.maps.SymbolPath.CIRCLE,
                                scale: 8, fillColor: "#34A853", fillOpacity: 1,
                                strokeColor: "#fff", strokeWeight: 2,
                            },
                            zIndex: 100,
                        };
                    }
                    const s = CLASS_STYLES[className]
                        ?? { strokeColor: "#9E9E9E", fillColor: "#9E9E9E", fillOpacity: 0.2 };
                    return {
                        ...s,
                        strokeWeight: 2,
                        zIndex: (className === "anomaly" || className === "Hotspot") ? 50 : 10,
                    };
                });

                // ── 클릭 시 속성 InfoWindow ───────────────────────
                const infoWindow = new g.maps.InfoWindow();
                map.data.addListener("click", (event) => {
                    const f = event.feature;
                    const keys = [
                        "name", "class_name", "width_m", "height_m", "area_m2",
                        "pixel_bbox", "rel_alt_m", "rtk_active", "angle_from_nadir_deg",
                    ];
                    const rows = keys
                        .map((k) => {
                            const v = f.getProperty(k);
                            return v != null
                                ? `<tr><td style="padding-right:8px;color:#666"><b>${k}</b></td><td>${JSON.stringify(v)}</td></tr>`
                                : null;
                        })
                        .filter(Boolean)
                        .join("");
                    infoWindow.setContent(`<table style="font-size:12px">${rows}</table>`);
                    infoWindow.setPosition(event.latLng);
                    infoWindow.open(map);
                });

                // ── 전체 feature에 맞춰 화면 이동 ─────────────────
                const bounds = new g.maps.LatLngBounds();
                map.data.forEach((feature) => {
                    feature.getGeometry().forEachLatLng((ll) => bounds.extend(ll));
                });
                if (!bounds.isEmpty()) map.fitBounds(bounds);
            })
            .catch((e) => {
                console.error(e);
                if (mapRef.current) {
                    mapRef.current.innerHTML =
                        `<div style="padding:24px;color:#c00">지도 로드 실패: ${e.message}</div>`;
                }
            });

        return () => { cancelled = true; };
    }, [geojson]);

    // 검출 개수 요약
    const numDetections = geojson?.metadata?.num_detections
        ?? geojson?.features?.filter((f) => f.properties?.type === "detection_box").length
        ?? 0;

    return (
        <div style={{ width: "100%" }}>
            <div style={{ fontSize: 12, color: "#666", marginBottom: 8 }}>
                검출 {numDetections}개 · 위성 뷰 · 폴리곤 클릭 시 상세 정보
            </div>
            <div
                ref={mapRef}
                style={{
                    width: "100%",
                    height: "60vh",
                    minHeight: 400,
                    borderRadius: 8,
                    background: "#f0f0f0",
                }}
            />
        </div>
    );
};
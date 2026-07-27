"""WGS84 ↔ 투영좌표계 변환.

기본 타깃은 EPSG:5186 (Korea 2000 / Central Belt). 필요 시 다른 EPSG 코드로
바꿔 사용 가능. 변환기는 stateless 하지만 ``pyproj.Transformer`` 객체 생성
비용이 약간 있으므로 (수십 ms) 파이프라인당 한 번만 만들어 재사용한다.
"""

from __future__ import annotations

from pyproj import Transformer


class CRSConverter:
    """단일 EPSG 페어에 대한 양방향 변환기.

    Parameters
    ----------
    target_epsg : 목적 좌표계 (한국 측량 표준 5186 권장).
    """

    def __init__(self, target_epsg: int = 5186):
        self.target_epsg = target_epsg
        self.to_proj = Transformer.from_crs(
            "EPSG:4326", f"EPSG:{target_epsg}", always_xy=True
        )
        self.to_wgs = Transformer.from_crs(
            f"EPSG:{target_epsg}", "EPSG:4326", always_xy=True
        )

    def forward(self, lon: float, lat: float) -> tuple[float, float]:
        """WGS84 (lon, lat) → 투영좌표 (X, Y)."""
        return self.to_proj.transform(lon, lat)

    def inverse(self, x: float, y: float) -> tuple[float, float]:
        """투영좌표 (X, Y) → WGS84 (lon, lat)."""
        return self.to_wgs.transform(x, y)


__all__ = ["CRSConverter"]
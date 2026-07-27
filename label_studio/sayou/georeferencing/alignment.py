"""
sayou/georeferencing/alignment.py

프레임별 위치 오프셋(cross_frame_residuals 산출물)을 georeferencing 출력에
적용해, 모든 프레임의 좌표를 **하나의 합의(consensus) 좌표계**로 맞춘다.

원리
────
cross_frame_residuals 의 네트워크 조정은 다음 관측식을 푼다.

    r_ij = mean(p_j - p_i) = t_j - t_i        (p = 각 프레임이 계산한 위치)

따라서 모든 프레임에 대해 `p_i - t_i` 가 같은 값(합의 위치)이 된다.

    보정된 좌표 = 계산된 좌표 - 그 프레임의 오프셋

검증: t_121=(0.55,0.85), t_169=(-0.30,-0.45) 인 합성 예에서 같은 모듈의
두 프레임 간 불일치가 1.544 m → 0.011 m 로 줄어든다.

화면(gps_ref 오버레이)에 적용하는 법
──────────────────────────────────
프론트는 대상 프레임의 footprint 4코너로 호모그래피를 세워 gps_ref 링을
역투영한다. 따라서 **양쪽이 같은 좌표계**여야 한다.

    소스 Task #121 의 검출 링   →  ring  - t_121
    대상 Task #169 의 footprint →  corner - t_169

둘 다 합의 좌표계가 되므로 오버레이가 맞는다. 한쪽만 보정하면 오히려
어긋나므로, 검출/footprint 두 경로에 **반드시 함께** 적용해야 한다.

지지 부족 프레임
────────────────
매칭이 적어 오프셋을 신뢰할 수 없는 프레임(trusted=false)은 그 값을 쓰지
않는다. 다만 오프셋은 공간적으로 매끄럽게 변하므로(위치 선형 사상 R² 0.54,
비등방 40배), 신뢰 가능한 이웃 프레임에서 IDW 보간해 채우는 편이
"보정 안 함"보다 낫다. 보정된 프레임과 안 된 프레임이 섞이면 그 경계에서
새로운 불연속이 생기기 때문이다.

사용
────
    from sayou.georeferencing.alignment import get_alignment

    al = get_alignment()                      # 기본 경로에서 1회 로드 후 캐시
    lon, lat = al.shift_lonlat(lon, lat, task_id, m_lon, m_lat)

환경변수 SAYOU_FRAME_OFFSETS 로 JSON 경로를 지정한다. 지정이 없거나 파일이
없으면 모든 보정이 비활성(no-op)이므로 안전하게 배포할 수 있다.
"""

import json
import logging
import math
import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# 오프셋 JSON 경로. 미설정이면 보정 비활성.
OFFSETS_PATH_ENV = 'SAYOU_FRAME_OFFSETS'

# 보간에 쓸 이웃 수, 그리고 이 거리(m)를 넘으면 보간하지 않는다.
IDW_NEIGHBORS = 4
IDW_MAX_DIST_M = 60.0


class FrameAlignment:
    """task_id → (동, 북) 오프셋. 신뢰 불가 프레임은 공간 보간으로 채운다."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        interpolate: bool = True,
        idw_neighbors: int = IDW_NEIGHBORS,
        idw_max_dist_m: float = IDW_MAX_DIST_M,
    ):
        self.interpolate = interpolate
        self.idw_neighbors = idw_neighbors
        self.idw_max_dist_m = idw_max_dist_m

        self._exact: Dict[int, Tuple[float, float]] = {}
        self._anchors: List[Tuple[float, float, float, float]] = []  # lat, lng, dE, dN
        self._positions: Dict[int, Tuple[float, float]] = {}
        self._cache: Dict[int, Optional[Tuple[float, float]]] = {}
        self._lock = threading.Lock()

        for rec in records:
            try:
                tid = int(rec['task_id'])
                de = float(rec['offset_east_m'])
                dn = float(rec['offset_north_m'])
            except (KeyError, TypeError, ValueError):
                continue

            lat, lng = rec.get('lat'), rec.get('lng')
            if lat is not None and lng is not None:
                self._positions[tid] = (float(lat), float(lng))

            if rec.get('trusted', True):
                self._exact[tid] = (de, dn)
                if lat is not None and lng is not None:
                    self._anchors.append((float(lat), float(lng), de, dn))

        logger.info(
            '[alignment] 오프셋 %d건 로드 (신뢰 %d, 앵커 %d, 보간=%s)',
            len(records), len(self._exact), len(self._anchors), interpolate,
        )

    # ----- 로딩 -----
    @classmethod
    def from_json(cls, path: str, **kw) -> 'FrameAlignment':
        with open(path) as fh:
            payload = json.load(fh)
        return cls(payload.get('frames', []), **kw)

    @classmethod
    def disabled(cls) -> 'FrameAlignment':
        return cls([])

    @property
    def enabled(self) -> bool:
        return bool(self._exact)

    # ----- 조회 -----
    def offset(self, task_id: Optional[int]) -> Optional[Tuple[float, float]]:
        """
        (동, 북) 미터 오프셋. 보정하지 않아야 하면 None.
        신뢰 가능한 값이 있으면 그대로, 없으면 이웃에서 IDW 보간.
        """
        if task_id is None or not self._exact:
            return None
        task_id = int(task_id)

        hit = self._exact.get(task_id)
        if hit is not None:
            return hit
        if not self.interpolate:
            return None

        with self._lock:
            if task_id in self._cache:
                return self._cache[task_id]
            val = self._interpolate(task_id)
            self._cache[task_id] = val
        return val

    def _interpolate(self, task_id: int) -> Optional[Tuple[float, float]]:
        pos = self._positions.get(task_id)
        if pos is None or not self._anchors:
            return None
        lat0, lng0 = pos
        m_lat = meters_per_deg_lat(lat0)
        m_lon = meters_per_deg_lon(lat0)

        scored = []
        for lat, lng, de, dn in self._anchors:
            d = math.hypot((lng - lng0) * m_lon, (lat - lat0) * m_lat)
            if d <= self.idw_max_dist_m:
                scored.append((d, de, dn))
        if not scored:
            logger.debug('[alignment] task=%s 반경 내 앵커 없음 — 보정 생략', task_id)
            return None

        scored.sort(key=lambda x: x[0])
        scored = scored[: self.idw_neighbors]

        # 거리가 0에 가까우면 그 값을 그대로
        if scored[0][0] < 1e-6:
            return (scored[0][1], scored[0][2])

        wsum = de_sum = dn_sum = 0.0
        for d, de, dn in scored:
            w = 1.0 / (d * d)
            wsum += w
            de_sum += w * de
            dn_sum += w * dn
        return (de_sum / wsum, dn_sum / wsum)

    # ----- 적용 -----
    def shift_lonlat(
        self,
        lon: float,
        lat: float,
        task_id: Optional[int],
        m_lon: float,
        m_lat: float,
    ) -> Tuple[float, float]:
        """계산된 (lon, lat) → 합의 좌표계. 보정 대상이 아니면 그대로 반환."""
        off = self.offset(task_id)
        if off is None:
            return lon, lat
        de, dn = off
        return (lon - de / m_lon, lat - dn / m_lat)

    def shift_ring(
        self,
        ring: Sequence[Sequence[float]],
        task_id: Optional[int],
        m_lon: float,
        m_lat: float,
    ) -> List[List[float]]:
        """[[lon, lat], ...] 링 전체를 보정. footprint / 검출 폴리곤 공용."""
        off = self.offset(task_id)
        if off is None:
            return [[float(p[0]), float(p[1])] for p in ring]
        de, dn = off
        return [[float(p[0]) - de / m_lon, float(p[1]) - dn / m_lat] for p in ring]

    def describe(self, task_id: Optional[int]) -> Dict[str, Any]:
        """GeoJSON metadata 에 실어 보낼 진단 정보."""
        if task_id is None or not self._exact:
            return {'applied': False, 'reason': 'alignment disabled'}
        tid = int(task_id)
        if tid in self._exact:
            de, dn = self._exact[tid]
            return {'applied': True, 'source': 'measured',
                    'east_m': round(de, 4), 'north_m': round(dn, 4)}
        off = self.offset(tid)
        if off is None:
            return {'applied': False, 'reason': 'no trusted offset or neighbour'}
        return {'applied': True, 'source': 'interpolated',
                'east_m': round(off[0], 4), 'north_m': round(off[1], 4)}


# ─────────────────────────────────────────────────────────────────────────────
# 측지 상수 (metadata.py 와 동일한 정의)
# ─────────────────────────────────────────────────────────────────────────────
_A = 6_378_137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2 - _F)


def meters_per_deg_lat(lat_deg: float) -> float:
    phi = math.radians(lat_deg)
    m = _A * (1 - _E2) / (1 - _E2 * math.sin(phi) ** 2) ** 1.5
    return m * math.pi / 180.0


def meters_per_deg_lon(lat_deg: float) -> float:
    phi = math.radians(lat_deg)
    n = _A / math.sqrt(1 - _E2 * math.sin(phi) ** 2)
    return n * math.cos(phi) * math.pi / 180.0


# ─────────────────────────────────────────────────────────────────────────────
# 프로세스 단위 캐시
# ─────────────────────────────────────────────────────────────────────────────
_default: Optional[FrameAlignment] = None
_default_lock = threading.Lock()


def get_alignment(path: Optional[str] = None, reload: bool = False) -> FrameAlignment:
    """
    기본 정합 객체. SAYOU_FRAME_OFFSETS 환경변수 경로에서 1회 로드 후 캐시한다.
    경로가 없거나 파일이 없으면 비활성 객체를 반환하므로 호출부는 분기 없이 써도 된다.
    """
    global _default
    if _default is not None and not reload and path is None:
        return _default

    target = path or os.environ.get(OFFSETS_PATH_ENV)
    with _default_lock:
        if target and os.path.exists(target):
            try:
                al = FrameAlignment.from_json(target)
            except Exception as e:
                logger.error('[alignment] %s 로드 실패: %s — 보정 비활성', target, e)
                al = FrameAlignment.disabled()
        else:
            if target:
                logger.warning('[alignment] 파일 없음: %s — 보정 비활성', target)
            al = FrameAlignment.disabled()
        if path is None:
            _default = al
    return al


def reset_alignment() -> None:
    """테스트/재로딩용."""
    global _default
    with _default_lock:
        _default = None
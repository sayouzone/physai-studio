"""
tasks/search_image_georeferencing.py

TaskImageGeoreferencing 모델(GeoDjango)로 폴리곤 커버리지 검색을 수행한다.
기존 raw SQL(polygon_image_search._do_search_pg)과 동일한 결과를,
psycopg2 직접 접근 대신 Django ORM으로 구현한 버전.

핵심 매핑 (raw SQL -> GeoDjango):
    ST_Intersects(footprint, region)        -> footprint__intersects=region
    ST_Covers(footprint, region)            -> Func(..., function='ST_Covers')
    ST_Area(ST_Intersection(a,b)::geography) -> Area(Cast(Intersection(...),
                                                         geography))
    coverage_ratio                          -> annotate 표현식 + Least(1.0, ...)

반환: [(TaskImageGeoreferencing 인스턴스, coverage_ratio, fully_covered), ...]
      또는 dict 리스트(as_dict=True) — API 직렬화용.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
좌표계 주의 (2026-07)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
여기서 쓰는 footprint 는 TaskImageGeoreferencing 에 **저장된** 값이다.
호출부(AnnotationGeoreferencingAPI)가 넘기는 region 은 매 요청마다 새로
계산되는 검출 폴리곤이다. 두 값이 같은 파이프라인에서 나오지 않으면
ST_Intersection / coverage_ratio 가 서로 다른 좌표계끼리 계산된다.

  레거시 인제스트 footprint : 34.38 x 22.92 m  (이방성 FOV + 지면평면)
  현재 파이프라인 footprint : 31.96 x 23.97 m  (등방 초점거리 + 패널면)
  → 스케일비 x1.076 / y0.956, 프레임 가장자리에서 약 1.3 m 차이

coverage_ratio 가 미묘하게 틀리면 hit 목록과 순위가 흔들리고,
min_coverage 임계 근처 이미지가 들쭉날쭉 포함/제외된다.
저장 footprint 를 재생성해야 한다 (rebuild_footprints 관리 명령).

수정 이력 (2026-07)
- find_images_covering_polygon 앞에 있던 진단용 print / raw cursor 블록 제거.
  요청마다 SHOW search_path, pg_class 조회, 전체 count(*) 를 실행하고 있었다.
  count(*) 는 테이블이 커질수록 선형으로 느려진다.
- 좌표 검증 추가 (lng/lat 순서, 범위).
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Sequence

from django.contrib.gis.db.models.functions import Area, Intersection
from django.contrib.gis.db.models import PolygonField
from django.contrib.gis.geos import Polygon
from django.db.models import BooleanField, F, FloatField, Func, Value
from django.db.models.functions import Least, NullIf

from tasks.models import TaskImageGeoreferencing

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 좌표 정규화
# ---------------------------------------------------------------------------

def _to_polygon(coordinates: Sequence[Sequence[float]]) -> Polygon:
    """[[lng, lat], ...] (닫힘 무관) -> SRID 4326 Polygon."""
    ring = [tuple(map(float, c)) for c in coordinates]
    if len(ring) < 3:
        raise ValueError('폴리곤은 최소 3개의 꼭짓점이 필요합니다.')

    # 좌표 순서 뒤바뀜은 조용히 빈 결과를 내므로 명시적으로 잡는다.
    for lng, lat in ring:
        if not (-180.0 <= lng <= 180.0 and -90.0 <= lat <= 90.0):
            raise ValueError(
                f'좌표 범위 이탈: ({lng}, {lat}). [[lng, lat], ...] 순서여야 합니다.'
            )

    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return Polygon(ring, srid=4326)


def _geom_value(geom: Polygon):
    """geometry 상수를 ORM 표현식으로. GeoDjango는 PolygonField output_field로
    Value를 넘겨야 psycopg2 어댑트 오류('can't adapt type Polygon') 없이 동작한다.
    """
    return Value(geom, output_field=PolygonField(srid=4326))


# ---------------------------------------------------------------------------
# GeoDjango 함수 헬퍼
# ---------------------------------------------------------------------------

class _GeographyArea(Func):
    """ST_Area(geom::geography) — 미터² 면적."""
    function = 'ST_Area'
    output_field = FloatField()


def _geography_area(geom_expr):
    """geometry 표현식을 geography로 캐스트 후 면적(m²)."""
    return _GeographyArea(
        Func(geom_expr, template='%(expressions)s::geography'))


class _StCovers(Func):
    """ST_Covers(a, b) -> boolean."""
    function = 'ST_Covers'
    output_field = BooleanField()


# ---------------------------------------------------------------------------
# 검색
# ---------------------------------------------------------------------------

def find_images_covering_polygon(
    coordinates: Sequence[Sequence[float]],
    min_coverage: float = 0.0,
    modality: Optional[str] = None,
    project_id: Optional[int] = None,
    rtk_fixed_only: bool = False,
    limit: int = 100,
    as_dict: bool = False,
):
    """폴리곤 좌표를 커버하는 TaskImageGeoreferencing 레코드 검색.

    Args:
        coordinates: [[lng, lat], ...] GeoJSON 순서 (닫힘 무관)
        min_coverage: 최소 커버리지 비율 (0=교차만, 1.0=완전 커버)
        modality: 'wide' | 'zoom' | 'ir'
        project_id: 프로젝트 필터
        rtk_fixed_only: RtkFlag=50 만
        limit: 최대 결과 수
        as_dict: True면 API용 dict 리스트 반환

    Returns:
        as_dict=False: [(obj, coverage_ratio, fully_covered), ...]
        as_dict=True:  [{...}, ...]  (coverage_ratio 내림차순)

    주의:
        footprint 는 저장값이다. region 과 다른 파이프라인 산출물이면
        coverage_ratio 가 부정확해진다 (파일 상단 주석 참고).
    """
    if not (0.0 <= min_coverage <= 1.0):
        raise ValueError('min_coverage는 0.0~1.0 범위여야 합니다.')

    region = _to_polygon(coordinates)
    region_val = _geom_value(region)

    # ST_Area(region::geography) — 상수(파라미터당 1회 평가)
    region_area = _geography_area(region_val)

    # ST_Area(ST_Intersection(footprint, region)::geography)
    inter_area = _geography_area(Intersection('footprint', region_val))

    # coverage_ratio = LEAST(1.0, inter_area / NULLIF(region_area, 0))
    coverage_expr = Least(
        Value(1.0),
        inter_area / NullIf(region_area, Value(0.0)),
        output_field=FloatField(),
    )

    qs = (TaskImageGeoreferencing.objects
          # Stage 1: GiST 인덱스 (bbox + 정밀) — ST_Intersects
          .filter(footprint__intersects=region))

    if modality:
        qs = qs.filter(modality=modality)
    if project_id is not None:
        qs = qs.filter(project_id=project_id)
    if rtk_fixed_only:
        qs = qs.filter(rtk_flag=50)

    qs = (qs
          .annotate(
              coverage_ratio=coverage_expr,
              fully_covered=_StCovers('footprint', region_val),
          )
          # Stage 2: 커버리지 임계
          .filter(coverage_ratio__gte=min_coverage)
          .order_by('-coverage_ratio', F('capture_time').desc(nulls_last=True)))

    qs = qs[:limit]

    if not as_dict:
        results = [(obj, round(obj.coverage_ratio, 4), obj.fully_covered)
                   for obj in qs]
        logger.debug('[geo-search] %d hits (min_coverage=%s)', len(results), min_coverage)
        return results

    results = []
    for obj in qs:
        # footprint를 GeoJSON coordinates로
        footprint_coords = json.loads(obj.footprint.geojson)['coordinates'][0]
        results.append({
            'id': obj.id,
            'task_id': obj.task_id,
            'project_id': obj.project_id,
            'path': obj.path,
            'filename': obj.filename,
            'modality': obj.modality,
            'capture_time': obj.capture_time,
            'rtk_flag': obj.rtk_flag,
            'coverage_ratio': round(obj.coverage_ratio, 4),
            'fully_covered': obj.fully_covered,
            'low_confidence': obj.low_confidence,
            'footprint': footprint_coords,
        })

    logger.debug('[geo-search] %d hits (min_coverage=%s)', len(results), min_coverage)
    return results


# ---------------------------------------------------------------------------
# 점 검색 (부가)
# ---------------------------------------------------------------------------

class _GeographyDistance(Func):
    """ST_Distance(a::geography, b::geography) — 미터 거리."""
    function = 'ST_Distance'
    output_field = FloatField()

    def __init__(self, geom_expr, other_geom):
        super().__init__(
            Func(geom_expr, template='%(expressions)s::geography'),
            Func(other_geom, template='%(expressions)s::geography'))


def find_images_containing_point(
    lat: float,
    lng: float,
    radius_m: float = 0.0,
    modality: Optional[str] = None,
    project_id: Optional[int] = None,
    rtk_fixed_only: bool = False,
    limit: int = 100,
):
    """(lat, lng)를 footprint가 포함/근접하는 이미지.

    radius_m=0 -> footprint__covers, radius_m>0 -> geography 거리 <= radius.
    반환: [(obj, distance_m), ...] 거리 오름차순.
    """
    from django.contrib.gis.geos import Point
    from django.contrib.gis.db.models import PointField

    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        raise ValueError(f'좌표 범위 이탈: lat={lat}, lng={lng}')

    point = Point(lng, lat, srid=4326)
    point_val = Value(point, output_field=PointField(srid=4326))

    qs = (TaskImageGeoreferencing.objects
          .annotate(distance_m=_GeographyDistance('footprint', point_val)))

    if radius_m > 0:
        qs = qs.filter(distance_m__lte=radius_m)
    else:
        qs = qs.filter(footprint__covers=point)

    if modality:
        qs = qs.filter(modality=modality)
    if project_id is not None:
        qs = qs.filter(project_id=project_id)
    if rtk_fixed_only:
        qs = qs.filter(rtk_flag=50)

    qs = qs.order_by('distance_m')[:limit]

    return [(obj, round(obj.distance_m, 2)) for obj in qs]
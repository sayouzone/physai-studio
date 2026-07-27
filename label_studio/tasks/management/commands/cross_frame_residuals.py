"""
tasks/management/commands/cross_frame_residuals.py

중첩 프레임에서 **같은 물리 모듈**을 매칭해 잔차 벡터를 모으고,
그 패턴으로 남은 georeferencing 오차의 원인을 분해한다.

잔차 패턴 해석
──────────────
    반경에 비례 + 방사 방향   → 투영 평면 높이 / 초점거리 잔차
    반경에 비례 + 접선 방향    → 짐벌 yaw 오차
    프레임 전체가 평행 이동    → 드론 위치(RTK/GPS) 기준 차이
    무작위                    → 라벨 편차 (물리 모델로 못 줄임)

네트워크 조정 (2026-07 추가)
────────────────────────────
실측에서 프레임들이 **내부적으로는 잘 맞는 여러 군**으로 갈리고, 군 사이가
순수 평행이동으로 어긋나는 패턴이 나왔다 (예: {121,122,123} 대 {140,141} 이
약 2.4 m). 이는 카메라 모델이 아니라 위치 기준(datum) 차이 — 대개 RTK 상태다.

그래서 프레임별 2D 평행이동 t_i 를 미지수로 두고

    minimize  Σ_pairs || (t_j - t_i) - r_ij ||²      (gauge: Σ t_i = 0)

를 최소제곱으로 푼다. |t_i| 가 큰 프레임이 기준에서 벗어난 프레임이고,
조정 후 잔차 RMS 가 크게 떨어지면 "남은 오차는 전부 프레임별 위치 오프셋"
이라는 뜻이다. 그 값이 곧 GeoAlignmentCorrection 에 넣을 보정량이다.

사용
────
    python manage.py cross_frame_residuals --project 3 --limit 200
    python manage.py cross_frame_residuals --tasks 121,122,123,140,141

통계 주의
─────────
쌍당 매칭이 3~5 개면 방사 기울기/yaw 의 개별 쌍 값은 노이즈다 (n=3 은
직선 적합의 자유도가 1). 쌍별 값은 참고만 하고 전체 집계를 보라.
이 명령은 n 이 부족한 쌍에 * 를 붙인다.
"""

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from django.core.management.base import BaseCommand, CommandError

from tasks.models import Task

logger = logging.getLogger(__name__)

# 이 개수 미만이면 쌍별 기울기/yaw 를 신뢰하지 않는다
RELIABLE_N = 8


def _plane_kwargs() -> dict:
    """api.py 의 _plane_kwargs() 와 같은 값을 써야 한다."""
    try:
        from tasks.api import _plane_kwargs as pk
        return pk(None)
    except Exception:
        return {}


class Frame:
    """한 이미지의 검출 위치(로컬 미터 평면)와 nadir 기준 정보."""

    def __init__(self, task_id: int, metadata, geojson, class_id: Optional[int]):
        self.task_id = task_id
        self.lat0 = metadata.gps.lat
        self.lng0 = metadata.gps.lng
        self.m_lat = metadata.m_per_deg_lat
        self.m_lon = metadata.m_per_deg_lon
        self.rtk_flag = getattr(metadata, 'rtk_flag', None)
        self.rtk_active = getattr(metadata, 'rtk_active', None)
        self.capture_time = getattr(metadata, 'capture_time', None)
        v = getattr(metadata, 'velocity', None) or []
        # DJI FlightXSpeed/YSpeed. 축 규약은 기종/펌웨어마다 달라
        # 여기서는 원값을 그대로 두고, 오프셋과의 2x2 회귀로 매핑을 추정한다.
        self.vel = np.array([float(v[0]), float(v[1])]) if len(v) >= 2 else None

        gp = (geojson.get('metadata') or {}).get('ground_plane') or {}
        self.agl = gp.get('agl_m')
        self.plane_alt = gp.get('altitude_m')
        self.plane_source = gp.get('source')

        self.points: List[np.ndarray] = []
        for f in geojson.get('features', []):
            props = f.get('properties') or {}
            if props.get('type') != 'detection_box':
                continue
            if class_id is not None and props.get('class_id') != class_id:
                continue
            ring = f['geometry']['coordinates'][0][:4]
            lon = sum(p[0] for p in ring) / 4
            lat = sum(p[1] for p in ring) / 4
            self.points.append(self.to_local(lon, lat))

    def to_local(self, lon: float, lat: float) -> np.ndarray:
        """드론 nadir 을 원점으로 한 (East, North) 미터."""
        return np.array([
            (lon - self.lng0) * self.m_lon,
            (lat - self.lat0) * self.m_lat,
        ])

    def offset_to(self, other: 'Frame') -> np.ndarray:
        """other 프레임 원점 기준으로 좌표계를 옮기기 위한 평행이동."""
        return np.array([
            (self.lng0 - other.lng0) * other.m_lon,
            (self.lat0 - other.lat0) * other.m_lat,
        ])

    def rtk_label(self) -> str:
        return {50: 'FIX', 16: 'FLOAT', 34: 'SINGLE', 0: 'GPS'}.get(
            self.rtk_flag, str(self.rtk_flag))


def match_mutual_nn(a_pts: np.ndarray, b_pts: np.ndarray, max_dist: float,
                    ratio: float = 0.7):
    """
    상호 최근접(mutual NN) + Lowe 비율 검정 매칭.

    태양광 배열은 **반복 격자**다. 모듈 간격(이 사이트 기준 N-S 약 1.2 m,
    E-W 약 2.0 m)보다 큰 max_dist 를 쓰면 옆 모듈로 잠기고, 그 오매칭이
    잔차를 모듈 간격의 배수 쪽으로 양자화시킨다. 실측 오프셋 히스토그램이
    0.8~1.0 m 와 1.6~1.8 m 에 봉우리를 보인 것이 그 신호였다.

    비율 검정: 최근접 d1 과 차근접 d2 에 대해 d1/d2 > ratio 면 모호하다고
    보고 버린다. 반복 구조에서 오매칭을 걸러내는 표준 방법.

    Returns: [(i, j, dist), ...], 그리고 기각 사유 카운트
    """
    stats = {'far': 0, 'not_mutual': 0, 'ambiguous': 0, 'kept': 0}
    if len(a_pts) == 0 or len(b_pts) == 0:
        return [], stats
    d = np.linalg.norm(a_pts[:, None, :] - b_pts[None, :, :], axis=2)
    a_best = d.argmin(axis=1)
    b_best = d.argmin(axis=0)

    pairs = []
    for i, j in enumerate(a_best):
        j = int(j)
        if d[i, j] > max_dist:
            stats['far'] += 1
            continue
        if b_best[j] != i:
            stats['not_mutual'] += 1
            continue
        # 차근접까지의 거리 (양쪽 방향 모두 확인)
        row = np.delete(d[i], j)
        col = np.delete(d[:, j], i)
        second = min(row.min() if row.size else np.inf,
                     col.min() if col.size else np.inf)
        if np.isfinite(second) and second > 0 and d[i, j] / second > ratio:
            stats['ambiguous'] += 1
            continue
        stats['kept'] += 1
        pairs.append((i, j, float(d[i, j])))
    return pairs, stats


def decompose(res: np.ndarray, rad_a: np.ndarray) -> dict:
    """잔차 벡터를 평행/방사/접선 성분으로 분해."""
    n = len(res)
    if n == 0:
        return {}
    mean = res.mean(axis=0)
    centered = res - mean

    r = np.linalg.norm(rad_a, axis=1)
    ok = r > 1e-6
    radial_u = np.zeros_like(rad_a)
    radial_u[ok] = rad_a[ok] / r[ok, None]
    tang_u = np.stack([-radial_u[:, 1], radial_u[:, 0]], axis=1)

    radial_c = (centered * radial_u).sum(axis=1)
    tang_c = (centered * tang_u).sum(axis=1)

    radial_slope = float(np.polyfit(r[ok], radial_c[ok], 1)[0]) if ok.sum() >= 3 else float('nan')
    tang_slope = float(np.polyfit(r[ok], tang_c[ok], 1)[0]) if ok.sum() >= 3 else float('nan')

    return {
        'n': n,
        'translation_m': float(np.linalg.norm(mean)),
        'translation_vec': mean,
        'radial_mean_m': float(np.abs(radial_c).mean()),
        'tangential_mean_m': float(np.abs(tang_c).mean()),
        'radial_slope': radial_slope,
        'yaw_deg': float(math.degrees(tang_slope)) if not math.isnan(tang_slope) else float('nan'),
        'rms_m': float(np.sqrt((np.linalg.norm(res, axis=1) ** 2).mean())),
        'mean_radius_m': float(r.mean()),
    }


def frame_support(n_frames: int, pair_obs) -> List[Tuple[int, int]]:
    """프레임별 (참여 쌍 수, 총 매칭 수). 오프셋 신뢰도 판단용."""
    sup = [[0, 0] for _ in range(n_frames)]
    for i, j, _r, n in pair_obs:
        for k in (i, j):
            sup[k][0] += 1
            sup[k][1] += n
    return [tuple(x) for x in sup]


def solve_frame_offsets(n_frames: int, pair_obs: List[Tuple[int, int, np.ndarray, int]],
                        damping: float = 0.0):
    """
    프레임별 2D 평행이동 t_i 를 최소제곱으로 푼다.

        (t_j - t_i) ≈ r_ij,   gauge: Σ t_i = 0

    x, y 가 독립이므로 따로 푼다. 관측 수만큼 가중(매칭 개수)을 준다.

    Returns: (N,2) 오프셋 배열
    """
    rows, rhs_x, rhs_y, weights = [], [], [], []
    for i, j, r, n in pair_obs:
        row = np.zeros(n_frames)
        row[j] = 1.0
        row[i] = -1.0
        rows.append(row)
        rhs_x.append(r[0])
        rhs_y.append(r[1])
        weights.append(math.sqrt(n))

    # gauge 고정: 평균 오프셋 = 0
    rows.append(np.ones(n_frames))
    rhs_x.append(0.0)
    rhs_y.append(0.0)
    weights.append(math.sqrt(len(pair_obs)) if pair_obs else 1.0)

    # 릿지 감쇠: 관측이 빈약한 프레임이 과적합으로 큰 오프셋을 갖지 않도록
    # 0 쪽으로 수축시킨다. 충분히 구속된 프레임은 거의 영향받지 않는다.
    if damping > 0:
        for k in range(n_frames):
            row = np.zeros(n_frames)
            row[k] = 1.0
            rows.append(row)
            rhs_x.append(0.0)
            rhs_y.append(0.0)
            weights.append(damping)

    A = np.array(rows) * np.array(weights)[:, None]
    tx = np.linalg.lstsq(A, np.array(rhs_x) * np.array(weights), rcond=None)[0]
    ty = np.linalg.lstsq(A, np.array(rhs_y) * np.array(weights), rcond=None)[0]
    return np.stack([tx, ty], axis=1)




def radial_profile(res_centered: np.ndarray, rad: np.ndarray, nbins: int = 8):
    """
    쌍별 중심화된 잔차의 **방사 성분을 반경 구간별로** 집계한다.

    형태로 원인을 가른다:
      반경에 선형        → 균일 스케일 잔재 (이론상 중심화로 제거되므로 드묾)
      바깥에서 급격히 증가 → 렌즈 왜곡 (r^3 항). DewarpData 가 없으면 유력.
      평평              → 계통 성분 없음

    Returns: [(r_center, mean_radial, n), ...]
    """
    r = np.linalg.norm(rad, axis=1)
    ok = r > 1e-6
    u = np.zeros_like(rad)
    u[ok] = rad[ok] / r[ok, None]
    radial_c = (res_centered * u).sum(axis=1)

    edges = np.quantile(r[ok], np.linspace(0, 1, nbins + 1))
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = ok & (r >= a) & (r < b)
        if m.sum() >= 10:
            out.append((float((a + b) / 2), float(radial_c[m].mean()), int(m.sum())))
    return out


def bearing_of(vec: np.ndarray) -> float:
    """(East, North) 벡터의 나침반 방위각 (도, 북=0, 동=90)."""
    return float((math.degrees(math.atan2(vec[0], vec[1])) + 360.0) % 360.0)


def explain_offsets(frames, offsets) -> dict:
    """
    네트워크 조정이 뽑은 프레임별 오프셋을 두 가설로 설명해 본다.

    가설 A — GNSS/카메라 시각 동기 오차
        기록 위치가 실제 노출 시점보다 Δt 어긋나면
            offset_i ≈ A · v_i            (v = 비행 속도)
        왕복 비행에서 라인마다 v 부호가 뒤집히므로 오프셋도 라인 단위로
        부호가 바뀐다. A 가 (회전 x 스칼라)에 가까우면 그 스칼라가 Δt.

    가설 B — 투영 평면 높이 / 스케일 잔차
        균일 방사 스케일 k 는 쌍 사이에서 평행이동으로 나타나고,
        네트워크 조정은 이를 t_i = k * (nadir_i - centroid) 로 흡수한다.
            offset_i ≈ k * (nadir_i - centroid)

    두 모델의 설명력(R^2)을 비교하면 어느 쪽이 지배적인지 갈린다.
    """
    out = {}
    off = np.asarray(offsets, dtype=float)
    tot = float((off ** 2).sum())
    if tot <= 0:
        return out

    def r2(pred):
        return 1.0 - float(((off - pred) ** 2).sum()) / tot

    # ── A: 속도 회귀 ──────────────────────────────────────────────
    have_v = [k for k, f in enumerate(frames) if f.vel is not None]
    if len(have_v) >= 6:
        V = np.array([frames[k].vel for k in have_v])
        O = off[have_v]
        M, *_ = np.linalg.lstsq(V, O, rcond=None)      # (2,2): O ~ V @ M
        pred = np.zeros_like(off)
        pred[have_v] = V @ M
        sv = np.linalg.svd(M, compute_uv=False)
        out['velocity'] = {
            'r2': r2(pred),
            'matrix': M.T.tolist(),                     # offset = M.T @ v
            # 왕복 비행은 속도가 한 축으로만 변하므로 M 이 rank-1 이 되기 쉽다.
            # 그때 작은 특이값은 0 에 가깝고 큰 쪽만 Δt 를 담는다 → max 를 쓴다.
            'singular_values_s': sv.tolist(),
            'dt_estimate_s': float(sv.max()),
            'rank_deficient': bool(sv.max() > 0 and sv.min() / sv.max() < 0.2),
            'n': len(have_v),
            'speed_mean': float(np.linalg.norm(V, axis=1).mean()),
        }

    # ── B: nadir 위치 회귀 ────────────────────────────────────────
    # 등방 스칼라 k 하나로는 '부지 경사' 를 못 잡는다. 경사는 방향성이 있어
    # 위치→오프셋 사상이 비등방(전단 포함)이 되므로 2x2 선형 사상을 적합한다.
    #   A 의 특이값이 둘 다 비슷 → 등방 스케일 (평면 높이 오차)
    #   특이값이 크게 다름       → 방향성 있는 경사/틸트
    ref = frames[0]
    nad = np.array([ref.to_local(f.lng0, f.lat0) for f in frames])
    nad = nad - nad.mean(axis=0)
    denom = float((nad ** 2).sum())
    if denom > 0:
        k = float((off * nad).sum() / denom)            # 등방 스칼라
        out['scale_isotropic'] = {
            'r2': r2(k * nad),
            'k': k,
            'plane_error_m': k * 43.5 / (1.0 + k) if k > -1 else float('nan'),
        }
        A, *_ = np.linalg.lstsq(nad, off, rcond=None)   # (2,2): off ~ nad @ A
        sv = np.linalg.svd(A, compute_uv=False)
        out['scale'] = {
            'r2': r2(nad @ A),
            'matrix': A.T.tolist(),                     # offset = A.T @ (nadir - c)
            'singular_values': sv.tolist(),
            'anisotropy': float(sv.max() / sv.min()) if sv.min() > 1e-12 else float('inf'),
            'span_m': float(np.linalg.norm(nad, axis=1).max() * 2),
        }

    return out


class Command(BaseCommand):
    help = '중첩 프레임 간 검출 잔차로 georeferencing 오차 모드를 진단'

    def add_arguments(self, parser):
        parser.add_argument('--project', type=int, default=None)
        parser.add_argument('--tasks', type=str, default=None, help='쉼표 구분 task id')
        parser.add_argument('--limit', type=int, default=100, help='분석할 최대 task 수')
        parser.add_argument('--class-id', type=int, default=1,
                            help='매칭할 클래스 (RGB 기준 1=anomaly)')
        parser.add_argument('--min-detections', type=int, default=3,
                            help='프레임을 쓰기 위한 최소 검출 수')
        parser.add_argument('--max-dist', type=float, default=1.5,
                            help='같은 모듈로 볼 최대 거리 (m). '
                                 '모듈 간격의 절반보다 작게 잡을 것')
        parser.add_argument('--ratio-test', type=float, default=0.7,
                            help='Lowe 비율 검정 임계 (d1/d2). 1.0 이면 비활성')
        parser.add_argument('--max-pair-sep', type=float, default=25.0,
                            help='프레임 쌍으로 볼 최대 드론 위치 거리 (m)')
        parser.add_argument('--outlier-threshold', type=float, default=0.8,
                            help='이 값(m) 이상 오프셋인 프레임을 이탈로 표시')
        parser.add_argument('--damping', type=float, default=1.0,
                            help='릿지 감쇠. 관측 빈약한 프레임의 오프셋을 0 쪽으로 '
                                 '수축. 0 이면 비활성')
        parser.add_argument('--min-support', type=int, default=25,
                            help='이 매칭 수 미만이면 오프셋을 신뢰하지 않음(표시만)')
        parser.add_argument('--export-offsets', type=str, default=None,
                            help='프레임별 오프셋을 JSON 으로 저장할 경로')

    def handle(self, *args, **opts):
        from sayou.annotation_georeferencer import AnnotationGeoreferencer
        from tasks.api import resolve_task_image_path

        if opts['tasks']:
            ids = [int(t) for t in opts['tasks'].split(',') if t.strip()]
            tasks = list(Task.objects.filter(pk__in=ids).select_related('project'))
        elif opts['project']:
            tasks = list(
                Task.objects.filter(project_id=opts['project'])
                .select_related('project')
                .order_by('id')[: opts['limit']]
            )
        else:
            raise CommandError('--project 또는 --tasks 중 하나는 필요합니다')

        if len(tasks) < 2:
            raise CommandError('프레임 쌍을 만들려면 최소 2개의 task 가 필요합니다')

        kwargs = _plane_kwargs()
        self.stdout.write(f'{len(tasks)} 개 task 로드 중 (plane_kwargs={kwargs}) ...')

        frames: List[Frame] = []
        plane_sources = defaultdict(int)
        rtk_counts = defaultdict(int)

        for t in tasks:
            try:
                path = resolve_task_image_path(t)
                gr = AnnotationGeoreferencer(path, annotation=None, task_id=t.id, **kwargs)
                ann = gr._get_annotation(t)
                if ann is None:
                    continue
                metadata, geojson = gr.get_geojson(annotation=ann)
            except Exception as e:
                self.stderr.write(f'  task={t.id} 건너뜀: {e}')
                continue

            fr = Frame(t.id, metadata, geojson, opts['class_id'])
            if len(fr.points) < opts['min_detections']:
                continue
            frames.append(fr)
            plane_sources[fr.plane_source] += 1
            rtk_counts[fr.rtk_label()] += 1

        if len(frames) < 2:
            raise CommandError('검출이 충분한 프레임이 2개 미만입니다')

        self.stdout.write(f'프레임 {len(frames)} 개 / 평면 출처: {dict(plane_sources)}')
        self.stdout.write(f'RTK 상태 분포: {dict(rtk_counts)}')
        if len(plane_sources) > 1:
            self.stdout.write(self.style.ERROR(
                '평면 출처가 섞여 있습니다. 프레임마다 다른 평면을 쓰면 잔차 분석이 무의미합니다.'
            ))
        if len(rtk_counts) > 1:
            self.stdout.write(self.style.WARNING(
                'RTK 상태가 섞여 있습니다. FIX(50) 이 아닌 프레임은 위치가 '
                '수십 cm~수 m 어긋나며, 이는 카메라 모델로 보정되지 않습니다.'
            ))

        idx_of = {fr.task_id: k for k, fr in enumerate(frames)}

        # ── 프레임 쌍 매칭 ────────────────────────────────────────────
        all_res, all_rad, pair_obs, centered_pool = [], [], [], []
        match_stats = defaultdict(int)

        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                A, B = frames[i], frames[j]
                sep = float(np.linalg.norm(B.offset_to(A)))
                if sep > opts['max_pair_sep']:
                    continue

                shift = B.offset_to(A)
                a_pts = np.array(A.points)
                b_pts = np.array(B.points) + shift

                pairs, mstats = match_mutual_nn(
                    a_pts, b_pts, opts['max_dist'], opts['ratio_test'])
                for k2, v2 in mstats.items():
                    match_stats[k2] += v2
                if len(pairs) < 3:
                    continue

                res = np.array([b_pts[j2] - a_pts[i2] for i2, j2, _ in pairs])
                rad = np.array([a_pts[i2] for i2, _, _ in pairs])
                all_res.append(res)
                all_rad.append(rad)
                # 쌍별 평행이동을 먼저 제거해야 방사/접선 성분이 오염되지 않는다.
                # (균일 스케일 오차는 쌍 사이에서 '평행이동'으로 나타나므로,
                #  전역 평균만 빼면 없는 방사 기울기가 만들어진다.)
                centered_pool.append(res - res.mean(axis=0))
                pair_obs.append((i, j, res.mean(axis=0), len(pairs)))

                st = decompose(res, rad)
                mark = ' ' if st['n'] >= RELIABLE_N else '*'
                self.stdout.write(
                    f"  task {A.task_id:>5}[{A.rtk_label():>5}] ↔ "
                    f"{B.task_id:<5}[{B.rtk_label():<5}] "
                    f"간격 {sep:5.1f} m  매칭 {st['n']:>3}{mark} "
                    f"RMS {st['rms_m']:5.2f}  평행 {st['translation_m']:5.2f}  "
                    f"방사기울기 {st['radial_slope']:+.4f}  yaw {st['yaw_deg']:+6.2f}°"
                )

        if not pair_obs:
            raise CommandError(
                '매칭된 프레임 쌍이 없습니다. --max-pair-sep 을 늘리거나 '
                '--class-id 를 확인하세요.'
            )

        self.stdout.write('')
        self.stdout.write(
            f"매칭 필터: 채택 {match_stats['kept']} / "
            f"거리초과 {match_stats['far']} / 비상호 {match_stats['not_mutual']} / "
            f"모호(비율검정) {match_stats['ambiguous']}"
        )
        if match_stats['ambiguous'] > match_stats['kept']:
            self.stdout.write(self.style.WARNING(
                '  모호 기각이 채택보다 많습니다 — 반복 격자에서 정상이지만, '
                '--max-dist 를 더 줄이면 남는 매칭의 품질이 올라갑니다.'
            ))

        res = np.vstack(all_res)
        rad = np.vstack(all_rad)
        st = decompose(res, rad)
        # 쌍별 중심화 후의 방사/접선 — 프레임 '내부' 왜곡만 남는다
        st_in = decompose(np.vstack(centered_pool), rad)

        self.stdout.write('')
        self.stdout.write('=' * 78)
        self.stdout.write(
            f"전체 — 프레임쌍 {len(pair_obs)} / 매칭 {st['n']} / 평균반경 {st['mean_radius_m']:.1f} m"
        )
        self.stdout.write(f"  RMS 잔차     : {st['rms_m']:.3f} m")
        self.stdout.write(
            f"  평행 성분     : {st['translation_m']:.3f} m  "
            f"[{st['translation_vec'][0]:+.3f}, {st['translation_vec'][1]:+.3f}] (E, N)"
        )
        self.stdout.write(
            f"  방사 기울기   : {st_in['radial_slope']:+.5f}  (쌍별 중심화 후 — 프레임 내부 왜곡)"
        )
        self.stdout.write(f"  yaw 추정     : {st_in['yaw_deg']:+.3f}°  (쌍별 중심화 후)")
        self.stdout.write(
            f"  [참고] 전역 중심화 값: 방사 {st['radial_slope']:+.5f}, yaw {st['yaw_deg']:+.3f}° "
            f"— 쌍별 평행이동에 오염되므로 해석하지 말 것"
        )
        prof = radial_profile(np.vstack(centered_pool), rad)
        if prof:
            self.stdout.write('')
            self.stdout.write('  방사 성분 프로파일 (쌍별 중심화 후, 반경 구간별 평균):')
            for rc, val, cnt in prof:
                bar = '█' * int(min(abs(val), 0.6) / 0.6 * 30)
                sign = '-' if val < 0 else '+'
                self.stdout.write(f"    r {rc:5.1f} m  {sign}{abs(val):.3f} m  {bar} (n={cnt})")
            # 형태 판정: |마지막|/|처음| 비율은 프로파일 부호가 뒤집히면 무의미하다
            # (쌍별 중심화 때문에 프로파일은 평균 0 근처를 가로지르는 것이 정상).
            # 선형 모형과 3차 모형을 표본수 가중 최소제곱으로 적합해 비교한다.
            rr = np.array([p_[0] for p_ in prof])
            yy = np.array([p_[1] for p_ in prof])
            ww = np.sqrt(np.array([p_[2] for p_ in prof], dtype=float))

            def _fit(X):
                b, *_ = np.linalg.lstsq(X * ww[:, None], yy * ww, rcond=None)
                resid = ((yy - X @ b) ** 2 * ww ** 2).sum()
                mean = (yy * ww ** 2).sum() / (ww ** 2).sum()
                tot = ((yy - mean) ** 2 * ww ** 2).sum()
                return b, (1 - resid / tot if tot > 0 else float('nan')), resid

            ones = np.ones_like(rr)
            b_lin, r2_lin, ss_lin = _fit(np.stack([ones, rr], 1))
            b_cub, r2_cub, ss_cub = _fit(np.stack([ones, rr, rr ** 3], 1))
            drop = 1 - ss_cub / ss_lin if ss_lin > 0 else 0.0
            rms_sys = float(np.sqrt((yy ** 2 * ww ** 2).sum() / (ww ** 2).sum()))

            self.stdout.write(
                f"    선형 적합 R² {r2_lin:.3f} / 3차 적합 R² {r2_cub:.3f} "
                f"(잔차 {drop*100:.0f}% 감소)"
            )
            if drop > 0.5 and abs(b_cub[2]) > 1e-6:
                self.stdout.write(self.style.WARNING(
                    f"    → r³ 항이 지배적입니다 (계수 {b_cub[2]:+.3e} m/m³). "
                    f"미보정 렌즈 왜곡입니다 (DewarpData 없음 → D=0)."
                ))
            else:
                self.stdout.write("    → 3차 항이 뚜렷하지 않습니다.")
            self.stdout.write(
                f"    이 계통 성분의 RMS 기여는 {rms_sys:.3f} m "
                f"(전체 RMS {st['rms_m']:.3f} m 중 분산 기여 "
                f"{(rms_sys / st['rms_m']) ** 2 * 100:.1f}%). "
                f"{'보정 이득이 작습니다.' if (rms_sys / st['rms_m']) ** 2 < 0.1 else '보정을 검토할 만합니다.'}"
            )

        if st['n'] < RELIABLE_N * 3:
            self.stdout.write(self.style.WARNING(
                f"  ! 매칭 {st['n']} 개는 통계적으로 얇습니다. --limit 을 늘리세요."
            ))

        # ── 네트워크 조정: 프레임별 위치 오프셋 ───────────────────────
        offsets = solve_frame_offsets(len(frames), pair_obs, opts['damping'])
        support = frame_support(len(frames), pair_obs)

        before = np.array([np.linalg.norm(r) for _, _, r, _ in pair_obs])
        after = np.array([
            np.linalg.norm((offsets[j] - offsets[i]) - r) for i, j, r, _ in pair_obs
        ])
        rms_before = float(np.sqrt((before ** 2).mean()))
        rms_after = float(np.sqrt((after ** 2).mean()))

        self.stdout.write('')
        self.stdout.write('=' * 78)
        self.stdout.write('프레임별 위치 오프셋 (네트워크 최소제곱, 평균=0 기준)')
        self.stdout.write(
            f"  쌍 평균잔차 RMS: 조정 전 {rms_before:.3f} m → 조정 후 {rms_after:.3f} m"
        )
        thr = opts['outlier_threshold']
        ranked = sorted(range(len(frames)), key=lambda k: -np.linalg.norm(offsets[k]))
        for k in ranked:
            mag = float(np.linalg.norm(offsets[k]))
            if mag < thr and k != ranked[0]:
                continue
            fr = frames[k]
            npair, nmatch = support[k]
            weak = nmatch < opts['min_support']
            flag = ''
            if weak:
                flag = self.style.WARNING(' ← 지지 부족(신뢰 불가)')
            elif mag >= thr:
                flag = self.style.WARNING(' ← 이탈')
            self.stdout.write(
                f"  task {fr.task_id:>5} [{fr.rtk_label():>5}]  "
                f"오프셋 {mag:5.2f} m  "
                f"[{offsets[k][0]:+.2f}, {offsets[k][1]:+.2f}] (E, N)  "
                f"쌍 {npair:>2} 매칭 {nmatch:>3}{flag}"
            )

        if opts['export_offsets']:
            import json
            payload = {
                'plane_kwargs': kwargs,
                'rms_before_m': round(rms_before, 4),
                'rms_after_m': round(rms_after, 4),
                'min_support': opts['min_support'],
                'frames': [
                    {
                        'task_id': frames[k].task_id,
                        'offset_east_m': round(float(offsets[k][0]), 4),
                        'offset_north_m': round(float(offsets[k][1]), 4),
                        # 드론 위치도 함께 저장한다. 오프셋은 공간적으로 매끄러우므로
                        # (위치 선형 사상 R² 0.5 수준), 지지가 부족한 프레임은
                        # 이웃 프레임에서 공간 보간할 수 있다. alignment.py 참고.
                        'lat': round(float(frames[k].lat0), 8),
                        'lng': round(float(frames[k].lng0), 8),
                        'pairs': support[k][0],
                        'matches': support[k][1],
                        'trusted': support[k][1] >= opts['min_support'],
                    }
                    for k in range(len(frames))
                ],
            }
            with open(opts['export_offsets'], 'w') as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            self.stdout.write(
                f"\n오프셋 {len(frames)} 건을 {opts['export_offsets']} 에 저장했습니다. "
                f"trusted=false 인 항목은 적용하지 마세요."
            )

        # ── 오프셋 원인 회귀 ──────────────────────────────────────────
        exp = explain_offsets(frames, offsets)
        if exp:
            self.stdout.write('')
            self.stdout.write('=' * 78)
            self.stdout.write('프레임 오프셋의 원인 회귀')
            v = exp.get('velocity')
            if v:
                self.stdout.write(
                    f"  속도 모델 (시각 동기)  R² {v['r2']:+.3f}  "
                    f"평균속도 {v['speed_mean']:.2f} m/s  "
                    f"Δt 추정 {v['dt_estimate_s']:.3f} s  "
                    f"(특이값 {v['singular_values_s'][0]:.3f}, {v['singular_values_s'][1]:.3f})"
                )
                if v.get('rank_deficient'):
                    self.stdout.write(
                        '    (속도가 한 축으로만 변해 그 축만 구속됩니다 — 왕복 비행에서 정상)'
                    )
            si = exp.get('scale_isotropic')
            if si:
                self.stdout.write(
                    f"  등방 스케일 (평면높이) R² {si['r2']:+.3f}  "
                    f"k {si['k']:+.5f}  → 평면 오차 {si['plane_error_m']:+.2f} m"
                )
            sc = exp.get('scale')
            if sc:
                self.stdout.write(
                    f"  위치 선형 사상 (경사)  R² {sc['r2']:+.3f}  "
                    f"특이값 {sc['singular_values'][0]:+.5f} / {sc['singular_values'][1]:+.5f}  "
                    f"비등방 {sc['anisotropy']:.1f}배  (측량범위 {sc['span_m']:.0f} m)"
                )
                M = np.array(sc['matrix'])
                U, S, Vt = np.linalg.svd(M)
                self.stdout.write(
                    f"    주방향: 위치 {bearing_of(Vt[0]):.0f}° 방향으로 이동하면 "
                    f"오프셋이 {bearing_of(U[:, 0]):.0f}° 방향으로 "
                    f"{S[0]:.4f} m/m 씩 변함 "
                    f"(측량범위 전체에서 {S[0] * sc['span_m']:.2f} m)"
                )
                if sc['r2'] > 0.4 and sc['anisotropy'] > 2.0:
                    self.stdout.write(self.style.WARNING(
                        '    → 오프셋이 위치에 방향성 있게 의존합니다. 사이트 전체에 '
                        '단일 수평 평면을 쓰는 것이 원인일 수 있습니다 (부지 경사). '
                        'DSM 또는 기울어진 평면 도입을 검토하세요.'
                    ))
            if v and sc:
                if v['r2'] > sc['r2'] + 0.15:
                    self.stdout.write(self.style.WARNING(
                        f"  → 속도 모델이 우세합니다. GNSS-카메라 시각 동기 오차 "
                        f"Δt≈{v['dt_estimate_s']:.3f} s 로 보입니다. "
                        f"노출 시점 위치를 gps_position - v*Δt 로 보정하세요."
                    ))
                elif sc['r2'] > v['r2'] + 0.15:
                    # 위치 의존이 우세. 등방이면 평면 '높이', 비등방이면 평면 '기울기'.
                    if sc['anisotropy'] <= 2.0 and si:
                        self.stdout.write(self.style.WARNING(
                            f"  → 위치 모델이 우세하고 거의 등방입니다. 투영 평면 "
                            f"높이를 {si['plane_error_m']:+.2f} m 조정하세요."
                        ))
                    else:
                        self.stdout.write(self.style.WARNING(
                            f"  → 위치 모델이 우세하고 비등방({sc['anisotropy']:.1f}배)입니다. "
                            f"단일 높이 조정으로는 해결되지 않습니다 — 부지 경사이므로 "
                            f"기울어진 평면 또는 DSM 이 필요합니다. "
                            f"측량범위 {sc['span_m']:.0f} m 에 걸친 사상 특이값 "
                            f"{sc['singular_values'][0]:+.5f} / {sc['singular_values'][1]:+.5f}."
                        ))
                else:
                    self.stdout.write(
                        '  → 두 모델 모두 지배적이지 않습니다. 프레임별 오프셋을 '
                        '그대로 GeoAlignmentCorrection 에 적용하는 편이 낫습니다.'
                    )

        # ── 판정 ─────────────────────────────────────────────────────
        self.stdout.write('')
        self.stdout.write('판정:')
        slope = st_in['radial_slope']
        if not math.isnan(slope) and abs(slope) > 0.01:
            self.stdout.write(self.style.WARNING(
                f'  · 방사 기울기 {slope:+.4f} (쌍별 중심화 후) — 반경의 '
                f'{abs(slope)*100:.1f}%. 주의: 균일 스케일 오차(평면 높이/초점거리)는 '
                f'쌍 사이에서 평행이동으로 나타나 이 값에 남지 않습니다. 따라서 이 성분은 '
                f'평면 높이가 아니라 (a) 미보정 렌즈 왜곡 — DewarpData 가 없어 D=0 이거나, '
                f'(b) 오매칭 잔재입니다. --max-dist 를 줄여 (b) 를 먼저 배제하세요.'
            ))
        else:
            self.stdout.write(
                f'  · 방사 기울기 {slope:+.4f} — 프레임 내부 방사 왜곡은 무시할 수준입니다.'
            )
        if abs(st_in['yaw_deg']) > 0.3:
            self.stdout.write(self.style.WARNING(
                f"  · yaw {st_in['yaw_deg']:+.2f}° — 짐벌 방위 편향. 평균반경에서 "
                f"{abs(math.radians(st_in['yaw_deg']))*st['mean_radius_m']:.2f} m 기여."
            ))

        drop = 1 - (rms_after / rms_before) if rms_before else 0
        if drop > 0.6:
            self.stdout.write(self.style.SUCCESS(
                f'  · 프레임별 평행이동만으로 잔차의 {drop*100:.0f}% 가 설명됩니다. '
                f'카메라 모델이 아니라 위치 기준(datum) 문제입니다 — RTK 상태를 확인하고, '
                f'위 오프셋을 GeoAlignmentCorrection 에 넣으면 됩니다.'
            ))
        else:
            self.stdout.write(
                f'  · 평행이동으로 설명되는 비율 {drop*100:.0f}%. '
                f'나머지는 라벨 편차 또는 프레임 내부 왜곡입니다.'
            )
"""프레임 등록 오차의 전역 네트워크 정렬.

왜 greedy 정렬로는 부족했나 (실측 근거)
---------------------------------------
``ortho.mosaic_frames`` 의 프레임별 정렬(캔버스와 1:1 비교) 을 실데이터
247 장에 돌린 결과 (``summary.json``):

* 보정된 프레임 **32/247 (13%)** — 215 장은 손도 못 댔다.
* 보정 크기 중앙값 0.699 m, **최대 0.979 m** — 상한 ``align_max_m=1.0`` 에
  분포가 잘려 있다 (0.7~1.0 구간에 16 장). 실제 오차는 1 m 를 넘는다.
* ZNCC 응답 중앙값 0.54, 최소 0.31 — 게이트(0.30) 경계에 몰려 있고,
  215 장은 게이트에서 탈락했다.

원인은 방식 자체다:

1. **캔버스 1:1 비교** — 아직 안 놓인 프레임과는 비교할 수 없고, 이미 놓인
   부분과의 겹침이 좁으면 측정이 실패한다. 그래서 대부분이 탈락했다.
2. **순차 탐욕** — 순서에 의존하고, 앞 프레임의 오차가 뒤로 전파된다.
3. **탐색 반경 한계** — 패널 행 주기(실측 2.72 m) 의 절반 미만으로 묶여
   있어 1 m 이상은 원리적으로 못 잡는다.

이 모듈의 방식
--------------
1. **모든 겹침 쌍**에 대해 상대 이동량 ``d_ij`` 를 측정한다 (캔버스가 아니라
   프레임끼리). 한 프레임이 이웃 4~8 장과 제약을 갖게 되므로 전 프레임이
   커버된다.
2. **coarse-to-fine** — 1단계는 행 주기보다 크게 블러해서 주기 구조를 지운
   뒤 넓은 반경(±``coarse_max_m``)으로 탐색한다. 주기 성분이 없으므로
   ±1~3 m 도 모호하지 않다. 2단계는 원해상도에서 좁은 반경으로 정밀화한다.
3. **전역 최소제곱** — ``o_i − o_j = d_ij`` 를 프레임 오프셋 ``o`` 에 대해
   풀되, 게이지(평균 0) 를 고정한다. IRLS 로 이상치를 눌러 준다.
   순서 의존성과 오차 전파가 사라진다.
4. **포화 마스킹** — 정반사로 날아간 픽셀(실측 원본에서 1.9~6.0%) 은 프레임
   마다 위치가 달라 정렬을 망친다. 매칭에서 제외한다.

한계
----
* 평행이동만 다룬다. 회전/축척은 BA 의 몫이다.
* 전역 게이지(평균 0) 는 절대 위치를 정하지 않는다 — 시임 연속성은 잡히지만
  전체가 함께 이동할 수 있다. 절대 정확도는 RTK/BA 가 담당한다.
* 겹침에 텍스처가 전혀 없는 쌍은 자동 탈락하고, 그 프레임이 고립되면
  오프셋 0 으로 남는다 (로그에 보고).
"""

from __future__ import annotations

import logging
from dataclasses import replace as dc_replace

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "AlignmentResult",
    "measure_pair_shift",
    "solve_offset_network",
    "align_frames_network",
]


class AlignmentResult:
    """네트워크 정렬 결과."""

    def __init__(self, offsets_en: np.ndarray, n_pairs: int, n_used: int,
                 residual_m: float, isolated: list[int]):
        self.offsets_en = offsets_en        # (N, 2) 프레임별 (dE, dN) 보정 (m)
        self.n_pairs = n_pairs
        self.n_used = n_used
        self.residual_m = residual_m
        self.isolated = isolated

    @property
    def magnitudes(self) -> np.ndarray:
        return np.hypot(self.offsets_en[:, 0], self.offsets_en[:, 1])

    def __repr__(self) -> str:
        m = self.magnitudes
        return (f"AlignmentResult(N={len(m)}, 쌍 {self.n_used}/{self.n_pairs}, "
                f"보정 중앙값 {np.median(m):.3f} m, 최대 {m.max():.3f} m, "
                f"잔차 {self.residual_m:.3f} m)")


# ---------------------------------------------------------------------------
# 쌍별 이동량 측정
# ---------------------------------------------------------------------------
def _sift_translation(a: np.ndarray, b: np.ndarray, mask: np.ndarray,
                      max_shift_px: float,
                      min_matches: int = 12
                      ) -> tuple[float, float, float, int] | None:
    """정사 격자 위 두 영상의 순수 평행이동 추정 (SIFT + RANSAC).

    템플릿 매칭 대신 특징점을 쓰는 이유:

    * **부호가 명확하다.** 대응점 ``p_a ↔ p_b`` 가 같은 정사 격자 위에 있으면
      ``b`` 를 ``a`` 에 맞추는 이동은 정의상 ``median(p_a − p_b)`` 다.
      템플릿 매칭은 minloc/반경/서브픽셀 부호를 세 번 뒤집어야 해서
      실수하기 쉽다 (실제로 이 파일의 1차 구현이 그렇게 틀렸다).
    * **주기 구조에 강하다.** 반복 패널 격자에서도 ratio test 가 모호한
      대응을 버리고, RANSAC 이 남은 이상치를 제거한다. 전역 FFT 피크는
      주기의 정수배로 통째로 끌려간다.
    * **포화 영역을 자연히 배제한다.** 마스크로 keypoint 를 걸러내면 된다.

    Returns ``(dx, dy, inlier_ratio, n_inliers)`` — 출력 픽셀 단위,
    ``b`` 에 더할 이동. 실패 시 ``None``.
    """
    au = np.clip(a, 0, 255).astype(np.uint8)
    bu = np.clip(b, 0, 255).astype(np.uint8)
    try:
        sift = cv2.SIFT_create(nfeatures=3000)
    except AttributeError:                              # pragma: no cover
        return None
    m8 = (mask > 0).astype(np.uint8) * 255
    ka, da = sift.detectAndCompute(au, m8)
    kb, db = sift.detectAndCompute(bu, m8)
    if da is None or db is None or len(ka) < min_matches or len(kb) < min_matches:
        return None

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(db, da, k=2)                      # query=b, train=a
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    if len(good) < min_matches:
        return None

    pb = np.array([kb[m.queryIdx].pt for m in good])
    pa = np.array([ka[m.trainIdx].pt for m in good])
    d = pa - pb                                          # b → a 이동

    # 탐색 반경 밖 대응은 기하학적으로 불가능 → 사전 제거.
    keep = np.hypot(d[:, 0], d[:, 1]) <= max_shift_px
    if keep.sum() < min_matches:
        return None
    d = d[keep]

    # RANSAC: 평행이동 1개 파라미터라 '중앙값 + 잔차 임계' 로 충분.
    med = np.median(d, axis=0)
    r = np.hypot(*(d - med).T)
    thr = max(2.0, float(np.percentile(r, 60)))
    inl = r <= thr
    if inl.sum() < min_matches:
        return None
    fin = d[inl].mean(axis=0)
    return float(fin[0]), float(fin[1]), float(inl.mean()), int(inl.sum())


def measure_pair_shift(fh_i, fh_j, img_i, img_j, gsd_m: float,
                       *,
                       max_shift_m: float = 3.0,
                       max_side: int = 900,
                       sat_threshold: int = 245,
                       min_inlier_ratio: float = 0.35,
                       min_inliers: int = 12,
                       **_ignored) -> tuple[float, float, float] | None:
    """두 프레임 겹침에서 상대 이동량 ``(dE, dN, weight)`` (m).

    ``(dE, dN)`` = 프레임 j 를 i 에 맞추기 위해 **j 에 더할 ENU 이동**.
    실패 시 ``None``.
    """
    bi, bj = fh_i.footprint_bounds(), fh_j.footprint_bounds()
    if bi is None or bj is None:
        return None
    x0 = max(bi[0], bj[0]); y0 = max(bi[1], bj[1])
    x1 = min(bi[2], bj[2]); y1 = min(bi[3], bj[3])
    if (x1 - x0) < 6.0 or (y1 - y0) < 6.0:
        return None

    span = max(x1 - x0, y1 - y0)
    g = max(span / max_side, gsd_m)
    nx, ny = int((x1 - x0) / g), int((y1 - y0) / g)
    if nx < 96 or ny < 96:
        return None

    def warp(fh, img):
        H = fh.ortho_pixel_matrix(x0, y1, g)
        return cv2.warpPerspective(
            img, H, (nx, ny),
            flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    wi, wj = warp(fh_i, img_i), warp(fh_j, img_j)
    if wi.ndim == 3:
        wi = wi.mean(axis=2); wj = wj.mean(axis=2)
    wi = wi.astype(np.float32); wj = wj.astype(np.float32)

    # 유효 + 비포화. 정반사는 프레임마다 위치가 달라 대응을 오염시킨다.
    mask = ((wi > 1) & (wj > 1)
            & (wi < sat_threshold) & (wj < sat_threshold))
    if mask.mean() < 0.2:
        return None

    res = _sift_translation(wi, wj, mask, max_shift_m / g,
                            min_matches=min_inliers)
    if res is None:
        return None
    dx, dy, ratio, n_inl = res
    if ratio < min_inlier_ratio:
        return None

    # 출력 격자: +x = 동, +y = 남 → ENU 는 (dx·g, −dy·g).
    return dx * g, -dy * g, float(ratio * np.sqrt(n_inl))


# ---------------------------------------------------------------------------
# 전역 네트워크 해
# ---------------------------------------------------------------------------
def solve_offset_network(n_frames: int,
                         pairs: list[tuple[int, int]],
                         shifts: np.ndarray,
                         weights: np.ndarray,
                         *,
                         n_irls: int = 4,
                         huber_m: float = 0.25) -> tuple[np.ndarray, float]:
    """``o_i − o_j = d_ij`` 최소제곱 (게이지: Σo = 0), IRLS 로버스트.

    Returns ``(offsets (N,2), 잔차 중앙값 m)``.
    """
    m = len(pairs)
    if m == 0:
        return np.zeros((n_frames, 2)), float("nan")

    rows = np.arange(m)
    A = np.zeros((m + 1, n_frames), dtype=np.float64)
    A[rows, [p[0] for p in pairs]] = 1.0
    A[rows, [p[1] for p in pairs]] = -1.0
    A[m, :] = 1.0                        # 게이지 제약

    w = np.asarray(weights, dtype=np.float64).copy()
    off = np.zeros((n_frames, 2))
    resid = float("nan")

    for _ in range(max(n_irls, 1)):
        W = np.concatenate([w, [w.max() * n_frames if w.size else 1.0]])
        Aw = A * W[:, None]
        sol = []
        for axis in range(2):
            b = np.concatenate([shifts[:, axis], [0.0]]) * W
            x, *_ = np.linalg.lstsq(Aw, b, rcond=None)
            sol.append(x)
        off = np.stack(sol, axis=1)

        pred = off[[p[0] for p in pairs]] - off[[p[1] for p in pairs]]
        r = np.hypot(*(pred - shifts).T)
        resid = float(np.median(r))
        # Huber 재가중.
        w = np.asarray(weights, dtype=np.float64) * np.minimum(
            1.0, huber_m / np.maximum(r, 1e-6))

    return off, resid


# ---------------------------------------------------------------------------
# 상위 진입점
# ---------------------------------------------------------------------------
def align_frames_network(frames: list,
                         image_paths: list,
                         gsd_m: float,
                         *,
                         max_neighbors: int = 6,
                         coarse_max_m: float = 3.0,
                         fine_max_m: float = 0.6,
                         row_period_m: float = 2.72,
                         min_zncc: float = 0.25,
                         max_offset_m: float = 4.0) -> tuple[list, AlignmentResult | None]:
    """겹침 쌍 측정 → 전역 해 → 보정된 ``FrameHomography`` 리스트.

    이미지는 쌍 처리 순서에 맞춰 소량만 캐시하므로 수백 장도 처리 가능하다.
    """
    n = len(frames)
    if n < 3:
        return frames, None

    C = np.array([f.camera_xyz[:2] for f in frames])
    # 인접쌍 후보: 각 프레임의 최근접 이웃.
    cand: set[tuple[int, int]] = set()
    for i in range(n):
        d = np.hypot(C[:, 0] - C[i, 0], C[:, 1] - C[i, 1])
        for j in np.argsort(d)[1:max_neighbors + 1]:
            j = int(j)
            cand.add((min(i, j), max(i, j)))
    cand_l = sorted(cand)

    cache: dict[int, np.ndarray] = {}

    def gray(idx):
        if idx not in cache:
            im = cv2.imread(str(image_paths[idx]), cv2.IMREAD_GRAYSCALE)
            cache[idx] = im
            if len(cache) > 12:
                cache.pop(next(iter(cache)))
        return cache[idx]

    pairs, shifts, weights = [], [], []
    for i, j in cand_l:
        gi, gj = gray(i), gray(j)
        if gi is None or gj is None:
            continue
        res = measure_pair_shift(frames[i], frames[j], gi, gj, gsd_m,
                                 coarse_max_m=coarse_max_m,
                                 fine_max_m=fine_max_m,
                                 row_period_m=row_period_m,
                                 min_zncc=min_zncc)
        if res is None:
            continue
        de, dn, z = res
        if np.hypot(de, dn) > coarse_max_m * 1.5:
            continue
        pairs.append((i, j)); shifts.append((de, dn)); weights.append(z)

    if len(pairs) < n // 2:
        logger.warning("네트워크 정렬: 유효 쌍이 %d개로 부족 (프레임 %d) — "
                       "겹침 텍스처가 없거나 포화가 심합니다. 정렬 생략.",
                       len(pairs), n)
        return frames, None

    off, resid = solve_offset_network(
        n, pairs, np.array(shifts), np.array(weights))

    # 물리적 타당성 — 이 범위를 넘는 프레임은 보정하지 않는다.
    mag = np.hypot(off[:, 0], off[:, 1])
    bad = mag > max_offset_m
    if bad.any():
        logger.warning("네트워크 정렬: 보정량이 %.1f m 를 넘는 프레임 %d장은 "
                       "제외합니다 (최대 %.2f m).", max_offset_m,
                       int(bad.sum()), float(mag.max()))
        off[bad] = 0.0

    seen = {i for p in pairs for i in p}
    isolated = [i for i in range(n) if i not in seen]

    out = []
    for k, fh in enumerate(frames):
        if abs(off[k, 0]) < 1e-6 and abs(off[k, 1]) < 1e-6:
            out.append(fh)
        else:
            out.append(dc_replace(fh, origin_xy=fh.origin_xy + off[k]))

    mag = np.hypot(off[:, 0], off[:, 1])
    logger.info("네트워크 정렬: 쌍 %d개, 프레임 %d장 보정 (중앙값 %.3f m, "
                "최대 %.3f m), 쌍 잔차 중앙값 %.3f m",
                len(pairs), int((mag > 1e-6).sum()),
                float(np.median(mag[mag > 1e-6])) if (mag > 1e-6).any() else 0.0,
                float(mag.max()), resid)
    if isolated:
        logger.warning("이웃 제약이 없는 고립 프레임 %d장 (보정 0 으로 유지): %s",
                       len(isolated), isolated[:8])
    if resid > 0.3:
        logger.warning("쌍 잔차 중앙값이 %.2f m 입니다 — 평행이동만으로는 "
                       "설명되지 않는 성분(회전/축척/지형 기복)이 남아 있습니다. "
                       "SfM/BA 품질을 확인하세요.", resid)

    return out, AlignmentResult(off, len(cand_l), len(pairs), resid, isolated)

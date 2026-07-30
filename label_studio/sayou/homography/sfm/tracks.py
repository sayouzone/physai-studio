"""페어별 매칭을 다중 시점 track 으로 연결.

track 의 정의
-------------
``track`` = 동일 지상점에 대응하는 관측들의 집합
         = ``[(image_idx, keypoint_idx, px, py), ...]``

Union-Find 로 ``(image_idx, keypoint_idx)`` 노드들을 연결해, 페어별로 끊겨 있던
매칭들을 하나의 그룹으로 묶는다. 그 후 그룹별로 다음 조건을 검사:

* **충돌 검사**: 한 이미지에서 두 개 이상의 keypoint 가 같은 track 에 들어가면
  매칭 오류로 보고 track 전체를 폐기 (다른 지상점이 한 track 으로 섞이는 사고).
* **길이 필터**: ``min_track_len`` 미만은 삼각측량 불가, ``max_track_len`` 초과는
  보통 반복 텍스처 (태양광 패널 그리드 등) 의 잘못된 매칭.
"""

from __future__ import annotations

import logging

import numpy as np
import time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)


class _UnionFind:
    """경로 압축 + 랭크 기반 Union-Find."""

    def __init__(self):
        self.parent: dict = {}
        self.rank: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        self.rank.setdefault(x, 0)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # 경로 압축.
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_tracks(matches: dict,
                 features: dict,
                 min_track_len: int = 2,
                 max_track_len: int = 30) -> list:
    """페어별 매칭들을 연결해 track 리스트를 만든다.

    track = 동일 지상점에 대응하는 관측들의 집합
          = [(image_idx, keypoint_idx, px, py), ...]

    구현 (vectorized)
    -----------------
    원본 dict 기반 Union-Find 는 Python 인터프리터 오버헤드 + 튜플 hash 비용
    + 캐시 미스로 매우 느렸다. 특히 GCE g2 같은 vCPU 환경 (Cascade Lake,
    낮은 클럭 + 작은 L3) 에서는 M4 Pro 대비 3~5 배 느림.

    재구현 핵심:

    1) ``(image_idx, keypoint_idx)`` 튜플을 flat 정수 ID 로 변환::

           node_id = image_idx * MAX_KP + keypoint_idx

       이렇게 하면 dict hash + 튜플 비용이 제거되고 모든 연산이 정수 배열 위에서
       이뤄진다.

    2) 모든 매칭을 ``(N, 2)`` int 엣지 배열 한 번에 모은 뒤
       ``scipy.sparse.csgraph.connected_components`` (C 구현) 로 한 호출에
       Union-Find 처리.

    3) 그룹별 관측 모으기는 ``np.argsort(labels)`` + ``np.split`` 로 벡터화.

    4) 충돌 검사 (한 이미지에서 두 keypoint 가 같은 track 에 들어감) 도
       그룹별로 numpy 로 처리.

    Parameters
    ----------
    matches : ``build_tie_points`` 의 출력
    features : ``build_tie_points`` 의 출력 (keypoint 픽셀좌표 조회용 — 본
        함수는 이미 ``matches`` 안의 ``pts_*`` 에 좌표가 있어서 직접 사용은
        하지 않지만, 기존 API 호환성을 위해 받는다)
    min_track_len : track 으로 인정할 최소 관측 수 (2 = 최소 삼각측량 가능)
    max_track_len : 비정상적으로 긴 track 제거 (보통 매칭 오류).
                    하나의 점이 30 장 이상에 나타나면 의심스러움.

    Returns
    -------
    tracks : list of track. 각 track 은
        ``[(image_idx, keypoint_idx, px, py), ...]`` 형식 — 원본 API 동일.
    """
    if not matches:
        logger.info("track 생성: 0개 (매칭이 비어있음)")
        return []

    t0 = time.perf_counter()

    # -----------------------------------------------------------------------
    # 1) 모든 관측을 평탄화 — 단일 numpy 배열로
    # -----------------------------------------------------------------------
    # 각 페어 (i, j) 의 매칭 K 개에 대해 i-쪽 관측 K 개 + j-쪽 관측 K 개.
    # 전체 관측 수 = sum(2 * K_pair) 이고 보통 수십만 ~ 수백만.
    n_obs_per_pair = [int(len(m[2])) for m in matches.values()]  # idx_i 길이
    total_obs = 2 * sum(n_obs_per_pair)

    if total_obs == 0:
        logger.info("track 생성: 0개 (매칭은 있으나 관측 없음)")
        return []

    # node_id 인코딩에 필요한 최대 keypoint 인덱스 산출.
    # 각 페어의 idx_i, idx_j 의 최대값 + 1 (안전 마진).
    max_kp = 1
    max_img = 0
    for (i, j), (_, _, idx_i, idx_j) in matches.items():
        if len(idx_i) > 0:
            max_kp = max(max_kp,
                         int(idx_i.max()) + 1,
                         int(idx_j.max()) + 1)
        max_img = max(max_img, i, j)
    n_images = max_img + 1

    # flat node id = image_idx * max_kp + keypoint_idx.
    # int64 로 안전한 범위: max_kp * n_images < 2^63. 보통 5000장 * 8000kp =
    # 4천만 → 한참 여유.
    obs_img = np.empty(total_obs, dtype=np.int32)
    obs_kp = np.empty(total_obs, dtype=np.int64)
    obs_px = np.empty(total_obs, dtype=np.float64)
    obs_py = np.empty(total_obs, dtype=np.float64)

    # 엣지 배열 — 그래프 입력. 페어당 K 개 엣지.
    edges_a = np.empty(total_obs // 2, dtype=np.int64)
    edges_b = np.empty(total_obs // 2, dtype=np.int64)

    off_obs = 0
    off_edge = 0
    for (i, j), (pts_i, pts_j, idx_i, idx_j) in matches.items():
        K = len(idx_i)
        if K == 0:
            continue
        idx_i_int = idx_i.astype(np.int64, copy=False)
        idx_j_int = idx_j.astype(np.int64, copy=False)
        node_i = i * max_kp + idx_i_int  # (K,)
        node_j = j * max_kp + idx_j_int  # (K,)

        # 엣지 (i-측 관측 ↔ j-측 관측).
        edges_a[off_edge:off_edge + K] = node_i
        edges_b[off_edge:off_edge + K] = node_j
        off_edge += K

        # i-측 관측 K 개 기록.
        s = off_obs
        e = s + K
        obs_img[s:e] = i
        obs_kp[s:e] = idx_i_int
        obs_px[s:e] = pts_i[:, 0]
        obs_py[s:e] = pts_i[:, 1]
        off_obs = e

        # j-측 관측 K 개 기록.
        s = off_obs
        e = s + K
        obs_img[s:e] = j
        obs_kp[s:e] = idx_j_int
        obs_px[s:e] = pts_j[:, 0]
        obs_py[s:e] = pts_j[:, 1]
        off_obs = e

    # 실제 사용한 부분만 자르기 (혹시 K=0 페어가 있었을 경우 대비).
    obs_img = obs_img[:off_obs]
    obs_kp = obs_kp[:off_obs]
    obs_px = obs_px[:off_obs]
    obs_py = obs_py[:off_obs]
    edges_a = edges_a[:off_edge]
    edges_b = edges_b[:off_edge]

    # 각 관측의 node_id (image, keypoint 조합).
    obs_node = obs_img.astype(np.int64) * max_kp + obs_kp

    t1 = time.perf_counter()

    # -----------------------------------------------------------------------
    # 2) connected_components 로 Union-Find 일괄 처리
    # -----------------------------------------------------------------------
    # 그래프 노드는 obs_node 의 unique 값들. 노드 ID 가 sparse (수십만 범위
    # 중 일부만 사용) 이므로 unique 매핑으로 0..n_unique-1 로 압축.
    # 압축하지 않으면 csgraph 가 max(node_id)+1 만큼 노드를 할당해 메모리
    # 폭발 (예: 10만 장 * 8000 = 8억 노드).
    all_nodes = np.concatenate([edges_a, edges_b, obs_node])
    unique_nodes, inv = np.unique(all_nodes, return_inverse=True)
    n_nodes = unique_nodes.shape[0]

    n_edges = edges_a.shape[0]
    edges_a_small = inv[:n_edges]
    edges_b_small = inv[n_edges:2 * n_edges]
    obs_node_small = inv[2 * n_edges:]  # 각 관측의 압축된 노드 ID

    # 무방향 그래프 sparse adjacency 행렬. 데이터는 dummy 1.
    data = np.ones(n_edges, dtype=np.uint8)
    graph = coo_matrix(
        (data, (edges_a_small, edges_b_small)),
        shape=(n_nodes, n_nodes),
    )
    # connected_components 는 directed=False 면 자동으로 양방향 처리.
    n_comp, labels = connected_components(
        graph, directed=False, return_labels=True,
    )

    t2 = time.perf_counter()

    # -----------------------------------------------------------------------
    # 3) component 별 관측 그룹화 — 전역 unique + argsort + split
    # -----------------------------------------------------------------------
    # 중요: 한 (image, keypoint) 가 여러 페어를 통해 여러 관측으로 등장할 수
    # 있다. 원본 dict 구현은 ``obs[(i, ki)] = ...`` 로 자동 중복 제거 후 검사
    # 했으므로, 본 함수도 unique 처리 후 검사한다.
    #
    # 성능 핵심: 그룹마다 ``np.unique`` 를 부르면 Python 호출 오버헤드가 폭발
    # (그룹 수십만 개 × unique 호출 비용 = 전체 시간의 60%). 대신 전역적으로
    # ``(label, node_id)`` 쌍에 대해 한 번만 unique 를 호출한다.

    # 각 관측에 component label 부여.
    obs_label = labels[obs_node_small]  # (total_obs,)

    # (label, node_id) 결합 키 — int64 한 단어로 패킹.
    # n_nodes < 2^32 가정 (수억 노드까지 안전).
    combined = obs_label.astype(np.int64) * (n_nodes + 1) + obs_node_small

    # 전역 unique — 같은 (label, node) 가 여러 관측에 등장하면 첫 번째만 남김.
    _, first_idx = np.unique(combined, return_index=True)
    first_idx = np.sort(first_idx)  # 원래 순서 보존.

    u_label = obs_label[first_idx]
    u_img = obs_img[first_idx]
    u_kp = obs_kp[first_idx]
    u_px = obs_px[first_idx]
    u_py = obs_py[first_idx]

    # label 별 정렬 후 split.
    sort_idx = np.argsort(u_label, kind="stable")
    sorted_labels = u_label[sort_idx]
    sorted_img = u_img[sort_idx]
    sorted_kp = u_kp[sort_idx]
    sorted_px = u_px[sort_idx]
    sorted_py = u_py[sort_idx]

    # 그룹 경계: label 이 바뀌는 인덱스.
    group_starts = np.concatenate([
        [0],
        np.nonzero(np.diff(sorted_labels))[0] + 1,
        [len(sorted_labels)],
    ])
    n_groups = len(group_starts) - 1

    t3a = time.perf_counter()

    # -----------------------------------------------------------------------
    # 4) 그룹별 필터링 (충돌/짧음/김) — 벡터화
    # -----------------------------------------------------------------------
    # 그룹별 길이 검사는 벡터로 한 번에. 충돌 검사는 그룹 내 image id 유일성을
    # bincount 로 일괄 처리.
    group_sizes = np.diff(group_starts)  # (n_groups,)

    # 길이 기반 1차 필터.
    keep_len = (group_sizes >= min_track_len) & (group_sizes <= max_track_len)
    dropped_short = int((group_sizes < min_track_len).sum())
    dropped_long = int((group_sizes > max_track_len).sum())

    # 충돌 검사 — 그룹별로 image 가 unique 한지. 그룹 단위 numpy 작업이지만
    # 그룹마다 한 번씩만 (이미 길이 필터 통과한 그룹만).
    # 빠른 충돌 검사: 그룹의 sorted_img 정렬 후 인접 중복이 있는지.
    keep_idx = np.nonzero(keep_len)[0]
    valid_mask = np.zeros(n_groups, dtype=bool)
    dropped_conflict = 0

    tracks: list[list[tuple[int, int, float, float]]] = []
    for g in keep_idx:
        s, e = group_starts[g], group_starts[g + 1]
        img_g = sorted_img[s:e]
        n = img_g.shape[0]

        # 충돌 검사:
        #   - size=2: 두 관측이 같은 component 에 속하려면 매칭 엣지로
        #     연결됐어야 하고 매칭은 항상 다른 두 이미지 사이라 충돌 불가.
        #     → 검사 스킵으로 21만 그룹 × np.unique 호출 회피.
        #   - size>=3: 같은 image 가 두 번 이상 등장하면 충돌 (다른
        #     keypoint 가 한 track 에 섞임). 작은 배열이면 sort 후 diff 가
        #     np.unique 보다 훨씬 빠름.
        if n >= 3:
            si = np.sort(img_g)
            if (si[1:] == si[:-1]).any():
                dropped_conflict += 1
                continue

        # 통과 — track 구성. tolist + zip 이 인덱싱 루프보다 약간 빠름.
        kp_g = sorted_kp[s:e]
        px_g = sorted_px[s:e]
        py_g = sorted_py[s:e]
        track = list(zip(
            img_g.tolist(), kp_g.tolist(),
            px_g.tolist(), py_g.tolist(),
        ))
        tracks.append(track)

    t3 = time.perf_counter()

    logger.info(
        "track 생성: %d개 (제거: 짧음 %d, 김 %d, 충돌 %d)",
        len(tracks), dropped_short, dropped_long, dropped_conflict,
    )
    logger.info(
        "  build_tracks 시간: 평탄화 %.2fs + CC %.2fs + unique/정렬 %.2fs "
        "+ 필터 %.2fs (총 %.2fs, 관측 %d, 노드 %d, 그룹 %d)",
        t1 - t0, t2 - t1, t3a - t2, t3 - t3a, t3 - t0,
        off_obs, n_nodes, n_comp,
    )
    if tracks:
        lengths = np.array([len(t) for t in tracks])
        logger.info("  track 길이 분포: 평균 %.1f, 최대 %d, 2장짜리 %d개",
                    lengths.mean(), lengths.max(), int((lengths == 2).sum()))
    return tracks


__all__ = ["build_tracks"]
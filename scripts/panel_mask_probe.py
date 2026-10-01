#!/usr/bin/env python3
"""패널 마스킹이 매칭 격차를 좁히는지 **측정만** 한다 (파이프라인 미변경).

    python scripts/panel_mask_probe.py \\
        --cameras <run>/cameras.npz --image-dir <RGB 또는 TM> \\
        --pairs 12

★ 왜 재고 나서 고치는가
  이 세션에서 소스를 먼저 고쳐 두 번 실패했다. 연직 완화 상한(P8)은
  원형 구멍을 만들었고, 노출 이득 정규화(P9)는 clip 이 먼저 걸려
  아무것도 바꾸지 못했다. 둘 다 "그럴듯한 가설"이었고 측정 없이
  넣었다. 이번에는 가설을 먼저 잰다.

★ 검증하려는 가설
  태양광 패널 유리면은 거울이라 거기 보이는 무늬는 하늘·구름의 반사다.
  카메라가 움직이면 무늬도 따라 움직이므로 **시점이 다르면 대응이
  물리적으로 성립하지 않는다.** 그래서 왕복 비행의 반대 방향끼리
  (비행선 교차) 매칭이 무너진다 — 실측:

      부지·센서          내부    교차    격차
      그린환경 TM        100%    94.4%   -5.6%p
      EWP RGB            86%     59%    -27%p
      옥산 RGB           83%     30%    -53%p
      사천 RGB(줌)       51%     12%    -39%p

  패널 내부를 빼고 램버시안(시점 무관) 영역만 쓰면 이 격차가
  좁아져야 한다. 좁아지지 않으면 가설이 틀린 것이고, 패치할 이유가
  없다.

★ 판정 기준
  교차 매칭의 inlier 수가 늘고 **격차(내부−교차)가 줄면** 성공.
  특징점 총수는 줄어도 무방하다 — 문제는 수가 아니라 질이다.
  실측 반례: --k-neighbors 를 8→16 으로 늘려 쌍을 두 배로 했더니
  내부 51→34%, 교차 12→8% 로 오히려 나빠졌다.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("panel_mask_probe")

MODES = ("none", "glint", "interior", "panel")


# ── 마스크 ────────────────────────────────────────────────────────────
def panel_binary(gray: np.ndarray, min_area_frac: float = 0.002,
                 close_px: int = 9, open_px: int = 5):
    """한 장에서 패널 영역을 이진화한다 (Otsu + 형태학).

    two_layer.segment_panels 와 같은 발상이지만, 그쪽은 완성된
    모자이크의 저해상 라벨맵에서 돌고 이쪽은 **원본 한 장**에서 돈다.

    밝은 쪽을 패널로 본다. 열화상처럼 반대인 경우를 대비해 넓은 쪽이
    90% 를 넘으면 뒤집는다 (패널이 화면 전부일 리는 없다).
    """
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (gray > thr).astype(np.uint8)
    if m.mean() > 0.90:
        m = 1 - m
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         np.ones((close_px, close_px), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         np.ones((open_px, open_px), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    min_px = max(int(min_area_frac * gray.size), 64)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lab], float(thr)


def build_mask(gray: np.ndarray, mode: str, sat_thr: int = 235,
               erode_frac: float = 0.006):
    """SIFT 에 넘길 마스크 (255 = 쓴다, 0 = 뺀다).

    none      전부 사용 (기준선)
    glint     포화·준포화만 제외 — 가장 보수적
    interior  패널 **내부**만 제외. 모듈 테두리·가대는 남긴다
              (테두리는 무광 금속이라 램버시안이지만, 격자가 주기적이라
               한 줄 건너 오매칭의 원인이기도 하다 — 양날)
    panel     패널 전체 제외. 지면·구조물만 남긴다 = 가장 공격적
    """
    h, w = gray.shape[:2]
    keep = np.full((h, w), 255, np.uint8)
    info = {}

    if mode == "none":
        info["masked_frac"] = 0.0
        return keep, info

    if mode == "glint":
        bad = gray >= sat_thr
        # 포화 주변은 아직 안 날아갔어도 시점 의존적이다
        bad = cv2.dilate(bad.astype(np.uint8),
                         np.ones((9, 9), np.uint8)).astype(bool)
        keep[bad] = 0
        info["masked_frac"] = float(bad.mean())
        return keep, info

    pm, thr = panel_binary(gray)
    info["otsu"] = thr
    info["panel_frac"] = float(pm.mean())

    if mode == "panel":
        bad = pm
    else:  # interior — 테두리 폭만큼 깎아 남긴다
        k = max(int(round(erode_frac * min(h, w))) | 1, 3)
        bad = cv2.erode(pm.astype(np.uint8),
                        np.ones((k, k), np.uint8)).astype(bool)
        info["erode_px"] = k

    # 포화는 어느 모드에서나 뺀다
    bad = bad | (gray >= sat_thr)
    keep[bad] = 0
    info["masked_frac"] = float(bad.mean())
    return keep, info


# ── 특징점·매칭 ───────────────────────────────────────────────────────
def extract(path: Path, mask_mode: str, max_features: int, scale: float):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    mask, info = build_mask(img, mask_mode)
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp, desc = sift.detectAndCompute(img, mask)
    return kp, desc, info, img.shape


def match_pair(f1, f2, ratio: float = 0.75, ransac_px: float = 3.0):
    """Lowe ratio + RANSAC. (대응 수, inlier 수) 반환."""
    kp1, d1 = f1[0], f1[1]
    kp2, d2 = f2[0], f2[1]
    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
        return 0, 0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    try:
        knn = bf.knnMatch(d1, d2, k=2)
    except cv2.error:
        return 0, 0
    good = [m for m, n in (p for p in knn if len(p) == 2)
            if m.distance < ratio * n.distance]
    if len(good) < 8:
        return len(good), 0
    p1 = np.float32([kp1[m.queryIdx].pt for m in good])
    p2 = np.float32([kp2[m.trainIdx].pt for m in good])
    _, inl = cv2.findHomography(p1, p2, cv2.RANSAC, ransac_px)
    return len(good), int(inl.sum()) if inl is not None else 0


# ── 쌍 선택 ───────────────────────────────────────────────────────────
def pick_pairs(xy: np.ndarray, paths: list, n_pairs: int):
    """진행 방향으로 비행선을 나눠 '내부' 와 '교차' 쌍을 같은 수로 뽑는다.

    ★ 교차 쌍이 핵심이다. 패널 반사는 시점 의존적이므로 반대 방향끼리
      매칭될 때 무너진다. 마스킹이 듣는지는 교차 쌍에서만 드러난다.
    """
    d = np.diff(xy, axis=0)
    hd = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    hd = np.concatenate([hd, hd[-1:]])
    turn = np.abs((np.diff(hd) + 180) % 360 - 180) > 60
    line = np.concatenate([[0], np.cumsum(turn)])

    # ★ 선회 한 번에 방향이 두 번 꺾여 1~2 프레임짜리 조각이 생긴다.
    #   합성 검증에서 왕복 3선이 5개로 세어졌다. 짧은 조각은 앞선에
    #   흡수시킨다.
    min_run = 3
    lab = line.copy()
    start = 0
    for k in range(1, len(lab) + 1):
        if k == len(lab) or lab[k] != lab[start]:
            if k - start < min_run and start > 0:
                lab[start:k] = lab[start - 1]
            start = k
    _, line = np.unique(lab, return_inverse=True)

    same, cross = [], []
    tree_xy = xy
    for i in range(len(xy)):
        dist = np.hypot(tree_xy[:, 0] - xy[i, 0], tree_xy[:, 1] - xy[i, 1])
        dist[i] = np.inf
        order = np.argsort(dist)[:12]
        for j in order:
            if dist[j] > 40.0:
                break
            pair = (min(i, j), max(i, j))
            if line[i] == line[j]:
                if pair not in same:
                    same.append(pair)
            else:
                if pair not in cross:
                    cross.append(pair)
    rng = np.random.default_rng(0)
    def take(lst):
        if not lst:
            return []
        idx = rng.choice(len(lst), min(n_pairs, len(lst)), replace=False)
        return [lst[k] for k in idx]
    return take(same), take(cross), int(line.max()) + 1


def main():
    ap = argparse.ArgumentParser(
        description="패널 마스킹이 매칭 격차를 좁히는지 측정 (파이프라인 미변경)")
    ap.add_argument("--cameras", required=True,
                    help="<run>/cameras.npz — 위치와 파일 경로를 여기서 읽는다")
    ap.add_argument("--image-dir", help="경로가 옮겨졌을 때 파일명을 찾을 폴더")
    ap.add_argument("--pairs", type=int, default=12,
                    help="내부·교차 각각 몇 쌍을 잴지 (기본 12)")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--max-features", type=int, default=8000)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="처리 속도용 축소 배율 (기본 0.5). 절대값 비교가 "
                         "아니라 모드 간 상대 비교이므로 무방")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    z = np.load(a.cameras, allow_pickle=True)
    if "cams_opt" not in z.files or "paths" not in z.files:
        sys.exit("cameras.npz 에 cams_opt / paths 가 없습니다 "
                 "(P3·P4 패치를 적용한 실행의 npz 를 쓰십시오)")
    xy = np.asarray(z["cams_opt"])[:, :2]
    raw = [Path(str(p)) for p in z["paths"]]
    if a.image_dir:
        d = Path(a.image_dir)
        raw = [d / p.name if not p.exists() else p for p in raw]
    missing = [p for p in raw if not p.exists()]
    if missing:
        sys.exit("이미지를 찾을 수 없습니다 (%d장). --image-dir 을 주십시오.\n  예: %s"
                 % (len(missing), missing[0]))

    same, cross, n_lines = pick_pairs(xy, raw, a.pairs)
    print("=" * 70)
    print("프레임 %d장, 비행선 %d개" % (len(raw), n_lines))
    print("측정 쌍: 비행선 내부 %d, 비행선 교차 %d  (축소 %.2f배)"
          % (len(same), len(cross), a.scale))
    if not cross:
        print("★ 교차 쌍이 없습니다 — 이 측정의 핵심이 빠집니다. "
              "비행 패턴을 확인하십시오.")
    print("=" * 70)

    results = {}
    for mode in a.modes:
        cache = {}

        def feat(i):
            if i not in cache:
                cache[i] = extract(raw[i], mode, a.max_features, a.scale)
            return cache[i]

        row = {}
        for label, pairs in (("내부", same), ("교차", cross)):
            inl, good, ok = [], [], 0
            for i, j in pairs:
                f1, f2 = feat(i), feat(j)
                if f1 is None or f2 is None:
                    continue
                g, n = match_pair(f1, f2)
                good.append(g); inl.append(n)
                ok += 1 if n >= 20 else 0
            row[label] = dict(
                n=len(inl),
                good=float(np.median(good)) if good else 0.0,
                inl=float(np.median(inl)) if inl else 0.0,
                rate=ok / len(inl) if inl else 0.0)
        kps = [len(f[0]) for f in cache.values() if f is not None]
        masked = [f[2].get("masked_frac", 0.0)
                  for f in cache.values() if f is not None]
        row["kp"] = float(np.median(kps)) if kps else 0.0
        row["masked"] = float(np.median(masked)) if masked else 0.0
        results[mode] = row

    print()
    print("  %-9s %8s %7s | %8s %8s %6s | %8s %8s %6s | %7s"
          % ("mode", "특징점", "가린면적",
             "내부대응", "내부inl", "성공률", "교차대응", "교차inl", "성공률",
             "격차"))
    print("  " + "-" * 92)
    base_gap = None
    for mode in a.modes:
        r = results[mode]
        gap = r["내부"]["rate"] - r["교차"]["rate"]
        if mode == "none":
            base_gap = gap
        print("  %-9s %8.0f %6.0f%% | %8.0f %8.0f %5.0f%% | %8.0f %8.0f %5.0f%% | %+6.0f%%p"
              % (mode, r["kp"], 100 * r["masked"],
                 r["내부"]["good"], r["내부"]["inl"], 100 * r["내부"]["rate"],
                 r["교차"]["good"], r["교차"]["inl"], 100 * r["교차"]["rate"],
                 100 * gap))

    print()
    print("=" * 70)
    if base_gap is None or "none" not in results:
        print("기준선(none)을 함께 재야 판정할 수 있습니다.")
        return 0
    best, best_gain = None, 0.0
    for mode in a.modes:
        if mode == "none":
            continue
        r = results[mode]
        gap = r["내부"]["rate"] - r["교차"]["rate"]
        d_gap = base_gap - gap                     # 줄어든 격차
        d_inl = r["교차"]["inl"] - results["none"]["교차"]["inl"]
        if d_gap > best_gain and d_inl >= 0:
            best, best_gain = mode, d_gap
    if best:
        print("★ %s 모드가 격차를 %.0f%%p 줄이고 교차 inlier 를 유지했습니다."
              % (best, 100 * best_gain))
        print("  가설이 지지됩니다 — 파이프라인 패치를 검토할 근거가 됩니다.")
        print("  다만 여기서 잰 것은 **쌍 단위 매칭**뿐입니다. track 길이와")
        print("  BA 수렴은 실제로 돌려 봐야 알 수 있습니다.")
    else:
        print("★ 어느 모드도 격차를 줄이지 못했습니다.")
        print("  패널 반사 가설이 이 부지에서는 성립하지 않습니다.")
        print("  교차 매칭이 나쁜 원인이 다른 데 있을 수 있습니다 —")
        print("  겹침 부족, 화각(줌), 반복 패턴 오매칭 등.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

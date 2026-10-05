#!/usr/bin/env python3
"""두 비행(촬영 세션)으로 나뉜 부지를 세 조각으로 처리하는 도구.

EWP-서오창IC-2 에서 손으로 한 절차를 부지에 상관없이 돌리게 묶었다.

  배경 (EWP 실측)
    두 비행이 겹친 줄 없이 '정확히 다음 줄'에서 이어졌다. 전체를 한 번에 풀면
    경계 부근 이음매가 조각나고(번들조정 2.33 px, 노출 이득 양쪽 clip), 세션
    하나만 풀면 경계 줄이 블록 가장자리라 기운 복제가 생겼다. 세션별로
    기준면을 따로 재고, 경계 띠만 좁게 함께 풀어 줄 사이 땅에서 붙였을 때
    가운데·동쪽 이음매가 풀렸다.

  하위 명령
    split      세션을 가르고 RGB_S0 · RGB_S1 · RGB_band 폴더와 session_info.json 생성
    override   plane_sweep 출력으로 override 후보 두 개(경사 평면 · 높이만) 계산
    bandplane  두 세션의 기준면에서 경계 띠 기준면 계산
    cuts       경계 띠 모자이크에서 자를 선 두 개(줄 사이 땅) 찾기
    merge      세 조각을 비행선과 나란한 선에서 잘라 붙이기
    compare    경계를 따라 몇 자리를 잘라 나란히 비교

  세 조각은 --no-exposure-comp 로 만드는 것을 권한다. 조각마다 노출 사슬을
  따로 맞추면 자른 선 양쪽 밝기가 달라진다(EWP 서쪽 숲의 가로선).
"""
import argparse
import json
import math
import os
import re
import sys
from datetime import datetime

import numpy as np
from pathlib import Path

PAT = re.compile(r"DJI_(\d{14})_(\d{4})")


# ─────────────────────────────────────────────────────────────────────
#  공통
# ─────────────────────────────────────────────────────────────────────
def load_cams(npz):
    z = np.load(npz, allow_pickle=True)
    c = z["cams_opt"]
    names = [os.path.basename(str(p)) for p in z["paths"]]
    return c[:, 0].astype(float), c[:, 1].astype(float), names


def sessions_of(names):
    """파일명 순서로 정렬해 순번이 다시 작아지거나 2분 넘게 비면 새 세션."""
    order = sorted(range(len(names)), key=lambda i: names[i])
    t, q = {}, {}
    for i in order:
        m = PAT.search(names[i])
        if not m:
            sys.exit("파일명이 DJI_YYYYMMDDhhmmss_NNNN 형식이 아닙니다: %s" % names[i])
        t[i] = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        q[i] = int(m.group(2))
    sess = np.zeros(len(names), int)
    s, prev = 0, None
    for i in order:
        if prev is not None and (q[i] <= q[prev] or (t[i] - t[prev]).total_seconds() > 120):
            s += 1
        sess[i] = s
        prev = i
    return sess, t, order


def frame_axes(x, y, sess, order):
    """주 비행 방향(180° 접기)과 단위벡터."""
    dx, dy, ok = [], [], []
    for a, b in zip(order[:-1], order[1:]):
        if sess[a] != sess[b]:
            continue
        ddx, ddy = x[b] - x[a], y[b] - y[a]
        if math.hypot(ddx, ddy) > 0.5:
            dx.append(ddx); dy.append(ddy)
    th = np.arctan2(np.array(dy), np.array(dx)) * 2
    h = 0.5 * math.atan2(np.sin(th).mean(), np.cos(th).mean())
    return h, math.cos(h), math.sin(h)


def legs_of(x, y, sess, order, k, hdeg, v):
    """세션 k 의 연속 직선 구간(주 비행 방향 ±20°, 5장 이상)."""
    ids = [i for i in order if sess[i] == k]
    legs, cur = [], []
    for a, b in zip(ids[:-1], ids[1:]):
        dx, dy = x[b] - x[a], y[b] - y[a]
        ang = math.degrees(math.atan2(dy, dx)) % 180
        dev = abs(((ang - hdeg) + 90) % 180 - 90)
        if dev < 20 and math.hypot(dx, dy) > 0.5:
            if not cur:
                cur = [a]
            cur.append(b)
        else:
            if len(cur) >= 5:
                legs.append(cur)
            cur = []
    if len(cur) >= 5:
        legs.append(cur)
    # 같은 줄의 조각을 합친다 (수직 좌표 1.5 m 안)
    lines = []
    for L in sorted(legs, key=lambda L: np.median(v[L])):
        c = float(np.median(v[L]))
        if lines and c - lines[-1][0] < 1.5:
            lines[-1][1].extend(L)
            lines[-1][0] = float(np.median(v[lines[-1][1]]))
        else:
            lines.append([c, list(L)])
    return lines


def plane_at(g, X, Y):
    ox, oy = g.get("origin_xy", [0.0, 0.0])
    return g["c"] + g["a"] * (X - ox) + g["b"] * (Y - oy)


def to_uv(X, Y, info):
    dx, dy = X - info["x0"], Y - info["y0"]
    return dx * info["ux"] + dy * info["uy"], -dx * info["uy"] + dy * info["ux"]


def from_uv(u, v, info):
    return (info["x0"] + u * info["ux"] - v * info["uy"],
            info["y0"] + u * info["uy"] + v * info["ux"])



def transit_runs(x, y, sess, order, names, hdeg, min_run=3, dev_deg=45.0, min_cross=None):
    """주 비행 방향에서 dev_deg 넘게 벗어난 이동이 min_run 장 이상 이어진 구간 = 비행선 밖.

    ★ EWP 열화상: 세션 1 이 끝나고 남북으로 부지를 가로지르며 돌아온 11장
      (0385~0395, 11:12)이 경계 한가운데 위를 지나 모자이크에 자주 뽑혔고,
      패널이 1~2 m 밀린 조각을 만들었다. 빼자 경계 띠가 이어졌다.
      처음에는 'Y 가 상대 세션 쪽으로 반 줄 넘게 넘어간 프레임'으로 골라 4장만
      잡았다(RGB 실험에서 기각된 이유). 이동 방향으로 고르면 11장 모두 잡힌다.
      줄 끝의 선회(1~2장)는 min_run 으로 거른다.
    ★ EWP RGB 전체에서는 선회에 3~5장이 찍혀 아홉 곳이 함께 잡혔다. 선회는
      비행선 한 칸만 옮겨 가고 복귀·이동은 여러 줄을 가로지르므로, 구간 처음과
      끝의 수직 거리가 min_cross(기본 비행선 간격 × 1.5)를 넘는 것만 남긴다.
    """
    off = {}
    for a, b in zip(order[:-1], order[1:]):
        if sess[a] != sess[b]:
            continue
        dx, dy = x[b] - x[a], y[b] - y[a]
        if math.hypot(dx, dy) < 0.5:
            continue
        ang = math.degrees(math.atan2(dy, dx)) % 180
        dev = abs(((ang - hdeg) + 90) % 180 - 90)
        if dev > dev_deg:
            off[a] = True; off[b] = True
    runs, cur = [], []
    for i in order:
        if off.get(i):
            if cur and sess[cur[-1]] != sess[i]:
                runs.append(cur); cur = []
            cur.append(i)
        else:
            if cur: runs.append(cur)
            cur = []
    if cur: runs.append(cur)
    runs = [r for r in runs if len(r) >= min_run]
    if min_cross:
        h = math.radians(hdeg); vv = lambda i: -x[i] * math.sin(h) + y[i] * math.cos(h)
        runs = [r for r in runs if abs(vv(r[-1]) - vv(r[0])) > min_cross]
    return runs

def link(src, dst):
    if not os.path.lexists(dst):
        os.symlink(src, dst)


# ─────────────────────────────────────────────────────────────────────
#  split
# ─────────────────────────────────────────────────────────────────────
def cmd_split(a):
    x, y, names = load_cams(a.cameras)
    sess, t, order = sessions_of(names)
    ns = int(sess.max()) + 1
    print("세션 %d개" % ns)
    for k in range(ns):
        ids = [i for i in order if sess[i] == k]
        print("  세션 %d  %4d장  %s ~ %s" % (k, len(ids), t[ids[0]].strftime("%H:%M"),
                                          t[ids[-1]].strftime("%H:%M")))
    if ns != 2:
        sys.exit("세션이 2개일 때만 처리합니다 (지금 %d개)." % ns)

    h, ux, uy = frame_axes(x, y, sess, order)
    hdeg = math.degrees(h) % 180
    x0, y0 = float(np.mean(x)), float(np.mean(y))
    u = (x - x0) * ux + (y - y0) * uy
    v = -(x - x0) * uy + (y - y0) * ux
    L0 = legs_of(x, y, sess, order, 0, hdeg, v)
    L1 = legs_of(x, y, sess, order, 1, hdeg, v)
    L0 = [(c, np.array(L)) for c, L in L0]; L1 = [(c, np.array(L)) for c, L in L1]
    if len(L0) < 2 or len(L1) < 2 or len(L0) + len(L1) < 5:
        sys.exit("비행선을 충분히 못 잡았습니다 (세션 0: %d줄, 세션 1: %d줄)." % (len(L0), len(L1)))
    allc = np.sort([c for c, _ in L0 + L1])
    d = np.diff(allc); spacing = float(np.median(d[d > 1.5]))
    side = 1.0 if np.mean([c for c, _ in L1]) > np.mean([c for c, _ in L0]) else -1.0
    # 경계에서 먼 순서가 아니라 '경계에서 가까운 순서'로 줄 정렬
    s0 = sorted([c for c, _ in L0], key=lambda c: -side * c)     # 경계 쪽부터
    s1 = sorted([c for c, _ in L1], key=lambda c: side * c)
    # 다시 찍은 줄은 비행 방향으로도 20 m 이상 겹쳐야 한다 (옥산 오판 방지)
    rep, split_same = 0, 0
    for cb, lb in L1:
        for ca, la in L0:
            if abs(ca - cb) >= 0.35 * spacing:
                continue
            ov = min(u[la].max(), u[lb].max()) - max(u[la].min(), u[lb].min())
            if ov >= 20: rep += 1
            else: split_same += 1
            break
    v0 = np.array([c for c, _ in L0]); v1 = np.array([c for c, _ in L1])
    vov = min(v0.max(), v1.max()) - max(v0.min(), v1.min())
    if split_same > 2 or vov > 3 * spacing:
        print("★ 경계가 비행선과 나란하지 않습니다 — %d줄에 걸쳐 두 세션이 구간을 나눠 찍었습니다"
              % split_same)
        print("  (수직 좌표 겹침 %.0f m). 두 선 자르기가 맞지 않습니다." % max(vov, 0))
        print("  site_triage 의 세션 지도로 경계 모양을 먼저 보십시오.")
        if not a.force:
            sys.exit("  그래도 진행하려면 --force")
    elif split_same:
        print("경계 줄 %d개를 두 세션이 구간을 나눠 찍었습니다 — 경계 띠 안에 들어가므로 진행합니다."
              % split_same)
    if a.band_lines < 4:
        print("  ※ 나뉜 경계 줄이 있으면 --band-lines 를 4 이상으로 두십시오.")
    print("비행 방향 %.0f°,  비행선 간격 %.1f m,  세션 0 %d줄 · 세션 1 %d줄"
          % (0.0 if hdeg >= 179.5 else hdeg, spacing, len(s0), len(s1)))
    umid = float(np.median(u))
    geo = {"x0": x0, "y0": y0, "ux": ux, "uy": uy}
    pa, pb = from_uv(umid, s0[0], geo), from_uv(umid, s1[0], geo)
    print("경계 줄  세션 0  X %.0f Y %.1f   ·   세션 1  X %.0f Y %.1f   (간격 %.1f m, 다시 찍은 줄 %d개)"
          % (pa[0], pa[1], pb[0], pb[1], abs(s1[0] - s0[0]), rep))

    # ★ K_Demo: 세션 1 이 북쪽 끝 2줄뿐. 두 줄짜리 세션을 따로 풀면 번들조정이
    #   약하므로, 줄이 band_lines 이하인 세션은 통째로 띠에 넣고 그 세션 조각은
    #   쓰지 않는다 (두 조각 처리: 다른 세션 + 띠).
    small0 = len(s0) <= a.band_lines
    small1 = len(s1) <= a.band_lines
    if small0 and small1:
        sys.exit("두 세션 모두 %d줄 이하입니다 — 세션을 나누지 말고 한 번에 푸십시오." % a.band_lines)
    n0 = len(s0) if small0 else min(a.band_lines, len(s0) - 1)
    n1 = len(s1) if small1 else min(a.band_lines, len(s1) - 1)
    lo = s0[n0 - 1] - side * spacing / 2           # 세션 0 쪽 띠 끝
    hi = s1[n1 - 1] + side * spacing / 2           # 세션 1 쪽 띠 끝
    vmin, vmax = min(lo, hi), max(lo, hi)
    inband = (v >= vmin) & (v <= vmax)
    # 자를 선 목표: 각 세션 안쪽 2~3번째 줄 사이 (EWP: 경계 줄 바로 옆에서 자르면 기운 복제)
    far = 3 * spacing                              # 작은 세션 쪽 선은 그 세션 바깥으로
    tA = (s0[-1] - side * far) if small0 else (0.5 * (s0[1] + s0[2]) if len(s0) > 2 else s0[1])
    tB = (s1[-1] + side * far) if small1 else (0.5 * (s1[2] + s1[3]) if len(s1) > 3 else s1[-1])
    two = small0 or small1
    if two:
        k = 1 if small1 else 0
        print("세션 %d 이 %d줄뿐입니다 — 두 조각(세션 %d + 경계 띠)으로 처리합니다." % (k, len(s1 if small1 else s0), 1 - k))
        print("  세션 %d 조각은 만들 필요가 없습니다. merge 의 --s%d 에 띠 모자이크를 그대로 주십시오." % (k, k))

    # 비행선 밖: 상대 세션 쪽으로 반 줄 넘게 넘어간 프레임 (EWP 복귀 구간 0392~0395)
    bmid = 0.5 * (s0[0] + s1[0])
    tset = {i for i in range(len(names))
            if (sess[i] == 0 and side * (v[i] - s0[0]) > spacing / 2) or
               (sess[i] == 1 and side * (s1[0] - v[i]) > spacing / 2)}
    for r in transit_runs(x, y, sess, order, names, hdeg, min_cross=1.5 * spacing):
        tset.update(r)
    transit = [(i, int(sess[i])) for i in sorted(tset, key=lambda i: names[i])]
    if transit:
        print("비행선 밖(상대 세션 쪽으로 넘어간) 프레임 %d장:" % len(transit))
        for i, k in transit[:14]:
            print("    세션 %d  %s  %s" % (k, names[i], t[i].strftime("%H:%M:%S")))
        if len(transit) > 14:
            print("    ... 외 %d장" % (len(transit) - 14))
        print("  EWP 열화상에서 이 구간(복귀 11장)이 경계를 깨뜨렸습니다 — --drop-transit 을 권합니다.")
    drop = {i for i, _ in transit} if a.drop_transit else set()

    src = {n: os.path.join(a.image_dir, n) for n in os.listdir(a.image_dir)}
    miss = [n for n in names if n not in src]
    if miss:
        sys.exit("사진 폴더에 없는 파일 %d장 (예: %s) — --image-dir 를 확인하십시오." % (len(miss), miss[0]))
    pre = a.prefix
    out = {k: os.path.join(a.out, "%s_%s" % (pre, k)) for k in ("S0", "S1", "band")}
    for p in out.values():
        os.makedirs(p, exist_ok=True)
    cnt = {"S0": 0, "S1": 0, "band": 0}
    for i, n in enumerate(names):
        if i in drop:
            continue
        k = "S0" if sess[i] == 0 else "S1"
        link(src[n], os.path.join(out[k], n)); cnt[k] += 1
        if inband[i]:
            link(src[n], os.path.join(out["band"], n)); cnt["band"] += 1
    for k in ("S0", "S1", "band"):
        print("  %-28s %4d장" % (out[k], cnt[k]))

    ub = u[inband]
    info = dict(heading_rad=h, ux=ux, uy=uy, x0=x0, y0=y0, side=side, spacing=spacing,
                lines_s0=s0, lines_s1=s1, repeated=rep, boundary_v=bmid,
                band_v=[vmin, vmax], cut_targets=[tA, tB],
                boundary_u=[float(ub.min()), float(ub.max())],
                boundary_xy=[float(np.mean(x[inband])), float(np.mean(y[inband]))],
                prefix=pre, transit=[names[i] for i, _ in transit],
                two_piece=bool(two), small_session=(1 if small1 else 0) if two else None)
    p = os.path.join(a.out, "session_info.json")
    json.dump(info, open(p, "w"), indent=1, ensure_ascii=False)
    print("저장 %s" % p)
    print()
    print("다음: 세 폴더를 각각 파이프라인으로 돌린 뒤, 세션마다 plane_sweep → override")


# ─────────────────────────────────────────────────────────────────────
#  override
# ─────────────────────────────────────────────────────────────────────
LINE = re.compile(r"X\s+(\d+(?:\.\d+)?)\s+Y\s+(\d+(?:\.\d+)?)\s+→\s+최적\s+([+-]?\d+\.\d+)\s*m.*?"
                  r"기복비율\s+([\d.]+)(.*)$")


def cmd_override(a):
    for p, what in ((a.summary, "summary.json"), (a.sweep, "plane_sweep 출력")):
        if not os.path.exists(p):
            sys.exit("%s 이 없습니다: %s\n  그 실행(piece)과 plane_sweep(sweep)이 끝났는지 확인하십시오." % (what, p))
    g = json.load(open(a.summary))["ground_plane"]
    txt = open(a.sweep, encoding="utf-8", errors="replace").read().splitlines()
    pts, edge, weak = [], [], []
    for s in txt:
        m = LINE.search(s)
        if not m:
            continue
        X, Y, off, rel, tail = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), m.group(5)
        if "범위 끝" in tail:
            edge.append((X, Y, off))
        elif "신뢰 낮음" in tail or rel < 0.6:
            weak.append((X, Y, off))
        else:
            pts.append((X, Y, off))
    n = len(pts) + len(edge) + len(weak)
    print("측정 지점 %d곳 — 신뢰 %d · 신뢰 낮음 %d · 범위 끝 %d" % (n, len(pts), len(weak), len(edge)))
    if n == 0:
        sys.exit("plane_sweep 출력에서 지점 줄을 못 찾았습니다. tee 로 저장한 전체 출력을 주십시오.")
    if edge and 3 * len(edge) >= n:
        up = sum(1 for e in edge if e[2] > 0)
        sys.exit("범위 끝이 %d/%d곳 — %s 방향으로 --z-min/--z-max 를 넓혀 다시 재십시오."
                 % (len(edge), n, "위(+)" if up * 2 >= len(edge) else "아래(-)"))
    if len(pts) < 5:
        sys.exit("신뢰 지점이 %d곳뿐입니다 — --starts 를 늘리거나 범위를 넓히십시오." % len(pts))
    P = np.array(pts)
    tgt = plane_at(g, P[:, 0], P[:, 1]) + P[:, 2]          # 지점마다 원하는 기준면 높이
    ox, oy = float(P[:, 0].mean()), float(P[:, 1].mean())
    A = np.column_stack([P[:, 0] - ox, P[:, 1] - oy, np.ones(len(P))])
    c, *_ = np.linalg.lstsq(A, tgt, rcond=None)
    r = tgt - A @ c
    sl = math.degrees(math.atan(math.hypot(c[0], c[1])))
    med = float(np.median(P[:, 2]))
    r2 = P[:, 2] - med
    print()
    print("후보 1  경사 평면   경사 %.2f°   잔차 최대 %.2f m  RMSE %.2f m"
          % (sl, np.abs(r).max(), math.sqrt((r ** 2).mean())))
    print('  export SAYOU_PLANE_OVERRIDE="%.9f,%.9f,%.4f,%.3f,%.3f"' % (c[0], c[1], c[2], ox, oy))
    gox, goy = g.get("origin_xy", [0.0, 0.0])
    print("후보 2  높이만 %+.2f m   잔차 최대 %.2f m  RMSE %.2f m"
          % (med, np.abs(r2).max(), math.sqrt((r2 ** 2).mean())))
    print('  export SAYOU_PLANE_OVERRIDE="%.9f,%.9f,%.4f,%.3f,%.3f"' % (g["a"], g["b"], g["c"] + med, gox, goy))
    print()
    if np.abs(r).max() > 1.0:
        print("경사 평면 잔차가 1 m 를 넘습니다 — 두 후보를 모두 돌려 plane_sweep 으로")
        print("수렴을 비교하십시오 (K_Demo 는 높이만, EWP 세션 0 은 경사 평면이 이겼습니다).")
    else:
        print("경사 평면 잔차가 1 m 안쪽입니다 — 후보 1 을 쓰셔도 됩니다.")


# ─────────────────────────────────────────────────────────────────────
#  bandplane
# ─────────────────────────────────────────────────────────────────────
def cmd_bandplane(a):
    info = json.load(open(a.info))
    g0 = json.load(open(a.s0))["ground_plane"]
    g1 = json.load(open(a.s1))["ground_plane"]
    ux, uy = info["ux"], info["uy"]
    u0, u1 = info["boundary_u"]
    vb = info["boundary_v"]
    zs0, zs1 = [], []
    for uu in np.linspace(u0, u1, 5):
        X, Y = from_uv(uu, vb, info)
        zs0.append(plane_at(g0, X, Y)); zs1.append(plane_at(g1, X, Y))
    zs0, zs1 = np.array(zs0), np.array(zs1)
    print("경계를 따라 다섯 자리의 기준면 높이")
    for z0, z1 in zip(zs0, zs1):
        print("   세션 0 %.2f   세션 1 %.2f   차이 %+.2f m" % (z0, z1, z1 - z0))
    dmax = float(np.abs(zs1 - zs0).max())
    if dmax > 1.0:
        print("★ 경계에서 두 세션의 기준면이 최대 %.1f m 다릅니다. 자른 선에서 줄이 어긋날 수" % dmax)
        print("  있습니다. 각 세션의 plane_sweep 경계 쪽 지점을 다시 확인하십시오.")
    bx, by = info["boundary_xy"]
    zc = 0.5 * (plane_at(g0, bx, by) + plane_at(g1, bx, by))
    aa, bb = 0.5 * (g0["a"] + g1["a"]), 0.5 * (g0["b"] + g1["b"])
    print()
    print("경계 띠 기준면 (두 세션 평균, 경계 중심 %.1f, %.1f)" % (bx, by))
    print('  export SAYOU_PLANE_OVERRIDE="%.9f,%.9f,%.4f,%.3f,%.3f"' % (aa, bb, zc, bx, by))


# ─────────────────────────────────────────────────────────────────────
#  cuts
# ─────────────────────────────────────────────────────────────────────
def cmd_cuts(a):
    info = json.load(open(a.info))
    if a.from_y:
        # 열화상에는 초록 땅이 없어 줄 사이를 못 찾는다 — 같은 부지 RGB 에서 확정한
        # 자를 선(투영 좌표 Y, 비행이 동서일 때)을 경계 중심 X 에서 v 로 옮긴다.
        bx = info["boundary_xy"][0]
        va = to_uv(bx, a.from_y[0], info)[1]; vb = to_uv(bx, a.from_y[1], info)[1]
        if info["side"] * (vb - va) <= 0:
            va, vb = vb, va
        print("다른 센서의 자를 선 Y %.2f · %.2f → v %+.2f · %+.2f (경계 중심 X %.1f 기준)"
              % (a.from_y[0], a.from_y[1], va, vb, bx))
        if abs(info["uy"]) > 0.05:
            print("  ※ 비행이 동서에서 %.0f° 기울어 있어 Y 가 일정한 선과 v 가 일정한 선이 다릅니다 — 결과를 눈으로 확인하십시오."
                  % abs(math.degrees(math.asin(info["uy"]))))
        print()
        print("export CUT_A=%.2f CUT_B=%.2f" % (va, vb))
        return
    import rasterio
    ux, uy, side = info["ux"], info["uy"], info["side"]
    src = rasterio.open(a.band)
    step = max(a.res, abs(src.transform.a))
    sc = abs(src.transform.a) / step
    H, W = max(1, int(src.height * sc)), max(1, int(src.width * sc))
    arr = src.read(indexes=[1, 2, 3], out_shape=(3, H, W)).astype(np.int16)
    T = src.transform * src.transform.scale(src.width / W, src.height / H)
    rows, cols = np.mgrid[0:H, 0:W]
    X = T.c + (cols + 0.5) * T.a
    Y = T.f + (rows + 0.5) * T.e
    U, V = to_uv(X, Y, info)
    r, gch, b = arr
    valid = (r + gch + b) > 0
    green = (gch > r + 8) & (gch > b + 8)
    uu0, uu1 = info["boundary_u"]
    inside = valid & (U > uu0 + 10) & (U < uu1 - 10)
    res = {}
    skip = {0: "A", 1: "B"}.get(info.get("small_session")) if info.get("two_piece") else None
    for name, tg in (("A", info["cut_targets"][0]), ("B", info["cut_targets"][1])):
        if name == skip:
            res[name] = (float(tg), 1.0)          # 작은 세션 바깥 — 실제로는 자르지 않음
            continue
        half = info["spacing"] * 0.6
        bins = np.arange(tg - half, tg + half + 0.1, 0.1)
        m = inside & (V >= bins[0]) & (V < bins[-1])
        idx = np.digitize(V[m], bins) - 1
        tot = np.bincount(idx, minlength=len(bins))[:len(bins) - 1]
        gr = np.bincount(idx, weights=green[m], minlength=len(bins))[:len(bins) - 1]
        frac = np.where(tot > 50, gr / np.maximum(tot, 1), 0)
        k = np.convolve(frac, np.ones(3) / 3, mode="same")
        ctr = 0.5 * (bins[:-1] + bins[1:])
        cand = np.where(k >= np.percentile(k, 70))[0]
        i = cand[np.argmin(np.abs(ctr[cand] - tg))]
        res[name] = (float(ctr[i]), float(k[i]))
    print("자를 선 (비행선에 수직인 좌표, 비행선과 나란한 선)")
    for nm, lab in (("A", "세션 0 → 띠"), ("B", "띠 → 세션 1")):
        vv, f = res[nm]
        if nm == skip:
            print("  %s  %s   (두 조각 처리 — 작은 세션 바깥이라 자르지 않음)" % (nm, lab))
            continue
        warn = "" if f >= 0.5 else "   ★ 땅 비율이 낮습니다 — 패널 줄이 비행선과 나란하지 않거나 줄 위를 지납니다"
        px, py = from_uv(0.5 * (uu0 + uu1), vv, info)
        print("  %s  %s   v = %+.2f  (선 위의 한 점 X %.1f Y %.2f)   땅 비율 %.0f%%%s"
              % (nm, lab, vv, px, py, 100 * f, warn))
    print()
    print("export CUT_A=%.2f CUT_B=%.2f" % (res["A"][0], res["B"][0]))
    print("  v 는 부지 중심 기준, 비행선에 수직인 좌표입니다. merge 에 이 값을 그대로 넘기십시오.")


# ─────────────────────────────────────────────────────────────────────
#  merge
# ─────────────────────────────────────────────────────────────────────
def cmd_merge(a):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window
    from rasterio.transform import from_origin
    info = json.load(open(a.info))
    ux, uy, side = info["ux"], info["uy"], info["side"]
    ca, cb = a.cut_a, a.cut_b
    if side * (cb - ca) <= 0:
        sys.exit("CUT_B 가 CUT_A 보다 세션 1 쪽에 있어야 합니다 (지금 A %.2f, B %.2f)." % (ca, cb))
    srcs = [rasterio.open(p) for p in (a.s0, a.band, a.s1)]
    crs = srcs[0].crs
    g = a.gsd or abs(srcs[0].transform.a)
    L = min(s.bounds.left for s in srcs); R = max(s.bounds.right for s in srcs)
    B = min(s.bounds.bottom for s in srcs); Tp = max(s.bounds.top for s in srcs)
    L = math.floor(L / g) * g; Tp = math.ceil(Tp / g) * g
    W = int(math.ceil((R - L) / g)); H = int(math.ceil((Tp - B) / g))
    tr = from_origin(L, Tp, g, g)
    vrts = [WarpedVRT(s, crs=crs, transform=tr, width=W, height=H,
                      resampling=Resampling.nearest, nodata=0) for s in srcs]
    nb = srcs[0].count; dt = srcs[0].dtypes[0]; idx = list(range(1, nb + 1))
    if any(x.count != nb for x in srcs):
        sys.exit("세 조각의 밴드 수가 다릅니다: %s" % [x.count for x in srcs])
    prof = dict(driver="GTiff", width=W, height=H, count=nb, dtype=dt, crs=crs,
                transform=tr, tiled=True, blockxsize=512, blockysize=512,
                compress="LZW", nodata=0, BIGTIFF="IF_SAFER")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    blk = 2048
    with rasterio.open(a.out, "w", **prof) as dst:
        for r0 in range(0, H, blk):
            for c0 in range(0, W, blk):
                h, w = min(blk, H - r0), min(blk, W - c0)
                win = Window(c0, r0, w, h)
                cols = np.arange(c0, c0 + w) + 0.5
                rows = np.arange(r0, r0 + h) + 0.5
                Xg = L + cols[None, :] * g
                Yg = Tp - rows[:, None] * g
                _, V = to_uv(Xg, Yg, info)
                s = side * V
                pick = np.where(s < side * ca, 0, np.where(s < side * cb, 1, 2))
                out = np.zeros((nb, h, w), dt)
                got = np.zeros((h, w), bool)
                data = [vr.read(indexes=idx, window=win) for vr in vrts]
                for k in range(3):                        # 정해진 조각부터
                    ok = (pick == k) & (np.abs(data[k]).sum(axis=0) > 0)
                    out[:, ok] = data[k][:, ok]; got |= ok
                for k in (1, 0, 2):                       # 비어 있으면 다른 조각으로 메움
                    ok = ~got & (np.abs(data[k]).sum(axis=0) > 0)
                    out[:, ok] = data[k][:, ok]; got |= ok
                dst.write(out, window=win)
    print("저장 %s  (%d x %d, GSD %.3f)" % (a.out, W, H, g))


# ─────────────────────────────────────────────────────────────────────
#  compare
# ─────────────────────────────────────────────────────────────────────
def cmd_compare(a):
    import rasterio
    from rasterio.windows import from_bounds
    from PIL import Image
    info = json.load(open(a.info))
    ux, uy = info["ux"], info["uy"]
    u0, u1 = info["boundary_u"]
    vb = info["boundary_v"]
    rows = []
    for f in np.linspace(0.2, 0.8, a.n):
        uu = u0 + f * (u1 - u0)
        cx, cy = from_uv(uu, vb, info)
        tiles = []
        for p in a.mosaics:
            s = rasterio.open(p)
            w = from_bounds(cx - 20, cy - 30, cx + 20, cy + 30, s.transform)
            if s.count >= 3:
                arr = s.read(indexes=[1, 2, 3], window=w, boundless=True, fill_value=0)
                im = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
            else:
                b = s.read(1, window=w, boundless=True, fill_value=0).astype(float)
                v = b[b != 0]
                lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0, 1)
                g = np.clip((b - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8); g[b == 0] = 0
                im = np.repeat(g[..., None], 3, axis=2)
            tiles.append(np.array(Image.fromarray(im).resize((400, 600))))
        rows.append(np.hstack(tiles))
    Image.fromarray(np.vstack(rows)).save(a.out, quality=85)
    print("저장 %s   행 = 경계를 따라 %d 자리, 열 = %s" % (a.out, a.n, " · ".join(os.path.basename(os.path.dirname(p)) for p in a.mosaics)))


# ─────────────────────────────────────────────────────────────────────
#  empty · common  (옥산에서 손으로 돌리던 것)
# ─────────────────────────────────────────────────────────────────────
def cmd_empty(a):
    """카메라 궤적(볼록껍질) 안에서 비어 있는 픽셀 비율. 옥산: 12.14% → 0.14%."""
    import rasterio
    from scipy.spatial import Delaunay
    from rasterio.windows import from_bounds
    c = np.load(a.cameras, allow_pickle=True)["cams_opt"][:, :2]
    hull = Delaunay(c)
    # 모든 모자이크를 '궤적 전체' 격자 위에서 센다 (파일 범위 밖 궤적도 빈 자리로)
    L, B0 = c[:, 0].min(), c[:, 1].min(); R, T0 = c[:, 0].max(), c[:, 1].max()
    W = max(1, int((R - L) / a.step)); H = max(1, int((T0 - B0) / a.step))
    xs = L + (np.arange(W) + 0.5) * (R - L) / W
    ys = T0 - (np.arange(H) + 0.5) * (T0 - B0) / H
    X, Y = np.meshgrid(xs, ys)
    inside = hull.find_simplex(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape) >= 0
    for p in a.mosaics:
        s = rasterio.open(p)
        arr = s.read(window=from_bounds(L, B0, R, T0, s.transform),
                     out_shape=(s.count, H, W), boundless=True, fill_value=0)
        empty = inside & (np.abs(arr).sum(axis=0) == 0)
        print("%-40s 궤적 안 빈 픽셀 %6.2f%%" % (os.path.relpath(p, os.path.dirname(os.path.dirname(p))),
                                             100 * empty.sum() / max(inside.sum(), 1)))


def cmd_common(a):
    """기존(ref)이 채운 자리만 남긴 비교본과, 기존에 빈 자리만 새 결과로 메운 합본.

    채운 면적이 다른 결과끼리 qc_tear 를 그대로 비교하면 판정이 뒤집힌다.
    옥산: 그대로 재면 새 결과가 14.11 대 12.05% 로 유의하게 나빴지만,
    공통 면적으로 재면 11.29 대 12.05% 로 같았다(새로 채운 어려운 자리 때문).
    """
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    ref = rasterio.open(a.ref)
    new = WarpedVRT(rasterio.open(a.new), crs=ref.crs, transform=ref.transform,
                    width=ref.width, height=ref.height, resampling=Resampling.nearest, nodata=0)
    nb = ref.count; idx = list(range(1, nb + 1))
    prof = ref.profile.copy()
    prof.update(driver="GTiff", count=nb, tiled=True, blockxsize=512, blockysize=512,
                compress="LZW", nodata=0, BIGTIFF="IF_SAFER")
    outs = [(a.out_common, "common")] + ([(a.out_hybrid, "hybrid")] if a.out_hybrid else [])
    for p, _ in outs:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    dsts = {k: rasterio.open(p, "w", **prof) for p, k in outs}
    B, filled, total = 2048, 0, 0
    for r0 in range(0, ref.height, B):
        for c0 in range(0, ref.width, B):
            w = Window(c0, r0, min(B, ref.width - c0), min(B, ref.height - r0))
            x = ref.read(indexes=idx, window=w); y = new.read(indexes=idx, window=w)
            rx, ry = np.abs(x).sum(axis=0) > 0, np.abs(y).sum(axis=0) > 0
            cm = y.copy(); cm[:, ~rx] = 0
            dsts["common"].write(cm, window=w)
            if "hybrid" in dsts:
                hb = x.copy(); hb[:, ~rx & ry] = y[:, ~rx & ry]
                dsts["hybrid"].write(hb, window=w)
            filled += int((~rx & ry).sum()); total += int(rx.sum())
    for d in dsts.values():
        d.close()
    print("저장 %s%s" % (a.out_common, ("  ·  " + a.out_hybrid) if a.out_hybrid else ""))
    print("새 결과가 기존보다 더 채운 면적: 기존 채움의 %.1f%%" % (100 * filled / max(total, 1)))
    print("qc_tear 는 기존(%s)과 common 을 비교하십시오." % os.path.basename(os.path.dirname(a.ref)))


def cmd_transit(a):
    x, y, names = load_cams(a.cameras)
    sess, t, order = sessions_of(names)
    h, ux, uy = frame_axes(x, y, sess, order)
    hdeg = math.degrees(h) % 180
    x0, y0 = float(np.mean(x)), float(np.mean(y))
    v = -(x - x0) * uy + (y - y0) * ux
    cs = sorted(c for k in range(int(sess.max()) + 1) for c, _ in legs_of(x, y, sess, order, k, hdeg, v))
    dd = np.diff(cs); spacing = float(np.median(dd[dd > 1.5])) if (dd > 1.5).any() else 0.0
    min_cross = a.min_cross if a.min_cross is not None else 1.5 * spacing
    runs = transit_runs(x, y, sess, order, names, hdeg, a.min_run, a.dev, min_cross)
    drop = set(i for r in runs for i in r)
    print("주 비행 방향 %.0f°, 비행선 간격 %.1f m — 벗어난 이동이 %d장 이상 이어지고 %.1f m 넘게 가로지른 구간 %d곳, 모두 %d장"
          % (0.0 if hdeg >= 179.5 else hdeg, spacing, a.min_run, min_cross, len(runs), len(drop)))
    for r in runs:
        print("   세션 %d  %s ~ %s  (%d장, %s ~ %s)" % (sess[r[0]], names[r[0]], names[r[-1]], len(r),
              t[r[0]].strftime("%H:%M:%S"), t[r[-1]].strftime("%H:%M:%S")))
    if not a.out:
        return
    src = {n: os.path.join(a.image_dir, n) for n in os.listdir(a.image_dir)}
    os.makedirs(a.out, exist_ok=True)
    keep = 0
    for i, n in enumerate(names):
        if i in drop or n not in src:
            continue
        link(os.path.realpath(src[n]), os.path.join(a.out, n)); keep += 1
    print("남김 %d장 → %s" % (keep, a.out))


def cmd_xreg(a):
    """두 모자이크가 겹치는 자리에서 칸마다 위치 차(위상 상관)를 잰다.

    같은 부지를 다른 날 찍은 두 회차(Site-2 29696 · 29719)끼리 재면 검측점 없이
    절대 위치의 일관성을, 같은 회차의 열화상 대 RGB 를 재면 겹쳐 볼 때의 위치
    차를 가늠한다. 센서가 달라도 쓰도록 밝기 대신 기울기 크기로 비교한다.
    """
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds
    A, Bs = rasterio.open(a.a), rasterio.open(a.b)
    # ★ ODM 정사영상은 UTM, 파이프라인은 한국 좌표계 — B 의 범위를 A 좌표계로 옮겨 겹침을 잰다
    bb = Bs.bounds if Bs.crs == A.crs else transform_bounds(Bs.crs, A.crs, *Bs.bounds, densify_pts=21)
    if Bs.crs != A.crs:
        print("좌표계가 다릅니다 — A %s · B %s. B 를 A 좌표계로 옮겨 비교합니다." % (A.crs.to_string(), Bs.crs.to_string()))
    L = max(A.bounds.left, bb[0]); R = min(A.bounds.right, bb[2])
    Bt = max(A.bounds.bottom, bb[1]); T = min(A.bounds.top, bb[3])
    if R - L < a.tile or T - Bt < a.tile:
        sys.exit("두 모자이크가 겹치는 범위가 칸 크기(%.0f m)보다 작습니다." % a.tile)
    g = a.res; W = int((R - L) / g); H = int((T - Bt) / g)
    tr = from_origin(L, T, g, g)
    def grab(src):
        vr = WarpedVRT(src, crs=A.crs, transform=tr, width=W, height=H, resampling=Resampling.average, nodata=0)
        arr = vr.read().astype(np.float32)
        valid = (arr[3] > 0) if arr.shape[0] == 4 else (np.abs(arr).sum(axis=0) > 0)
        gray = arr[:3].mean(axis=0) if arr.shape[0] >= 3 else arr[0]
        gy, gx = np.gradient(gray)
        return np.hypot(gx, gy), valid
    ga, va = grab(A); gb, vb = grab(Bs)

    def pcorr(p, q):
        """위상 상관 지도(가운데가 0 이동)를 돌려준다."""
        w = np.outer(np.hanning(p.shape[0]), np.hanning(p.shape[1])).astype(np.float32)
        p = (p - p.mean()) * w; q = (q - q.mean()) * w
        c = np.fft.fft2(p) * np.conj(np.fft.fft2(q)); c /= np.abs(c) + 1e-9
        return np.fft.fftshift(np.real(np.fft.ifft2(c)))

    def peak(cc, cy, cx, rad):
        """(cy, cx) 둘레 rad 칸 안에서만 봉우리를 찾아 부화소 위치와 PSR 을 돌려준다."""
        h, w = cc.shape
        y0, y1 = max(1, int(cy - rad)), min(h - 1, int(cy + rad) + 1)
        x0, x1 = max(1, int(cx - rad)), min(w - 1, int(cx + rad) + 1)
        sub = cc[y0:y1, x0:x1]
        iy, ix = np.unravel_index(np.argmax(sub), sub.shape); iy += y0; ix += x0
        pk = cc[iy, ix]; psr = (pk - cc.mean()) / (cc.std() + 1e-9)
        def par(v0, v1, v2):
            d = v0 - 2 * v1 + v2
            return 0.0 if abs(d) < 1e-9 else 0.5 * (v0 - v2) / d
        return iy + par(cc[iy-1, ix], pk, cc[iy+1, ix]), ix + par(cc[iy, ix-1], pk, cc[iy, ix+1]), psr

    # 1단계 — 겹치는 범위 전체를 거칠게. 길·가장자리처럼 반복되지 않는 것이 결정한다.
    k = max(1, int(round(a.coarse / g)))
    def pool(m, vm):
        hh, ww = m.shape[0] // k * k, m.shape[1] // k * k
        mm = np.where(vm, m, 0)[:hh, :ww].reshape(hh // k, k, ww // k, k).mean(axis=(1, 3))
        return mm
    ca, cb = pool(ga, va), pool(gb, vb)
    ccg = pcorr(ca, cb); hh, ww = ccg.shape
    gy, gx, gpsr = peak(ccg, hh / 2, ww / 2, max(hh, ww))
    dy0, dx0 = (gy - hh // 2) * k, (gx - ww // 2) * k          # 세밀 격자 화소 단위
    print("1단계 (전체, %.2f m) — B 가 A 보다 동쪽 %+.2f m · 북쪽 %+.2f m, PSR %.1f"
          % (a.coarse, -dx0 * g, dy0 * g, gpsr))
    if gpsr < a.min_psr:
        print("  ★ 1단계 신뢰가 낮습니다 — 전체 이동을 잘못 잡았을 수 있어 2단계도 믿기 어렵습니다.")
        print("    --coarse 1.0 으로 더 거칠게 재 보십시오.")

    # 2단계 — 칸마다, 1단계 값 둘레 ±search m 안에서만
    n = int(a.tile / g); rad = a.search / g
    out = []
    for r0 in range(0, H - n + 1, n):
        for c0 in range(0, W - n + 1, n):
            if not (va[r0:r0+n, c0:c0+n].mean() > 0.9 and vb[r0:r0+n, c0:c0+n].mean() > 0.9):
                continue
            cc = pcorr(ga[r0:r0+n, c0:c0+n], gb[r0:r0+n, c0:c0+n])
            py, px, psr = peak(cc, n // 2 + dy0, n // 2 + dx0, rad)
            dy, dx = py - n // 2, px - n // 2
            # A 를 기준으로 B 가 동쪽(+x) · 북쪽(+y)으로 얼마나 밀렸나 (m)
            out.append((L + (c0 + n / 2) * g, T - (r0 + n / 2) * g, -dx * g, dy * g, psr))
    good = [o for o in out if o[4] >= a.min_psr]
    print("겹치는 범위 %.0f × %.0f m,  칸 %.0f m,  해상도 %.2f m" % (R - L, T - Bt, a.tile, g))
    print("잰 칸 %d개,  신뢰(PSR ≥ %.0f) %d개" % (len(out), a.min_psr, len(good)))
    if not good:
        print("  ★ 신뢰할 칸이 없습니다 — 칸을 키우거나(--tile) 겹치는 구역을 확인하십시오."); return None
    ex = np.array([o[2] for o in good]); ny = np.array([o[3] for o in good])
    d = np.hypot(ex, ny)
    print("B 가 A 보다  동쪽 %+.3f m · 북쪽 %+.3f m (중앙값),  거리 중앙값 %.3f m · 90%% %.3f m"
          % (np.median(ex), np.median(ny), np.median(d), np.percentile(d, 90)))
    print("칸마다 흩어짐  동서 %.3f m · 남북 %.3f m (표준편차)" % (ex.std(), ny.std()))
    if a.list:
        for o in good:
            print("   X %.0f Y %.0f   동 %+.3f  북 %+.3f m   PSR %.1f" % o)
    return dict(east=float(np.median(ex)), north=float(np.median(ny)), n=len(good),
                sd_e=float(ex.std()), sd_n=float(ny.std()), crs_a=A.crs, crs_b=Bs.crs)


def cmd_align(a):
    """기준 모자이크(같은 회차 열화상)에 맞춰 옮길 모자이크(RGB)의 좌표 원점만 옮긴다.

    ★ RGB 번들조정이 모든 카메라를 함께 1~2° 기울여 모자이크가 통째로 0.6~1.6 m
      밀렸다(Site-2 · Site-1 · 사천 · 그린환경). 열화상은 여섯 부지 모두 ODM 과
      0.2~0.45 m 로 맞았고, RGB 의 어긋남은 대부분 평행 이동이었다(남는 흩어짐
      약 0.2 m). 영상은 그대로 두고 위치 정보만 고치므로 이음매는 바뀌지 않는다.
    """
    import shutil, rasterio
    from rasterio.transform import Affine
    r = cmd_xreg(argparse.Namespace(a=a.ref, b=a.mov, res=a.res, tile=a.tile, min_psr=a.min_psr,
                                    coarse=a.coarse, search=a.search, list=False))
    if r is None:
        sys.exit("측정 실패 — 옮기지 않았습니다.")
    if r["crs_a"] != r["crs_b"]:
        sys.exit("두 모자이크의 좌표계가 다릅니다 — 같은 좌표계일 때만 옮깁니다.")
    sd = max(r["sd_e"], r["sd_n"])
    if (r["n"] < a.min_tiles or sd > a.max_sd) and not a.force:
        sys.exit("★ 측정이 약합니다 (신뢰 칸 %d개, 흩어짐 %.2f m) — 옮기지 않았습니다. 확인 후 --force."
                 % (r["n"], sd))
    print()
    mag = float(np.hypot(r["east"], r["north"]))
    if mag < a.min_shift and not a.force:
        # ★ K_Demo 0.23 m · EWP 0.10 m 를 옮기자 ODM 대비 0.20 → 0.17, 0.10 → 0.20 m — 측정 오차 수준
        print("어긋남 %.2f m 가 최소 이동량 %.2f m 보다 작습니다 — 측정 오차 수준이라 옮기지 않았습니다." % (mag, a.min_shift))
        return
    print("옮길 양: 동쪽 %+.3f m · 북쪽 %+.3f m  (측정한 어긋남의 반대)" % (-r["east"], -r["north"]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    shutil.copyfile(a.mov, a.out)
    with rasterio.open(a.out, "r+") as d:
        t = d.transform
        d.transform = Affine(t.a, t.b, t.c - r["east"], t.d, t.e, t.f - r["north"])
        # ★ 이미 정렬한 결과를 다시 정렬하면(Site-2-29719: 20 m 칸 뒤 30 m 칸) 앞의 이동을 덮어썼다.
        #   ALIGN_SHIFT_* 는 원래 결과로부터의 누적 이동, ALIGN_LAST_* 는 이번 이동.
        old = d.tags()
        pe = float(old.get("ALIGN_SHIFT_E_M", 0.0)); pn = float(old.get("ALIGN_SHIFT_N_M", 0.0))
        steps = int(old.get("ALIGN_STEPS", 1 if "ALIGN_SHIFT_E_M" in old else 0)) + 1
        d.update_tags(ALIGNED_TO=os.path.abspath(a.ref),
                      ALIGN_SHIFT_E_M="%.4f" % (pe - r["east"]), ALIGN_SHIFT_N_M="%.4f" % (pn - r["north"]),
                      ALIGN_LAST_E_M="%.4f" % -r["east"], ALIGN_LAST_N_M="%.4f" % -r["north"],
                      ALIGN_STEPS=str(steps), ALIGN_TILES=str(r["n"]))
    print("저장 %s  (영상은 그대로, 좌표 원점만 옮김)" % a.out)
    if steps > 1:
        print("  %d번째 정렬 — 원래 결과로부터 누적 이동: 동쪽 %+.3f m · 북쪽 %+.3f m" % (steps, pe - r["east"], pn - r["north"]))


def _hull(pts):
    pts = sorted(set(map(tuple, pts)))
    if len(pts) < 3:
        return pts
    def cross(o, a, b): return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def cmd_overlay(a):
    """옥상처럼 높이가 다른 층을, 그 층의 기준면으로 만든 모자이크에서 가져와 덮는다.

    ★ Site-2-29696: 경사지 위 건물 옥상의 패널이 땅 기준면보다 13~18 m 높아 배열이
      조각났다. 옥상 기준면(+17 m)으로 만들면 이어진다. 옥상 윤곽은 레이저거리계로
      기준면보다 min_dh 넘게 높게 잰 프레임 위치를 무리로 묶은 볼록 윤곽 + buffer.
    """
    import rasterio
    from rasterio.warp import transform as wtransform
    from PIL import Image, ImageDraw
    B, Tp = rasterio.open(a.base), rasterio.open(a.top)
    rows = [l.rstrip("\n").split("\t") for l in open(a.exif) if l.strip()]
    lat = np.array([float(r[0]) for r in rows]); lon = np.array([float(r[1]) for r in rows])
    alt = np.array([float(r[2]) for r in rows]); lrf = np.array([float(r[4]) for r in rows])
    xs, ys = wtransform("EPSG:4326", B.crs, lon.tolist(), lat.tolist())
    xs, ys = np.array(xs), np.array(ys)
    pa, pb, pc, pox, poy = [float(v) for v in a.plane.split(",")]
    dh = (alt - lrf) - (pc + pa * (xs - pox) + pb * (ys - poy))
    sel = np.where((lrf > 1) & (dh > a.min_dh))[0]
    if len(sel) < 3:
        sys.exit("기준면보다 %.0f m 넘게 높은 프레임이 %d장뿐 — 층 윤곽을 만들 수 없습니다." % (a.min_dh, len(sel)))
    # 가까운 점끼리 묶기 (union-find)
    par = list(range(len(sel)))
    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            if np.hypot(xs[sel[i]] - xs[sel[j]], ys[sel[i]] - ys[sel[j]]) < a.gap:
                par[find(i)] = find(j)
    groups = {}
    for i in range(len(sel)): groups.setdefault(find(i), []).append(sel[i])
    polys = []
    for g in sorted(groups.values(), key=len, reverse=True):
        if len(g) < a.min_frames:
            continue
        P = np.c_[xs[g], ys[g]]; h = np.array(_hull(P)); cen = h.mean(axis=0)
        v = h - cen; nrm = np.hypot(v[:, 0], v[:, 1])[:, None] + 1e-9
        h = h + v / nrm * a.buffer                               # 둘레로 buffer m 넓힘
        polys.append(h)
        print("층 윤곽 %d: 프레임 %d장, 높이 차 중앙 %+.1f m, 범위 X %.0f~%.0f Y %.0f~%.0f"
              % (len(polys), len(g), np.median(dh[g]), h[:, 0].min(), h[:, 0].max(), h[:, 1].min(), h[:, 1].max()))
    if not polys:
        sys.exit("프레임 %d장 이상인 무리가 없습니다." % a.min_frames)
    # 기준 모자이크 격자에서 윤곽 안을 칠함
    W, H = B.width, B.height; inv = ~B.transform
    mask = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(mask)
    for h in polys:
        dr.polygon([tuple(inv * (x, y)) for x, y in h], fill=255)
    m = np.array(mask) > 0
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    vr = WarpedVRT(Tp, crs=B.crs, transform=B.transform, width=W, height=H, resampling=Resampling.nearest, nodata=0)
    base = B.read(); top = vr.read()
    if top.shape[0] != base.shape[0]:
        sys.exit("두 모자이크의 밴드 수가 다릅니다.")
    use = m & (np.abs(top).sum(axis=0) > 0)
    out = base.copy(); out[:, use] = top[:, use]
    prof = B.profile.copy()
    prof.update(driver="GTiff", tiled=True, blockxsize=256, blockysize=256, compress="deflate")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with rasterio.open(a.out, "w", **prof) as d:
        d.write(out)
        d.update_tags(OVERLAY_TOP=os.path.abspath(a.top), OVERLAY_BASE=os.path.abspath(a.base),
                      OVERLAY_MIN_DH=str(a.min_dh), OVERLAY_POLYS=str(len(polys)))
    print("옥상 쪽에서 가져온 픽셀 %.1f%% (윤곽 안 %.1f%%)  →  저장 %s" % (100 * use.mean(), 100 * m.mean(), a.out))
    if a.mask_out:
        Image.fromarray((m * 255).astype(np.uint8)).save(a.mask_out)
        print("윤곽 그림 %s" % a.mask_out)


def _seam_stats(mosaic_path, labels_path=None, offset_m=None, max_samples=200000, seed=0):
    """라벨 지도의 경계(이음매)를 가로질러 밝기 단차를 잰다."""
    import rasterio
    from rasterio.enums import Resampling
    mp = Path(mosaic_path)
    lp = Path(labels_path) if labels_path else mp.with_name(mp.stem + "_labels.tif")
    if not lp.exists():
        return None, "라벨 지도 없음 (%s) — 이 기능 이후의 파이프라인으로 다시 만들어야 함" % lp.name
    L = rasterio.open(lp); lab = L.read(1); rg = abs(L.transform.a)
    M = rasterio.open(mp)
    # 모자이크를 라벨 해상도의 절반 격자로 읽음 (라벨 원점과 같음)
    g2 = rg / 2.0
    H2 = int(round(lab.shape[0] * 2)); W2 = int(round(lab.shape[1] * 2))
    from rasterio.vrt import WarpedVRT
    from rasterio.transform import Affine
    tr2 = Affine(g2, 0, L.transform.c, 0, -g2, L.transform.f)
    vr = WarpedVRT(M, crs=L.crs, transform=tr2, width=W2, height=H2, resampling=Resampling.average,
                   nodata=0, dtype="float32")          # 정수로 평균하면 약한 무늬에서 반올림이 단차 비를 비튼다
    arr = vr.read().astype(np.float32)
    valid = np.abs(arr).sum(axis=0) > 0
    gray = arr[:3].mean(axis=0) if arr.shape[0] >= 3 else arr[0]
    rgbish = arr.shape[0] >= 3 and float(np.std(arr[0][valid] - arr[2][valid])) > 4.0
    panel = None
    if rgbish:
        r, g, b = arr[0], arr[1], arr[2]
        panel = valid & (b > r + 12) & (b >= g) & (arr[:3].max(axis=0) < 170)
    d = max(1, int(round((offset_m or 1.5 * rg) / g2)))       # 경계에서 양쪽으로 떨어진 거리 (반격자 칸)
    rng = np.random.default_rng(seed)
    # 경계 칸: 오른쪽 · 아래 이웃과 라벨이 다름 (둘 다 유효)
    ok = lab >= 0
    hb = ok[:, :-1] & ok[:, 1:] & (lab[:, :-1] != lab[:, 1:])
    vb = ok[:-1, :] & ok[1:, :] & (lab[:-1, :] != lab[1:, :])
    def take(mask, axis):
        ys, xs = np.nonzero(mask)
        if len(ys) > max_samples:
            k = rng.choice(len(ys), max_samples, replace=False); ys, xs = ys[k], xs[k]
        # 경계 위치(반격자): 가로 경계는 두 칸 사이 x = 2x+1.5 쯤, 세로는 y
        if axis == 1:
            cy, cx = 2 * ys + 1, 2 * xs + 2
            a = (cy, cx - d); b = (cy, cx + d - 1)
        else:
            cy, cx = 2 * ys + 2, 2 * xs + 1
            a = (cy - d, cx); b = (cy + d - 1, cx)
        inb = (a[0] >= 0) & (a[1] >= 0) & (b[0] < H2) & (b[1] < W2)
        a = (a[0][inb], a[1][inb]); b = (b[0][inb], b[1][inb]); c = (cy[inb], cx[inb])
        both = valid[a] & valid[b]
        step = np.abs(gray[a] - gray[b])[both]
        onp = panel[(c[0][both], c[1][both])] if panel is not None else None
        return step, onp
    s1, p1 = take(hb, 1); s2, p2 = take(vb, 0)
    seam_step = np.r_[s1, s2]
    # 비교: 경계가 아닌 곳에서 같은 간격의 밝기 차 (가로 · 세로)
    inner = ok.copy(); inner[:, :-1] &= ~hb; inner[:, 1:] &= ~hb; inner[:-1, :] &= ~vb; inner[1:, :] &= ~vb
    ys, xs = np.nonzero(inner)
    if len(ys) > max_samples:
        k = rng.choice(len(ys), max_samples, replace=False); ys, xs = ys[k], xs[k]
    base = []
    for dy, dx in ((0, 1), (1, 0)):
        cy, cx = 2 * ys + 1, 2 * xs + 1
        a = (cy - d * dy, cx - d * dx); b = (cy + d * dy - dy, cx + d * dx - dx)
        inb = (a[0] >= 0) & (a[1] >= 0) & (b[0] < H2) & (b[1] < W2)
        a = (a[0][inb], a[1][inb]); b = (b[0][inb], b[1][inb])
        both = valid[a] & valid[b]
        base.append(np.abs(gray[a] - gray[b])[both])
    base = np.concatenate(base)
    if seam_step.size < 50 or base.size < 50:
        return None, "표본이 너무 적음"
    area_m2 = float(ok.sum()) * rg * rg
    length_m = float(hb.sum() + vb.sum()) * rg
    bm = float(np.median(base)) + 1e-6
    out = dict(seam_step_ratio=float(np.median(seam_step)) / bm,
               strong_step_frac=float(np.mean(seam_step > np.percentile(base, 90))),
               seam_density_m_per_100m2=100.0 * length_m / max(area_m2, 1e-6),
               seam_on_panel_frac=(float(np.mean(np.r_[p1, p2])) if panel is not None and (p1 is not None) else None),
               n_seam=int(seam_step.size), label_gsd_m=rg, frames=int(len(np.unique(lab[ok]))))
    return out, None


def _seam_shift(mosaic_path, labels_path=None, n_samples=3000, along_m=1.6, side_m=0.30, gap_m=0.04,
                max_shift_m=0.30, seed=0):
    """이음매 양쪽 띠의 무늬가 경계를 따라 몇 m 밀렸는지 — 기하 어긋남.

    ★ 밝기 단차 비는 갈평 세 결과(base · 2차 안내 매칭 · 호모그래피 검증)를 2.22~2.24 로 구분하지
      못했다. 같은 사진을 노출 보정 없이 쓰니 경계의 밝기 차는 같다. 눈으로 본 차이는 경계에서
      패널 줄이 끊기거나 엇나가는 **기하**였다. 경계 양쪽에 경계를 따라 가는 띠를 잡고, 두 띠의
      밝기 분포를 경계 방향으로 밀어 가며 정규화 상관이 가장 큰 이동을 찾는다.
    """
    import rasterio
    from rasterio.windows import from_bounds
    mp = Path(mosaic_path); lp = Path(labels_path) if labels_path else mp.with_name(mp.stem + "_labels.tif")
    if not lp.exists():
        return None
    L = rasterio.open(lp); lab = L.read(1); rg = abs(L.transform.a)
    M = rasterio.open(mp); g = abs(M.transform.a)
    ok = lab >= 0
    hb = ok[:, :-1] & ok[:, 1:] & (lab[:, :-1] != lab[:, 1:])          # 좌우로 라벨이 바뀜 → 경계는 세로
    vb = ok[:-1, :] & ok[1:, :] & (lab[:-1, :] != lab[1:, :])          # 위아래로 바뀜 → 경계는 가로
    rng = np.random.default_rng(seed)
    pts = [(*p, 'v') for p in zip(*np.nonzero(hb))] + [(*p, 'h') for p in zip(*np.nonzero(vb))]
    if not pts:
        return None
    pick = rng.choice(len(pts), min(n_samples, len(pts)), replace=False)
    S = int(round(max_shift_m / g)); shifts = []; n_edge = 0
    for k in pick:
        r, c, kind = pts[k]
        if kind == 'v':
            X = L.transform.c + (c + 1) * rg; Y = L.transform.f - (r + 0.5) * rg
            win = from_bounds(X - side_m - gap_m, Y - along_m / 2, X + side_m + gap_m, Y + along_m / 2, M.transform)
        else:
            X = L.transform.c + (c + 0.5) * rg; Y = L.transform.f - (r + 1) * rg
            win = from_bounds(X - along_m / 2, Y - side_m - gap_m, X + along_m / 2, Y + side_m + gap_m, M.transform)
        try:
            a = M.read(indexes=list(range(1, min(M.count, 3) + 1)), window=win, boundless=True, fill_value=0).astype(np.float32)
        except Exception:
            continue
        if a.size == 0:
            continue
        gray = a.mean(axis=0); valid = np.abs(a).sum(axis=0) > 0
        if kind == 'h':                                              # 경계 방향을 세로축으로 맞춤
            gray = gray.T; valid = valid.T
        H, W = gray.shape
        if H < 3 * S or W < 6:
            continue
        mid = W // 2; gp = max(1, int(round(gap_m / g))); sp = max(2, int(round(side_m / g)))
        A = gray[:, max(0, mid - gp - sp):mid - gp]; B = gray[:, mid + gp:mid + gp + sp]
        vA = valid[:, max(0, mid - gp - sp):mid - gp].all(axis=1); vB = valid[:, mid + gp:mid + gp + sp].all(axis=1)
        if not (vA.all() and vB.all()):
            continue
        pa = A.mean(axis=1); pb = B.mean(axis=1)
        if pa.std() < 4.0 or pb.std() < 4.0:                         # 경계를 가로지르는 무늬가 없음
            continue
        # ★ 탐색 범위를 무늬 주기의 45% 안으로 — 한 주기 옆에 상관이 걸리는 일을 구조적으로 막는다.
        #   Site-2 열화상(GSD 0.05 m)은 상한 0.30 m 가 6 화소뿐이라 표본의 52% 가 끝에 걸렸다.
        Se = S
        q = pa - pa.mean(); ac = np.correlate(q, q, mode="full")[len(q) - 1:]
        ac = ac / (ac[0] + 1e-9)
        pk = [t for t in range(2, len(ac) - 1) if ac[t] > ac[t - 1] and ac[t] >= ac[t + 1] and ac[t] > 0.3]
        if pk:
            Se = max(2, min(S, int(0.45 * pk[0])))
        core = slice(Se, H - Se); x = pa[core] - pa[core].mean()
        nccs = []
        for sft in range(-Se, Se + 1):
            y = pb[Se + sft:H - Se + sft]; y = y - y.mean()
            den = np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9
            nccs.append(float((x * y).sum() / den))
        nccs = np.array(nccs); kb = int(np.argmax(nccs)); best = float(nccs[kb]); bs = kb - Se
        if best >= 0.6:
            # ★ 탐색 끝에 붙은 표본은 측정 실패 (갈평 세 결과의 90% 값이 상한 0.30 m 근처였다)
            if abs(bs) >= Se - 0 and Se > 1 and (kb == 0 or kb == len(nccs) - 1):
                n_edge += 1
            else:
                # 화소 이하 정밀도 — 최대점 둘레 포물선 (열화상은 0.05 m 단위로만 나오던 것)
                sub = 0.0
                if 0 < kb < len(nccs) - 1:
                    c0, c1, c2 = nccs[kb - 1], nccs[kb], nccs[kb + 1]
                    den2 = (c0 - 2 * c1 + c2)
                    if abs(den2) > 1e-9:
                        sub = float(np.clip(0.5 * (c0 - c2) / den2, -0.5, 0.5))
                shifts.append(abs(bs + sub) * g)
    n_all = len(shifts) + n_edge
    if len(shifts) < 30:
        return dict(n=len(shifts), edge_frac=(n_edge / n_all if n_all else None))
    sh = np.array(shifts)
    rb = np.random.default_rng(1)
    bt = rb.choice(sh, (500, len(sh)), replace=True)
    med_ci = np.percentile(np.median(bt, axis=1), [2.5, 97.5])
    fr_ci = np.percentile(np.mean(bt > 0.05, axis=1), [2.5, 97.5])
    return dict(n=len(sh), median_m=float(np.median(sh)), p90_m=float(np.percentile(sh, 90)),
                frac_over_5cm=float(np.mean(sh > 0.05)), median_ci=tuple(map(float, med_ci)),
                frac_ci=tuple(map(float, fr_ci)), edge_frac=n_edge / n_all,
                frac_over_2gsd=float(np.mean(sh > 2 * g)), gsd=g)


def cmd_seams(a):
    """라벨 지도로 이음매 자체를 잰다 — 밝기 단차 · 뚜렷한 단차 비율 · 밀도 · 패널 위 비율.

    ★ qc_tear(줄 어긋남)는 갈평 RGB · Site-2 열화상에서 눈으로 나빠진 이음새를
      '구분되지 않음'으로 냈다. 이음매가 어디를 지나는지(라벨 경계)를 알면 그 자리의
      밝기 단차를 직접 잴 수 있다.
    """
    print("%-22s %7s %9s │ %-22s %-22s %7s %8s %6s %7s" % ("결과", "단차 비", "뚜렷한단차",
                                                      "어긋남 중앙값 (95%)", ">5cm (95%)", ">2GSD", "어긋남90", "표본", "끝 걸림"))
    first = None
    for mp in a.mosaics:
        r, why = _seam_stats(mp, offset_m=a.offset_m)
        name = Path(mp).parent.name
        if r is None:
            print("%-24s %s" % (name, why)); continue
        gsh = _seam_shift(mp, n_samples=a.samples) or {}
        if "median_m" in gsh:
            mark = ""
            if first is None:
                first = gsh
            else:
                def _ov(k):
                    return not (gsh[k][1] < first[k][0] or gsh[k][0] > first[k][1])
                # ★ 중앙값 구간 또는 비율 구간 중 하나라도 겹치지 않으면 구분된다 (갈평: gal_h 와 gal_gm6 는
                #   중앙값 구간이 0.052~0.065 대 0.033~0.045 로 갈렸는데 비율만 보아 '구분 안 됨'으로 냈다)
                if _ov("median_ci") and _ov("frac_ci"):
                    mark = "  ← 첫 결과와 구분 안 됨"
                else:
                    mark = "  ← 첫 결과보다 나쁨" if gsh["median_m"] > first["median_m"] else "  ← 첫 결과보다 좋음"
            geo = "%.3f (%.3f~%.3f)    %4.1f%% (%4.1f~%4.1f%%) %7.1f%% %8.3f %6d %6.0f%%%s" % (
                gsh["median_m"], gsh["median_ci"][0], gsh["median_ci"][1], 100 * gsh["frac_over_5cm"],
                100 * gsh["frac_ci"][0], 100 * gsh["frac_ci"][1], 100 * gsh["frac_over_2gsd"],
                gsh["p90_m"], gsh["n"], 100 * gsh["edge_frac"], mark)
        else:
            geo = "표본 부족 (%d)" % gsh.get("n", 0)
        print("%-22s %7.2f %8.1f%% │ %s" % (name, r["seam_step_ratio"], 100 * r["strong_step_frac"], geo))
    print()
    print("끝 걸림: 탐색 상한에 상관이 걸린 표본(반복 줄 한 주기 옆) — 어긋남 계산에서 뺌")
    print("어긋남: 이음매 양쪽 띠의 무늬가 경계를 따라 밀린 거리 m (중앙값 · 90%) 와 5 cm 넘는 비율 — 기하")
    print("단차 비: 이음매를 가로지른 밝기 차 ÷ 이음매 아닌 곳의 같은 간격 밝기 차 (1 이면 이음매가 안 보임)")
    print("뚜렷한 단차: 이음매 단차 중 '이음매 아닌 곳' 상위 10% 를 넘는 비율 (10% 이면 구분 안 됨)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("split"); p.set_defaults(f=cmd_split)
    p.add_argument("--image-dir", required=True, help="사진 폴더 (예: $IMAGE_DIR/RGB)")
    p.add_argument("--cameras", required=True, help="전체 실행의 cameras.npz (투영 좌표)")
    p.add_argument("--out", required=True, help="부지 폴더 (여기에 하위 폴더와 session_info.json 생성)")
    p.add_argument("--prefix", default="RGB", help="만들 폴더 이름 앞머리 (열화상이면 TM)")
    p.add_argument("--band-lines", type=int, default=5, help="경계 띠에 넣을 줄 수 (세션마다, 기본 5)")
    p.add_argument("--drop-transit", action="store_true", help="비행선 밖 프레임을 뺌")
    p.add_argument("--force", action="store_true", help="경계가 비행선과 나란하지 않아도 진행")
    p = sp.add_parser("override"); p.set_defaults(f=cmd_override)
    p.add_argument("--sweep", required=True, help="plane_sweep 전체 출력을 tee 로 저장한 파일")
    p.add_argument("--summary", required=True, help="plane_sweep 이 잰 실행의 summary.json")
    p = sp.add_parser("bandplane"); p.set_defaults(f=cmd_bandplane)
    p.add_argument("--info", required=True); p.add_argument("--s0", required=True); p.add_argument("--s1", required=True)
    p = sp.add_parser("cuts"); p.set_defaults(f=cmd_cuts)
    p.add_argument("--info", required=True); p.add_argument("--band", default=None, help="경계 띠 mosaic.tif (--from-y 면 불필요)")
    p.add_argument("--res", type=float, default=0.1, help="검색 해상도 m (기본 0.1)")
    p.add_argument("--from-y", type=float, nargs=2, default=None,
                   help="다른 센서(RGB)에서 확정한 자를 선의 Y 두 개 — 땅을 찾지 않고 옮겨 씀 (열화상용)")
    p = sp.add_parser("merge"); p.set_defaults(f=cmd_merge)
    p.add_argument("--info", required=True)
    p.add_argument("--s0", required=True); p.add_argument("--band", required=True); p.add_argument("--s1", required=True)
    p.add_argument("--cut-a", type=float, required=True); p.add_argument("--cut-b", type=float, required=True)
    p.add_argument("--gsd", type=float, default=None); p.add_argument("--out", required=True)
    p = sp.add_parser("compare"); p.set_defaults(f=cmd_compare)
    p.add_argument("--info", required=True); p.add_argument("--mosaics", nargs="+", required=True)
    p.add_argument("--n", type=int, default=3); p.add_argument("--out", required=True)
    p = sp.add_parser("transit"); p.set_defaults(f=cmd_transit)
    p.add_argument("--cameras", required=True, help="그 사진들로 돌린 실행의 cameras.npz")
    p.add_argument("--image-dir", default=None, help="사진 폴더 (--out 과 함께)")
    p.add_argument("--out", default=None, help="비행선 밖 구간을 뺀 새 폴더 (없으면 목록만)")
    p.add_argument("--min-run", type=int, default=3, help="연속 몇 장 이상을 비행선 밖으로 볼지 (기본 3, 선회는 1~2장)")
    p.add_argument("--dev", type=float, default=45.0, help="주 방향에서 벗어난 각도 기준 (기본 45°)")
    p.add_argument("--min-cross", type=float, default=None, help="구간이 가로지른 수직 거리 기준 m (기본 비행선 간격 × 1.5 — 선회 제외)")
    p = sp.add_parser("xreg"); p.set_defaults(f=cmd_xreg)
    p.add_argument("--a", required=True, help="기준 mosaic.tif"); p.add_argument("--b", required=True, help="비교 mosaic.tif")
    p.add_argument("--res", type=float, default=0.05, help="비교 해상도 m (기본 0.05)")
    p.add_argument("--tile", type=float, default=20.0, help="칸 크기 m (기본 20)")
    p.add_argument("--min-psr", type=float, default=8.0, help="신뢰 기준 (기본 8)")
    p.add_argument("--coarse", type=float, default=0.5, help="1단계 전체 비교 해상도 m (기본 0.5)")
    p.add_argument("--search", type=float, default=0.5, help="2단계 칸마다 찾는 반경 m (기본 0.5 — 모듈 반 칸보다 좁게)")
    p.add_argument("--list", action="store_true", help="칸마다 출력")
    p = sp.add_parser("align"); p.set_defaults(f=cmd_align)
    p.add_argument("--ref", required=True, help="기준 mosaic.tif (같은 회차 열화상)")
    p.add_argument("--mov", required=True, help="옮길 mosaic.tif (RGB)")
    p.add_argument("--out", required=True, help="옮긴 결과 mosaic.tif")
    # ★ 센서가 다른 영상끼리는 30 m 칸이 반복되지 않는 것(길 · 가장자리 · 설비)을 더 담는다.
    #   다섯 부지에서 1단계와 칸 중앙값이 0.2 m 안으로 모임 (20 m · PSR 8 에서는 29696 이 0.3 m 어긋남)
    p.add_argument("--res", type=float, default=0.05); p.add_argument("--tile", type=float, default=30.0)
    p.add_argument("--min-psr", type=float, default=6.0); p.add_argument("--coarse", type=float, default=0.5)
    p.add_argument("--search", type=float, default=0.5)
    p.add_argument("--min-tiles", type=int, default=5, help="신뢰 칸이 이보다 적으면 옮기지 않음 (기본 5)")
    p.add_argument("--max-sd", type=float, default=0.4, help="칸마다 흩어짐이 이보다 크면 옮기지 않음 m (기본 0.4)")
    p.add_argument("--min-shift", type=float, default=0.3, help="어긋남이 이보다 작으면 옮기지 않음 m (기본 0.3 — 측정 오차 수준)")
    p.add_argument("--force", action="store_true", help="측정이 약하거나 이동이 작아도 옮김")
    p = sp.add_parser("overlay"); p.set_defaults(f=cmd_overlay)
    p.add_argument("--base", required=True, help="땅 기준면 mosaic.tif")
    p.add_argument("--top", required=True, help="옥상 기준면 mosaic.tif")
    p.add_argument("--exif", required=True, help="site_triage 와 같은 exif tsv (위도·경도·고도·상대고도·LRF)")
    p.add_argument("--plane", required=True, help="땅 기준면 a,b,c,x0,y0 — 레이저거리계 높이 차를 이것 기준으로 잼")
    p.add_argument("--min-dh", type=float, default=10.0, help="기준면보다 이만큼 넘게 높으면 옥상 (기본 10 m)")
    p.add_argument("--gap", type=float, default=12.0, help="같은 옥상으로 묶는 프레임 간격 m (기본 12)")
    p.add_argument("--min-frames", type=int, default=5, help="옥상으로 볼 최소 프레임 수 (기본 5)")
    p.add_argument("--buffer", type=float, default=4.0, help="윤곽을 둘레로 넓히는 폭 m (기본 4)")
    p.add_argument("--out", required=True); p.add_argument("--mask-out", default=None, help="윤곽 그림 png")
    p = sp.add_parser("seams"); p.set_defaults(f=cmd_seams)
    p.add_argument("--mosaics", nargs="+", required=True, help="mosaic.tif 들 (옆에 mosaic_labels.tif 가 있어야 함)")
    p.add_argument("--offset-m", type=float, default=None, help="이음매에서 양쪽으로 떨어진 거리 m (기본 라벨 칸의 1.5배)")
    p.add_argument("--samples", type=int, default=3000, help="기하 어긋남을 잴 경계 표본 수 (기본 3000)")
    p = sp.add_parser("empty"); p.set_defaults(f=cmd_empty)
    p.add_argument("--cameras", required=True, help="궤적을 정할 cameras.npz")
    p.add_argument("--mosaics", nargs="+", required=True)
    p.add_argument("--step", type=float, default=0.45, help="검사 해상도 m (기본 0.45)")
    p = sp.add_parser("common"); p.set_defaults(f=cmd_common)
    p.add_argument("--ref", required=True, help="기존 mosaic.tif")
    p.add_argument("--new", required=True, help="새 mosaic.tif (세 조각 결과)")
    p.add_argument("--out-common", required=True, help="기존이 채운 자리만 남긴 새 결과")
    p.add_argument("--out-hybrid", default=None, help="(선택) 기존 + 빈 자리만 새 결과")
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()

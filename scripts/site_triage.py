#!/usr/bin/env python3
"""부지 사전 진단 — 파이프라인을 돌리기 **전에** 튜닝 가능한 부지인지 판정한다.

    exiftool -q -n -T -GPSLatitude -GPSLongitude -GPSAltitude \\
        -RelativeAltitude -LRFTargetDistance \\
        -FocalLengthIn35mmFormat -ImageWidth <RGB>/*.JPG > exif.tsv
    python scripts/site_triage.py exif.tsv

★ 초점거리 열(6·7번)을 반드시 넣으십시오. 없으면 광각을 가정하는데,
  줌(_Z)·열화상(_T) 데이터에서 프레임 크기가 통째로 틀립니다. 실측:
  어느 부지를 줌(f35 61mm)으로 찍었는데 광각(24mm)으로 가정해 덮힘
  반경을 50 m 로 잡았고(실제 15 m), 그 탓에 "촬영 부족"이라는 반대
  판정이 나왔다.

★ 왜 필요한가
  옥산_1호에서 30분짜리 실행을 아홉 번 돌린 뒤에야 "촬영이 부족해서
  후처리로 못 고친다"는 결론에 도달했다. 그 판단에 필요한 숫자는 전부
  EXIF 안에 있었고 30초면 나왔다.

  판정 기준은 EWP-서오창IC-2(성공)와 옥산_1호(실패)의 실측 대비다.

                        EWP TM    EWP RGB   옥산 RGB
    촬영 밀도            12.5      12.5       7.5    프레임/1000m2
    최소 k 중앙값        0.06      0.09       0.15
    카메라밴드/모자이크   ~1.0      ~1.0       0.54
    LRF 평면적합 RMSE    0.82      0.82       3.70   m
    LRF 기복             13        13         38     m
    → 결과              성공      부분성공    실패

  최소 k 가 이 중 가장 강력하다. k 는 기복변위 배율이므로
  패널 높이 1.5 m 기준 지상 오차 = 1.5 x k 다.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

# EWP-서오창IC-2 실측 기준값 (튜닝이 통했던 부지)
REF = dict(density=12.5, min_k=0.09, band_ratio=0.95, lrf_rmse=0.82)


def load(path, lat_col=0, lon_col=1):
    """exiftool TSV. 열: lat lon GPSAlt RelAlt LRF [f35 ImageWidth]"""
    d = np.genfromtxt(path, dtype=float)
    if d.ndim == 1:
        d = d[None, :]
    if d.shape[1] < 3:
        sys.exit("열이 %d개뿐입니다 — lat lon GPSAlt [RelAlt] [LRF] 순서로 "
                 "뽑으셨는지 확인하십시오" % d.shape[1])
    ncol = d.shape[1]
    lat, lon, alt = d[:, lat_col], d[:, lon_col], d[:, 2]
    rel = d[:, 3] if d.shape[1] > 3 else np.full(len(d), np.nan)
    lrf = d[:, 4] if d.shape[1] > 4 else np.full(len(d), np.nan)
    f35 = float(np.nanmedian(d[:, 5])) if d.shape[1] > 5 else float("nan")
    wpx = float(np.nanmedian(d[:, 6])) if d.shape[1] > 6 else float("nan")
    # 국소 평면 근사 — 부지 규모(<2 km)에서 오차 무시 가능
    lat0 = np.nanmedian(lat)
    x = np.radians(lon - np.nanmedian(lon)) * 6378137.0 * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * 6356752.0
    return x, y, alt, rel, lrf, f35, wpx, ncol


def load_npz(cam_path, alt_path):
    """검증용 — cameras.npz(투영좌표) + alt.tsv(RelAlt LRF GPSAlt)."""
    z = np.load(cam_path, allow_pickle=True)
    c = z["cams_opt"]
    a = np.genfromtxt(alt_path, dtype=float)
    if len(a) != len(c):
        sys.exit("cameras.npz %d장 vs alt.tsv %d행 — 개수가 다릅니다"
                 % (len(c), len(a)))
    return (c[:, 0], c[:, 1], a[:, 2], a[:, 0], a[:, 1],
            float("nan"), float("nan"), 7)


def fit_plane_trimmed(x, y, z, ok):
    """대칭 절단 최소제곱 — 법면·수목 이상치를 걷어낸다."""
    ox, oy = float(np.mean(x)), float(np.mean(y))
    X, Y = x - ox, y - oy
    m = ok.copy()
    coef = np.zeros(3)
    for _ in range(15):
        if m.sum() < 10:
            break
        A = np.column_stack([X[m], Y[m], np.ones(m.sum())])
        coef, *_ = np.linalg.lstsq(A, z[m], rcond=None)
        r = z - (coef[0] * X + coef[1] * Y + coef[2])
        s = 1.4826 * np.median(np.abs(r[m] - np.median(r[m])))
        if not np.isfinite(s) or s <= 0:
            break
        nm = ok & (np.abs(r - np.median(r[m])) < 2.5 * s)
        if nm.sum() == m.sum():
            m = nm
            break
        m = nm
    r = z - (coef[0] * X + coef[1] * Y + coef[2])
    slope = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
    rmse = float(np.sqrt((r[m] ** 2).mean())) if m.any() else float("nan")
    return coef, (ox, oy), slope, rmse, m


def min_k_map(x, y, height, half_diag, cell=4.0):
    """격자마다 '가장 가까운 카메라까지의 수평거리 / 고도' = 도달 가능한 최소 k.

    ★ **카메라 궤적 안팎을 반드시 나눠 봐야 한다.** 처음에는 모자이크
      전체의 중앙값 하나만 냈는데, 어느 부지에서 0.465 가 나와 "촬영
      부족" 으로 판정했다. 나눠 보니 궤적 안은 0.052(EWP 0.09 보다 우수),
      밖은 0.654 였고 밖이 면적의 75% 였다. 촬영이 부족한 게 아니라
      모자이크가 촬영 범위보다 넓게 잡힌 것이었다.

      판정은 **궤적 안쪽 값**으로 한다. 밖은 잘라내면 되는 문제다.
    """
    try:
        from scipy.spatial import cKDTree, ConvexHull, Delaunay
    except Exception:
        return None
    gx = np.arange(x.min() - half_diag, x.max() + half_diag + cell, cell)
    gy = np.arange(y.min() - half_diag, y.max() + half_diag + cell, cell)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    cam = np.column_stack([x, y])
    d, _ = cKDTree(cam).query(pts, k=1)
    covered = d <= half_diag              # 어떤 프레임에라도 덮이는 자리
    try:
        hull = ConvexHull(cam)
        tri = Delaunay(cam[hull.vertices])
        in_hull = tri.find_simplex(pts) >= 0
        hull_area = float(hull.volume)
    except Exception:
        in_hull = covered
        hull_area = float("nan")
    return dict(k_in=d[covered & in_hull] / height,
                k_out=d[covered & ~in_hull] / height,
                n_cov=int(covered.sum()), hull_area=hull_area)


def bar(v, ref, better_low, width=22):
    """기준 대비 막대. 기준=1.0 위치에 | 를 찍는다."""
    ratio = (ref / v) if better_low else (v / ref)
    n = int(np.clip(ratio, 0, 2) / 2 * width)
    return "#" * n + "." * (width - n) + ("  %.2fx" % ratio)


def main():
    ap = argparse.ArgumentParser(
        description="파이프라인 실행 전 부지 진단 (EXIF 만으로 30초)")
    ap.add_argument("tsv", nargs="?", help="exiftool TSV (lat lon GPSAlt RelAlt LRF)")
    ap.add_argument("--cameras", help="검증용: cameras.npz (투영좌표)")
    ap.add_argument("--alt", help="검증용: alt.tsv (RelAlt LRF GPSAlt)")
    ap.add_argument("--panel-h", type=float, default=1.5,
                    help="패널 높이 (기복변위 환산용, 기본 1.5 m)")
    ap.add_argument("--cell", type=float, default=4.0, help="최소 k 격자 (m)")
    ap.add_argument("--f35", type=float, default=None,
                    help="35mm 환산 초점거리(mm). TSV 에 열이 없을 때 한 장만 "
                         "읽어 지정: exiftool -FocalLengthIn35mmFormat <파일>")
    ap.add_argument("--f-px", type=float, default=None,
                    help="초점거리(px)를 직접 지정. 기존 실행이 있으면 "
                         "summary.json 의 focal_calibration.f_px_final")
    ap.add_argument("--image-w", type=float, default=None,
                    help="영상 가로 픽셀 (기본 4000)")
    a = ap.parse_args()

    if a.cameras and a.alt:
        x, y, alt, rel, lrf, f35, wpx, ncol = load_npz(a.cameras, a.alt)
    elif a.tsv:
        x, y, alt, rel, lrf, f35, wpx, ncol = load(a.tsv)
    else:
        ap.error("TSV 를 주거나 --cameras/--alt 를 함께 주십시오")

    n = len(x)
    ok = np.isfinite(lrf) & (lrf > 5) & (lrf < 300)
    gz = alt - lrf                       # LRF 가 맞힌 지점의 표고
    height = float(np.nanmedian(lrf[ok])) if ok.sum() > 20 else \
        float(np.nanmedian(rel))
    if not np.isfinite(height) or height <= 0:
        sys.exit("비행고도를 구할 수 없습니다 — LRF 도 RelativeAltitude 도 "
                 "읽히지 않았습니다")

    print("=" * 68)
    print("부지 사전 진단 — 프레임 %d장, 비행고도 %.1f m" % (n, height))
    print("=" * 68)

    # ── 1. 촬영 밀도 ────────────────────────────────────────────────
    W = x.max() - x.min()
    H = y.max() - y.min()
    # ── 프레임 지상 크기 — 초점거리에 따라 3배까지 달라진다 ──────────
    if a.image_w:
        wpx = a.image_w
    if not np.isfinite(wpx) or wpx <= 0:
        wpx = 4000.0
    if a.f_px:
        f_px, src = a.f_px, "지정값"
    elif a.f35:
        f_px, src = a.f35 / 36.0 * wpx, "지정 f35 %.1fmm" % a.f35
    elif np.isfinite(f35) and f35 > 0:
        f_px, src = f35 / 36.0 * wpx, "EXIF f35 %.1fmm" % f35
    else:
        # ★ 경고만 내면 계속 놓친다. 실측에서 같은 실수가 세 번 반복됐고,
        #   줌(f35 47mm)을 광각(24mm)으로 가정해 프레임 면적을 6배로
        #   잡는 바람에 3번 항목이 매번 틀린 답을 냈다. 멈춘다.
        # ★ 초점거리에 의존하지 않는 항목은 그대로 내준다. 멈추기만 하고
        #   "유효하다"고 말만 하면 쓸모가 없다. 의존하는 것은 프레임 크기
        #   (덮힘 반경) → 3번 궤적 밖 비율뿐이다.
        f_px, src, frame_w, frame_h = None, None, None, None
        half_diag = None
    if f_px:
        frame_w, frame_h = wpx * height / f_px, 0.75 * wpx * height / f_px
        half_diag = 0.5 * float(np.hypot(frame_w, frame_h))
    print()
    print("0. 카메라")
    if not f_px:
        print("   ★ 초점거리를 읽을 수 없습니다 (TSV %d열, 7열 필요)." % ncol)
        print("     아래 1·2·4·5번은 초점거리와 무관하므로 유효합니다.")
        print("     3번(궤적 밖 비율)만 생략합니다.")
        print()
        print("     전체를 보시려면 아래 중 하나:")
        print("       exiftool ... -FocalLengthIn35mmFormat -ImageWidth ... > exif.tsv")
        print("       site_triage.py exif.tsv --f35 <mm> --image-w <px>")
        print("       site_triage.py exif.tsv --f-px <summary.json 의 f_px_final>")
        # 격자 범위만 정하는 용도의 임시값 — 3번은 생략하므로 판정에 안 쓰인다
        half_diag = 60.0
    else:
        print("   f_px %.0f (%s),  영상 %.0f px" % (f_px, src, wpx))
        print("   프레임 지상 %.1f x %.1f m,  덮힘 반경 %.1f m"
              % (frame_w, frame_h, half_diag))
    if frame_w and frame_w < 50.0:
        print("   ★ 프레임이 좁습니다 (성공 사례 EWP 는 83 m).")
        print("     같은 비행의 광각(_W) 파일이 있으면 그쪽을 쓰십시오.")
        print("     화각이 좁으면 한 장에 담기는 특징이 줄어 매칭이 무너집니다.")
        print("     실측(줌 f35 47mm, 프레임 33 m): 비행선 교차 매칭 12%,")
        print("     프레임의 19%가 tie point 0개, 어긋남 0.65 m.")
        print("     ※ --k-neighbors 를 16 으로 올리는 것은 **해롭습니다**")
        print("       (실측: 내부 51->34%, 교차 12->8%, 어긋남 0.652->0.680 m).")
        print("       후보를 늘리면 겹침 적은 먼 쌍이 성공률을 희석합니다.")
        print("     → 줌이어도 지면이 평탄하면(4번 RMSE < 0.5 m) 결과는")
        print("       좋습니다. 실측: 교차 매칭 1.7% 인데 어긋남 0.047 m.")


    mk = min_k_map(x, y, height, half_diag, a.cell)
    hull_area = mk["hull_area"] if mk else float("nan")
    # ★ 밀도는 **카메라 궤적 면적** 기준. 바운딩박스로 재면 가늘고 긴
    #   궤적에서 과소평가된다.
    dens = 1000.0 * n / hull_area if np.isfinite(hull_area) and hull_area > 0 \
        else float("nan")
    print()
    print("1. 촬영 밀도 (카메라 궤적 면적 기준)")
    if f_px:
        print("   카메라 범위 %.0f x %.0f m,  궤적 면적 %.0f m2,  덮힘 반경 %.0f m"
              % (W, H, hull_area, half_diag))
    else:
        print("   카메라 범위 %.0f x %.0f m,  궤적 면적 %.0f m2"
              % (W, H, hull_area))
    print("   %.1f 프레임/1000m2   %s" % (dens, bar(dens, REF["density"], False)))

    # ── 2. 최소 k — 가장 중요 ──────────────────────────────────────
    print()
    print("2. 도달 가능한 최소 k  ★ 가장 중요")
    if mk is None:
        print("   scipy 가 없어 건너뜁니다 (pip install scipy)")
        med_k = float("nan")
        frac_out = float("nan")
    else:
        ki, ko = mk["k_in"], mk["k_out"]
        med_k = float(np.median(ki)) if len(ki) else float("nan")
        frac_out = len(ko) / max(mk["n_cov"], 1)
        print("   격자 %.1f m,  덮히는 칸 %d" % (a.cell, mk["n_cov"]))
        print("   궤적 안  %5d칸 (%2.0f%%)   중앙값 %.3f  p90 %.3f   %s"
              % (len(ki), 100 * (1 - frac_out), med_k,
                 np.percentile(ki, 90) if len(ki) else float("nan"),
                 bar(med_k, REF["min_k"], True)))
        if len(ko):
            print("   궤적 밖  %5d칸 (%2.0f%%)   중앙값 %.3f  p90 %.3f"
                  % (len(ko), 100 * frac_out,
                     np.median(ko), np.percentile(ko, 90)))
        print("   → 궤적 안 기복변위 중앙값 %.2f m (패널 높이 %.1f m 기준)"
              % (a.panel_h * med_k, a.panel_h))
        if f_px and frac_out > 0.35:
            print("   ★ 모자이크의 %.0f%% 가 카메라가 지나간 적 없는 자리입니다."
                  % (100 * frac_out))
            print("     그 자리의 원형 무늬는 잘라내는 것이 답입니다 — "
                  "설정으로 못 고칩니다.")

    # ── 3. 카메라 밴드 vs 부지 ─────────────────────────────────────
    print()
    print("3. 카메라가 부지를 덮는가")
    if not f_px:
        print("   생략 — 초점거리를 몰라 프레임 크기를 정할 수 없습니다.")
        ratio = float("nan")
        frac_out = float("nan")
    else:
        ratio = 1.0 - frac_out if np.isfinite(frac_out) else float("nan")
    if f_px:
        print("   모자이크 중 카메라 궤적 안 비율 %.2f   %s"
              % (ratio, bar(ratio, REF["band_ratio"], False)))
        print("   → 궤적 + 여유 %.0f m 로 잘라내면 나머지가 사라집니다:"
              % half_diag)
        print("      gdal_translate -projwin <xmin-%.0f> <ymax+%.0f> "
              "<xmax+%.0f> <ymin-%.0f> in.tif out.tif"
              % (half_diag * 0.25, half_diag * 0.25,
                 half_diag * 0.25, half_diag * 0.25))

    # ── 4. LRF 지형 ────────────────────────────────────────────────
    print()
    print("4. 지형 — 단일 평면으로 표현되는가")
    if ok.sum() < 20:
        print("   LRF 유효 %d장뿐 — 판정 불가" % ok.sum())
        rmse = relief = float("nan")
    else:
        _, _, slope, rmse, m = fit_plane_trimmed(x, y, gz, ok)
        relief = float(np.percentile(gz[ok], 97) - np.percentile(gz[ok], 3))
        print("   LRF 지면 표고 %.1f ~ %.1f m (p3~p97),  기복 %.1f m"
              % (np.percentile(gz[ok], 3), np.percentile(gz[ok], 97), relief))
        print("   전체 범위 %.1f ~ %.1f m" % (gz[ok].min(), gz[ok].max()))
        print("   평면적합 경사 %.3f도,  RMSE %.2f m (inlier %d/%d)   %s"
              % (slope, rmse, m.sum(), ok.sum(),
                 bar(rmse, REF["lrf_rmse"], True)))
        if rmse > 1.5:
            print("   ★ RMSE 가 1.5 m 를 넘습니다 — 단일 평면 모델이 "
                  "성립하지 않습니다. 기준면을 어떻게 잡아도 이만큼 남습니다.")
        elif rmse < 0.5:
            print("   → 평면 모델이 잘 맞습니다. 다만 **경사는 반드시 "
                  "plane_sweep 으로 확인하십시오.**")
        else:
            print("   → 평면 모델은 성립하지만 경사는 확인이 필요합니다.")
        # ★ 한때 "RMSE<1.0 이면 LRF 경사를 쓰라" 고 안내했으나 반증됐다.
        #   실측: LRF 1.288°, 파이프라인 채택 0.830°, plane_sweep 실측
        #   2.237°. 후자를 적용하니 어긋남 0.158 → 0.102 m (35% 개선).
        #   LRF 경사가 맞은 부지도 있었으나(EWP: LRF 1.561 vs 채택 1.504)
        #   일반 규칙으로 쓸 수 없다. plane_sweep 으로 재는 것이 유일하다.

    # ── 5. P7 필요 여부 ────────────────────────────────────────────
    print()
    print("5. RelativeAltitude vs LRF  (P7 패치 필요 여부)")
    if ok.sum() > 20 and np.isfinite(rel).sum() > 20:
        sp_rel = float(np.percentile(rel, 90) - np.percentile(rel, 10))
        sp_lrf = float(np.percentile(lrf[ok], 90) - np.percentile(lrf[ok], 10))
        print("   RelativeAltitude 폭 %.2f m   LRF 폭 %.2f m" % (sp_rel, sp_lrf))
        if sp_rel < 0.5 < sp_lrf:
            print("   ★ RelativeAltitude 에 지형 정보가 없습니다 — "
                  "**P7 필수**. 없으면 초점거리가 잘못된 방향으로 보정됩니다.")
        else:
            print("   두 값이 비슷하게 변합니다 — P7 영향은 작습니다.")
    else:
        print("   판정 불가")

    # ── 종합 ───────────────────────────────────────────────────────
    print()
    print("=" * 68)
    # ★ 촬영 문제와 지형 문제를 나눠야 한다. 처방이 정반대다.
    #   촬영 부족  → 후처리로 못 고침. 기본 설정이 최선 (옥산 실측).
    #   지형 비평면 → 경사 보정이 실제로 듣는다. 실측(경사 21도 산지):
    #     촬영은 좋았는데(밀도 24.0, 최소 k 0.057) 지형만 걸렸고,
    #     평면을 0도 → 13.6도 로 고쳐 어긋남 0.392 → 0.265 m (32%).
    #     그런데 "촬영 부족, 기본이 최선" 으로 판정해 정반대를 권했다.
    shoot, terrain = [], []
    if np.isfinite(med_k) and med_k > 0.12:
        shoot.append("궤적 안 최소 k %.3f (기준 0.09, 한계 0.12)" % med_k)
    if np.isfinite(dens) and dens < 9.0:
        shoot.append("촬영 밀도 %.1f (기준 12.5, 한계 9.0)" % dens)
    if np.isfinite(rmse) and rmse > 1.5:
        terrain.append("LRF 평면 RMSE %.2f m (한계 1.5)" % rmse)
    fails = shoot + terrain
    crop = bool(f_px) and np.isfinite(frac_out) and frac_out > 0.35

    if not fails:
        print("★ 판정: 촬영은 충분합니다. 기준면·초점 조정으로 개선 여지가 "
              "있습니다.")
        if crop:
            print("  단, 모자이크의 %.0f%% 가 궤적 밖입니다 — **잘라내십시오.**"
                  % (100 * frac_out))
            print("  그 부분의 원형 무늬는 설정으로 고쳐지지 않습니다.")
        print("  절차 — 기본 실행 → plane_sweep(--starts 10곳 이상) →")
        print("         신뢰지점 오프셋이 ±0.3 m 밖이면 P5 override →")
        print("         qc_tear --window 로 확정")
    elif shoot and terrain:
        print("★ 판정: **촬영도 지형도 한계 밖**. 후처리로 고칠 수 없습니다.")
        for f in fails:
            print("   - " + f)
        print()
        print("  기본 설정으로 한 번 돌리고 궤적 밖은 잘라내십시오.")
        print("  (옥산_1호 실측: 9가지 설정 중 기본을 이긴 것이 없었습니다)")
    elif shoot:
        print("★ 판정: **촬영 부족**. 후처리로 고칠 수 없습니다.")
        for f in shoot:
            print("   - " + f)
        print()
        print("  기본 설정이 최선이었습니다. 한 번 돌리고 궤적 밖은")
        print("  잘라내십시오.")
        print()
        print("  재촬영 조건: 비행선 간격을 절반으로, 부지 짧은 축 전체를")
        print("  덮도록. 그러면 최소 k 가 0.06 대로 내려갑니다.")
    else:
        print("★ 판정: **촬영은 충분, 지형이 평면이 아님.**")
        for f in terrain:
            print("   - " + f)
        print()
        print("  기본 설정을 쓰지 마십시오 — **경사 보정이 실제로 듣습니다.**")
        print("  실측(경사 21도 산지): 평면 0도 → 13.6도 로 어긋남")
        print("  0.392 → 0.265 m (32%). 다만 잔차는 남습니다.")
        print()
        print("  절차 — 기본 실행 → plane_sweep(--starts 10곳, --z 범위를")
        print("         기복만큼 넓게) → P5 override → 재측정해 수렴 확인")
        print()
        print("  그래도 남는 잔차는 평면 모델의 한계입니다. DSM 기반")
        print("  도구(WebODM 등) 비교를 검토하십시오.")
        if np.isfinite(relief) and relief > 10:
            print()
            print("  촬영 쪽 개선: 기복 %.0f m 부지에서 수평 비행을 하면"
                  % relief)
            print("  고도가 자리마다 달라져 충전율이 떨어집니다. 지형추종")
            print("  (terrain-follow) 비행을 검토하십시오.")
    print("=" * 68)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""sayou 패키지에 최소 수정 3건을 적용한다 (되돌리기 가능).

무엇을 고치는가
--------------
★ P1 / P1b 를 고를 때 — **max_shift_m 은 block_m 의 1/3 이하여야 한다.**
  ``assess_mosaic`` 은 블록 안에서 ±max_shift 만큼 밀며 상관을 잰다.
  탐색 폭 2×max_shift 가 블록 높이에 가까워지면 남는 표본이 없어
  ``len(A) < 40`` 으로 **모든 블록이 걸러지고 지표가 통째로 사라진다.**
  (실측: block 3.8 m 에 max_shift 4.0 m 를 넣었더니 summary 의 quality
   가 아예 생성되지 않았다.)

  그리고 블록을 키우면 블록 수가 제곱으로 줄어든다. 20개 미만이면
  ``assess_mosaic`` 이 None 을 돌려준다. 적용 전에 현재 블록 수를 보고
  결정하라 — 200개 이상이면 P1, 100~200이면 P1b, 100개 미만이면
  **어느 쪽도 쓰지 말고** 다른 계측기를 쓰라.

**P1. mosaic_qc 측정 범위** (`homography/mosaic_qc.py`)
    ``block_m 3.8 → 12.5``, ``max_shift_m 1.00 → 4.00``,
    ``min_panel_frac 0.30 → 0.25``.

    실측 행 피치가 **6.03 m** 인데 블록이 3.8 m 라 한 블록이 한 주기도
    담지 못했고, 탐색 상한 1.00 m 는 실제 어긋남 2~3 m 의 절반도 안 됐다.
    그래서 ``panel_misalign_median`` 이 0.031 m 로 나오는 동안 눈으로는
    수 m 가 보였다. 이 값을 안 고치면 아래 어떤 조치도 개선 여부를 잴 수
    없다.

    ※ 블록 면적이 11배가 되므로 블록 수가 1241 → 100 대로 줄어든다.
      절대값 비교보다 ``qc_row_phase.py`` 를 주지표로 쓰고, 이 값은
      보조로 보라.

**P2. 자세 스무딩의 앵커 게이트** (`homography/attitude_smoothing.py`)
    ``smooth_weak_attitudes`` 에 ``ref_rotations`` / ``max_anchor_angle_deg``
    를 추가한다. 앵커(양옆의 관측 충분한 프레임)가 **초기 자세 기준으로**
    대상 프레임과 30° 넘게 다르면 앵커로 쓰지 않는다.

    근거: 이 비행은 왕복 스캔이라 kappa 가 +90°/−94° 를 오가고 선회가
    85곳이다. 약한 프레임이 선회 근처에 있으면 **비행선을 넘어 SLERP**
    하게 되고, 그러면 스무딩이 오차를 만들어 넣는다. 실제 실행의
    ``angle_change_median 35.31°, max 127.92°`` 가 그 의심의 근거다.

    판정 기준을 BA 결과가 아니라 **초기값(RTK+짐벌)** 으로 두는 것이
    핵심이다. BA 결과는 지금 의심하고 있는 대상이므로 기준이 될 수 없다.

**P3. cameras.npz 에 관측 좌표 저장** (`homography/pipeline.py`)
    ``obs_cam`` (프레임 인덱스), ``obs_uv`` (영상좌표) 를 추가한다.
    ``frame_conditioning.py`` 가 이걸 읽어 **구경문제로 자세가 구속되지
    않는 프레임**을 찾는다. 개수로는 안 잡히는 프레임들이다.

사용법
------
```bash
python patch_sayou.py --root <설치루트>/sayou           # 확인만
python patch_sayou.py --root <설치루트>/sayou --apply   # 적용
python patch_sayou.py --root <설치루트>/sayou --revert  # 되돌리기
```

``--root`` 는 ``homography/`` 가 들어 있는 ``sayou`` 폴더다. 예를 들어
``label_studio/sayou``. 적용 시 ``*.py.sayoubak`` 백업을 남긴다.

환경변수
--------
``SAYOU_ANCHOR_ANGLE_DEG`` — P2 의 각도 게이트 (기본 30). ``999`` 를 주면
게이트가 사실상 꺼져 패치 전과 같은 동작이 된다. A-B 비교에 쓴다.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BAK = ".sayoubak"
TAG = "[sayou-patch]"

# ---------------------------------------------------------------------------
# P2 가 삽입할 코드 조각
# ---------------------------------------------------------------------------
_P2_HELPER = '''

def _quat_angle_deg(qa, qb) -> float:
    """[sayou-patch] 두 쿼터니언 사이 회전각 (deg). 부호 모호성 처리 포함."""
    d = float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb)))
    return float(np.degrees(2.0 * np.arccos(min(abs(d), 1.0))))
'''

PATCHES = [
    # -- P1b: 탐색 상한만 올린다 (블록 크기는 그대로) -------------------------
    #  ★ P1 은 블록을 12.5 m 로 키우는데, RGB 처럼 패널 마스크가 3.8% 밖에
    #    안 잡혀 블록이 50~70개인 결과에서는 20개 미만으로 떨어져 지표가
    #    통째로 생략된다. 그럴 때는 P1 대신 이것만 쓴다.
    #    실측 근거: RGB 여덟 실행 전부 misalign_p90 = 0.9900 — max_shift_m
    #    1.00 에 clip 된 값이다. 10% 넘는 블록이 1 m 이상 어긋나 있는데
    #    얼마인지 잰 적이 없다는 뜻이다.
    #  P1 과 함께 쓸 수 없다. 둘 중 하나만 고르라.
    dict(
        name="P1b mosaic_qc 중간 설정",
        path="homography/homography/mosaic_qc.py",
        old="""                  block_m: float = 3.8,
                  min_panel_frac: float = 0.30,
                  max_shift_m: float = 1.00,""",
        new="""                  block_m: float = 8.0,       # [sayou-patch] 3.8 -> 8.0
                  min_panel_frac: float = 0.20,   # [sayou-patch] 0.30 -> 0.20
                  max_shift_m: float = 2.5,   # [sayou-patch] 1.00 -> 2.5""",
    ),
    # -- P1 ----------------------------------------------------------------
    dict(
        name="P1 mosaic_qc 측정 범위",
        path="homography/homography/mosaic_qc.py",
        old="""def assess_mosaic(path, gsd_m: float, *,
                  block_m: float = 3.8,
                  min_panel_frac: float = 0.30,
                  max_shift_m: float = 1.00,""",
        new="""def assess_mosaic(path, gsd_m: float, *,
                  block_m: float = 12.5,      # [sayou-patch] 3.8 -> 12.5
                  min_panel_frac: float = 0.25,   # [sayou-patch] 0.30 -> 0.25
                  max_shift_m: float = 4.00,  # [sayou-patch] 1.00 -> 4.00""",
    ),
    # -- P2a: 헬퍼 함수 ------------------------------------------------------
    dict(
        name="P2a 각도 헬퍼 추가",
        path="homography/homography/attitude_smoothing.py",
        old="""def smooth_weak_attitudes(rotations: list,""",
        new=_P2_HELPER.strip("\n") + """


def smooth_weak_attitudes(rotations: list,""",
    ),
    # -- P2b: 시그니처 -------------------------------------------------------
    dict(
        name="P2b 스무딩 시그니처",
        path="homography/homography/attitude_smoothing.py",
        old="""                          min_obs: int = 60,
                          max_gap: int = 8):""",
        new="""                          min_obs: int = 60,
                          max_gap: int = 8,
                          ref_rotations=None,          # [sayou-patch]
                          max_anchor_angle_deg: float = 30.0):  # [sayou-patch]""",
    ),
    # -- P2c: 기준 쿼터니언 --------------------------------------------------
    dict(
        name="P2c 기준 쿼터니언 준비",
        path="homography/homography/attitude_smoothing.py",
        old="""    quats = [rotmat_to_quat(R) for R in rotations]
    new_rot = list(rotations)""",
        new="""    quats = [rotmat_to_quat(R) for R in rotations]
    # [sayou-patch] 앵커 적격성은 **초기 자세(RTK+짐벌)** 로 판단한다.
    #   BA 결과로 판단하면, 지금 의심하고 있는 값으로 의심 대상을 거르는
    #   순환이 된다. ref 가 없으면 게이트를 적용하지 않는다(기존 동작).
    ref_q = None
    if ref_rotations is not None and len(ref_rotations) == n:
        ref_q = [rotmat_to_quat(R) for R in ref_rotations]
    n_anchor_reject = 0
    new_rot = list(rotations)""",
    ),
    # -- P2d: 앵커 선택 게이트 -----------------------------------------------
    dict(
        name="P2d 앵커 각도 게이트",
        path="homography/homography/attitude_smoothing.py",
        old="""        if li is not None and (i - li) > max_gap:
            li = None
        if ri is not None and (ri - i) > max_gap:
            ri = None""",
        new="""        if li is not None and (i - li) > max_gap:
            li = None
        if ri is not None and (ri - i) > max_gap:
            ri = None

        # [sayou-patch] 비행선을 넘는 앵커를 배제한다. 왕복 스캔에서
        #   선회를 사이에 둔 앵커로 SLERP 하면 자세가 통째로 틀어진다.
        if ref_q is not None and max_anchor_angle_deg < 180.0:
            if li is not None and _quat_angle_deg(
                    ref_q[li], ref_q[i]) > max_anchor_angle_deg:
                li = None
                n_anchor_reject += 1
            if ri is not None and _quat_angle_deg(
                    ref_q[ri], ref_q[i]) > max_anchor_angle_deg:
                ri = None
                n_anchor_reject += 1
            # 양쪽이 남았는데 서로 다른 비행선이면 보간 자체가 무의미하다.
            if (li is not None and ri is not None
                    and _quat_angle_deg(ref_q[li],
                                        ref_q[ri]) > max_anchor_angle_deg):
                ri = None
                n_anchor_reject += 1""",
    ),
    # -- P2e: info 보고 ------------------------------------------------------
    dict(
        name="P2e 게이트 통계 보고",
        path="homography/homography/attitude_smoothing.py",
        old="""        "skipped_no_anchor": n_skipped,
        "min_obs": min_obs,
    }""",
        new="""        "skipped_no_anchor": n_skipped,
        "min_obs": min_obs,
        # [sayou-patch]
        "anchor_rejected": int(n_anchor_reject),
        "max_anchor_angle_deg": float(max_anchor_angle_deg),
    }""",
    ),
    # -- P3a: 변수 초기화 ----------------------------------------------------
    dict(
        name="P3a obs_export 초기화",
        path="homography/pipeline.py",
        old="""    obs_cnt_final = None
    focal_info = None""",
        new="""    obs_cnt_final = None
    obs_export = None            # [sayou-patch] 프레임 조건수 진단용
    focal_info = None""",
    ),
    # -- P3b: 관측 좌표 수집 -------------------------------------------------
    dict(
        name="P3b 관측 좌표 수집",
        path="homography/pipeline.py",
        old="""                obs_cnt_final = obs_cnt.copy()
                obs_stats = {""",
        new="""                obs_cnt_final = obs_cnt.copy()
                # [sayou-patch] 관측 영상좌표를 남긴다. 개수만으로는 구경문제
                #   (도로 위 프레임의 대응이 한 직선에만 몰리는 것) 를 볼 수
                #   없다. frame_conditioning.py 가 이걸 읽는다.
                try:
                    obs_export = (
                        np.fromiter((o[0] for o in observations),
                                    dtype=np.int64, count=len(observations)),
                        np.asarray([o[2] for o in observations],
                                   dtype=np.float32))
                except Exception:
                    obs_export = None
                obs_stats = {""",
    ),
    # -- P3c: 스무딩 호출에 기준 자세 전달 -----------------------------------
    dict(
        name="P3c 스무딩 호출에 초기 자세 전달",
        path="homography/pipeline.py",
        old="""            _R_all, _smooth_info = smooth_weak_attitudes(
                _R_all, obs_cnt_final,
                min_obs=attitude_smooth_min_obs,
                max_gap=attitude_smooth_max_gap)""",
        new="""            # [sayou-patch] 앵커 적격성 판단용 초기 자세를 함께 넘긴다.
            import os as _os
            _anchor_deg = float(_os.environ.get("SAYOU_ANCHOR_ANGLE_DEG", 30.0))
            _R_ref = [_rotation_from_opk(*initial_cameras[_i, 3:6])
                      for _i in range(len(metas))]
            _R_all, _smooth_info = smooth_weak_attitudes(
                _R_all, obs_cnt_final,
                min_obs=attitude_smooth_min_obs,
                max_gap=attitude_smooth_max_gap,
                ref_rotations=_R_ref,
                max_anchor_angle_deg=_anchor_deg)""",
    ),
    # -- P5: 기준면 직접 지정 --------------------------------------------------
    #  ★ 실측(EWP RGB, plane_sweep 4지점)에서 현재 평면이 실제 정합면보다
    #    낮고 경사도 절반이었다.
    #        현재  a -0.009003  b +0.024666  c 168.927   경사 1.504°
    #        보정  a -0.023678  b +0.049080  c 171.713   경사 3.119°
    #    네 지점 적합 잔차가 0.26 m 로 작아 **기울어진 평면 하나로 부지
    #    전체가 맞는다.** 그런데 계수를 직접 지정할 CLI 가 없다.
    #    --flat-plane 은 경사를 0 으로, --panel-top 은 c 만 건드린다.
    #
    #    환경변수로 넣는다:
    #      export SAYOU_PLANE_OVERRIDE="a,b,c,origin_x,origin_y"
    dict(
        name="P5 기준면 직접 지정",
        path="homography/pipeline.py",
        old="""    # LRF/메타데이터 기준 고도 — BA 점군 평면의 교차검증에 쓴다.
    ref_zs = [z for z in (estimate_ground_z(m) for m in metas) if z is not None]""",
        new="""    # [sayou-patch] 계수를 직접 지정하면 추정을 건너뛴다.
    import os as _os
    _ov = _os.environ.get("SAYOU_PLANE_OVERRIDE", "").strip()
    if _ov:
        try:
            _v = [float(t) for t in _ov.replace(" ", "").split(",")]
            if len(_v) != 5:
                raise ValueError("a,b,c,origin_x,origin_y 다섯 개가 필요합니다")
            _pl = GroundPlane(a=_v[0], b=_v[1], c=_v[2],
                              origin_xy=(_v[3], _v[4]))
            logger.warning("기준면을 환경변수로 지정했습니다 — 추정을 "
                           "건너뜁니다: a=%+.6f b=%+.6f c=%.3f 경사 %.3f°",
                           _v[0], _v[1], _v[2],
                           np.degrees(np.arctan(np.hypot(_v[0], _v[1]))))
            # [sayou-patch] ★ 원점이 이 부지 카메라에서 멀면 **다른 부지의
            #   값이 셸에 남아 새어 들어온 것**이다. 실측: 직전 부지 override
            #   (원점 188450,279043)가 130 m 떨어진 다음 부지에 그대로
            #   적용돼 기준면이 8.09 m 높아졌고, plane_sweep 이 -8.83 m 를
            #   요구했다. 그 값을 진짜 기준면 오차로 착각해 세 번을 헛돌았다.
            try:
                _cams = np.asarray(cams_opt, dtype=float)
                _cx = float(np.median(_cams[:, 0]))
                _cy = float(np.median(_cams[:, 1]))
                _d = float(np.hypot(_cx - _v[3], _cy - _v[4]))
                _dz = _v[0] * (_cx - _v[3]) + _v[1] * (_cy - _v[4])
                logger.warning("  원점에서 카메라 중심까지 %.0f m "
                               "(경사로 인한 표고차 %+.2f m)", _d, _dz)
                # 거리가 아니라 **표고차**로 판정한다. 실측 누출 사례는
                # 175 m 로 거리 문턱(200 m)을 못 넘었지만 표고차가 8.09 m
                # 였다. 피해는 표고차로 발생한다.
                if abs(_dz) > 2.0:
                    logger.error("  ★★ 지정한 원점이 이 부지 카메라 중심에서 "
                                 "%.0f m 떨어져 있고, 경사 때문에 기준면이 "
                                 "%+.2f m 어긋납니다.", _d, _dz)
                    logger.error("     **다른 부지의 값이 셸에 남아 있지 "
                                 "않은지 확인하십시오.**")
                    logger.error("     이 부지 값이 아니라면: "
                                 "unset SAYOU_PLANE_OVERRIDE")
            except Exception:
                pass
            return _pl, "override", None
        except Exception as _exc:
            logger.error("SAYOU_PLANE_OVERRIDE 를 읽지 못했습니다 (%s): %s",
                         _ov, _exc)
            raise

    # LRF/메타데이터 기준 고도 — BA 점군 평면의 교차검증에 쓴다.
    ref_zs = [z for z in (estimate_ground_z(m) for m in metas) if z is not None]""",
    ),
    # -- P5b: override 일 때 자동보정 차단 ------------------------------------
    #  ★ P5 만으로는 부족했다. 계수를 지정해도 뒤이어 "기준면 자동보정" 이
    #    돌아 c 를 다시 움직인다. 실측(12_mid): 170.750 을 지정했는데
    #    Δz=-2.996 이 ±3.0 한계 안이라 적용돼 **167.754** 로 돌았고,
    #    plane_source 가 override → overlap_calibrated 로 바뀌었다.
    #    지정했으면 지정한 값으로 돌아야 한다.
    dict(
        name="P5b override 시 자동보정 차단",
        path="homography/pipeline.py",
        old="""    if auto_plane and plane_source == "ba_points":
        logger.info("기준면 자동보정 생략 — BA 점군 평면을 신뢰합니다.")
        auto_plane = False""",
        new="""    if auto_plane and plane_source == "override":
        # [sayou-patch] 계수를 직접 지정했으면 그대로 쓴다.
        logger.info("기준면 자동보정 생략 — 계수를 직접 지정했습니다.")
        auto_plane = False

    if auto_plane and plane_source == "ba_points":
        logger.info("기준면 자동보정 생략 — BA 점군 평면을 신뢰합니다.")
        auto_plane = False""",
    ),
    # -- P6: 2층 패널 마스크 보정 제한 ------------------------------------------
    #  ★ two_layer 는 점군의 "상층 비율" 에 맞춰 영상 임계를 강제로 조정한다.
    #    개발 당시엔 점군 50.6% vs Otsu 47.4% 로 잘 맞아 안전했지만,
    #    이 RGB 에서는 점군 11.8% vs Otsu 47.1% 로 35%p 나 벌어졌다.
    #    그대로 따르면 패널 면적의 3/4 이 지면 높이로 워프돼 2층의 이득이
    #    통째로 뒤집힌다 (실측 11_layer: 어긋남 0.119 → 0.170 m).
    #
    #    점군이 항상 옳지는 않다. RGB 는 SIFT 특징이 풀·자갈에서 나오고
    #    패널면은 매끈해 점이 적게 잡히므로 상층 비율이 과소평가된다.
    #    Otsu 와 크게 어긋나면 **점군을 버리고 Otsu 를 쓴다.**
    #
    #    SAYOU_PANEL_FRACTION 으로 직접 지정할 수도 있다.
    dict(
        name="P6 2층 패널 마스크 보정 제한",
        path="homography/homography/two_layer.py",
        old="""    if target_fraction is not None and 0.05 < target_fraction < 0.95:""",
        new="""    # [sayou-patch] 사람이 지정하면 그것을 쓴다.
    import os as _os
    _pf = _os.environ.get("SAYOU_PANEL_FRACTION", "").strip()
    if _pf:
        try:
            target_fraction = float(_pf)
            logger.info("패널 면적비를 지정받았습니다: %.1f%%",
                        target_fraction * 100)
        except ValueError:
            logger.error("SAYOU_PANEL_FRACTION 을 읽지 못했습니다: %s", _pf)

    # [sayou-patch] 점군 상층 비율이 Otsu 와 크게 어긋나면 점군을 버린다.
    if (not _pf and target_fraction is not None
            and abs(target_fraction - frac) > 0.25):
        logger.warning(
            "점군 상층 비율(%.1f%%)이 영상 Otsu(%.1f%%)와 %.0f%%p 어긋납니다 "
            "— **점군을 신뢰하지 않고 Otsu 를 씁니다.** RGB 는 패널면이 "
            "매끈해 특징점이 적어 상층 비율이 과소평가됩니다. 다르게 하려면 "
            "SAYOU_PANEL_FRACTION 으로 지정하십시오.",
            target_fraction * 100, frac * 100,
            abs(target_fraction - frac) * 100)
        target_fraction = None

    if target_fraction is not None and 0.05 < target_fraction < 0.95:""",
    ),
    # -- P7: 초점 보정 기준을 LRF 우선으로 -------------------------------------
    #  ★ _metadata_distance 가 RelativeAltitude 와 LRF 를 **평균**낸다.
    #    실측(EWP-서오창IC-2 RGB, 909장):
    #        RelativeAltitude  중앙값 50.634  p10 50.603  p90 50.674  (폭 7 cm)
    #        LRF               중앙값 46.018  p10 43.494  p90 56.866  (폭 13.4 m)
    #    정속·정고도 비행이라 RelativeAltitude 는 이륙점 기준 기압고도이고
    #    **지형 정보가 없다.** 표고차 13 m 인 부지에서 이것을 "카메라–지면
    #    거리" 로 쓸 수 없다. 그런데 둘을 평균내어 48.278 이 되고, 초점
    #    자기보정이 그 값을 쫓아가 f 를 2704 → 3004 (+11%) 로 밀어 올렸다.
    #
    #    불일치 가드는 25% 인데 실제 차이는 9% 라 걸리지 않는다.
    #    같은 패키지의 rtk.estimate_ground_z 는 이미 LRF 를 1순위로 쓰며
    #    "RelativeAltitude 는 이륙지점 기준이라 오차가 크다" 고 적어 두었다.
    #    같은 규칙을 여기에도 적용한다.
    dict(
        name="P7 초점 보정 기준 LRF 우선",
        path="homography/homography/focal.py",
        old="""    vals = [v for v in (rel, lrf) if v is not None and 1.0 < v < 1000.0]
    if not vals:
        return None
    if len(vals) == 2 and abs(vals[0] - vals[1]) > 0.25 * max(vals):
        return None                       # 두 센서 불일치 — 신뢰 불가
    return float(np.median(vals))""",
        new="""    # [sayou-patch] LRF 가 있으면 그것만 쓴다. RelativeAltitude 와 평균내면
    #   지형 정보가 없는 값이 절반 섞여 기준이 통째로 편향된다.
    if lrf is not None and 1.0 < lrf < 1000.0:
        return float(lrf)

    vals = [v for v in (rel, lrf) if v is not None and 1.0 < v < 1000.0]
    if not vals:
        return None
    if len(vals) == 2 and abs(vals[0] - vals[1]) > 0.25 * max(vals):
        return None                       # 두 센서 불일치 — 신뢰 불가
    return float(np.median(vals))""",
    ),
    # -- P8: 연직 완화에 상한 -------------------------------------------------
    #  ★ 픽셀별 연직 완화에 상한이 없어 실측(옥산_1호 RGB)에서 k 가 5.20
    #    까지 열렸다. 시선각 79°, 패널 높이 1.5 m 기준 기복변위 7.8 m 다.
    #
    #    상한은 **inf 복원 뒤에, 유한한 mk 에만** 걸어야 한다. 앞에 걸면
    #    아무 프레임도 덮지 않는 자리(mk=inf)까지 함께 잡았다가 다음 줄이
    #    inf 로 되돌려 놓는다 — 첫 시도에서 "34.8% 를 비웁니다" 라고
    #    보고했지만 그중 23.0%p 가 그런 자리였고 충전율은 1.19%p 만
    #    움직였다. qc_tear 두 구역 모두 구간이 겹쳐 효과가 없었다.
    #
    #      export SAYOU_OFFNADIR_CEILING=0.6     # 시선각 31°
    dict(
        name="P8 연직 완화 상한",
        path="homography/homography/ortho.py",
        old="""        k_allow[~np.isfinite(mk)] = np.inf""",
        new="""        k_allow[~np.isfinite(mk)] = np.inf
        # [sayou-patch] 완화 상한. 유한한 mk 에만 걸어 '덮는 프레임은 있으나
        #   너무 비스듬한' 자리만 비운다. mk=inf 는 어차피 안 채워진다.
        import os as _os
        _ceil = float(_os.environ.get("SAYOU_OFFNADIR_CEILING", "0") or 0)
        if _ceil > 0:
            _over = np.isfinite(mk) & (k_allow > _ceil)
            k_allow[_over] = -1.0      # 어떤 프레임도 통과 못 함 = 비움
            logger.info("연직 완화 상한 k<=%.2f — 덮이지만 너무 비스듬한 "
                        "%.1f%% 픽셀을 비웁니다 (시선각 %.0f도 초과)",
                        _ceil, 100.0 * float(_over.mean()),
                        float(np.degrees(np.arctan(_ceil))))""",
    ),
    # -- P4: 스무딩된 자세도 저장 ---------------------------------------------
    #  ★ cams_opt 은 BA 직후 값이고, 자세 스무딩은 그 뒤 호모그래피 단계에서만
    #    _R_all 에 적용된다. 그래서 cameras.npz 만 보면 스무딩 효과가 전혀
    #    안 보인다. 실측 확인: 스무딩 대상인 약한 프레임 111장의
    #    cams_opt−initial 회전각이 전부 정확히 0.000° 였다.
    #    이 상태에서는 debug_single_frame --use-ba 로 P2 의 효과를 잴 수 없다.
    dict(
        name="P4 스무딩된 자세 저장",
        path="homography/pipeline.py",
        old="""    frames, frame_metas, frame_src_idx = [], [], []
    for i, (m, k) in enumerate(zip(metas, intrinsics_obj)):""",
        new="""    # [sayou-patch] 스무딩까지 반영된 자세를 따로 남긴다.
    try:
        _sm = np.asarray(cams_opt, dtype=np.float64).copy()
        for _i, _R in enumerate(_R_all):
            if _i < len(_sm):
                _sm[_i, 3:6] = _opk_from_rotation(_R)
        cams_smoothed = _sm
    except Exception:
        cams_smoothed = None

    frames, frame_metas, frame_src_idx = [], [], []
    for i, (m, k) in enumerate(zip(metas, intrinsics_obj)):""",
    ),
    dict(
        name="P4b 역변환 헬퍼",
        path="homography/pipeline.py",
        old="""def _rotation_from_opk(""",
        new="""def _opk_from_rotation(R):
    \"\"\"[sayou-patch] _rotation_from_opk 의 역변환 (omega, phi, kappa).

    _rotation_from_opk 가 Rz(k) @ Ry(p) @ Rx(o) 순서를 쓰므로 그 역을 푼다.
    짐벌 자세가 phi = ±90° 근처면 특이(gimbal lock)라 그때는 kappa=0 으로
    두고 omega 에 몰아준다 — 회전행렬 자체는 보존된다.
    \"\"\"
    R = np.asarray(R, dtype=np.float64)
    phi = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    if abs(np.cos(phi)) < 1e-8:
        return float(np.arctan2(-R[1, 2], R[1, 1])), float(phi), 0.0
    omega = np.arctan2(-R[1, 2], R[2, 2])
    kappa = np.arctan2(-R[0, 1], R[0, 0])
    return float(omega), float(phi), float(kappa)


def _rotation_from_opk(""",
    ),
    # -- P3d: npz 저장 -------------------------------------------------------
    dict(
        name="P3d cameras.npz 에 관측 저장",
        path="homography/pipeline.py",
        old="""            initial=np.asarray(initial_cameras, dtype=np.float64),
            paths=np.array([str(m.origin_path) for m in metas]))""",
        new="""            initial=np.asarray(initial_cameras, dtype=np.float64),
            paths=np.array([str(m.origin_path) for m in metas]),
            # [sayou-patch]
            **({} if obs_export is None else
               {"obs_cam": obs_export[0], "obs_uv": obs_export[1]}))""",
    ),
    dict(
        name="P4c npz 에 추가",
        path="homography/pipeline.py",
        old="""            **({} if obs_export is None else
               {"obs_cam": obs_export[0], "obs_uv": obs_export[1]}))""",
        new="""            **({} if obs_export is None else
               {"obs_cam": obs_export[0], "obs_uv": obs_export[1]}),
            **({} if cams_smoothed is None else
               {"cams_smoothed": cams_smoothed}))""",
    ),
]


def _resolve(root: Path, rel: str) -> Path:
    p = root / rel
    if p.exists():
        return p
    # 설치 구조가 다를 수 있으니 한 번 더 찾아본다.
    hits = list(root.rglob(Path(rel).name))
    return hits[0] if len(hits) == 1 else p


def run(root: Path, apply: bool, revert: bool, skip=(), only=()) -> int:
    if revert:
        n = 0
        for f in sorted(root.rglob("*" + BAK)):
            tgt = f.with_suffix("")
            shutil.copy2(f, tgt)
            f.unlink()
            print(f"  되돌림: {tgt}")
            n += 1
        print(f"\n백업 {n}개 복원." if n else "\n복원할 백업이 없습니다.")
        return 0

    if not only and "P1" not in skip and "P1b" not in skip:
        print("  ! P1 과 P1b 는 같은 줄을 고칩니다 — 하나만 고르십시오.")
        print("    블록이 200개 이상이면 --skip P1b, 적으면 --skip P1.")
        return 2
    files: dict[Path, str] = {}
    report = []
    for p in PATCHES:
        pid = p["name"].split()[0]
        # P1 / P1b 는 서로 다른 항목이고, P2a~P2e 는 한 묶음이다.
        grp = pid if pid in ("P1", "P1b") else pid[:2]
        if only and grp not in only:
            continue
        if grp in skip:
            report.append((p["name"], "건너뜀", "--skip"))
            continue
        path = _resolve(root, p["path"])
        if not path.exists():
            report.append((p["name"], "파일없음", str(path)))
            continue
        if path not in files:
            files[path] = path.read_text()
        src = files[path]
        if p["new"] in src:
            report.append((p["name"], "이미적용", path.name))
            continue
        cnt = src.count(p["old"])
        if cnt != 1:
            report.append((p["name"], f"앵커{cnt}개", path.name))
            continue
        files[path] = src.replace(p["old"], p["new"])
        report.append((p["name"], "적용가능", path.name))

    w = max(len(r[0]) for r in report)
    for name, state, extra in report:
        mark = {"적용가능": "+", "이미적용": "=", "건너뜀": "."}.get(state, "!")
        print(f"  {mark} {name:<{w}}  {state:<8} {extra}")

    bad = [r for r in report if r[1] not in ("적용가능", "이미적용", "건너뜀")]
    if bad:
        print("\n★ 앵커를 못 찾은 항목이 있습니다. 설치본이 이 zip 과 다른")
        print("  버전일 수 있습니다. 해당 파일을 보내주시면 앵커를 맞추겠습니다.")
        print("  나머지만 적용해도 되지만, 적용 전에 확인하시는 편이 낫습니다.")

    if not apply:
        print("\n--apply 를 붙이면 실제로 적용합니다 (백업 *.sayoubak 생성).")
        return 1 if bad else 0

    for path, text in files.items():
        if text == path.read_text():
            continue
        bak = Path(str(path) + BAK)
        if not bak.exists():
            shutil.copy2(path, bak)
        path.write_text(text)
        print(f"  기록: {path}")
    print("\n적용 완료. 되돌리려면 --revert.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="homography/ 가 들어 있는 sayou 폴더")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="적용하지 않을 항목 (예: --skip P1). RGB 처럼 패널 "
                         "블록이 원래 적은 결과에서는 P1 을 빼십시오")
    ap.add_argument("--only", nargs="*", default=[],
                    help="이 항목만 적용 (예: --only P2 P3)")
    a = ap.parse_args()
    if not a.root.exists():
        print(f"경로 없음: {a.root}")
        return 2
    return run(a.root, a.apply, a.revert, tuple(a.skip), tuple(a.only))


if __name__ == "__main__":
    sys.exit(main())

"""SIFT 특징점·매칭 결과 캐시.

왜 필요한가 (실측 근거)
-----------------------
20분 49초 실행의 내역:

| 단계 | 시간 | 비중 |
|---|---|---|
| **SIFT 추출** | 8m11s | 40.9% |
| 모자이크 | 6m03s | 30.2% |
| BA 전체 | 3m56s | 19.7% |
| **매칭** | 1m30s | 7.5% |

**SIFT 추출 + 매칭이 전체의 48%** 인데, 이 단계는 **입력 이미지가 같으면
결과가 항상 동일**합니다. 파라미터 튜닝을 반복하는 동안 매번 같은 계산을
다시 하고 있었습니다 (이 프로젝트에서만 19회).

캐시하면 재실행이 20m49s → **약 10m20s** 로 줄어듭니다.

캐시 무효화
-----------
결과를 바꿀 수 있는 것만 키에 넣습니다:

* 이미지 파일 목록 (경로 + 크기 + mtime)
* 특징점 파라미터 (max_features 등)
* 인접쌍 목록 (k_neighbors 가 바뀌면 쌍이 달라짐)

하나라도 다르면 캐시를 무시하고 새로 계산합니다. **초점거리 보정이나
BA 설정을 바꿔도 SIFT 결과는 달라지지 않으므로** 그런 튜닝에서는 캐시가
그대로 유효합니다 — 바로 그 경우를 노린 것입니다.

주의
----
* 캐시 파일은 수백 MB 가 될 수 있습니다 (관측 172만 개). 출력 폴더에
  ``.feature_cache/`` 로 저장되며 ``--no-cache`` 로 끄거나 폴더를 지우면
  됩니다.
* ``pickle`` 을 쓰므로 **신뢰할 수 있는 자기 캐시만** 읽어야 합니다.
  다른 곳에서 받은 캐시 파일을 넣지 마세요.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import time

import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["FeatureCache"]

_CACHE_VERSION = 3


class FeatureCache:
    """tie point 계산 결과를 디스크에 캐시."""

    def __init__(self, cache_dir: Path | str | None, enabled: bool = True):
        self.enabled = bool(enabled and cache_dir is not None)
        self.dir = Path(cache_dir) / ".feature_cache" if cache_dir else None
        if self.enabled:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("캐시 폴더를 만들 수 없어 캐시를 끕니다: %s", exc)
                self.enabled = False

    # -- 키 ---------------------------------------------------------------
    @staticmethod
    def _key(metas, pairs, extra: dict) -> str:
        h = hashlib.sha256()
        h.update(f"v{_CACHE_VERSION}".encode())
        for m in metas:
            p = Path(getattr(m, "origin_path", "") or "")
            try:
                st = p.stat()
                sig = f"{p.name}:{st.st_size}:{int(st.st_mtime)}"
            except OSError:
                sig = f"{p.name}:?"
            h.update(sig.encode())
        h.update(repr(sorted(map(tuple, pairs))).encode())
        for k in sorted(extra):
            h.update(f"{k}={extra[k]}".encode())
        return h.hexdigest()[:16]

    def path_for(self, metas, pairs, extra: dict) -> Path | None:
        if not self.enabled:
            return None
        return self.dir / f"tie_{self._key(metas, pairs, extra)}.pkl"

    # -- 읽기/쓰기 ---------------------------------------------------------
    def load(self, metas, pairs, extra: dict):
        p = self.path_for(metas, pairs, extra)
        if p is None or not p.exists():
            return None
        t0 = time.perf_counter()
        try:
            with p.open("rb") as fh:
                payload = pickle.load(fh)
        except Exception as exc:                       # 손상/버전 불일치
            logger.warning("캐시를 읽지 못해 새로 계산합니다 (%s). 파일: %s",
                           exc, p.name)
            return None
        if payload.get("version") != _CACHE_VERSION:
            return None
        n_pairs = len(payload.get("matches", {}))
        logger.info("특징점·매칭 캐시 적중: %s (%s, 쌍 %d개) — SIFT 추출과 "
                    "매칭을 건너뜁니다 (%.1fs 에 로드)",
                    p.name, _fmt_size(p.stat().st_size), n_pairs,
                    time.perf_counter() - t0)
        return payload["matches"], payload["features"]

    def save(self, metas, pairs, extra: dict, matches, features) -> None:
        p = self.path_for(metas, pairs, extra)
        if p is None:
            return
        t0 = time.perf_counter()
        try:
            # cv2.KeyPoint 는 pickle 이 안 되므로 튜플로 직렬화.
            # ★ descriptor 를 uint8 로 저장한다.
            #   OpenCV SIFT descriptor 는 float32 지만 **값이 0~255 정수**다
            #   (검증: 정수와의 최대 차이 0.000000, uint8 왕복 오차
            #   0.000000). 4배 작아지고 손실이 전혀 없다 —
            #   실측 1.6 GB → 약 0.4 GB.
            #   keypoint 좌표는 서브픽셀이므로 float32 로 묶어 저장한다.
            feats_ser = {}
            for idx, (kps, desc, shape) in features.items():
                kp_arr = np.array(
                    [(k.pt[0], k.pt[1], k.size, k.angle, k.response,
                      k.octave, k.class_id) for k in kps], dtype=np.float32)
                d = desc
                if d is not None and d.dtype != np.uint8:
                    if float(np.abs(d - np.round(d)).max()) < 1e-4 \
                            and 0.0 <= float(d.min()) \
                            and float(d.max()) <= 255.0:
                        d = d.astype(np.uint8)
                feats_ser[idx] = (kp_arr, d, shape)
            tmp = p.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                pickle.dump({"version": _CACHE_VERSION, "matches": matches,
                             "features": feats_ser}, fh,
                            protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(p)
        except Exception as exc:
            logger.warning("캐시 저장 실패 (계속 진행): %s", exc)
            return
        logger.info("특징점·매칭 캐시 저장: %s (%s, %.1fs). 다음 실행에서 "
                    "SIFT 추출·매칭 단계를 건너뜁니다.",
                    p.name, _fmt_size(p.stat().st_size),
                    time.perf_counter() - t0)

    @staticmethod
    def restore_features(feats_ser):
        """저장된 튜플을 ``cv2.KeyPoint`` 로 되돌린다."""
        import cv2
        out = {}
        for idx, (kps, desc, shape) in feats_ser.items():
            d = desc
            if d is not None and d.dtype == np.uint8:
                d = d.astype(np.float32)      # 매칭은 float32 를 기대한다
            out[idx] = ([cv2.KeyPoint(x=float(a), y=float(b), size=float(c),
                                      angle=float(e2), response=float(f2),
                                      octave=int(g2), class_id=int(h2))
                         for a, b, c, e2, f2, g2, h2 in kps], d, shape)
        return out


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"

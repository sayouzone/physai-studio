"""특징점·매칭 결과를 바꾸는 모든 것의 지문 — 캐시 키용.

왜 필요한가
-----------
캐시 키는 원래 사진 목록 · 인접쌍 수 · 이웃 수만 담았다(P10 을 적용하면 매칭
배율도). 그래서 **비율 검사, RANSAC 모델, 문턱값, 특징점 상한을 코드에서
바꾸거나 매칭 관련 패치를 켜고 꺼도** 캐시가 이전 매칭을 그대로 돌려준다.
실험이 "모든 지표가 소수점까지 같다"로 끝나고, 바꾼 것이 효과가 없다고
잘못 판단하게 된다. 이 파이프라인에서 이미 겪은 일이다(--panel-unit 을 줬는데
결과가 완전히 같았던 경우 — 그때는 모듈 누락이 원인이었다).

설정값을 하나씩 키에 넣는 방식은 새 설정이 생길 때마다 빠뜨리기 쉽다. 그래서
**매칭 코드 파일 자체의 해시**를 키에 넣는다. 코드에 박힌 값(비율 0.75,
FM_RANSAC 1.0 px, 대응 20개 등)이 바뀌든, patch_sayou 가 함수를 고치든 지문이
저절로 바뀐다. 코드 밖에서 결과를 바꾸는 것(OpenCV 판, GPU 경로, 환경변수)은
따로 넣는다.

지문에 넣지 않는 것
------------------
* ``rtk_match_check`` · ``guided_match`` — 캐시는 그 **앞**의 매칭 결과를
  저장한다. 쌍 검사는 캐시를 읽은 뒤에 적용되므로 키에 넣을 필요가 없다.
  (안내 매칭을 tie point 생성 단계에 넣게 되면 그때 _SOURCES 에 추가할 것.)
* 초점거리 · BA · 모자이크 설정 — 매칭 결과를 바꾸지 않는다. 캐시가 노리는
  것이 바로 이런 튜닝을 빠르게 반복하는 경우다.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

__all__ = ["matching_signature", "MATCHING_SOURCES"]

# 매칭 결과(tie point)를 만드는 코드. 이 파일들의 내용이 바뀌면 캐시를 다시 만든다.
MATCHING_SOURCES = ("extract.py", "match.py", "pairs.py", "sift.py")

# 코드 밖에서 매칭 결과를 바꾸는 환경변수
_ENV_KEYS = ("SAYOU_MATCH_SCALE",)


def _code_fingerprint() -> str:
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in MATCHING_SOURCES:
        p = here / name
        h.update(name.encode())
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()[:12]


def matching_signature() -> dict:
    """캐시 키에 더할 값. 모두 문자열이라 키 해시에 그대로 들어간다."""
    sig = {"match_code": _code_fingerprint()}
    try:
        import cv2
        sig["opencv"] = str(cv2.__version__)
    except Exception:                                   # pragma: no cover
        sig["opencv"] = "?"
    try:
        from ..gpu_backend import HAS_CV_CUDA
        sig["cv_cuda"] = "1" if HAS_CV_CUDA else "0"
    except Exception:                                   # pragma: no cover
        sig["cv_cuda"] = "?"
    for k in _ENV_KEYS:
        sig["env_" + k] = os.environ.get(k, "")
    return sig

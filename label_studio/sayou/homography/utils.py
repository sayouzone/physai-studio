"""공통 유틸리티.

``fmt_elapsed`` 는 ``sfm.bundle_adjustment``, ``features.pairs``, ``pipeline``
세 모듈이 import 하지만 저장소에 파일이 없어 ImportError 가 난다::

    ModuleNotFoundError: No module named 'solar_thermal.georeferencing.utils'

이미 로컬에 같은 이름의 모듈이 있다면 이 파일은 덮어쓰지 말고 무시할 것.
"""

from __future__ import annotations

__all__ = ["fmt_elapsed", "fmt_bytes"]


def fmt_elapsed(seconds: float) -> str:
    """경과 시간을 사람이 읽기 좋은 문자열로.

    로그에서 단계별 시간을 비교하기 쉽도록 자릿수를 맞춘다::

        0.42s / 12.7s / 3m 04s / 1h 22m 09s
    """
    if seconds < 0:
        return "0.00s"
    if seconds < 60:
        return f"{seconds:.2f}s"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def fmt_bytes(n: int) -> str:
    """바이트 수 → 사람이 읽기 좋은 단위 (누산기/맵 메모리 로그용)."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"

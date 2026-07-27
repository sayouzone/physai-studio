import time


def fmt_elapsed(seconds: float) -> str:
    return f"{seconds:.1f}s ({time.strftime('%H:%M:%S', time.gmtime(seconds))})"

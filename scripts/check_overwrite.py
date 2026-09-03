#!/usr/bin/env python3
"""배포 아카이브를 풀기 전에 **기존 파일을 덮어쓰는지** 확인한다.

왜 필요한가: 제가 만든 ``quality.py`` 가 같은 패키지의 기존 ``quality.py``
(RTK 품질 — ``RTKQuality``) 를 덮어써 ``ImportError`` 를 냈습니다. 파일을
수동으로 옮기는 환경에서는 이름 충돌이 조용히 일어납니다.

사용법::

    python scripts/check_overwrite.py <아카이브.tar.gz> <설치 루트>

설치 루트 예: label_studio/sayou/  (그 아래 homography/ 가 있는 곳)
"""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    arc, root = Path(sys.argv[1]), Path(sys.argv[2])
    if not arc.exists() or not root.exists():
        print(f"경로를 찾을 수 없습니다: {arc if not arc.exists() else root}")
        return 2

    with tarfile.open(arc) as tf:
        names = [n for n in tf.getnames() if n.endswith(".py")]

    existing, new = [], []
    for n in names:
        stem = Path(n).name
        hits = list(root.rglob(stem))
        (existing if hits else new).append((stem, hits))

    print(f"아카이브 파이썬 파일 {len(names)}개\n")
    if new:
        print(f"[새 파일 {len(new)}개] — 충돌 없음")
        for s, _ in new:
            print(f"    + {s}")
    if existing:
        print(f"\n[기존 파일을 덮어씀 {len(existing)}개] — 내용 확인 필요")
        for s, hits in existing:
            print(f"    ! {s}")
            for h in hits:
                print(f"        → {h}")
        print("\n덮어쓰기 전에 해당 파일들을 백업하거나 git 상태를 확인하세요.")
        print("같은 이름이라도 **다른 기능**일 수 있습니다.")
    return 1 if existing else 0


if __name__ == "__main__":
    raise SystemExit(main())

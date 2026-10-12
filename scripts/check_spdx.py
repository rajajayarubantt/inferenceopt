# SPDX-License-Identifier: Apache-2.0
"""Fail if a source file lacks the SPDX license header in its first lines.

Usage: python scripts/check_spdx.py FILE [FILE ...]  (pre-commit passes staged files)
"""

from __future__ import annotations

import sys
from pathlib import Path

HEADER = "SPDX-License-Identifier: Apache-2.0"
SKIP_SUFFIXES = (".d.ts",)
SKIP_NAMES = {"_version.py"}


def main(argv: list[str]) -> int:
    missing = []
    for name in argv:
        path = Path(name)
        if path.name in SKIP_NAMES or path.name.endswith(SKIP_SUFFIXES) or not path.is_file():
            continue
        if path.suffix == ".py" and path.stat().st_size == 0:
            continue
        head = path.read_text(encoding="utf-8", errors="replace").splitlines()[:3]
        if not any(HEADER in line for line in head):
            missing.append(name)
    for name in missing:
        print(f"{name}: missing '{HEADER}' header")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

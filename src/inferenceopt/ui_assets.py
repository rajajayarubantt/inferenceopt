# SPDX-License-Identifier: Apache-2.0
"""Location of the built web UI bundled into the package."""

from __future__ import annotations

from pathlib import Path

UI_DIR = Path(__file__).parent / "_ui"


def ui_index() -> Path | None:
    index = UI_DIR / "index.html"
    return index if index.is_file() else None

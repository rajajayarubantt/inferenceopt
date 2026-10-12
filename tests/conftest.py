# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from inferenceopt.services import AppContext

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never touch the developer's real config or database."""
    home = tmp_path / "home"
    monkeypatch.setenv("INFERENCEOPT_HOME", str(home))
    for name in list(os.environ):
        if name.startswith("INFERENCEOPT_") and not name.startswith(
            ("INFERENCEOPT_HOME", "INFERENCEOPT_TEST_")
        ):
            monkeypatch.delenv(name)
    return home


@pytest.fixture
def ctx() -> Iterator[AppContext]:
    context = AppContext.open()
    yield context
    context.close()

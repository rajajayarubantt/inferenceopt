# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inferenceopt.cli import app
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.unit

runner = CliRunner()


def test_version_json() -> None:
    result = runner.invoke(app, ["version", "--json"])
    assert result.exit_code == 0
    assert "inferenceopt" in json.loads(result.stdout)


def test_doctor_json_passes() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.stdout
    names = {c["name"] for c in json.loads(result.stdout)["checks"]}
    assert {"python", "database", "web ui"} <= names


def test_config_show_hides_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCEOPT_LLM__API_KEY", "sk-thisisaverysecretkey123")
    result = runner.invoke(app, ["config", "show", "--json"])
    assert result.exit_code == 0
    assert "thisisaverysecretkey" not in result.stdout
    assert json.loads(result.stdout)["llm"]["api_key"] == {"is_set": True}


def test_job_validate_ok() -> None:
    result = runner.invoke(app, ["job", "validate", str(REPO_ROOT / "configs/example-job.yaml")])
    assert result.exit_code == 0


def test_job_validate_bad(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("job:\n  name: BAD NAME\n", encoding="utf-8")
    result = runner.invoke(app, ["job", "validate", str(bad)])
    assert result.exit_code == 1
    assert "INVALID" in result.stdout


def test_bad_config_exit_code(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("nonsense: 1\n", encoding="utf-8")
    result = runner.invoke(app, ["doctor", "--config", str(cfg)])
    assert result.exit_code == 2


def test_ui_refuses_remote_bind_without_flag() -> None:
    result = runner.invoke(app, ["ui", "--host", "0.0.0.0", "--no-browser"])
    assert result.exit_code == 2

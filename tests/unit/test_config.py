# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from inferenceopt.config import ConfigError, load_config
from inferenceopt.config.redaction import REDACTED, redact, redact_text, register_secret
from inferenceopt.logs import configure_logging
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.unit


def test_defaults_without_file() -> None:
    cfg = load_config()
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.allow_remote is False
    assert cfg.llm.enabled is False


def test_example_config_loads() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "example-config.yaml")
    assert cfg.server.port == 8765


def test_env_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("server:\n  port: 9000\n", encoding="utf-8")
    monkeypatch.setenv("INFERENCEOPT_SERVER__PORT", "9100")
    assert load_config(path).server.port == 9100


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("sever:\n  port: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown top-level keys: sever"):
        load_config(path)


def test_unknown_nested_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("server:\n  prot: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_explicit_file_is_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_secret_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCEOPT_LLM__API_KEY", "sk-supersecretvalue1234567")
    cfg = load_config()
    assert "supersecret" not in repr(cfg)
    assert cfg.llm.api_key is not None


@pytest.mark.parametrize(
    "text",
    [
        "key sk-abcdefghijklmnopqrstuvwx used",
        "Authorization: Bearer abcdefghijklmnop",
        "token hf_ABCDEFGHIJKLMNOPQRSTUVWX",
        "ghp_ABCDEFGHIJKLMNOPQRSTUVWX",
    ],
)
def test_redact_text_patterns(text: str) -> None:
    assert REDACTED in redact_text(text)


def test_redact_mapping_by_key() -> None:
    out = redact({"api_key": "plain", "nested": {"password": "x"}, "port": 1})
    assert out == {"api_key": REDACTED, "nested": {"password": REDACTED}, "port": 1}


def test_registered_secret_redacted_in_logs(capsys: pytest.CaptureFixture[str]) -> None:
    register_secret("my-session-token-xyz")
    configure_logging("INFO")
    logging.getLogger("inferenceopt.test").info("url is /?token=%s", "my-session-token-xyz")
    err = capsys.readouterr().err
    assert "my-session-token-xyz" not in err
    assert REDACTED in err


def test_filter_keeps_args_tuple_for_uvicorn_access_log() -> None:
    from inferenceopt.config.redaction import RedactingFilter

    register_secret("launch-token-abcdef")
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "GET", "/?token=launch-token-abcdef", "1.1", 200),
        None,
    )
    assert RedactingFilter().filter(record)
    assert isinstance(record.args, tuple) and len(record.args) == 5
    assert "launch-token-abcdef" not in record.getMessage()

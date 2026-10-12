# SPDX-License-Identifier: Apache-2.0
"""Remove secrets from strings, nested data and log records."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Mapping
from typing import Any

from pydantic import SecretStr

REDACTED = "***REDACTED***"

_SECRET_KEY = re.compile(
    r"(api[_-]?key|token|secret|password|passwd|authorization|credential|cookie)", re.I
)
_SECRET_VALUE_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),  # OpenAI/Anthropic-style keys
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),  # Hugging Face tokens
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub tokens
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key ids
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/]{8,}=*"),
]

_known_secrets: set[str] = set()
_lock = threading.Lock()


def register_secret(value: str | SecretStr | None) -> None:
    """Redact this exact value wherever it appears from now on (e.g. a session token)."""
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    if raw and len(raw) >= 6:
        with _lock:
            _known_secrets.add(raw)


def redact_text(text: str) -> str:
    with _lock:
        known = sorted(_known_secrets, key=len, reverse=True)
    for secret in known:
        text = text.replace(secret, REDACTED)
    for pattern in _SECRET_VALUE_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def redact(value: Any) -> Any:
    """Return a copy of ``value`` with secrets replaced. Keys that look secret are masked."""
    if isinstance(value, SecretStr):
        return REDACTED
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            k: (
                REDACTED
                if _SECRET_KEY.search(str(k)) and isinstance(v, str | int | float | bytes)
                else redact(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list | tuple | set):
        return type(value)(redact(v) for v in value)
    return value


class RedactingFilter(logging.Filter):
    """Logging filter that redacts the message template and each argument.

    Arguments are redacted in place rather than pre-formatted, because some formatters
    (e.g. uvicorn's access log) unpack ``record.args`` themselves.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact(a) for a in record.args)
        elif isinstance(record.args, Mapping):
            record.args = redact(record.args)
        return True

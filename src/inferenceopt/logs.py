# SPDX-License-Identifier: Apache-2.0
"""Structured logging setup. Every handler redacts secrets before output."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from inferenceopt.config.redaction import RedactingFilter, redact_text

_HANDLER_NAME = "inferenceopt"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(payload)


def configure_logging(level: str = "INFO", *, json_format: bool = False) -> None:
    """Configure the ``inferenceopt`` logger. Safe to call more than once."""
    logger = logging.getLogger("inferenceopt")
    logger.setLevel(level)
    for handler in list(logger.handlers):
        if handler.get_name() == _HANDLER_NAME:
            logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.addFilter(RedactingFilter())
    handler.setFormatter(
        JsonFormatter()
        if json_format
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False

# SPDX-License-Identifier: Apache-2.0
"""Inference runtime descriptions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from inferenceopt.schemas.base import Versioned


class SupportLevel(StrEnum):
    TESTED = "tested"
    """Validated end to end on real hardware in CI or by maintainers."""
    EXPERIMENTAL = "experimental"
    """Implemented and contract-tested, but not validated on real hardware."""
    UNSUPPORTED = "unsupported"
    """Not available on this platform."""


class RuntimeProfile(Versioned):
    name: str
    version: str | None = None
    support_level: SupportLevel
    executable: str | None = None
    platforms: list[str] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

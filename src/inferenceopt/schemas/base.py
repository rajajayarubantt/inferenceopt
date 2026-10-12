# SPDX-License-Identifier: Apache-2.0
"""Shared base classes for all public data contracts."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Schema(BaseModel):
    """Strict base: unknown fields are errors, so typos in job files fail loudly."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Versioned(Schema):
    """A persisted or exchanged document. Bump ``schema_version`` on breaking changes."""

    schema_version: int = 1


class Provenance(StrEnum):
    """Where a value came from. Never present an inferred value as detected."""

    DETECTED = "detected"
    INFERRED = "inferred"
    USER_SPECIFIED = "user_specified"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def utcnow() -> datetime:
    return datetime.now(tz=UTC)

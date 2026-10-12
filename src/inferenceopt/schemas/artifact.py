# SPDX-License-Identifier: Apache-2.0
"""Artifacts and deployment manifests (Plan.md §13)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from inferenceopt.schemas.base import Schema, Versioned, utcnow


class Artifact(Schema):
    path: str
    kind: Literal["config", "script", "container", "report", "lockfile", "model", "other"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class DeploymentManifest(Versioned):
    trial_id: str
    model_id: str
    runtime: str
    runtime_version: str | None = None
    configuration: dict[str, Any]
    benchmark_command: str
    artifacts: list[Artifact] = Field(default_factory=list)
    tested_properties: list[str] = Field(default_factory=list)
    untested_properties: list[str] = Field(default_factory=list)
    inferenceopt_version: str
    created_at: datetime = Field(default_factory=utcnow)

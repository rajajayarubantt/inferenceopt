# SPDX-License-Identifier: Apache-2.0
"""Hardware inventory (Plan.md §6)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from inferenceopt.schemas.base import Provenance, Schema, Versioned, new_id, utcnow


class DeviceBackend(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"
    OTHER = "other"


class Device(Schema):
    device_id: str
    backend: DeviceBackend
    name: str
    architecture: str | None = None
    compute_capability: str | None = None
    memory_total_bytes: int | None = Field(default=None, ge=0)
    memory_free_bytes: int | None = Field(default=None, ge=0)
    driver_version: str | None = None
    supported_dtypes: list[str] = Field(default_factory=list)
    field_sources: dict[str, Provenance] = Field(default_factory=dict)
    """Provenance per field name. Fields not listed are ``detected``."""


class HostInfo(Schema):
    os: str
    os_version: str
    python_version: str
    cpu_arch: str
    cpu_model: str | None = None
    cpu_flags: list[str] = Field(default_factory=list)
    logical_cores: int | None = None
    physical_cores: int | None = None
    ram_total_bytes: int | None = None
    ram_available_bytes: int | None = None


class HardwareProfile(Versioned):
    id: str = Field(default_factory=lambda: new_id("hw"))
    captured_at: datetime = Field(default_factory=utcnow)
    host: HostInfo
    devices: list[Device] = Field(default_factory=list)
    runtime_versions: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    """Human-readable explanations, e.g. why a capability could not be detected."""

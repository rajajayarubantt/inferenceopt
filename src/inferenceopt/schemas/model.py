# SPDX-License-Identifier: Apache-2.0
"""LLM model references and metadata (Plan.md §7)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from inferenceopt.schemas.base import Versioned

WeightFormat = Literal["gguf", "safetensors", "pytorch", "tensorrt_engine", "other"]


class ModelReference(Versioned):
    id: str
    """Registry key, e.g. ``hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF@<rev>``."""
    source: Literal["huggingface", "local", "url"]
    repo: str | None = None
    revision: str | None = None
    """Immutable revision (commit hash) where available."""
    path: Path | None = None
    architecture: str | None = None
    parameter_count: int | None = Field(default=None, ge=0)
    weight_format: WeightFormat | None = None
    quantization: str | None = None
    context_length_max: int | None = Field(default=None, gt=0)
    """Advertised maximum. Never assume it is feasible on the current machine."""
    license: str | None = None
    license_url: str | None = None
    requires_custom_code: bool = False
    supported_runtimes: list[str] = Field(default_factory=list)
    sha256: dict[str, str] = Field(default_factory=dict)
    """File name to SHA-256 hex digest."""

# SPDX-License-Identifier: Apache-2.0
"""LLM model references and metadata (Plan.md §7)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from inferenceopt.schemas.base import Provenance, Schema, Versioned

WeightFormat = Literal["gguf", "safetensors", "pytorch", "tensorrt_engine", "other"]
ModelSource = Literal["huggingface", "local", "url"]
RegistryOrigin = Literal["curated", "user"]


class ModelArchitecture(Schema):
    """Shape parameters needed for memory estimation. All values describe one model copy."""

    family: str | None = None
    """Architecture name as the model declares it, e.g. ``llama`` or ``qwen2``."""
    num_layers: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    vocab_size: int | None = Field(default=None, gt=0)
    sliding_window: int | None = Field(default=None, gt=0)
    """If set, attention only looks back this many tokens, which caps the KV cache."""


class ModelReference(Versioned):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")
    """Registry key, e.g. ``hf:Qwen/Qwen2.5-0.5B-Instruct`` or ``local:my-model-q4``."""
    display_name: str
    source: ModelSource
    origin: RegistryOrigin = "user"
    repo: str | None = None
    revision: str | None = None
    """Immutable revision (commit hash) where available."""
    path: str | None = None
    """Local file or directory. Only for ``source == "local"``."""
    weight_format: WeightFormat | None = None
    quantization: str | None = None
    """E.g. ``Q4_K_M`` (GGUF) or ``bf16`` (safetensors)."""
    parameter_count: int | None = Field(default=None, ge=0)
    weight_bytes: int | None = Field(default=None, ge=0)
    """Size of the weight files on disk, when known."""
    architecture: ModelArchitecture | None = None
    context_length_max: int | None = Field(default=None, gt=0)
    """Advertised maximum. Never assume it is feasible on the current machine."""
    license: str | None = None
    license_url: str | None = None
    requires_custom_code: bool = False
    supported_runtimes: list[str] = Field(default_factory=list)
    sha256: dict[str, str] = Field(default_factory=dict)
    """File name to SHA-256 hex digest."""
    field_sources: dict[str, Provenance] = Field(default_factory=dict)
    """Provenance per field. Curated entries are ``user_specified`` until verified."""
    notes: list[str] = Field(default_factory=list)

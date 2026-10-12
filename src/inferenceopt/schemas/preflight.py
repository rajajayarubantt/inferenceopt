# SPDX-License-Identifier: Apache-2.0
"""Memory estimates and preflight (will-it-fit) reports (Plan.md §7.2, §7.3)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from inferenceopt.schemas.base import Schema

KVCacheDType = Literal["f32", "f16", "bf16", "q8_0", "q4_0", "fp8"]
Verdict = Literal["fits", "tight", "does_not_fit", "unknown"]


class EstimateInputs(Schema):
    context_tokens: int = Field(default=4096, gt=0)
    concurrency: int = Field(default=1, gt=0)
    """Sequences resident at the same time, each with its own KV cache."""
    kv_cache_dtype: KVCacheDType = "f16"
    batch_tokens: int = Field(default=512, gt=0)
    """Tokens processed per forward pass (llama.cpp ``n_ubatch``), sizes scratch buffers."""
    flash_attention: bool = True
    weight_bits_per_param: float | None = Field(default=None, gt=0)
    """Overrides the bits/param derived from the model's quantization."""


class MemoryComponent(Schema):
    name: str
    bytes: int = Field(ge=0)
    basis: str
    """How this number was obtained, in one line."""


class MemoryEstimate(Schema):
    """Always an estimate: actual memory must be measured (Plan.md §7.2)."""

    kind: Literal["estimate"] = "estimate"
    low_bytes: int
    expected_bytes: int
    high_bytes: int
    components: list[MemoryComponent]
    assumptions: list[str]
    inputs: EstimateInputs


class DeviceFit(Schema):
    device_id: str
    device_name: str
    available_bytes: int | None
    """Free memory now (minus a safety reserve), or None if unknown."""
    verdict: Verdict
    reason: str


class RuntimeCompatibility(Schema):
    runtime: str
    compatible: bool
    reasons: list[str] = Field(default_factory=list)


class PreflightReport(Schema):
    model_id: str
    estimate: MemoryEstimate | None
    devices: list[DeviceFit]
    runtimes: list[RuntimeCompatibility]
    verdict: Verdict
    """Best verdict across devices."""
    messages: list[str] = Field(default_factory=list)

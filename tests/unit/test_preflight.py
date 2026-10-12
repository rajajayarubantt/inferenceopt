# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from inferenceopt.models import check_runtime, preflight
from inferenceopt.schemas import EstimateInputs, ModelArchitecture, ModelReference
from tests.helpers import GIB, hardware_profile

pytestmark = pytest.mark.unit

ARCH = ModelArchitecture(
    num_layers=32, hidden_size=4096, num_attention_heads=32, num_kv_heads=8, head_dim=128
)


def gguf_model(weight_gib: float) -> ModelReference:
    return ModelReference(
        id="local:m",
        display_name="m",
        source="local",
        weight_format="gguf",
        weight_bytes=int(weight_gib * GIB),
        architecture=ARCH,
        context_length_max=8192,
    )


def test_fits_tight_and_does_not_fit() -> None:
    inputs = EstimateInputs(context_tokens=4096)
    hw = hardware_profile(gpu_free_gib=20)
    assert preflight(gguf_model(4), hw, inputs, ["llamacpp"]).verdict == "fits"
    gpu = next(d for d in preflight(gguf_model(17.0), hw, inputs, ["llamacpp"]).devices
               if d.device_id == "gpu0")  # fmt: skip
    assert gpu.verdict == "tight"
    report = preflight(gguf_model(40), hw, inputs, ["llamacpp"])
    assert report.verdict == "does_not_fit"


def test_partial_offload_hint() -> None:
    hw = hardware_profile(gpu_free_gib=6, gpu_total_gib=6, ram_free_gib=16)
    report = preflight(gguf_model(8), hw, EstimateInputs(), ["llamacpp"])
    gpu = next(d for d in report.devices if d.device_id == "gpu0")
    assert gpu.verdict == "does_not_fit"
    assert any("offload" in m for m in report.messages)


def test_context_above_model_maximum_warns() -> None:
    report = preflight(gguf_model(1), hardware_profile(), EstimateInputs(context_tokens=16384))
    assert any("exceeds" in m for m in report.messages)


def test_unknown_free_memory() -> None:
    hw = hardware_profile()
    hw.devices[1] = hw.devices[1].model_copy(update={"memory_free_bytes": None})
    report = preflight(gguf_model(1), hw, EstimateInputs(), ["llamacpp"])
    assert next(d for d in report.devices if d.device_id == "gpu0").verdict == "unknown"


def test_runtime_rules() -> None:
    safetensors = gguf_model(1).model_copy(update={"weight_format": "safetensors"})
    linux, windows = hardware_profile(os="Linux"), hardware_profile(os="Windows")
    cpu_only = hardware_profile(gpu_free_gib=None)

    assert check_runtime("llamacpp", gguf_model(1), windows).compatible
    assert not check_runtime("llamacpp", safetensors, linux).compatible
    assert check_runtime("vllm", safetensors, linux).compatible
    assert not check_runtime("vllm", safetensors, windows).compatible
    assert not check_runtime("vllm", safetensors, cpu_only).compatible
    assert check_runtime("tensorrt_llm", safetensors, linux).compatible
    assert not check_runtime("tensorrt_llm", gguf_model(1), linux).compatible

    custom = safetensors.model_copy(update={"requires_custom_code": True})
    result = check_runtime("vllm", custom, linux)
    assert not result.compatible and any("custom code" in r for r in result.reasons)


def test_no_compatible_runtime_means_does_not_fit() -> None:
    safetensors = gguf_model(1).model_copy(update={"weight_format": "safetensors"})
    report = preflight(safetensors, hardware_profile(os="Windows"), EstimateInputs())
    assert report.verdict == "does_not_fit"
    assert any("No runtime" in m for m in report.messages)


def test_model_without_architecture() -> None:
    model = gguf_model(1).model_copy(update={"architecture": None})
    report = preflight(model, hardware_profile(), EstimateInputs(), ["llamacpp"])
    assert report.estimate is None
    assert report.verdict == "unknown"

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inferenceopt.models import EstimateError, estimate_memory
from inferenceopt.models.estimator import KV_BYTES_PER_ELEMENT
from inferenceopt.schemas import EstimateInputs, ModelArchitecture, ModelReference

pytestmark = pytest.mark.property

ARCH = ModelArchitecture(
    family="llama",
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    num_kv_heads=8,
    head_dim=128,
    vocab_size=128256,
)


def model(**kw: object) -> ModelReference:
    base: dict[str, object] = {
        "id": "test:m",
        "display_name": "m",
        "source": "local",
        "parameter_count": 8_000_000_000,
        "quantization": "bf16",
        "architecture": ARCH,
    }
    base.update(kw)
    return ModelReference.model_validate(base)


def test_kv_cache_matches_formula() -> None:
    est = estimate_memory(model(), EstimateInputs(context_tokens=8192, concurrency=2))
    kv = next(c for c in est.components if c.name == "kv_cache")
    assert kv.bytes == 2 * 32 * 8 * 128 * 8192 * 2 * 2  # = 2 GiB


def test_weights_from_bits_and_from_file_size() -> None:
    est = estimate_memory(model(), EstimateInputs())
    assert est.components[0].bytes == 16_000_000_000
    est = estimate_memory(model(weight_bytes=4_000_000_000), EstimateInputs())
    assert est.components[0].bytes == 4_000_000_000
    est = estimate_memory(model(), EstimateInputs(weight_bits_per_param=4))
    assert est.components[0].bytes == 4_000_000_000


def test_sliding_window_caps_expected_but_not_high() -> None:
    windowed = model(architecture=ARCH.model_copy(update={"sliding_window": 4096}))
    inputs = EstimateInputs(context_tokens=32768)
    a, b = estimate_memory(model(), inputs), estimate_memory(windowed, inputs)
    assert b.expected_bytes < a.expected_bytes
    assert b.high_bytes == a.high_bytes


def test_missing_metadata_raises() -> None:
    with pytest.raises(EstimateError, match="architecture"):
        estimate_memory(model(architecture=None), EstimateInputs())
    with pytest.raises(EstimateError, match="bits per weight"):
        estimate_memory(model(quantization="mystery"), EstimateInputs())
    with pytest.raises(EstimateError, match="parameter count"):
        estimate_memory(model(parameter_count=None), EstimateInputs())


inputs_st = st.builds(
    EstimateInputs,
    context_tokens=st.integers(1, 262_144),
    concurrency=st.integers(1, 256),
    kv_cache_dtype=st.sampled_from(list(KV_BYTES_PER_ELEMENT)),
    batch_tokens=st.integers(1, 8192),
    flash_attention=st.booleans(),
)


@given(inputs_st)
def test_range_is_ordered(inputs: EstimateInputs) -> None:
    est = estimate_memory(model(), inputs)
    assert 0 < est.low_bytes <= est.expected_bytes <= est.high_bytes
    assert est.expected_bytes == sum(c.bytes for c in est.components)


@given(inputs_st, st.integers(1, 4096))
def test_monotonic_in_context(inputs: EstimateInputs, extra: int) -> None:
    more = inputs.model_copy(update={"context_tokens": inputs.context_tokens + extra})
    assert (
        estimate_memory(model(), more).expected_bytes
        >= estimate_memory(model(), inputs).expected_bytes
    )


@given(inputs_st, st.integers(1, 64))
def test_monotonic_in_concurrency(inputs: EstimateInputs, extra: int) -> None:
    more = inputs.model_copy(update={"concurrency": inputs.concurrency + extra})
    assert (
        estimate_memory(model(), more).expected_bytes
        > estimate_memory(model(), inputs).expected_bytes
    )


@given(inputs_st)
def test_smaller_kv_dtype_never_costs_more(inputs: EstimateInputs) -> None:
    ordered = sorted(KV_BYTES_PER_ELEMENT, key=KV_BYTES_PER_ELEMENT.__getitem__)
    sizes = [
        estimate_memory(model(), inputs.model_copy(update={"kv_cache_dtype": d})).expected_bytes
        for d in ordered
    ]
    assert sizes == sorted(sizes)

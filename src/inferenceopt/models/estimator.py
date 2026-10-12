# SPDX-License-Identifier: Apache-2.0
"""Memory estimation (Plan.md §7.2).

Every number here is an estimate with an explicit basis and a low/expected/high range.
Actual memory must be measured; benchmark trials always record measured peaks.
"""

from __future__ import annotations

from inferenceopt.schemas import (
    EstimateInputs,
    MemoryComponent,
    MemoryEstimate,
    ModelReference,
)

MIB = 1024**2

# Approximate average bits per weight, including quantization scales.
BITS_PER_WEIGHT = {
    "f32": 32.0, "fp32": 32.0, "float32": 32.0,
    "f16": 16.0, "fp16": 16.0, "float16": 16.0, "bf16": 16.0, "bfloat16": 16.0,
    "fp8": 8.0, "int8": 8.0, "q8_0": 8.5,
    "q6_k": 6.56, "q5_k_m": 5.69, "q5_k_s": 5.54, "q5_0": 5.5, "q5_1": 6.0,
    "q4_k_m": 4.89, "q4_k_s": 4.58, "q4_0": 4.5, "q4_1": 5.0, "iq4_xs": 4.25, "iq4_nl": 4.5,
    "q3_k_l": 4.27, "q3_k_m": 3.91, "q3_k_s": 3.5, "q2_k": 3.35,
    "awq": 4.25, "gptq": 4.25,
}  # fmt: skip

KV_BYTES_PER_ELEMENT = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "q8_0": 34 / 32,  # 32 int8 values + one f16 scale per block
    "q4_0": 18 / 32,  # 32 4-bit values + one f16 scale per block
}

# Runtime/driver context, allocator pools and library code. Wide on purpose.
OVERHEAD_LOW, OVERHEAD_EXPECTED, OVERHEAD_HIGH = 150 * MIB, 300 * MIB, 800 * MIB
DEFAULT_VOCAB = 32_000


class EstimateError(ValueError):
    """Not enough metadata to estimate memory for this model."""


def _weights(model: ModelReference, inputs: EstimateInputs) -> MemoryComponent:
    if inputs.weight_bits_per_param is None and model.weight_bytes:
        return MemoryComponent(
            name="weights", bytes=model.weight_bytes, basis="size of the weight files on disk"
        )
    bits = inputs.weight_bits_per_param
    label = "override"
    if bits is None and model.quantization:
        label = model.quantization
        bits = BITS_PER_WEIGHT.get(model.quantization.lower())
    if bits is None:
        raise EstimateError(
            f"unknown bits per weight for quantization {model.quantization!r}; "
            "pass weight_bits_per_param"
        )
    params = model.parameter_count
    if params is None:
        raise EstimateError("parameter count is unknown")
    return MemoryComponent(
        name="weights",
        bytes=int(params * bits / 8),
        basis=f"{params:,} params x {bits:g} bits ({label})",
    )


def kv_cache_bytes(model: ModelReference, inputs: EstimateInputs, *, cap_window: bool) -> int:
    """KV ≈ 2 x layers x kv_heads x head_dim x cached_tokens x bytes/elem x sequences."""
    arch = model.architecture
    assert arch is not None
    tokens = inputs.context_tokens
    if cap_window and arch.sliding_window:
        tokens = min(tokens, arch.sliding_window)
    per_elem = KV_BYTES_PER_ELEMENT[inputs.kv_cache_dtype]
    return int(
        2 * arch.num_layers * arch.num_kv_heads * arch.head_dim * tokens * per_elem
        * inputs.concurrency
    )  # fmt: skip


def estimate_memory(model: ModelReference, inputs: EstimateInputs) -> MemoryEstimate:
    arch = model.architecture
    if arch is None:
        raise EstimateError("model architecture (layers, heads, head size) is unknown")

    assumptions = [
        "Estimate only; real usage depends on the runtime, its version and settings.",
        "Assumes a standard decoder-only transformer with one KV cache per sequence.",
        f"Runtime overhead assumed {OVERHEAD_LOW // MIB}-{OVERHEAD_HIGH // MIB} MiB.",
    ]
    weights = _weights(model, inputs)

    kv = kv_cache_bytes(model, inputs, cap_window=True)
    kv_full = kv_cache_bytes(model, inputs, cap_window=False)
    kv_basis = (
        f"2 x {arch.num_layers} layers x {arch.num_kv_heads} KV heads x {arch.head_dim} dims"
        f" x {inputs.context_tokens:,} tokens x {inputs.kv_cache_dtype}"
        f" x {inputs.concurrency} sequence(s)"
    )
    if arch.sliding_window and arch.sliding_window < inputs.context_tokens:
        kv_basis += f", capped at the {arch.sliding_window:,}-token sliding window"
        assumptions.append(
            "Sliding-window cap assumed for all layers; models that mix full and windowed "
            "attention need more (the high bound uses the full context)."
        )

    vocab = arch.vocab_size or DEFAULT_VOCAB
    if arch.vocab_size is None:
        assumptions.append(f"Vocabulary size unknown; assumed {DEFAULT_VOCAB:,}.")
    compute = inputs.batch_tokens * (vocab + 4 * arch.hidden_size) * 4
    compute_basis = f"{inputs.batch_tokens} batch tokens x (logits + activations) in f32"
    if not inputs.flash_attention:
        compute += inputs.batch_tokens * inputs.context_tokens * arch.num_attention_heads * 4
        compute_basis += " + attention scores (no flash attention)"

    components = [
        weights,
        MemoryComponent(name="kv_cache", bytes=kv, basis=kv_basis),
        MemoryComponent(name="compute_buffers", bytes=compute, basis=compute_basis),
        MemoryComponent(name="runtime_overhead", bytes=OVERHEAD_EXPECTED, basis="assumed"),
    ]
    return MemoryEstimate(
        low_bytes=int(weights.bytes * 0.98 + kv + compute * 0.5 + OVERHEAD_LOW),
        expected_bytes=sum(c.bytes for c in components),
        high_bytes=int(weights.bytes * 1.05 + kv_full + compute * 2 + OVERHEAD_HIGH),
        components=components,
        assumptions=assumptions,
        inputs=inputs,
    )

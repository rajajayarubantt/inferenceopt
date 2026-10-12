# SPDX-License-Identifier: Apache-2.0
"""Static compatibility checks and will-it-fit reports (Plan.md §7.3 stages 1-2).

These checks are deliberately conservative and run before any download or trial. Runtime
adapters (Phase 3) refine runtime compatibility with real capability probes.
"""

from __future__ import annotations

from inferenceopt.models.estimator import EstimateError, estimate_memory
from inferenceopt.schemas import (
    DeviceBackend,
    DeviceFit,
    EstimateInputs,
    HardwareProfile,
    MemoryEstimate,
    ModelReference,
    PreflightReport,
    RuntimeCompatibility,
    Verdict,
)

MIB = 1024**2
KNOWN_RUNTIMES = ("llamacpp", "vllm", "tensorrt_llm")
_RANK: dict[Verdict, int] = {"fits": 3, "tight": 2, "does_not_fit": 1, "unknown": 0}
_RUNTIME_BACKENDS = {
    "llamacpp": {DeviceBackend.CPU, DeviceBackend.CUDA, DeviceBackend.METAL, DeviceBackend.ROCM},
    "vllm": {DeviceBackend.CUDA, DeviceBackend.ROCM},
    "tensorrt_llm": {DeviceBackend.CUDA},
}


def check_runtime(
    runtime: str, model: ModelReference, hardware: HardwareProfile
) -> RuntimeCompatibility:
    reasons: list[str] = []
    backends = {d.backend for d in hardware.devices}
    fmt = model.weight_format

    if runtime == "llamacpp":
        if fmt != "gguf":
            reasons.append(
                "llama.cpp needs a GGUF file; use a GGUF variant of this model or convert it "
                "with llama.cpp's convert_hf_to_gguf.py."
            )
    elif runtime == "vllm":
        if hardware.host.os != "Linux":
            reasons.append(
                f"vLLM runs on Linux (use WSL2 on Windows); this host is {hardware.host.os}."
            )
        if not backends & _RUNTIME_BACKENDS["vllm"]:
            reasons.append("vLLM needs an NVIDIA (CUDA) or AMD (ROCm) GPU; none detected.")
        if fmt not in ("safetensors", "pytorch", None):
            reasons.append(f"vLLM expects Hugging Face weights; this model is {fmt}.")
    elif runtime == "tensorrt_llm":
        if hardware.host.os != "Linux":
            reasons.append(f"TensorRT-LLM runs on Linux; this host is {hardware.host.os}.")
        gpus = [d for d in hardware.devices if d.backend == DeviceBackend.CUDA]
        if not gpus:
            reasons.append("TensorRT-LLM needs an NVIDIA GPU; none detected.")
        if fmt not in ("safetensors", "tensorrt_engine", None):
            reasons.append(f"TensorRT-LLM expects Hugging Face weights or an engine; got {fmt}.")
    else:
        reasons.append(f"unknown runtime {runtime!r}")

    if model.requires_custom_code and runtime != "llamacpp":
        reasons.append("Model requires custom code (trust_remote_code); review it before use.")
        return RuntimeCompatibility(runtime=runtime, compatible=False, reasons=reasons)
    return RuntimeCompatibility(runtime=runtime, compatible=not reasons, reasons=reasons)


def _fit(
    estimate: MemoryEstimate, free: int | None, total: int | None
) -> tuple[Verdict, str, int | None]:
    if free is None:
        return "unknown", "free memory could not be detected", None
    reserve = max(int((total or free) * 0.05), 256 * MIB)
    usable = max(free - reserve, 0)

    def gib(n: int) -> str:
        return f"{n / 1024**3:.2f} GiB"

    span = f"needs {gib(estimate.low_bytes)}-{gib(estimate.high_bytes)}, {gib(usable)} usable"
    if estimate.high_bytes <= usable:
        return "fits", f"Fits even at the high estimate ({span}).", usable
    if estimate.expected_bytes <= usable:
        return "tight", f"Fits at the expected estimate but not the high one ({span}).", usable
    return "does_not_fit", f"Does not fit ({span}).", usable


def preflight(
    model: ModelReference,
    hardware: HardwareProfile,
    inputs: EstimateInputs,
    runtimes: list[str] | None = None,
) -> PreflightReport:
    messages: list[str] = []
    compat = [check_runtime(r, model, hardware) for r in (runtimes or list(KNOWN_RUNTIMES))]
    usable_backends = set().union(
        *(_RUNTIME_BACKENDS.get(c.runtime, set()) for c in compat if c.compatible)
    )

    try:
        estimate: MemoryEstimate | None = estimate_memory(model, inputs)
    except EstimateError as exc:
        estimate = None
        messages.append(f"No memory estimate: {exc}.")

    if model.context_length_max and inputs.context_tokens > model.context_length_max:
        messages.append(
            f"Requested context {inputs.context_tokens:,} exceeds the model's advertised "
            f"maximum of {model.context_length_max:,} tokens."
        )

    fits: list[DeviceFit] = []
    for device in hardware.devices:
        if estimate is None:
            fits.append(
                DeviceFit(
                    device_id=device.device_id,
                    device_name=device.name,
                    available_bytes=None,
                    verdict="unknown",
                    reason="no memory estimate",
                )
            )
            continue
        verdict, reason, usable = _fit(
            estimate, device.memory_free_bytes, device.memory_total_bytes
        )
        if usable_backends and device.backend not in usable_backends:
            reason += " No compatible runtime uses this device."
        fits.append(
            DeviceFit(
                device_id=device.device_id,
                device_name=device.name,
                available_bytes=usable,
                verdict=verdict,
                reason=reason,
            )
        )

    eligible = [
        f for f, d in zip(fits, hardware.devices, strict=True)
        if not usable_backends or d.backend in usable_backends
    ]  # fmt: skip
    best: Verdict = max((f.verdict for f in eligible), key=_RANK.__getitem__, default="unknown")

    gpu_fits = [
        f for f, d in zip(fits, hardware.devices, strict=True) if d.backend != DeviceBackend.CPU
    ]
    cpu_fit = next(
        (f for f, d in zip(fits, hardware.devices, strict=True) if d.backend == DeviceBackend.CPU),
        None,
    )
    llamacpp_ok = any(c.runtime == "llamacpp" and c.compatible for c in compat)
    if (
        estimate is not None
        and llamacpp_ok
        and cpu_fit is not None
        and any(g.verdict == "does_not_fit" for g in gpu_fits)
    ):
        combined = sum(g.available_bytes or 0 for g in gpu_fits) + (cpu_fit.available_bytes or 0)
        if estimate.expected_bytes <= combined:
            messages.append(
                "Too large for GPU memory alone, but llama.cpp can offload some layers to the "
                "GPU and keep the rest in system RAM (slower than full GPU)."
            )
    if not any(c.compatible for c in compat):
        messages.append("No runtime is compatible with this model on this machine.")
        best = "does_not_fit" if best != "unknown" else best
    messages.append("Multi-device splits (tensor parallelism) are not evaluated yet.")

    return PreflightReport(
        model_id=model.id,
        estimate=estimate,
        devices=fits,
        runtimes=compat,
        verdict=best,
        messages=messages,
    )

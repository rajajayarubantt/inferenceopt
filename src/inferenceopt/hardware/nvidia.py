# SPDX-License-Identifier: Apache-2.0
"""NVIDIA GPU detection through NVML (read-only, no admin rights needed)."""

from __future__ import annotations

import contextlib
from typing import Any

from inferenceopt.hardware.base import DetectionResult
from inferenceopt.schemas import Device, DeviceBackend, HostInfo, Provenance

# (minimum compute capability, architecture name). Checked from newest to oldest.
_ARCHITECTURES = [
    ((10, 0), "Blackwell"),
    ((9, 0), "Hopper"),
    ((8, 9), "Ada Lovelace"),
    ((8, 0), "Ampere"),
    ((7, 5), "Turing"),
    ((7, 0), "Volta"),
    ((6, 0), "Pascal"),
    ((5, 0), "Maxwell"),
]

# (minimum compute capability, dtype with hardware support).
_DTYPES = [
    ((0, 0), "f32"),
    ((5, 3), "f16"),
    ((6, 1), "int8"),
    ((8, 0), "bf16"),
    ((8, 9), "fp8"),
    ((10, 0), "fp4"),
]


def architecture_for(cc: tuple[int, int]) -> str | None:
    return next((name for floor, name in _ARCHITECTURES if cc >= floor), None)


def dtypes_for(cc: tuple[int, int]) -> list[str]:
    return [dtype for floor, dtype in _DTYPES if cc >= floor]


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class NvidiaDetector:
    name = "nvidia"

    def __init__(self, nvml: Any | None = None) -> None:
        self._nvml = nvml  # injectable for tests

    def _load(self) -> Any | None:
        if self._nvml is not None:
            return self._nvml
        try:
            import pynvml
        except ImportError:
            return None
        return pynvml

    def detect(self, host: HostInfo) -> DetectionResult:
        result = DetectionResult()
        nvml = self._load()
        if nvml is None:
            result.notes.append("NVIDIA: nvidia-ml-py is not installed; GPU detection skipped.")
            return result
        try:
            nvml.nvmlInit()
        except Exception as exc:  # NVMLError_LibraryNotFound, DriverNotLoaded, ...
            result.notes.append(
                f"NVIDIA: no usable driver ({_text(exc) or type(exc).__name__}). "
                "If this machine has an NVIDIA GPU, install the NVIDIA driver."
            )
            return result
        try:
            result.runtime_versions["nvidia_driver"] = _text(nvml.nvmlSystemGetDriverVersion())
            try:
                cuda = int(nvml.nvmlSystemGetCudaDriverVersion_v2())
                result.runtime_versions["cuda_driver"] = f"{cuda // 1000}.{(cuda % 1000) // 10}"
            except Exception:
                result.notes.append("NVIDIA: could not read the CUDA driver version.")
            for index in range(int(nvml.nvmlDeviceGetCount())):
                result.devices.append(self._device(nvml, index, result.notes))
        finally:
            with contextlib.suppress(Exception):  # shutdown failures are irrelevant
                nvml.nvmlShutdown()
        return result

    def _device(self, nvml: Any, index: int, notes: list[str]) -> Device:
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        mem = nvml.nvmlDeviceGetMemoryInfo(handle)
        sources = {
            "architecture": Provenance.INFERRED,
            "supported_dtypes": Provenance.INFERRED,
        }
        try:
            major, minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)
            cc: tuple[int, int] | None = (int(major), int(minor))
        except Exception:
            cc = None
            notes.append(f"NVIDIA gpu{index}: compute capability unavailable.")
        return Device(
            device_id=f"gpu{index}",
            backend=DeviceBackend.CUDA,
            name=_text(nvml.nvmlDeviceGetName(handle)),
            architecture=architecture_for(cc) if cc else None,
            compute_capability=f"{cc[0]}.{cc[1]}" if cc else None,
            memory_total_bytes=int(mem.total),
            memory_free_bytes=int(mem.free),
            driver_version=_text(nvml.nvmlSystemGetDriverVersion()),
            supported_dtypes=dtypes_for(cc) if cc else [],
            field_sources=sources,
        )

# SPDX-License-Identifier: Apache-2.0
"""Apple Silicon GPU (Metal). Memory is unified with host RAM."""

from __future__ import annotations

import platform

from inferenceopt.hardware.base import DetectionResult
from inferenceopt.schemas import Device, DeviceBackend, HostInfo, Provenance


class AppleDetector:
    name = "apple"

    def detect(self, host: HostInfo) -> DetectionResult:
        result = DetectionResult()
        if host.os != "Darwin" or host.cpu_arch != "arm64":
            return result
        result.devices.append(
            Device(
                device_id="gpu0",
                backend=DeviceBackend.METAL,
                name=f"{host.cpu_model or 'Apple Silicon'} GPU",
                architecture="apple-silicon",
                memory_total_bytes=host.ram_total_bytes,
                memory_free_bytes=host.ram_available_bytes,
                supported_dtypes=["f32", "f16", "bf16", "int8"],
                field_sources={
                    "memory_total_bytes": Provenance.INFERRED,
                    "memory_free_bytes": Provenance.INFERRED,
                    "supported_dtypes": Provenance.INFERRED,
                },
            )
        )
        result.runtime_versions["macos"] = platform.mac_ver()[0] or "unknown"
        result.notes.append(
            "Apple Silicon: GPU memory is unified with system RAM, and macOS limits how much "
            "of it the GPU may use. The GPU memory shown is inferred from RAM; measure it."
        )
        return result

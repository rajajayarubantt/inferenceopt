# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from inferenceopt import hardware
from inferenceopt.hardware.apple import AppleDetector
from inferenceopt.hardware.base import DetectionResult
from inferenceopt.hardware.host import normalize_flags
from inferenceopt.hardware.nvidia import NvidiaDetector, architecture_for, dtypes_for
from inferenceopt.schemas import Device, DeviceBackend, HostInfo, Provenance

pytestmark = pytest.mark.unit

HOST = HostInfo(os="Linux", os_version="6", python_version="3.12", cpu_arch="x86_64")


class FakeNVML:
    def __init__(self, gpus: list[dict[str, Any]], fail_init: bool = False) -> None:
        self.gpus = gpus
        self.fail_init = fail_init
        self.shutdown_called = False

    def nvmlInit(self) -> None:
        if self.fail_init:
            raise RuntimeError("NVML Shared Library Not Found")

    def nvmlShutdown(self) -> None:
        self.shutdown_called = True

    def nvmlSystemGetDriverVersion(self) -> bytes:
        return b"550.54"

    def nvmlSystemGetCudaDriverVersion_v2(self) -> int:
        return 12040

    def nvmlDeviceGetCount(self) -> int:
        return len(self.gpus)

    def nvmlDeviceGetHandleByIndex(self, i: int) -> dict[str, Any]:
        return self.gpus[i]

    def nvmlDeviceGetName(self, h: dict[str, Any]) -> str:
        return str(h["name"])

    def nvmlDeviceGetMemoryInfo(self, h: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(total=h["total"], free=h["free"])

    def nvmlDeviceGetCudaComputeCapability(self, h: dict[str, Any]) -> tuple[int, int]:
        return h["cc"]  # type: ignore[no-any-return]


def test_nvidia_detects_devices() -> None:
    nvml = FakeNVML(
        [
            {"name": "RTX 4090", "total": 24 * 2**30, "free": 23 * 2**30, "cc": (8, 9)},
            {"name": "T4", "total": 16 * 2**30, "free": 15 * 2**30, "cc": (7, 5)},
        ]
    )
    result = NvidiaDetector(nvml).detect(HOST)
    assert nvml.shutdown_called
    assert result.runtime_versions == {"nvidia_driver": "550.54", "cuda_driver": "12.4"}
    ada, turing = result.devices
    assert (ada.device_id, ada.architecture, ada.compute_capability) == (
        "gpu0",
        "Ada Lovelace",
        "8.9",
    )
    assert "fp8" in ada.supported_dtypes and "bf16" in ada.supported_dtypes
    assert "bf16" not in turing.supported_dtypes
    assert ada.field_sources["supported_dtypes"] == Provenance.INFERRED
    assert ada.memory_free_bytes == 23 * 2**30


def test_nvidia_without_driver_is_a_note_not_an_error() -> None:
    result = NvidiaDetector(FakeNVML([], fail_init=True)).detect(HOST)
    assert result.devices == []
    assert "no usable driver" in result.notes[0]


@pytest.mark.parametrize(
    ("cc", "arch"),
    [
        ((12, 0), "Blackwell"),
        ((9, 0), "Hopper"),
        ((8, 6), "Ampere"),
        ((6, 1), "Pascal"),
        ((3, 0), None),
    ],
)
def test_architecture_names(cc: tuple[int, int], arch: str | None) -> None:
    assert architecture_for(cc) == arch


def test_dtypes_grow_with_capability() -> None:
    assert dtypes_for((7, 0)) == ["f32", "f16", "int8"]
    assert dtypes_for((10, 0))[-1] == "fp4"


def test_normalize_flags() -> None:
    raw = ["SSE4.1", "avx2", "asimddp", "pni", "AVX512VNNI", "mmx", "FEAT_I8MM"]
    assert normalize_flags(raw) == ["avx2", "avx512_vnni", "dotprod", "i8mm", "sse3", "sse4_1"]


def test_apple_only_on_apple_silicon() -> None:
    assert AppleDetector().detect(HOST).devices == []
    mac = HOST.model_copy(update={"os": "Darwin", "cpu_arch": "arm64", "ram_total_bytes": 2**34})
    (device,) = AppleDetector().detect(mac).devices
    assert device.backend == DeviceBackend.METAL
    assert device.field_sources["memory_total_bytes"] == Provenance.INFERRED


class Boom:
    name = "boom"

    def detect(self, host: HostInfo) -> DetectionResult:
        raise RuntimeError("kaput")


class FakeGPU:
    name = "fake"

    def detect(self, host: HostInfo) -> DetectionResult:
        return DetectionResult(
            devices=[Device(device_id="gpu0", backend=DeviceBackend.OTHER, name="X")],
            notes=["fake note"],
        )


def test_scan_survives_failing_detectors_and_dedupes_ids() -> None:
    profile = hardware.scan([Boom(), FakeGPU(), FakeGPU()])
    ids = [d.device_id for d in profile.devices]
    assert ids[0] == "cpu0"
    assert len(ids) == len(set(ids)) == 3
    assert any("boom: detector failed" in n for n in profile.notes)
    assert profile.host.ram_total_bytes and profile.host.ram_total_bytes > 0


def test_real_scan_runs_on_this_machine() -> None:
    profile = hardware.scan()
    assert profile.devices[0].device_id == "cpu0"
    assert profile.host.logical_cores

# SPDX-License-Identifier: Apache-2.0
"""Hardware discovery (Plan.md §6). Read-only; never needs administrator rights."""

from __future__ import annotations

import logging
import shutil
from importlib.metadata import entry_points

from inferenceopt.hardware.apple import AppleDetector
from inferenceopt.hardware.base import DetectionResult, Detector
from inferenceopt.hardware.host import detect_host
from inferenceopt.hardware.nvidia import NvidiaDetector
from inferenceopt.schemas import HardwareProfile

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "inferenceopt.hardware"


def builtin_detectors() -> list[Detector]:
    return [NvidiaDetector(), AppleDetector()]


def plugin_detectors() -> list[Detector]:
    found: list[Detector] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            found.append(ep.load()())
        except Exception as exc:
            log.warning("hardware plugin %s failed to load: %s", ep.name, exc)
    return found


def scan(detectors: list[Detector] | None = None) -> HardwareProfile:
    """Detect the host and all accelerators. Detector failures become notes, not errors."""
    host, host_result = detect_host()
    devices = list(host_result.devices)
    runtime_versions = dict(host_result.runtime_versions)
    notes = list(host_result.notes)

    for detector in (
        detectors if detectors is not None else builtin_detectors() + plugin_detectors()
    ):
        try:
            result = detector.detect(host)
        except Exception as exc:
            notes.append(f"{detector.name}: detector failed ({type(exc).__name__}: {exc}).")
            continue
        taken = {d.device_id for d in devices}
        for device in result.devices:
            if device.device_id in taken:  # keep ids unique across detectors
                suffix = sum(d.device_id.startswith(device.device_id) for d in devices)
                device = device.model_copy(update={"device_id": f"{device.device_id}_{suffix}"})
            devices.append(device)
            taken.add(device.device_id)
        runtime_versions.update(result.runtime_versions)
        notes.extend(result.notes)

    if shutil.which("rocm-smi") or shutil.which("amd-smi"):
        notes.append("AMD ROCm tools found, but AMD GPU detection is not implemented yet.")

    return HardwareProfile(
        host=host, devices=devices, runtime_versions=runtime_versions, notes=notes
    )


__all__ = ["DetectionResult", "Detector", "scan"]

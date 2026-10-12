# SPDX-License-Identifier: Apache-2.0
"""Detector plugin contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from inferenceopt.schemas import Device, HostInfo


@dataclass
class DetectionResult:
    devices: list[Device] = field(default_factory=list)
    runtime_versions: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class Detector(Protocol):
    """An accelerator detector. Register third-party ones under the entry-point group
    ``inferenceopt.hardware``. Detectors must be read-only, need no admin rights, and
    report what they could not detect in ``notes`` instead of guessing."""

    name: str

    def detect(self, host: HostInfo) -> DetectionResult: ...

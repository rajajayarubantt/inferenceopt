# SPDX-License-Identifier: Apache-2.0
"""Versioned public data contracts shared by the CLI, API, UI and storage."""

from inferenceopt.schemas.artifact import Artifact, DeploymentManifest
from inferenceopt.schemas.base import Provenance, Schema, Versioned
from inferenceopt.schemas.hardware import Device, DeviceBackend, HardwareProfile, HostInfo
from inferenceopt.schemas.job import (
    Constraints,
    DeploymentOptions,
    JobFile,
    Objective,
    OptimizationJob,
    Permissions,
    PlannerOptions,
    SearchSpec,
    Workload,
)
from inferenceopt.schemas.metrics import METRICS, MetricDef
from inferenceopt.schemas.model import ModelReference
from inferenceopt.schemas.planner import PlannerCall
from inferenceopt.schemas.runtime import RuntimeProfile, SupportLevel
from inferenceopt.schemas.trial import (
    BenchmarkRun,
    CandidateConfiguration,
    FailureRecord,
    QualityRun,
    Trial,
    TrialStatus,
)

__all__ = [
    "METRICS",
    "Artifact",
    "BenchmarkRun",
    "CandidateConfiguration",
    "Constraints",
    "DeploymentManifest",
    "DeploymentOptions",
    "Device",
    "DeviceBackend",
    "FailureRecord",
    "HardwareProfile",
    "HostInfo",
    "JobFile",
    "MetricDef",
    "ModelReference",
    "Objective",
    "OptimizationJob",
    "Permissions",
    "PlannerCall",
    "PlannerOptions",
    "Provenance",
    "QualityRun",
    "RuntimeProfile",
    "Schema",
    "SearchSpec",
    "SupportLevel",
    "Trial",
    "TrialStatus",
    "Versioned",
    "Workload",
]

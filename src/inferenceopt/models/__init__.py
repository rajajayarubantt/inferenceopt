# SPDX-License-Identifier: Apache-2.0
"""Model registry, local model inspection, memory estimation and preflight checks."""

from inferenceopt.models.estimator import EstimateError, estimate_memory
from inferenceopt.models.local import ModelInspectionError, inspect_path
from inferenceopt.models.preflight import KNOWN_RUNTIMES, check_runtime, preflight
from inferenceopt.models.registry import (
    ModelExistsError,
    ModelNotFoundError,
    ModelRegistry,
    curated_models,
)

__all__ = [
    "KNOWN_RUNTIMES",
    "EstimateError",
    "ModelExistsError",
    "ModelInspectionError",
    "ModelNotFoundError",
    "ModelRegistry",
    "check_runtime",
    "curated_models",
    "estimate_memory",
    "inspect_path",
    "preflight",
]

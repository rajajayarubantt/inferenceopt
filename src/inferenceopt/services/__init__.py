# SPDX-License-Identifier: Apache-2.0
"""Application services.

The CLI, the HTTP API (and therefore the web UI) and the Python API are thin wrappers
over these functions. Add new capabilities here first, then expose them everywhere.
"""

from inferenceopt.services.context import AppContext
from inferenceopt.services.inventory import (
    add_local_model,
    current_hardware,
    get_model,
    inspect_local_model,
    latest_hardware,
    list_models,
    remove_model,
    run_preflight,
    scan_hardware,
)
from inferenceopt.services.jobs import (
    JobValidation,
    ValidationIssue,
    validate_job_data,
    validate_job_file,
    validate_job_text,
)
from inferenceopt.services.system import (
    DoctorCheck,
    DoctorReport,
    VersionInfo,
    doctor,
    public_config,
    version_info,
)

__all__ = [
    "AppContext",
    "DoctorCheck",
    "DoctorReport",
    "JobValidation",
    "ValidationIssue",
    "VersionInfo",
    "add_local_model",
    "current_hardware",
    "doctor",
    "get_model",
    "inspect_local_model",
    "latest_hardware",
    "list_models",
    "public_config",
    "remove_model",
    "run_preflight",
    "scan_hardware",
    "validate_job_data",
    "validate_job_file",
    "validate_job_text",
    "version_info",
]

# SPDX-License-Identifier: Apache-2.0
"""Application services.

The CLI, the HTTP API (and therefore the web UI) and the Python API are thin wrappers
over these functions. Add new capabilities here first, then expose them everywhere.
"""

from inferenceopt.services.context import AppContext
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
    "doctor",
    "public_config",
    "validate_job_data",
    "validate_job_file",
    "validate_job_text",
    "version_info",
]

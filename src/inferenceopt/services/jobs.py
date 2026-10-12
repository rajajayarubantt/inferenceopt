# SPDX-License-Identifier: Apache-2.0
"""Job file validation. Execution arrives in Phase 4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from inferenceopt.schemas import JobFile, OptimizationJob


class ValidationIssue(BaseModel):
    location: str
    message: str


class JobValidation(BaseModel):
    ok: bool
    issues: list[ValidationIssue] = []
    job: OptimizationJob | None = None


def validate_job_data(data: Any) -> JobValidation:
    try:
        job = JobFile.model_validate(data).job
    except ValidationError as exc:
        issues = [
            ValidationIssue(
                location=".".join(str(p) for p in err["loc"]) or "(root)", message=err["msg"]
            )
            for err in exc.errors()
        ]
        return JobValidation(ok=False, issues=issues)
    return JobValidation(ok=True, job=job)


def validate_job_text(text: str) -> JobValidation:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return JobValidation(
            ok=False, issues=[ValidationIssue(location="(yaml)", message=str(exc))]
        )
    return validate_job_data(data)


def validate_job_file(path: Path) -> JobValidation:
    return validate_job_text(path.read_text(encoding="utf-8"))

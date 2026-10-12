# SPDX-License-Identifier: Apache-2.0
"""Job endpoints. Phase 1 provides validation only."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from inferenceopt import services

router = APIRouter(prefix="/jobs", tags=["jobs"])


class ValidateJobRequest(BaseModel):
    yaml: str


@router.post("/validate")
def validate_job(body: ValidateJobRequest) -> services.JobValidation:
    return services.validate_job_text(body.yaml)

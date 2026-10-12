# SPDX-License-Identifier: Apache-2.0
"""Health, version, doctor and configuration endpoints."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel

from inferenceopt import services
from inferenceopt.server.deps import Ctx

router = APIRouter(tags=["system"])


class Health(BaseModel):
    status: Literal["ok"] = "ok"


@router.get("/health")
def health() -> Health:
    """Liveness probe. Unauthenticated and reveals nothing."""
    return Health()


@router.get("/version")
def version() -> services.VersionInfo:
    return services.version_info()


@router.get("/doctor")
def doctor(ctx: Ctx) -> services.DoctorReport:
    return services.doctor(ctx)


@router.get("/config")
def config(ctx: Ctx) -> dict[str, Any]:
    """Current configuration. Secret values are never returned, only whether they are set."""
    return services.public_config(ctx)

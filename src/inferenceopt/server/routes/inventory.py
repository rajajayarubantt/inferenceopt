# SPDX-License-Identifier: Apache-2.0
"""Hardware, model registry and preflight endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from inferenceopt import services
from inferenceopt.models import ModelExistsError, ModelInspectionError, ModelNotFoundError
from inferenceopt.schemas import EstimateInputs, HardwareProfile, ModelReference, PreflightReport
from inferenceopt.server.deps import Ctx

router = APIRouter(tags=["inventory"])


# --- hardware -------------------------------------------------------------------------


@router.get("/hardware", responses={404: {"description": "No scan has been saved yet"}})
def get_hardware(ctx: Ctx) -> HardwareProfile:
    """The most recent saved hardware scan."""
    profile = services.latest_hardware(ctx)
    if profile is None:
        raise HTTPException(status_code=404, detail="no hardware scan yet")
    return profile


@router.post("/hardware/scan")
def scan_hardware(ctx: Ctx) -> HardwareProfile:
    """Scan the hardware now (read-only) and save the result."""
    return services.scan_hardware(ctx)


# --- models ---------------------------------------------------------------------------


class LocalModelRequest(BaseModel):
    path: str = Field(min_length=1, description="Absolute path to a .gguf file or HF directory")
    id: str | None = Field(default=None, description="Registry id (default: derived from name)")


def _local_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise HTTPException(status_code=422, detail="path must be absolute")
    return path


@router.get("/models")
def list_models(ctx: Ctx) -> list[ModelReference]:
    return services.list_models(ctx)


@router.post("/models/inspect")
def inspect_model(body: LocalModelRequest) -> ModelReference:
    """Read a local model's metadata (headers only) without registering it."""
    try:
        return services.inspect_local_model(_local_path(body.path))
    except ModelInspectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/models", status_code=201)
def add_model(body: LocalModelRequest, ctx: Ctx) -> ModelReference:
    """Register a local model (reads headers only; nothing is copied or downloaded)."""
    try:
        return services.add_local_model(ctx, _local_path(body.path), body.id)
    except ModelInspectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/models/{model_id:path}")
def get_model(model_id: str, ctx: Ctx) -> ModelReference:
    try:
        return services.get_model(ctx, model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"model {model_id!r} not found") from exc


@router.delete("/models/{model_id:path}", status_code=204)
def remove_model(model_id: str, ctx: Ctx) -> Response:
    """Remove a user-added model from the registry. Files on disk are never touched."""
    try:
        services.remove_model(ctx, model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"model {model_id!r} not found") from exc
    except ModelExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=204)


# --- preflight ------------------------------------------------------------------------


class PreflightRequest(BaseModel):
    model_id: str
    inputs: EstimateInputs = Field(default_factory=EstimateInputs)
    runtimes: list[str] | None = None
    rescan: bool = False


@router.post("/preflight")
def preflight(body: PreflightRequest, ctx: Ctx) -> PreflightReport:
    """Estimate memory and check whether the model fits each device. Never loads weights."""
    try:
        return services.run_preflight(
            ctx, body.model_id, body.inputs, body.runtimes, rescan=body.rescan
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"model {body.model_id!r} not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

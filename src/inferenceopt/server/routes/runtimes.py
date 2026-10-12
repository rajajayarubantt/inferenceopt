# SPDX-License-Identifier: Apache-2.0
"""Runtime endpoints: status, settings, launch previews, smoke tests, llama.cpp installs.

Anything that starts a process or downloads files requires ``confirm: true`` in the
request body, enforced here on the server rather than only in the UI.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from inferenceopt.models import ModelNotFoundError
from inferenceopt.runtimes.base import RuntimeSettings
from inferenceopt.runtimes.llamacpp.install import InstallError
from inferenceopt.runtimes.registry import UnknownRuntimeError
from inferenceopt.server.deps import Ctx
from inferenceopt.services import runtimes as svc

router = APIRouter(prefix="/runtimes", tags=["runtimes"])


def _not_found(name: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"unknown runtime {name!r}")


def _require_confirm(confirm: bool, what: str) -> None:
    if not confirm:
        raise HTTPException(status_code=428, detail=f"{what} requires confirm: true")


@router.get("")
def list_runtimes(ctx: Ctx) -> list[svc.RuntimeStatus]:
    return svc.list_runtimes(ctx)


@router.get("/{name}")
def get_runtime(name: str, ctx: Ctx) -> svc.RuntimeStatus:
    try:
        return svc.runtime_status(ctx, name)
    except UnknownRuntimeError as exc:
        raise _not_found(name) from exc


@router.put("/{name}/settings")
def save_runtime_settings(name: str, body: RuntimeSettings, ctx: Ctx) -> svc.RuntimeStatus:
    """Where to find the runtime. Executables must have the runtime's own binary name."""
    try:
        svc.save_settings(ctx, name, body)
        return svc.runtime_status(ctx, name)
    except UnknownRuntimeError as exc:
        raise _not_found(name) from exc
    except svc.InvalidSettingsError as exc:
        raise HTTPException(status_code=422, detail=exc.errors) from exc


class LaunchRequest(BaseModel):
    model_id: str
    config: dict[str, Any] = Field(default_factory=dict)


@router.post("/{name}/launch-preview")
def launch_preview(name: str, body: LaunchRequest, ctx: Ctx) -> svc.LaunchPreview:
    """Validate a configuration and show the exact command. Starts nothing."""
    try:
        return svc.launch_preview(ctx, name, body.model_id, body.config)
    except UnknownRuntimeError as exc:
        raise _not_found(name) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"model {body.model_id!r} not found") from exc


class SmokeTestRequest(LaunchRequest):
    max_tokens: int = Field(default=24, ge=1, le=256)
    confirm: bool = False


@router.post("/{name}/smoke-test")
def smoke_test(name: str, body: SmokeTestRequest, ctx: Ctx) -> svc.SmokeTestResult:
    """Start the runtime, generate a few tokens, stop it. Not a benchmark."""
    _require_confirm(body.confirm, "starting a runtime")
    try:
        return svc.smoke_test(ctx, name, body.model_id, body.config, max_tokens=body.max_tokens)
    except UnknownRuntimeError as exc:
        raise _not_found(name) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"model {body.model_id!r} not found") from exc


class InstallPlanResponse(BaseModel):
    tag: str
    variant: str
    assets: list[str]
    total_bytes: int


Channel = Literal["stable", "nightly"]
Variant = Literal["auto", "cpu", "cuda", "vulkan"]


@router.get("/llamacpp/install-plan")
def llamacpp_install_plan(
    ctx: Ctx, channel: Channel = "stable", variant: Variant = "auto"
) -> InstallPlanResponse:
    """What an install would download (for the consent dialog). Downloads nothing."""
    try:
        plan = svc.plan_llamacpp_install(ctx, channel, variant)
    except InstallError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return InstallPlanResponse(
        tag=plan.tag,
        variant=plan.variant,
        assets=[a.name for a in plan.assets],
        total_bytes=plan.total_bytes,
    )


class InstallRequest(BaseModel):
    channel: Channel = "stable"
    variant: Variant = "auto"
    confirm: bool = False


@router.post("/llamacpp/install", status_code=202)
def install_llamacpp(body: InstallRequest, ctx: Ctx) -> svc.InstallTask:
    """Download an official llama.cpp build (SHA-256 verified) in the background."""
    _require_confirm(body.confirm, "downloading llama.cpp")
    return svc.start_llamacpp_install(ctx, body.channel, body.variant)


@router.get("/llamacpp/install/{task_id}")
def llamacpp_install_status(task_id: str) -> svc.InstallTask:
    task = svc.get_install_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="no such install task")
    return task

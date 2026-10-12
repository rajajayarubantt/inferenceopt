# SPDX-License-Identifier: Apache-2.0
"""Runtime services: discovery, settings, launch previews, smoke tests, installs."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from inferenceopt.config import data_dir
from inferenceopt.runtimes.base import (
    Installation,
    LaunchSpec,
    ParamSpec,
    RuntimeSettings,
)
from inferenceopt.runtimes.llamacpp import install as llama_install
from inferenceopt.runtimes.openai_client import stream_chat
from inferenceopt.runtimes.process import LaunchError, free_port, start_server
from inferenceopt.runtimes.registry import adapters, get_adapter
from inferenceopt.schemas import FailureRecord, SupportLevel
from inferenceopt.schemas.base import utcnow
from inferenceopt.services.context import AppContext
from inferenceopt.services.inventory import current_hardware, get_model
from inferenceopt.storage.orm import RuntimeSettingsRow

# --- status -----------------------------------------------------------------------------


class RuntimeStatus(BaseModel):
    name: str
    display_name: str
    support_level: SupportLevel
    platforms: list[str]
    weight_formats: list[str]
    homepage: str
    settings: RuntimeSettings
    installation: Installation | None
    notes: list[str]
    parameters: list[ParamSpec]


def get_settings(ctx: AppContext, name: str) -> RuntimeSettings:
    get_adapter(name)  # raises UnknownRuntimeError
    with ctx.db.session() as s:
        row = s.get(RuntimeSettingsRow, name)
        return RuntimeSettings.model_validate(row.payload) if row else RuntimeSettings()


class InvalidSettingsError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def save_settings(ctx: AppContext, name: str, settings: RuntimeSettings) -> RuntimeSettings:
    errors = get_adapter(name).validate_settings(settings)
    if errors:
        raise InvalidSettingsError(errors)
    with ctx.db.session() as s:
        row = s.get(RuntimeSettingsRow, name)
        payload = settings.model_dump(mode="json")
        if row is None:
            s.add(RuntimeSettingsRow(name=name, updated_at=utcnow(), payload=payload))
        else:
            row.payload, row.updated_at = payload, utcnow()
    return settings


def runtime_status(ctx: AppContext, name: str) -> RuntimeStatus:
    adapter = get_adapter(name)
    settings = get_settings(ctx, name)
    install, notes = adapter.detect(settings)
    return RuntimeStatus(
        name=adapter.name,
        display_name=adapter.display_name,
        support_level=adapter.support_level,
        platforms=list(adapter.platforms),
        weight_formats=list(adapter.weight_formats),
        homepage=adapter.homepage,
        settings=settings,
        installation=install,
        notes=notes,
        parameters=adapter.parameters(install),
    )


def list_runtimes(ctx: AppContext) -> list[RuntimeStatus]:
    return [runtime_status(ctx, name) for name in adapters()]


# --- launch preview ---------------------------------------------------------------------


class LaunchPreview(BaseModel):
    runtime: str
    model_id: str
    config: dict[str, Any]
    """Defaults overlaid with the requested configuration."""
    errors: list[str]
    spec: LaunchSpec | None


def _prepare(
    ctx: AppContext, name: str, model_id: str, config: dict[str, Any], port: int
) -> LaunchPreview:
    adapter = get_adapter(name)
    model = get_model(ctx, model_id)
    settings = get_settings(ctx, name)
    install, notes = adapter.detect(settings)
    errors = adapter.validate(config, model, current_hardware(ctx), install)
    if install is None:
        errors.insert(0, f"{adapter.display_name} is not installed: {' '.join(notes)}")
    spec = None
    if install is not None and not errors:
        spec = adapter.build_launch(install, model, config, port=port)
        if settings.extra_env and install.mode == "native":
            spec = spec.model_copy(update={"env": {**spec.env, **settings.extra_env}})
    return LaunchPreview(
        runtime=name,
        model_id=model_id,
        config=adapter.resolve(config, install),
        errors=errors,
        spec=spec,
    )


def launch_preview(
    ctx: AppContext, name: str, model_id: str, config: dict[str, Any]
) -> LaunchPreview:
    """Validate and show exactly what would run. Never starts anything."""
    return _prepare(ctx, name, model_id, config, port=8080)


# --- smoke test -------------------------------------------------------------------------

SMOKE_PROMPT = "Reply with one short sentence: what is 2 + 2?"


class SmokeTestResult(BaseModel):
    ok: bool
    runtime: str
    model_id: str
    runtime_version: str | None = None
    errors: list[str] = Field(default_factory=list)
    """Validation errors; nothing was started if these are present."""
    spec: LaunchSpec | None = None
    load_time_s: float | None = None
    ttft_s: float | None = None
    e2e_s: float | None = None
    completion_tokens: int | None = None
    token_count_source: str | None = None
    decode_tokens_per_s: float | None = None
    output_excerpt: str | None = None
    peak_host_memory_bytes: int | None = None
    failure: FailureRecord | None = None
    log_path: str | None = None
    log_tail: str | None = None
    started_at: datetime = Field(default_factory=utcnow)


def smoke_test(
    ctx: AppContext,
    name: str,
    model_id: str,
    config: dict[str, Any] | None = None,
    *,
    max_tokens: int = 24,
) -> SmokeTestResult:
    """Start the runtime, wait for health, run one short completion, stop it.

    Single-request smoke check only (Plan.md §7.3 stage 3); not a benchmark.
    """
    config = config or {}
    preview = _prepare(ctx, name, model_id, config, port=free_port())
    install, _ = get_adapter(name).detect(get_settings(ctx, name))
    result = SmokeTestResult(
        ok=False,
        runtime=name,
        model_id=model_id,
        runtime_version=install.version if install else None,
        errors=preview.errors,
        spec=preview.spec,
    )
    if preview.errors or preview.spec is None:
        return result

    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = data_dir() / "logs" / "smoke" / f"{stamp}-{name}-{uuid.uuid4().hex[:6]}.log"
    result.log_path = str(log_path)
    try:
        server = start_server(preview.spec, log_path)
    except LaunchError as exc:
        result.failure = exc.failure
        result.log_tail = exc.failure.log_excerpt
        return result

    try:
        result.load_time_s = server.load_time_s

        async def run() -> Any:
            async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
                return await stream_chat(
                    client,
                    preview.spec.api_base,  # type: ignore[union-attr]
                    preview.spec.served_model_name,  # type: ignore[union-attr]
                    [{"role": "user", "content": SMOKE_PROMPT}],
                    max_tokens=max_tokens,
                )

        stream = asyncio.run(run())
        result.ttft_s = stream.ttft_s
        result.e2e_s = stream.e2e_s
        result.completion_tokens = stream.completion_tokens
        result.token_count_source = stream.token_count_source
        result.output_excerpt = stream.text[:300]
        if stream.ttft_s is not None and stream.completion_tokens and stream.completion_tokens > 1:
            decode_time = stream.e2e_s - stream.ttft_s
            if decode_time > 0:
                result.decode_tokens_per_s = (stream.completion_tokens - 1) / decode_time
        if not stream.ok or not stream.text.strip():
            result.failure = FailureRecord(
                stage="generate",
                kind="invalid_output",
                message=stream.error or "the server returned no text",
            )
        else:
            result.ok = True
    finally:
        result.peak_host_memory_bytes = server.peak_host_memory()
        server.stop()
        result.log_tail = server.log_tail(40)
    return result


# --- llama.cpp installs ------------------------------------------------------------------


class InstallTask(BaseModel):
    id: str
    status: Literal["planning", "downloading", "done", "failed"] = "planning"
    tag: str | None = None
    variant: str | None = None
    assets: list[str] = Field(default_factory=list)
    total_bytes: int | None = None
    downloaded_bytes: int = 0
    current_asset: str | None = None
    executable: str | None = None
    error: str | None = None


_tasks: dict[str, InstallTask] = {}
_tasks_lock = threading.Lock()


def plan_llamacpp_install(
    ctx: AppContext, channel: str = "stable", variant: llama_install.Variant = "auto"
) -> llama_install.InstallPlan:
    hardware = current_hardware(ctx)
    tag, assets = llama_install.resolve_release(channel)
    return llama_install.plan_install(
        tag, assets, variant=variant, cuda_driver=hardware.runtime_versions.get("cuda_driver")
    )


def install_llamacpp(
    plan: llama_install.InstallPlan,
    progress: Callable[[str, int, int | None], None] | None = None,
) -> Path:
    return llama_install.install(plan, progress)


def start_llamacpp_install(
    ctx: AppContext, channel: str = "stable", variant: llama_install.Variant = "auto"
) -> InstallTask:
    """Run an install in a background thread; poll with ``get_install_task``."""
    task = InstallTask(id=uuid.uuid4().hex[:12])
    with _tasks_lock:
        _tasks[task.id] = task

    def work() -> None:
        try:
            plan = plan_llamacpp_install(ctx, channel, variant)
            task.tag, task.variant = plan.tag, plan.variant
            task.assets = [a.name for a in plan.assets]
            task.total_bytes = plan.total_bytes
            task.status = "downloading"
            finished: dict[str, int] = {}

            def progress(name: str, done: int, total: int | None) -> None:
                task.current_asset = name
                finished[name] = done
                task.downloaded_bytes = sum(finished.values())

            task.executable = str(install_llamacpp(plan, progress))
            task.status = "done"
        except Exception as exc:
            task.error = f"{type(exc).__name__}: {exc}"
            task.status = "failed"

    threading.Thread(target=work, name=f"install-{task.id}", daemon=True).start()
    return task


def get_install_task(task_id: str) -> InstallTask | None:
    with _tasks_lock:
        return _tasks.get(task_id)

# SPDX-License-Identifier: Apache-2.0
"""Version, environment and health checks (``doctor``)."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from typing import Any, Literal

from pydantic import BaseModel, SecretStr

from inferenceopt import __version__
from inferenceopt.services.context import AppContext
from inferenceopt.ui_assets import ui_index

CheckStatus = Literal["ok", "info", "warn", "fail"]


class VersionInfo(BaseModel):
    inferenceopt: str
    python: str
    platform: str


class DoctorCheck(BaseModel):
    name: str
    status: CheckStatus
    detail: str
    hint: str | None = None


class DoctorReport(BaseModel):
    version: VersionInfo
    checks: list[DoctorCheck]

    @property
    def ok(self) -> bool:
        return all(c.status != "fail" for c in self.checks)


def version_info() -> VersionInfo:
    return VersionInfo(
        inferenceopt=__version__,
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
    )


def _optional_package(name: str, extra: str, purpose: str) -> DoctorCheck:
    if importlib.util.find_spec(name) is not None:
        return DoctorCheck(name=f"extra: {extra}", status="ok", detail=f"{name} installed")
    return DoctorCheck(
        name=f"extra: {extra}",
        status="info",
        detail=f"{name} not installed ({purpose})",
        hint=f'pip install "inferenceopt[{extra}]"',
    )


def _executable(name: str, label: str, hint: str) -> DoctorCheck:
    path = shutil.which(name)
    if path:
        return DoctorCheck(name=label, status="ok", detail=path)
    return DoctorCheck(name=label, status="info", detail=f"{name} not found on PATH", hint=hint)


def doctor(ctx: AppContext) -> DoctorReport:
    checks: list[DoctorCheck] = []

    checks.append(
        DoctorCheck(
            name="python",
            status="ok" if sys.version_info >= (3, 11) else "fail",
            detail=platform.python_version(),
            hint=None if sys.version_info >= (3, 11) else "InferenceOpt requires Python 3.11+",
        )
    )

    checks.append(
        DoctorCheck(
            name="config",
            status="ok",
            detail=str(ctx.config_path)
            if ctx.config_path.is_file()
            else f"no file at {ctx.config_path}; using defaults",
        )
    )

    current, head = ctx.db.current_revision(), ctx.db.head_revision()
    checks.append(
        DoctorCheck(
            name="database",
            status="ok" if current == head else "fail",
            detail=f"{ctx.db.path} (schema {current}, latest {head})",
        )
    )

    index = ui_index()
    checks.append(
        DoctorCheck(
            name="web ui",
            status="ok" if index else "warn",
            detail=str(index.parent) if index else "UI assets not found in this install",
            hint=None if index else "From a source checkout: cd ui && pnpm install && pnpm build",
        )
    )

    checks.append(_optional_package("litellm", "llm", "LLM planner"))
    checks.append(_optional_package("optuna", "optimize", "Bayesian search"))
    checks.append(
        _executable(
            "llama-server",
            "runtime: llama.cpp",
            "Install llama.cpp or set its path in Settings (adapter arrives in Phase 3)",
        )
    )
    checks.append(
        _executable("docker", "docker", "Optional: used for isolated workers and deployment")
    )

    return DoctorReport(version=version_info(), checks=checks)


def public_config(ctx: AppContext) -> dict[str, Any]:
    """Configuration safe to show in the UI: secrets become ``{"is_set": bool}``."""

    def walk(value: Any) -> Any:
        if isinstance(value, SecretStr):
            return {"is_set": bool(value.get_secret_value())}
        if isinstance(value, BaseModel):
            return {k: walk(getattr(value, k)) for k in type(value).model_fields}
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if hasattr(value, "__fspath__"):
            return str(value)
        return value

    data: dict[str, Any] = walk(ctx.config)
    # Secret fields that are None are still secrets; report them as unset, not null.
    data["server"]["token"] = {"is_set": ctx.config.server.token is not None}
    data["llm"]["api_key"] = {"is_set": ctx.config.llm.api_key is not None}
    return data

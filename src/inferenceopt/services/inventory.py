# SPDX-License-Identifier: Apache-2.0
"""Hardware inventory, model registry and preflight services (Phase 2)."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from inferenceopt import hardware
from inferenceopt.models import (
    KNOWN_RUNTIMES,
    ModelRegistry,
    inspect_path,
)
from inferenceopt.models import (
    preflight as run_preflight_check,
)
from inferenceopt.schemas import (
    EstimateInputs,
    HardwareProfile,
    ModelReference,
    PreflightReport,
)
from inferenceopt.services.context import AppContext
from inferenceopt.storage.orm import HardwareProfileRow

# --- hardware -------------------------------------------------------------------------


def scan_hardware(ctx: AppContext, *, save: bool = True) -> HardwareProfile:
    profile = hardware.scan()
    if save:
        with ctx.db.session() as s:
            s.add(
                HardwareProfileRow(
                    id=profile.id,
                    captured_at=profile.captured_at,
                    payload=profile.model_dump(mode="json"),
                )
            )
    return profile


def latest_hardware(ctx: AppContext) -> HardwareProfile | None:
    with ctx.db.session() as s:
        row = s.scalars(
            select(HardwareProfileRow).order_by(HardwareProfileRow.captured_at.desc()).limit(1)
        ).first()
        return HardwareProfile.model_validate(row.payload) if row else None


def current_hardware(ctx: AppContext) -> HardwareProfile:
    """The latest saved scan, or a fresh (saved) scan if there is none."""
    return latest_hardware(ctx) or scan_hardware(ctx)


# --- models ---------------------------------------------------------------------------


def list_models(ctx: AppContext) -> list[ModelReference]:
    return ModelRegistry(ctx.db).list()


def get_model(ctx: AppContext, model_id: str) -> ModelReference:
    return ModelRegistry(ctx.db).get(model_id)


def inspect_local_model(path: Path) -> ModelReference:
    """Read a local model's metadata without registering it."""
    return inspect_path(path)


def add_local_model(ctx: AppContext, path: Path, model_id: str | None = None) -> ModelReference:
    return ModelRegistry(ctx.db).add_local(path, model_id)


def remove_model(ctx: AppContext, model_id: str) -> None:
    ModelRegistry(ctx.db).remove(model_id)


# --- preflight ------------------------------------------------------------------------


def run_preflight(
    ctx: AppContext,
    model_id: str,
    inputs: EstimateInputs | None = None,
    runtimes: list[str] | None = None,
    *,
    rescan: bool = False,
) -> PreflightReport:
    model = get_model(ctx, model_id)
    profile = scan_hardware(ctx) if rescan else current_hardware(ctx)
    unknown = sorted(set(runtimes or []) - set(KNOWN_RUNTIMES))
    if unknown:
        raise ValueError(f"unknown runtime(s): {', '.join(unknown)}")
    return run_preflight_check(model, profile, inputs or EstimateInputs(), runtimes)

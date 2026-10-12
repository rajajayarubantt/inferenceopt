# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from inferenceopt import services
from inferenceopt.models import ModelExistsError, ModelNotFoundError, curated_models
from inferenceopt.services import AppContext
from tests.helpers import llama_gguf

pytestmark = pytest.mark.unit


def test_curated_registry_is_valid() -> None:
    models = curated_models()
    assert len(models) >= 5
    assert len({m.id for m in models}) == len(models)
    for m in models:
        assert m.architecture is not None
        assert m.origin == "curated"
        assert m.license


def test_add_get_list_remove(ctx: AppContext, tmp_path: Path) -> None:
    path = llama_gguf(tmp_path / "tiny.gguf")
    added = services.add_local_model(ctx, path)
    assert services.get_model(ctx, added.id) == added
    assert added.id in {m.id for m in services.list_models(ctx)}

    with pytest.raises(ModelExistsError):
        services.add_local_model(ctx, path)
    services.add_local_model(ctx, path, "local:second-name")

    services.remove_model(ctx, added.id)
    with pytest.raises(ModelNotFoundError):
        services.get_model(ctx, added.id)


def test_curated_models_are_protected(ctx: AppContext) -> None:
    curated_id = curated_models()[0].id
    with pytest.raises(ModelExistsError):
        services.remove_model(ctx, curated_id)


def test_hardware_scan_is_saved(ctx: AppContext) -> None:
    assert services.latest_hardware(ctx) is None
    profile = services.scan_hardware(ctx)
    assert services.latest_hardware(ctx) == profile
    assert services.current_hardware(ctx) == profile


def test_preflight_service_with_local_model(ctx: AppContext, tmp_path: Path) -> None:
    model = services.add_local_model(ctx, llama_gguf(tmp_path / "tiny.gguf"))
    report = services.run_preflight(ctx, model.id, runtimes=["llamacpp"])
    assert report.estimate is not None
    assert report.verdict in {"fits", "tight"}
    with pytest.raises(ValueError, match="unknown runtime"):
        services.run_preflight(ctx, model.id, runtimes=["nope"])

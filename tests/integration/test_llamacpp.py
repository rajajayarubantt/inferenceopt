# SPDX-License-Identifier: Apache-2.0
"""Real llama.cpp runs. Skipped unless a llama-server is available.

Configure with environment variables:
    INFERENCEOPT_TEST_LLAMA_MODE    native (default) or docker
    INFERENCEOPT_TEST_LLAMA_SERVER  path to llama-server (native; default: managed install/PATH)
    INFERENCEOPT_TEST_LLAMA_IMAGE   image (docker), e.g. .../llama.cpp:server-cuda-b11429
    INFERENCEOPT_TEST_GGUF          model file (default: scripts/fetch_test_model.py location)
    INFERENCEOPT_TEST_GPU_LAYERS    n_gpu_layers to use (default 0 = CPU)

Run: uv run pytest -m integration_llamacpp
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from inferenceopt import services
from inferenceopt.runtimes.base import RuntimeSettings
from inferenceopt.runtimes.llamacpp.adapter import LlamaCppAdapter
from inferenceopt.services import AppContext
from inferenceopt.services import runtimes as svc

pytestmark = pytest.mark.integration_llamacpp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from fetch_test_model import DEFAULT_DIR, FILENAME  # noqa: E402

MODE = os.environ.get("INFERENCEOPT_TEST_LLAMA_MODE", "native")
GPU_LAYERS = int(os.environ.get("INFERENCEOPT_TEST_GPU_LAYERS", "0"))


def _settings() -> RuntimeSettings:
    if MODE == "docker":
        return RuntimeSettings(
            mode="docker",
            image=os.environ.get(
                "INFERENCEOPT_TEST_LLAMA_IMAGE", "ghcr.io/ggml-org/llama.cpp:server-cuda"
            ),
        )
    return RuntimeSettings(executable=os.environ.get("INFERENCEOPT_TEST_LLAMA_SERVER"))


@pytest.fixture
def llama_ctx(ctx: AppContext) -> Iterator[tuple[AppContext, str]]:
    gguf = Path(os.environ.get("INFERENCEOPT_TEST_GGUF", str(DEFAULT_DIR / FILENAME)))
    if not gguf.is_file():
        pytest.skip(f"test model not found ({gguf}); run scripts/fetch_test_model.py")
    settings = _settings()
    install, notes = LlamaCppAdapter().detect(settings)
    if install is None or install.features.get("blocked_by_os_policy"):
        pytest.skip(f"llama-server unavailable: {' '.join(notes)}")
    svc.save_settings(ctx, "llamacpp", settings)
    model = services.add_local_model(ctx, gguf)
    yield ctx, model.id


def test_detects_real_build(llama_ctx: tuple[AppContext, str]) -> None:
    ctx, _ = llama_ctx
    status = svc.runtime_status(ctx, "llamacpp")
    assert status.installation is not None
    assert status.installation.version and status.installation.version.startswith("b")
    assert status.installation.features["help_available"]


def test_smoke_test_generates_text(llama_ctx: tuple[AppContext, str]) -> None:
    ctx, model_id = llama_ctx
    result = svc.smoke_test(
        ctx, "llamacpp", model_id, {"ctx_size": 2048, "n_gpu_layers": GPU_LAYERS}, max_tokens=16
    )
    assert result.ok, (result.errors, result.failure, result.log_tail)
    assert result.completion_tokens and result.completion_tokens > 0
    assert result.output_excerpt and result.output_excerpt.strip()
    assert result.load_time_s is not None and result.ttft_s is not None


def test_bad_model_file_is_a_recorded_failure(
    llama_ctx: tuple[AppContext, str], tmp_path: Path
) -> None:
    ctx, _ = llama_ctx
    broken = tmp_path / "broken.gguf"
    from tests.helpers import llama_gguf

    llama_gguf(broken)  # valid header, but no tensor data: the runtime must fail to load it
    model = services.add_local_model(ctx, broken, "local:broken")
    result = svc.smoke_test(ctx, "llamacpp", model.id, {"ctx_size": 512, "n_gpu_layers": 0})
    assert not result.ok
    assert result.failure is not None
    assert result.failure.kind in {"config", "crash"}

# SPDX-License-Identifier: Apache-2.0
"""Runtime services end to end with a fake adapter that launches the fake OpenAI server."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from inferenceopt import services
from inferenceopt.runtimes.base import (
    Installation,
    LaunchSpec,
    ParamSpec,
    RuntimeAdapter,
    RuntimeSettings,
)
from inferenceopt.schemas import ModelReference, SupportLevel
from inferenceopt.services import AppContext
from inferenceopt.services import runtimes as svc
from tests.helpers import llama_gguf

pytestmark = pytest.mark.unit

FAKE = Path(__file__).resolve().parents[1] / "fakes" / "fake_openai_server.py"


class FakeAdapter(RuntimeAdapter):
    name = "fake"
    display_name = "Fake runtime"
    support_level = SupportLevel.EXPERIMENTAL
    platforms = ("Windows", "Linux", "Darwin")
    weight_formats = ("gguf",)
    homepage = "https://example.invalid"
    executable_names = ("python", "python.exe")

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode

    def detect(self, settings: RuntimeSettings) -> tuple[Installation | None, list[str]]:
        return Installation(executable=sys.executable, version="1.0", source="path"), []

    def parameters(self, install: Installation | None) -> list[ParamSpec]:
        return [ParamSpec(name="tokens", type="int", minimum=1, maximum=64, default=6,
                          description="Tokens the fake server emits.")]  # fmt: skip

    def build_launch(
        self, install: Installation, model: ModelReference, config: dict[str, Any], *,
        port: int, host: str = "127.0.0.1",
    ) -> LaunchSpec:  # fmt: skip
        cfg = self.resolve(config, install)
        return LaunchSpec(
            argv=[install.executable, str(FAKE), "--port", str(port), "--mode", self.mode,
                  "--tokens", str(cfg["tokens"]), "--token-delay", "0.01"],
            port=port, api_base=f"http://127.0.0.1:{port}/v1", served_model_name="fake-model",
            startup_timeout_s=20,
        )  # fmt: skip


@pytest.fixture
def model_id(ctx: AppContext, tmp_path: Path) -> str:
    return services.add_local_model(ctx, llama_gguf(tmp_path / "tiny.gguf")).id


def use(monkeypatch: pytest.MonkeyPatch, adapter: RuntimeAdapter) -> None:
    monkeypatch.setattr(svc, "get_adapter", lambda name: adapter)


def test_smoke_test_ok(ctx: AppContext, model_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeAdapter())
    result = svc.smoke_test(ctx, "fake", model_id, {"tokens": 5})
    assert result.ok, (result.failure, result.errors)
    assert result.completion_tokens == 5
    assert result.output_excerpt and result.output_excerpt.startswith("tok0")
    assert result.load_time_s is not None and result.ttft_s is not None
    assert result.decode_tokens_per_s and result.decode_tokens_per_s > 0
    assert result.log_path and Path(result.log_path).is_file()
    if sys.platform == "win32":
        assert result.peak_host_memory_bytes and result.peak_host_memory_bytes > 0


def test_smoke_test_reports_oom(
    ctx: AppContext, model_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, FakeAdapter(mode="oom"))
    result = svc.smoke_test(ctx, "fake", model_id)
    assert not result.ok
    assert result.failure is not None and result.failure.kind == "oom"
    assert "out of memory" in (result.log_tail or "")


def test_invalid_config_starts_nothing(
    ctx: AppContext, model_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, FakeAdapter())
    result = svc.smoke_test(ctx, "fake", model_id, {"tokens": 999})
    assert not result.ok and result.spec is None and result.log_path is None
    assert any("<= 64" in e for e in result.errors)


def test_launch_preview(ctx: AppContext, model_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeAdapter())
    preview = svc.launch_preview(ctx, "fake", model_id, {})
    assert preview.errors == [] and preview.config == {"tokens": 6}
    assert preview.spec is not None and preview.spec.argv[0] == sys.executable


def test_settings_round_trip_and_rejection(ctx: AppContext, tmp_path: Path) -> None:
    assert svc.get_settings(ctx, "vllm") == RuntimeSettings()
    saved = svc.save_settings(
        ctx, "vllm", RuntimeSettings(mode="docker", image="vllm/vllm-openai:v1")
    )
    assert svc.get_settings(ctx, "vllm") == saved
    with pytest.raises(svc.InvalidSettingsError):
        svc.save_settings(ctx, "llamacpp", RuntimeSettings(executable=sys.executable))


def test_list_runtimes_reports_all(ctx: AppContext) -> None:
    names = {r.name for r in svc.list_runtimes(ctx)}
    assert names == {"llamacpp", "vllm", "tensorrt_llm"}

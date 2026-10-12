# SPDX-License-Identifier: Apache-2.0
"""Process supervisor + streaming client against the fake OpenAI server (no GPU needed)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

from inferenceopt.runtimes.base import LaunchSpec
from inferenceopt.runtimes.openai_client import StreamResult, list_models, stream_chat
from inferenceopt.runtimes.process import LaunchError, classify_failure, free_port, start_server

pytestmark = pytest.mark.unit

FAKE = Path(__file__).resolve().parents[1] / "fakes" / "fake_openai_server.py"


def spec(*extra: str, timeout: float = 20) -> LaunchSpec:
    port = free_port()
    return LaunchSpec(
        argv=[sys.executable, str(FAKE), "--port", str(port), *extra],
        port=port,
        api_base=f"http://127.0.0.1:{port}/v1",
        served_model_name="fake-model",
        startup_timeout_s=timeout,
    )


def chat(api_base: str, **kw: object) -> StreamResult:
    async def run() -> StreamResult:
        async with httpx.AsyncClient(timeout=30, trust_env=False) as c:
            return await stream_chat(
                c,
                api_base,
                "fake-model",
                [{"role": "user", "content": "hi"}],
                **kw,  # type: ignore[arg-type]
            )

    return asyncio.run(run())


def test_start_stream_stop(tmp_path: Path) -> None:
    server = start_server(spec("--tokens", "6", "--mode", "slow_start"), tmp_path / "s.log")
    try:
        assert server.alive()
        assert server.load_time_s is not None and server.load_time_s >= 0.9  # waited for 200
        result = chat(server.spec.api_base, max_tokens=50)
        assert result.ok, result.error
        assert result.text.startswith("tok0 ")
        assert result.completion_tokens == 6 and result.token_count_source == "usage"
        assert result.ttft_s is not None and 0 < result.ttft_s <= result.e2e_s
        assert len(result.inter_chunk_s) == 5
        assert result.server_timings == {"predicted_n": 6}

        async def models() -> list[str]:
            async with httpx.AsyncClient(trust_env=False) as c:
                return await list_models(c, server.spec.api_base)

        assert asyncio.run(models()) == ["fake-model"]
    finally:
        server.stop()
    assert not server.alive()
    assert "listening" in server.log_tail()


def test_token_count_falls_back_to_chunks(tmp_path: Path) -> None:
    server = start_server(spec("--tokens", "4", "--no-usage"), tmp_path / "s.log")
    try:
        result = chat(server.spec.api_base)
        assert result.completion_tokens == 4 and result.token_count_source == "chunks"
    finally:
        server.stop()


@pytest.mark.parametrize(("mode", "kind"), [("crash", "config"), ("oom", "oom")])
def test_early_exit_is_classified(tmp_path: Path, mode: str, kind: str) -> None:
    with pytest.raises(LaunchError) as info:
        start_server(spec("--mode", mode), tmp_path / "s.log")
    assert info.value.failure.kind == kind
    assert info.value.failure.log_excerpt


def test_never_ready_times_out(tmp_path: Path) -> None:
    with pytest.raises(LaunchError) as info:
        start_server(spec("--mode", "never_ready", timeout=1.5), tmp_path / "s.log")
    assert info.value.failure.kind == "timeout"


def test_missing_executable(tmp_path: Path) -> None:
    bad = spec().model_copy(update={"argv": [str(tmp_path / "nope.exe")]})
    with pytest.raises(LaunchError) as info:
        start_server(bad, tmp_path / "s.log")
    assert info.value.failure.kind == "config"


def test_http_error_is_reported(tmp_path: Path) -> None:
    server = start_server(spec(), tmp_path / "s.log")
    try:
        result = chat(server.spec.api_base.replace("/v1", "/nope"))
        assert not result.ok and "404" in (result.error or "")
    finally:
        server.stop()


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("CUDA error: out of memory", "oom"),
        ("error loading model", "config"),
        ("bind: address already in use", "config"),
        ("segfault", "crash"),
    ],
)
def test_classify_failure(text: str, kind: str) -> None:
    assert classify_failure(text) == kind

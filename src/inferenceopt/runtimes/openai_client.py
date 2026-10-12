# SPDX-License-Identifier: Apache-2.0
"""Streaming client for OpenAI-compatible servers (llama-server, vLLM, trtllm-serve).

Timing is measured on the client: time to first token (TTFT) is when the first non-empty
content delta arrives; inter-token latency (ITL) is the gap between content deltas. When a
server batches several tokens into one delta, ITL is per-chunk, and the result says so.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Literal

import httpx


@dataclass
class StreamResult:
    text: str = ""
    ttft_s: float | None = None
    e2e_s: float = 0.0
    chunk_times_s: list[float] = field(default_factory=list)
    """Arrival time of each content delta, relative to the request start."""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    token_count_source: Literal["usage", "chunks"] = "chunks"  # noqa: S105 - not a secret
    finish_reason: str | None = None
    server_timings: dict[str, Any] | None = None
    """Server-reported timings when available (llama-server's ``timings`` object)."""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def inter_chunk_s(self) -> list[float]:
        t = self.chunk_times_s
        return [b - a for a, b in pairwise(t)]


def _request_body(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    seed: int | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None:
        body["seed"] = seed
    body.update(extra or {})
    return body


async def stream_chat(
    client: httpx.AsyncClient,
    api_base: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 64,
    temperature: float = 0.0,
    seed: int | None = 0,
    extra: dict[str, Any] | None = None,
) -> StreamResult:
    result = StreamResult()
    body = _request_body(model, messages, max_tokens, temperature, seed, extra)
    parts: list[str] = []
    start = time.perf_counter()
    try:
        async with client.stream(
            "POST", f"{api_base.rstrip('/')}/chat/completions", json=body
        ) as response:
            if response.status_code != 200:
                detail = (await response.aread()).decode("utf-8", errors="replace")[:500]
                result.error = f"HTTP {response.status_code}: {detail}"
                return result
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                _consume(chunk, result, parts, time.perf_counter() - start)
    except httpx.HTTPError as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.e2e_s = time.perf_counter() - start
        result.text = "".join(parts)
    if result.completion_tokens is None and result.chunk_times_s:
        result.completion_tokens = len(result.chunk_times_s)
        result.token_count_source = "chunks"  # noqa: S105
    return result


def _consume(chunk: dict[str, Any], result: StreamResult, parts: list[str], now: float) -> None:
    if "error" in chunk:
        result.error = str(chunk["error"])
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        content = delta.get("content") or ""
        if content:
            if result.ttft_s is None:
                result.ttft_s = now
            result.chunk_times_s.append(now)
            parts.append(content)
        if choice.get("finish_reason"):
            result.finish_reason = choice["finish_reason"]
    usage = chunk.get("usage")
    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
        result.prompt_tokens = usage.get("prompt_tokens")
        result.completion_tokens = usage.get("completion_tokens")
        result.token_count_source = "usage"  # noqa: S105
    if isinstance(chunk.get("timings"), dict):
        result.server_timings = chunk["timings"]


async def list_models(client: httpx.AsyncClient, api_base: str) -> list[str]:
    response = await client.get(f"{api_base.rstrip('/')}/models")
    response.raise_for_status()
    return [m.get("id", "") for m in response.json().get("data", [])]

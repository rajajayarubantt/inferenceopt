# SPDX-License-Identifier: Apache-2.0
"""Adapter registry: built-in adapters plus ``inferenceopt.runtimes`` entry points."""

from __future__ import annotations

import logging
from functools import cache
from importlib.metadata import entry_points

from inferenceopt.runtimes.base import RuntimeAdapter
from inferenceopt.runtimes.llamacpp.adapter import LlamaCppAdapter
from inferenceopt.runtimes.tensorrt_llm import TensorRTLLMAdapter
from inferenceopt.runtimes.vllm import VLLMAdapter

log = logging.getLogger(__name__)
ENTRY_POINT_GROUP = "inferenceopt.runtimes"


class UnknownRuntimeError(KeyError):
    pass


@cache
def adapters() -> dict[str, RuntimeAdapter]:
    found: dict[str, RuntimeAdapter] = {
        a.name: a for a in (LlamaCppAdapter(), VLLMAdapter(), TensorRTLLMAdapter())
    }
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in found:
            continue  # built-ins are registered under the same names in pyproject.toml
        try:
            adapter = ep.load()()
        except Exception as exc:
            log.warning("runtime plugin %s failed to load: %s", ep.name, exc)
            continue
        found[adapter.name] = adapter
    return found


def get_adapter(name: str) -> RuntimeAdapter:
    try:
        return adapters()[name]
    except KeyError:
        raise UnknownRuntimeError(name) from None

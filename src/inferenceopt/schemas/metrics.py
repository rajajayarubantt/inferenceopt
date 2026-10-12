# SPDX-License-Identifier: Apache-2.0
"""Metric registry. Objectives and constraints may only reference metrics defined here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["minimize", "maximize"]


@dataclass(frozen=True, slots=True)
class MetricDef:
    name: str
    unit: str
    better: Direction
    description: str


_DEFS = [
    MetricDef("load_time_s", "s", "minimize", "Model load time until the server is healthy."),
    MetricDef("ttft_ms_p50", "ms", "minimize", "Median time to first token."),
    MetricDef("ttft_ms_p95", "ms", "minimize", "95th percentile time to first token."),
    MetricDef("itl_ms_p50", "ms", "minimize", "Median inter-token latency."),
    MetricDef("itl_ms_p95", "ms", "minimize", "95th percentile inter-token latency."),
    MetricDef("latency_ms_p50", "ms", "minimize", "Median end-to-end request latency."),
    MetricDef("latency_ms_p95", "ms", "minimize", "95th percentile end-to-end latency."),
    MetricDef("latency_ms_p99", "ms", "minimize", "99th percentile end-to-end latency."),
    MetricDef(
        "output_tokens_per_second",
        "tok/s",
        "maximize",
        "Aggregate output throughput across all concurrent requests.",
    ),
    MetricDef(
        "per_request_output_tokens_per_second",
        "tok/s",
        "maximize",
        "Median decode speed seen by a single request.",
    ),
    MetricDef("prefill_tokens_per_second", "tok/s", "maximize", "Prompt processing throughput."),
    MetricDef("max_concurrency_within_slo", "requests", "maximize", "Highest SLO-meeting load."),
    MetricDef("max_context_tokens", "tokens", "maximize", "Largest context that loads and runs."),
    MetricDef("peak_vram_gib", "GiB", "minimize", "Peak accelerator memory."),
    MetricDef("peak_ram_gib", "GiB", "minimize", "Peak host memory."),
    MetricDef("energy_j_per_token", "J/token", "minimize", "Only with trustworthy telemetry."),
    MetricDef("error_rate", "fraction", "minimize", "Failed requests / total requests."),
]

METRICS: dict[str, MetricDef] = {m.name: m for m in _DEFS}

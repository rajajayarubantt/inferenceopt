# SPDX-License-Identifier: Apache-2.0
"""Audit record for each LLM planner call (Plan.md §5.5)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from inferenceopt.schemas.base import Versioned, new_id, utcnow


class PlannerCall(Versioned):
    id: str = Field(default_factory=lambda: new_id("plan"))
    job_id: str
    provider: str
    model: str
    prompt_template_version: str
    request_hash: str
    response_hash: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    outcome: Literal["accepted", "rejected_invalid", "error", "skipped_budget"]
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

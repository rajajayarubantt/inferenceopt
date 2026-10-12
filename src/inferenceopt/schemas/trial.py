# SPDX-License-Identifier: Apache-2.0
"""Candidates, trials and their measurements (Plan.md §9, §10, §15)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from inferenceopt.schemas.base import Schema, Versioned, new_id, utcnow

CandidateOrigin = Literal["baseline", "grid", "random", "bayesian", "planner", "manual"]


class CandidateConfiguration(Versioned):
    id: str = Field(default_factory=lambda: new_id("cand"))
    runtime: str
    model: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    origin: CandidateOrigin
    parent_ids: list[str] = Field(default_factory=list)


class TrialStatus(StrEnum):
    PENDING = "pending"
    PREFLIGHT = "preflight"
    STARTING = "starting"
    WARMING = "warming"
    MEASURING = "measuring"
    EVALUATING = "evaluating"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            TrialStatus.ACCEPTED,
            TrialStatus.REJECTED,
            TrialStatus.FAILED,
            TrialStatus.CANCELLED,
        }


class BenchmarkRun(Schema):
    workload_id: str
    warm: bool
    repetitions: int = Field(ge=1)
    summary: dict[str, float] = Field(default_factory=dict)
    """Metric name (see ``schemas.metrics``) to value."""
    raw_samples_path: str | None = None
    """Relative path to raw per-request samples inside the trial directory."""


class QualityRun(Schema):
    suite: str
    suite_version: str
    score: float
    baseline_score: float | None = None
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


FailureKind = Literal[
    "oom", "timeout", "crash", "invalid_output", "constraint", "config", "cancelled", "other"
]


class FailureRecord(Schema):
    stage: str
    kind: FailureKind
    message: str
    log_excerpt: str | None = None


class Trial(Versioned):
    id: str = Field(default_factory=lambda: new_id("trial"))
    job_id: str
    candidate: CandidateConfiguration
    status: TrialStatus = TrialStatus.PENDING
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    hardware_profile_id: str | None = None
    environment_fingerprint: str | None = None
    benchmark_runs: list[BenchmarkRun] = Field(default_factory=list)
    quality_runs: list[QualityRun] = Field(default_factory=list)
    decision_reason: str | None = None
    failure: FailureRecord | None = None
    planner_call_id: str | None = None

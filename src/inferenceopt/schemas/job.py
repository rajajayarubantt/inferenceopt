# SPDX-License-Identifier: Apache-2.0
"""Optimization job specification (Plan.md §8)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from inferenceopt.schemas.base import Schema, Versioned
from inferenceopt.schemas.metrics import METRICS, Direction


class Objective(Schema):
    metric: str
    direction: Direction

    @field_validator("metric")
    @classmethod
    def _known_metric(cls, value: str) -> str:
        if value not in METRICS:
            known = ", ".join(sorted(METRICS))
            raise ValueError(f"unknown metric {value!r}; known metrics: {known}")
        return value


class Constraints(Schema):
    peak_memory_gib_max: float | None = Field(default=None, gt=0)
    context_tokens_min: int | None = Field(default=None, gt=0)
    output_tokens_per_second_min: float | None = Field(default=None, gt=0)
    ttft_ms_p95_max: float | None = Field(default=None, gt=0)
    latency_ms_p95_max: float | None = Field(default=None, gt=0)
    error_rate_max: float = Field(default=0.0, ge=0, le=1)
    quality_regression_max_percent: float | None = Field(default=1.0, ge=0)


class Workload(Schema):
    trace: str | None = None
    """Name of a versioned workload trace. If set, the ranges below are ignored."""
    concurrency: list[int] = Field(default_factory=lambda: [1], min_length=1)
    prompt_tokens: list[int] = Field(default_factory=lambda: [512], min_length=1)
    generation_tokens: list[int] = Field(default_factory=lambda: [128], min_length=1)
    request_rate_per_s: float | None = Field(default=None, gt=0)
    seed: int = 0

    @field_validator("concurrency", "prompt_tokens", "generation_tokens")
    @classmethod
    def _positive(cls, values: list[int]) -> list[int]:
        if any(v <= 0 for v in values):
            raise ValueError("all values must be positive")
        return values


class SearchSpec(Schema):
    strategy: Literal["grid", "random", "constrained_search", "bayesian"] = "constrained_search"
    max_trials: int = Field(default=40, gt=0)
    max_wall_time_minutes: float = Field(default=180, gt=0)
    repetitions: int = Field(default=3, ge=1)


class PlannerOptions(Schema):
    enabled: bool = False


class DeploymentOptions(Schema):
    generate_container: bool = True
    export_model_artifact: bool = False


class Permissions(Schema):
    allow_downloads: bool = False
    allow_network_in_worker: bool = False


class OptimizationJob(Versioned):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    model: str | list[str] = Field(min_length=1)
    hardware: list[str] = Field(default_factory=lambda: ["auto"], min_length=1)
    runtimes: list[str] = Field(default_factory=list)
    """Allowlist. Empty means every runtime whose support level is ``tested``."""
    objective: Objective
    constraints: Constraints = Field(default_factory=Constraints)
    workload: Workload = Field(default_factory=Workload)
    search: SearchSpec = Field(default_factory=SearchSpec)
    llm_planner: PlannerOptions = Field(default_factory=PlannerOptions)
    deployment: DeploymentOptions = Field(default_factory=DeploymentOptions)
    permissions: Permissions = Field(default_factory=Permissions)
    require_approval: bool = True

    @model_validator(mode="after")
    def _objective_direction_is_sensible(self) -> OptimizationJob:
        better = METRICS[self.objective.metric].better
        if self.objective.direction != better:
            raise ValueError(
                f"objective {self.objective.metric!r} should be {better}d, "
                f"not {self.objective.direction}d"
            )
        return self


class JobFile(Schema):
    """Top-level wrapper for job YAML files: ``job: {...}``."""

    job: OptimizationJob

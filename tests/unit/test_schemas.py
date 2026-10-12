# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferenceopt.schemas import (
    METRICS,
    CandidateConfiguration,
    HardwareProfile,
    HostInfo,
    Objective,
    OptimizationJob,
    Trial,
    TrialStatus,
)
from inferenceopt.services import validate_job_file, validate_job_text
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.unit


def test_example_job_validates() -> None:
    result = validate_job_file(REPO_ROOT / "configs" / "example-job.yaml")
    assert result.ok, result.issues
    assert result.job is not None
    assert result.job.objective.metric == "latency_ms_p95"


def test_unknown_metric_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown metric"):
        Objective(metric="speed", direction="maximize")


def test_wrong_direction_rejected() -> None:
    with pytest.raises(ValidationError, match="should be minimized"):
        OptimizationJob(
            name="x", model="m", objective={"metric": "latency_ms_p95", "direction": "maximize"}
        )


def test_typo_field_reported_with_location() -> None:
    result = validate_job_text(
        "job:\n  name: a\n  model: m\n  objective: {metric: ttft_ms_p50, direction: minimize}\n"
        "  workload: {concurency: [1]}\n"
    )
    assert not result.ok
    assert any(i.location == "job.workload.concurency" for i in result.issues)


def test_invalid_yaml_reported() -> None:
    result = validate_job_text("job: [unclosed")
    assert not result.ok
    assert result.issues[0].location == "(yaml)"


@pytest.mark.parametrize("bad", [[0], [-1], []])
def test_workload_values_must_be_positive(bad: list[int]) -> None:
    with pytest.raises(ValidationError):
        OptimizationJob(
            name="x",
            model="m",
            objective={"metric": "ttft_ms_p50", "direction": "minimize"},
            workload={"concurrency": bad},
        )


def test_every_metric_direction_is_valid() -> None:
    assert all(m.better in {"minimize", "maximize"} for m in METRICS.values())


def test_trial_round_trips_through_json() -> None:
    trial = Trial(
        job_id="job_1",
        candidate=CandidateConfiguration(runtime="llamacpp", model="m", origin="baseline"),
    )
    assert Trial.model_validate_json(trial.model_dump_json()) == trial
    assert not trial.status.is_terminal
    assert TrialStatus.ACCEPTED.is_terminal


def test_hardware_profile_has_version_and_id() -> None:
    profile = HardwareProfile(
        host=HostInfo(os="Linux", os_version="6", python_version="3.12", cpu_arch="x86_64")
    )
    assert profile.schema_version == 1
    assert profile.id.startswith("hw_")

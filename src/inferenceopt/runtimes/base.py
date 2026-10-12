# SPDX-License-Identifier: Apache-2.0
"""Runtime adapter contract (Plan.md §3.1, Phase 3).

An adapter describes one inference runtime: how to find it, which parameters it accepts,
how to validate a configuration, and how to turn a configuration into a ``LaunchSpec``.
Adapters never start processes themselves and never build shell strings; the process
supervisor runs the structured ``LaunchSpec``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from inferenceopt.schemas import HardwareProfile, ModelReference, Schema, SupportLevel

ParamType = Literal["int", "float", "bool", "choice", "str"]
_IMAGE = re.compile(r"[a-z0-9][a-z0-9._/:@-]{0,254}")
# Runtime tuning variables only; never loader/path variables such as LD_PRELOAD or PATH.
_ENV_ALLOWED = re.compile(
    r"CUDA_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES|OMP_NUM_THREADS|HF_HOME|"
    r"(GGML|LLAMA|VLLM|TRTLLM|NCCL|TLLM)_[A-Z0-9_]+"
)


class ParamSpec(Schema):
    """One tunable runtime parameter. The optimizer may only touch parameters listed here."""

    name: str
    type: ParamType
    description: str
    default: Any = None
    choices: list[str] | None = None
    minimum: float | None = None
    maximum: float | None = None
    tunable: bool = True
    """False for parameters users may set but search should leave alone."""

    @model_validator(mode="after")
    def _choices_for_choice(self) -> ParamSpec:
        if self.type == "choice" and not self.choices:
            raise ValueError(f"{self.name}: choice parameters need choices")
        return self

    def check(self, value: Any) -> str | None:
        """Return an error message, or None if ``value`` is acceptable."""
        if self.type == "bool":
            return None if isinstance(value, bool) else f"{self.name} must be true or false"
        if self.type == "int" and (not isinstance(value, int) or isinstance(value, bool)):
            return f"{self.name} must be an integer"
        if self.type == "float" and (not isinstance(value, int | float) or isinstance(value, bool)):
            return f"{self.name} must be a number"
        if self.type in ("str", "choice") and not isinstance(value, str):
            return f"{self.name} must be a string"
        if self.type == "choice" and self.choices and value not in self.choices:
            return f"{self.name} must be one of {', '.join(self.choices)}"
        if self.type in ("int", "float"):
            if self.minimum is not None and value < self.minimum:
                return f"{self.name} must be >= {self.minimum:g}"
            if self.maximum is not None and value > self.maximum:
                return f"{self.name} must be <= {self.maximum:g}"
        return None


class RuntimeSettings(Schema):
    """Where to find a runtime. Persisted per runtime; edited from the CLI or UI."""

    mode: Literal["native", "docker"] = "native"
    executable: str | None = None
    """Path to the runtime binary (native mode). Auto-detected when unset."""
    python_env: str | None = None
    """Python environment containing the runtime package (vLLM native mode)."""
    image: str | None = None
    """Container image (docker mode)."""
    extra_env: dict[str, str] = Field(default_factory=dict)


class Installation(Schema):
    """A runtime that was actually found on this machine."""

    executable: str
    version: str | None = None
    mode: Literal["native", "docker"] = "native"
    source: Literal["settings", "managed", "path", "docker"]
    devices: list[str] = Field(default_factory=list)
    """Backends/devices the build reports, e.g. ``["CUDA0: NVIDIA GeForce RTX 3050"]``."""
    features: dict[str, Any] = Field(default_factory=dict)
    """Probed feature flags, e.g. ``{"flash_attn_value": True}``."""


class LaunchSpec(Schema):
    """Exactly what will be executed. Shown verbatim in dry runs and stored with trials."""

    argv: list[str] = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    host: str = "127.0.0.1"
    port: int = Field(ge=1, le=65535)
    health_path: str = "/health"
    api_base: str
    """OpenAI-compatible base URL, e.g. ``http://127.0.0.1:8080/v1``."""
    served_model_name: str
    startup_timeout_s: float = 300
    container_name: str | None = None
    """Set for docker launches so the supervisor can stop the container."""


class RuntimeAdapter(ABC):
    name: str
    display_name: str
    support_level: SupportLevel
    platforms: tuple[str, ...]
    """``platform.system()`` values this runtime can run on natively."""
    weight_formats: tuple[str, ...]
    homepage: str
    executable_names: tuple[str, ...] = ()
    """Allowed basenames for a user-configured executable. Anything else is refused, so the
    settings API can never be used to run an arbitrary program."""

    def validate_settings(self, settings: RuntimeSettings) -> list[str]:
        errors = []
        if settings.executable:
            exe = Path(settings.executable)
            if exe.name.lower() not in {n.lower() for n in self.executable_names}:
                allowed = ", ".join(self.executable_names) or "none"
                errors.append(f"executable must be named one of: {allowed}")
            elif not exe.is_absolute():
                errors.append("executable must be an absolute path")
            elif not exe.is_file():
                errors.append(f"executable not found: {exe}")
        if settings.python_env and not Path(settings.python_env).is_dir():
            errors.append(f"python_env is not a directory: {settings.python_env}")
        if settings.image and not _IMAGE.fullmatch(settings.image):
            errors.append("image must be a plain image reference such as repo/name:tag@sha256:...")
        for key in settings.extra_env:
            if not _ENV_ALLOWED.fullmatch(key):
                errors.append(f"environment variable {key!r} is not allowed")
        return errors

    @abstractmethod
    def parameters(self, install: Installation | None) -> list[ParamSpec]:
        """Parameter schema. May depend on the installed version's probed features."""

    @abstractmethod
    def detect(self, settings: RuntimeSettings) -> tuple[Installation | None, list[str]]:
        """Find the runtime. Returns the installation (or None) and human-readable notes."""

    @abstractmethod
    def build_launch(
        self,
        install: Installation,
        model: ModelReference,
        config: dict[str, Any],
        *,
        port: int,
        host: str = "127.0.0.1",
    ) -> LaunchSpec:
        """Translate a validated configuration into a launch spec."""

    def defaults(self, install: Installation | None) -> dict[str, Any]:
        return {p.name: p.default for p in self.parameters(install) if p.default is not None}

    def validate(
        self,
        config: dict[str, Any],
        model: ModelReference,
        hardware: HardwareProfile | None,
        install: Installation | None,
    ) -> list[str]:
        """Schema checks plus runtime-specific cross-field rules. Empty list means valid."""
        specs = {p.name: p for p in self.parameters(install)}
        errors = [f"unknown parameter {k!r}" for k in config if k not in specs]
        errors += [e for k, v in config.items() if k in specs and (e := specs[k].check(v))]
        if model.weight_format and model.weight_format not in self.weight_formats:
            errors.append(
                f"{self.display_name} cannot load {model.weight_format} weights "
                f"(supported: {', '.join(self.weight_formats)})"
            )
        if not errors:
            errors += self.cross_validate(config, model, hardware, install)
        return errors

    def cross_validate(
        self,
        config: dict[str, Any],
        model: ModelReference,
        hardware: HardwareProfile | None,
        install: Installation | None,
    ) -> list[str]:
        return []

    def resolve(self, config: dict[str, Any], install: Installation | None) -> dict[str, Any]:
        """Defaults overlaid with the user's configuration."""
        return {**self.defaults(install), **config}


class RuntimeAdapterError(Exception):
    """A runtime could not be found, configured or launched."""


class ConfigurationError(RuntimeAdapterError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors

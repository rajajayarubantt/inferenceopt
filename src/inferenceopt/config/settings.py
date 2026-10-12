# SPDX-License-Identifier: Apache-2.0
"""Application configuration.

Precedence, highest first: explicit keyword arguments, ``INFERENCEOPT_*`` environment
variables (nested with ``__``, e.g. ``INFERENCEOPT_SERVER__PORT``), then the YAML file.
Secrets are typed as ``SecretStr`` so they never appear in reprs, logs or API responses.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from inferenceopt.config.paths import default_config_file

_config_file: ContextVar[Path | None] = ContextVar("_config_file", default=None)


class ConfigError(Exception):
    """Raised when the configuration file or environment is invalid."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerConfig(_Section):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    allow_remote: bool = False
    """Must be true to bind to a non-loopback address. Token auth stays on regardless."""
    allowed_hosts: list[str] = Field(default_factory=list)
    """Extra Host header values to accept, in addition to loopback names."""
    token: SecretStr | None = None
    """Fixed session token. If unset, a random token is generated on each launch."""
    open_browser: bool = True


class StorageConfig(_Section):
    database_path: Path | None = None
    """SQLite file. Defaults to the per-user data directory."""


class LLMConfig(_Section):
    enabled: bool = False
    planner_model: str | None = None
    """Any LiteLLM model string or alias, e.g. ``openai/gpt-4.1`` or ``ollama/llama3``."""
    api_base: str | None = None
    api_key: SecretStr | None = None
    """Prefer the provider's own environment variable; this exists for UI-entered keys."""
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0)
    max_planning_tokens: int = Field(default=4000, gt=0)
    max_calls_per_job: int = Field(default=20, ge=0)
    allow_external_data: bool = False
    send_hardware_metadata: bool = True
    send_model_names: bool = True
    send_benchmark_traces: bool = False


class LoggingConfig(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_format: bool = False


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERENCEOPT_",
        env_nested_delimiter="__",
        # Top-level env vars such as INFERENCEOPT_HOME are not config fields; unknown
        # keys in the YAML file are rejected separately in load_config().
        extra="ignore",
    )

    server: ServerConfig = Field(default_factory=ServerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        path = _config_file.get()
        if path is not None and path.is_file():
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=path))
        return tuple(sources)


def _check_top_level_keys(path: Path) -> None:
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if data is None:
        return
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    unknown = sorted(set(data) - set(AppConfig.model_fields))
    if unknown:
        raise ConfigError(f"{path}: unknown top-level keys: {', '.join(unknown)}")


def load_config(path: Path | None = None, **overrides: Any) -> AppConfig:
    """Load configuration from ``path`` (default: the per-user config file) and env."""
    resolved = path or default_config_file()
    if path is not None and not resolved.is_file():
        raise ConfigError(f"config file not found: {resolved}")
    if resolved.is_file():
        _check_top_level_keys(resolved)
    token = _config_file.set(resolved)
    try:
        return AppConfig(**overrides)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration ({resolved}):\n{exc}") from exc
    finally:
        _config_file.reset(token)

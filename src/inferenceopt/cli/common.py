# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by CLI command modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from inferenceopt import services
from inferenceopt.config import ConfigError

console = Console()
err_console = Console(stderr=True)

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Config file (default: per-user config.yaml)."),
]
JsonOption = Annotated[bool, typer.Option("--json", help="Machine-readable JSON output.")]


def open_context(config: Path | None) -> services.AppContext:
    try:
        return services.AppContext.open(config)
    except ConfigError as exc:
        err_console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(2) from exc


def print_json(data: Any) -> None:
    console.print_json(json.dumps(data, default=str))


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{n} B"


def fmt_count(n: int | None) -> str:
    if n is None:
        return "?"
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.2f}{suffix}"
    return str(n)

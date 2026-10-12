# SPDX-License-Identifier: Apache-2.0
"""Command-line interface. Every command is a thin wrapper over ``inferenceopt.services``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from inferenceopt import services
from inferenceopt.cli.common import (
    ConfigOption,
    JsonOption,
    console,
    err_console,
    open_context,
    print_json,
)
from inferenceopt.cli.inventory import hardware_app, models_app, preflight
from inferenceopt.cli.runtimes import runtimes_app
from inferenceopt.config.redaction import redact

app = typer.Typer(
    name="inferenceopt",
    help="Hardware-aware, measurement-driven LLM inference optimization.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Inspect configuration.", no_args_is_help=True)
job_app = typer.Typer(help="Work with optimization job files.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(job_app, name="job")
app.add_typer(hardware_app, name="hardware")
app.add_typer(models_app, name="models")
app.add_typer(runtimes_app, name="runtimes")
app.command()(preflight)

_STATUS_STYLE = {"ok": "green", "info": "blue", "warn": "yellow", "fail": "red"}


@app.command()
def version(as_json: JsonOption = False) -> None:
    """Show version information."""
    info = services.version_info()
    if as_json:
        print_json(info.model_dump())
    else:
        console.print(f"inferenceopt {info.inferenceopt} (Python {info.python}, {info.platform})")


@app.command()
def doctor(config: ConfigOption = None, as_json: JsonOption = False) -> None:
    """Check the installation and environment."""
    ctx = open_context(config)
    try:
        report = services.doctor(ctx)
    finally:
        ctx.close()
    if as_json:
        print_json(report.model_dump())
    else:
        table = Table(title=f"inferenceopt {report.version.inferenceopt}", show_lines=False)
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Detail", overflow="fold")
        for check in report.checks:
            style = _STATUS_STYLE[check.status]
            detail = escape(check.detail)
            if check.hint:
                detail += f"\n[dim]hint: {escape(check.hint)}[/]"
            table.add_row(escape(check.name), f"[{style}]{check.status}[/]", detail)
        console.print(table)
    if not report.ok:
        raise typer.Exit(1)


@config_app.command("show")
def config_show(config: ConfigOption = None, as_json: JsonOption = False) -> None:
    """Print the effective configuration (secrets are never shown)."""
    ctx = open_context(config)
    try:
        data = redact(services.public_config(ctx))
    finally:
        ctx.close()
    if as_json:
        print_json(data)
    else:
        console.print(f"[dim]config file: {ctx.config_path}[/]")
        print_json(data)


@job_app.command("validate")
def job_validate(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    as_json: JsonOption = False,
) -> None:
    """Validate a job YAML file against the schema."""
    result = services.validate_job_file(file)
    if as_json:
        print_json(result.model_dump(mode="json"))
    elif result.ok and result.job is not None:
        console.print(f"[green]OK[/] {file} is valid (job {result.job.name!r})")
    else:
        console.print(f"[red]INVALID[/] {file} is invalid:")
        for issue in result.issues:
            console.print(f"  [bold]{escape(issue.location)}[/]: {escape(issue.message)}")
    if not result.ok:
        raise typer.Exit(1)


@app.command()
def ui(
    config: ConfigOption = None,
    host: Annotated[str | None, typer.Option(help="Bind address (default 127.0.0.1).")] = None,
    port: Annotated[int | None, typer.Option(help="Port (default 8765).")] = None,
    no_browser: Annotated[bool, typer.Option("--no-browser", help="Don't open a browser.")] = False,
    allow_remote: Annotated[
        bool, typer.Option("--allow-remote", help="Allow binding to a non-loopback address.")
    ] = False,
) -> None:
    """Start the web UI."""
    from inferenceopt.server.run import RemoteBindError, serve

    ctx = open_context(config)
    try:
        serve(
            ctx,
            host=host,
            port=port,
            open_browser=False if no_browser else None,
            allow_remote=allow_remote or None,
            console=console,
        )
    except RemoteBindError as exc:
        err_console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from exc
    finally:
        ctx.close()


def main() -> None:
    app()

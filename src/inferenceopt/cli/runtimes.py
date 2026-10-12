# SPDX-License-Identifier: Apache-2.0
"""``runtimes`` commands: list, inspect, configure, install, launch preview, smoke test."""

from __future__ import annotations

import json
import shlex
from typing import Annotated, Any, Literal

import typer
from rich.markup import escape
from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn
from rich.table import Table

from inferenceopt.cli.common import (
    ConfigOption,
    JsonOption,
    console,
    err_console,
    fmt_bytes,
    open_context,
    print_json,
)
from inferenceopt.models import ModelNotFoundError
from inferenceopt.runtimes.base import RuntimeSettings
from inferenceopt.runtimes.llamacpp.install import InstallError
from inferenceopt.runtimes.registry import UnknownRuntimeError
from inferenceopt.services import runtimes as svc

runtimes_app = typer.Typer(
    help="Inference runtimes (llama.cpp, vLLM, TensorRT-LLM).", no_args_is_help=True
)

_LEVEL_STYLE = {"tested": "green", "experimental": "yellow", "unsupported": "red"}
SetOption = Annotated[
    list[str] | None,
    typer.Option(
        "--set", help="Runtime parameter KEY=VALUE (repeatable). Values are JSON or text."
    ),
]


def _parse_sets(items: list[str] | None) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            err_console.print(f"[red]--set expects KEY=VALUE, got {escape(item)!r}[/]")
            raise typer.Exit(2)
        try:
            config[key] = json.loads(raw)
        except json.JSONDecodeError:
            config[key] = raw
    return config


def _unknown(name: str) -> typer.Exit:
    err_console.print(
        f"[red]unknown runtime {escape(name)!r}[/] (see `inferenceopt runtimes list`)"
    )
    return typer.Exit(2)


@runtimes_app.command("list")
def runtimes_list(config: ConfigOption = None, as_json: JsonOption = False) -> None:
    """Show each runtime, whether it was found, and its support level."""
    ctx = open_context(config)
    try:
        statuses = svc.list_runtimes(ctx)
    finally:
        ctx.close()
    if as_json:
        print_json([s.model_dump(mode="json") for s in statuses])
        return
    table = Table()
    for col in ("Runtime", "Support", "Found", "Version", "Devices"):
        table.add_column(col, overflow="fold")
    for s in statuses:
        inst = s.installation
        level = s.support_level.value
        table.add_row(
            f"{s.display_name} [dim]({s.name})[/]",
            f"[{_LEVEL_STYLE[level]}]{level}[/]",
            escape(f"{inst.mode}: {inst.executable}") if inst else "[red]no[/]",
            inst.version or "?" if inst else "",
            escape(", ".join(inst.devices)) if inst else "",
        )
    console.print(table)
    for s in statuses:
        for note in s.notes:
            console.print(f"[yellow]{s.name}:[/] {escape(note)}")


@runtimes_app.command("inspect")
def runtimes_inspect(name: str, config: ConfigOption = None, as_json: JsonOption = False) -> None:
    """Show a runtime's settings, detection result and tunable parameters."""
    ctx = open_context(config)
    try:
        status = svc.runtime_status(ctx, name)
    except UnknownRuntimeError as exc:
        raise _unknown(name) from exc
    finally:
        ctx.close()
    if as_json:
        print_json(status.model_dump(mode="json"))
        return
    console.print(f"[bold]{status.display_name}[/]  support: {status.support_level.value}")
    console.print(f"settings: {escape(status.settings.model_dump_json(exclude_defaults=True))}")
    if inst := status.installation:
        console.print(f"found: {escape(inst.executable)} ({inst.mode}, {inst.version or '?'})")
        for d in inst.devices:
            console.print(f"  device {escape(d)}")
    for note in status.notes:
        console.print(f"[yellow]note:[/] {escape(note)}")
    table = Table(title="Parameters")
    for col in ("Name", "Type", "Default", "Range / choices", "Description"):
        table.add_column(col, overflow="fold")
    for p in status.parameters:
        span = (
            ", ".join(p.choices)
            if p.choices
            else (
                f"{p.minimum:g} - {p.maximum:g}"
                if p.minimum is not None and p.maximum is not None
                else ""
            )
        )
        table.add_row(
            p.name, p.type, "" if p.default is None else str(p.default), span, p.description
        )
    console.print(table)


@runtimes_app.command("configure")
def runtimes_configure(
    name: str,
    mode: Annotated[str | None, typer.Option(help="native or docker")] = None,
    executable: Annotated[
        str | None, typer.Option(help="Absolute path to the runtime binary.")
    ] = None,
    python_env: Annotated[
        str | None, typer.Option(help="Python environment (vLLM native).")
    ] = None,
    image: Annotated[str | None, typer.Option(help="Container image (docker mode).")] = None,
    reset: Annotated[bool, typer.Option("--reset", help="Clear all settings first.")] = False,
    config: ConfigOption = None,
) -> None:
    """Tell InferenceOpt where to find a runtime."""
    ctx = open_context(config)
    try:
        current = RuntimeSettings() if reset else svc.get_settings(ctx, name)
        updates = {
            k: v
            for k, v in {
                "mode": mode,
                "executable": executable,
                "python_env": python_env,
                "image": image,
            }.items()
            if v is not None
        }
        svc.save_settings(ctx, name, current.model_copy(update=updates))
        status = svc.runtime_status(ctx, name)
    except UnknownRuntimeError as exc:
        raise _unknown(name) from exc
    except svc.InvalidSettingsError as exc:
        for e in exc.errors:
            err_console.print(f"[red]{escape(e)}[/]")
        raise typer.Exit(2) from exc
    finally:
        ctx.close()
    found = status.installation
    console.print(
        f"Saved. {status.display_name}: "
        + (
            f"found {escape(found.executable)} ({found.version or '?'})"
            if found
            else "[red]not found[/]"
        )
    )
    for note in status.notes:
        console.print(f"[yellow]note:[/] {escape(note)}")


@runtimes_app.command("install")
def runtimes_install(
    name: Annotated[str, typer.Argument(help="Only llamacpp can be installed automatically.")],
    channel: Annotated[
        str, typer.Option(help="stable, nightly, or an exact build tag.")
    ] = "stable",
    variant: Annotated[str, typer.Option(help="auto, cpu, cuda or vulkan.")] = "auto",
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask for confirmation.")] = False,
    config: ConfigOption = None,
) -> None:
    """Download an official llama.cpp build (verified against GitHub's SHA-256 digests)."""
    if name != "llamacpp":
        err_console.print(
            "Only llamacpp can be installed automatically; see the vLLM/TensorRT-LLM docs."
        )
        raise typer.Exit(2)
    if variant not in ("auto", "cpu", "cuda", "vulkan"):
        err_console.print("--variant must be auto, cpu, cuda or vulkan")
        raise typer.Exit(2)
    v: Literal["auto", "cpu", "cuda", "vulkan"] = variant  # type: ignore[assignment]
    ctx = open_context(config)
    try:
        plan = svc.plan_llamacpp_install(ctx, channel, v)
    except InstallError as exc:
        err_console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    finally:
        ctx.close()
    console.print(f"llama.cpp {plan.tag} ({plan.variant}) from github.com/ggml-org/llama.cpp:")
    for a in plan.assets:
        console.print(f"  {a.name}  {fmt_bytes(a.size)}  sha256 {(a.sha256 or 'MISSING')[:16]}...")
    if not yes and not typer.confirm(f"Download {fmt_bytes(plan.total_bytes)}?"):
        raise typer.Exit(1)
    columns = ("[progress.description]{task.description}", BarColumn(), DownloadColumn())
    with Progress(*columns, TransferSpeedColumn(), console=console) as progress:
        tasks: dict[str, Any] = {}

        def on_progress(asset: str, done: int, total: int | None) -> None:
            if asset not in tasks:
                tasks[asset] = progress.add_task(asset, total=total)
            progress.update(tasks[asset], completed=done)

        try:
            path = svc.install_llamacpp(plan, on_progress)
        except InstallError as exc:
            err_console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(1) from exc
    console.print(f"[green]Installed[/] {escape(str(path))}")


@runtimes_app.command("launch-preview")
def runtimes_launch_preview(
    name: str,
    model_id: str,
    sets: SetOption = None,
    config: ConfigOption = None,
    as_json: JsonOption = False,
) -> None:
    """Validate a configuration and print the exact command. Starts nothing."""
    ctx = open_context(config)
    try:
        preview = svc.launch_preview(ctx, name, model_id, _parse_sets(sets))
    except UnknownRuntimeError as exc:
        raise _unknown(name) from exc
    except ModelNotFoundError as exc:
        err_console.print(f"[red]model {escape(model_id)!r} not found[/]")
        raise typer.Exit(1) from exc
    finally:
        ctx.close()
    if as_json:
        print_json(preview.model_dump(mode="json"))
    elif preview.spec:
        console.print(escape(shlex.join(preview.spec.argv)), soft_wrap=True)
    for e in preview.errors:
        err_console.print(f"[red]{escape(e)}[/]")
    if preview.errors:
        raise typer.Exit(1)


@runtimes_app.command("smoke-test")
def runtimes_smoke_test(
    name: str,
    model_id: str,
    sets: SetOption = None,
    max_tokens: Annotated[int, typer.Option(help="Tokens to generate.")] = 24,
    config: ConfigOption = None,
    as_json: JsonOption = False,
) -> None:
    """Start the runtime, generate a few tokens, stop it. A sanity check, not a benchmark."""
    ctx = open_context(config)
    try:
        with console.status(f"Starting {name}..."):
            result = svc.smoke_test(ctx, name, model_id, _parse_sets(sets), max_tokens=max_tokens)
    except UnknownRuntimeError as exc:
        raise _unknown(name) from exc
    except ModelNotFoundError as exc:
        err_console.print(f"[red]model {escape(model_id)!r} not found[/]")
        raise typer.Exit(1) from exc
    finally:
        ctx.close()
    if as_json:
        print_json(result.model_dump(mode="json"))
    else:
        for e in result.errors:
            err_console.print(f"[red]{escape(e)}[/]")
        if result.ok:
            console.print(f"[green]OK[/] {name} {result.runtime_version or ''}")
            console.print(
                f"  load {result.load_time_s:.2f}s, TTFT {(result.ttft_s or 0) * 1000:.0f} ms, "
                f"{result.completion_tokens} tokens ({result.token_count_source}), "
                f"decode {result.decode_tokens_per_s or 0:.1f} tok/s (single request)"
            )
            console.print(f"  output: {escape(result.output_excerpt or '')!r}")
        elif result.failure:
            err_console.print(f"[red]{result.failure.kind}[/]: {escape(result.failure.message)}")
            if result.log_tail:
                err_console.print(f"[dim]{escape(result.log_tail[-2000:])}[/]")
        if result.log_path:
            console.print(f"[dim]log: {escape(result.log_path)}[/]")
    if not result.ok:
        raise typer.Exit(1)

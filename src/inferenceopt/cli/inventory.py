# SPDX-License-Identifier: Apache-2.0
"""``hardware``, ``models`` and ``preflight`` commands."""

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
    fmt_bytes,
    fmt_count,
    open_context,
    print_json,
)
from inferenceopt.models import ModelExistsError, ModelInspectionError, ModelNotFoundError
from inferenceopt.schemas import EstimateInputs, HardwareProfile, PreflightReport, Provenance

hardware_app = typer.Typer(help="Discover CPUs, GPUs and memory.", no_args_is_help=True)
models_app = typer.Typer(help="Browse and register models.", no_args_is_help=True)

_VERDICT_STYLE = {"fits": "green", "tight": "yellow", "does_not_fit": "red", "unknown": "blue"}


def _print_hardware(profile: HardwareProfile) -> None:
    h = profile.host
    console.print(
        f"[bold]{escape(h.cpu_model or h.cpu_arch)}[/]  ({h.os} {h.os_version}, {h.cpu_arch})"
    )
    console.print(
        f"Cores: {h.physical_cores or '?'} physical / {h.logical_cores or '?'} logical   "
        f"RAM: {fmt_bytes(h.ram_available_bytes)} free of {fmt_bytes(h.ram_total_bytes)}"
    )
    console.print(f"CPU features: {', '.join(h.cpu_flags) or 'unknown'}")
    table = Table(title="Devices")
    for col in ("ID", "Backend", "Name", "Arch / CC", "Memory free / total", "Dtypes"):
        table.add_column(col)
    for d in profile.devices:
        inferred = "*" if d.field_sources.get("supported_dtypes") == Provenance.INFERRED else ""
        arch = d.architecture or ""
        if d.compute_capability:
            arch += f" / {d.compute_capability}"
        table.add_row(
            d.device_id,
            d.backend.value,
            escape(d.name),
            arch,
            f"{fmt_bytes(d.memory_free_bytes)} / {fmt_bytes(d.memory_total_bytes)}",
            ", ".join(d.supported_dtypes) + inferred,
        )
    console.print(table)
    console.print("[dim]* inferred from the device type, not measured[/]")
    for k, v in profile.runtime_versions.items():
        console.print(f"{k}: {v}")
    for note in profile.notes:
        console.print(f"[yellow]note:[/] {escape(note)}")


@hardware_app.command("scan")
def hardware_scan(
    config: ConfigOption = None,
    as_json: JsonOption = False,
    no_save: Annotated[bool, typer.Option("--no-save", help="Don't store the result.")] = False,
) -> None:
    """Scan the hardware (read-only) and save the result."""
    ctx = open_context(config)
    try:
        profile = services.scan_hardware(ctx, save=not no_save)
    finally:
        ctx.close()
    if as_json:
        print_json(profile.model_dump(mode="json"))
    else:
        _print_hardware(profile)


@hardware_app.command("show")
def hardware_show(config: ConfigOption = None, as_json: JsonOption = False) -> None:
    """Show the most recent saved scan."""
    ctx = open_context(config)
    try:
        profile = services.latest_hardware(ctx)
    finally:
        ctx.close()
    if profile is None:
        err_console.print("No saved scan yet. Run: inferenceopt hardware scan")
        raise typer.Exit(1)
    if as_json:
        print_json(profile.model_dump(mode="json"))
    else:
        console.print(f"[dim]scanned {profile.captured_at:%Y-%m-%d %H:%M UTC}[/]")
        _print_hardware(profile)


@models_app.command("list")
def models_list(config: ConfigOption = None, as_json: JsonOption = False) -> None:
    """List curated and user-added models."""
    ctx = open_context(config)
    try:
        models = services.list_models(ctx)
    finally:
        ctx.close()
    if as_json:
        print_json([m.model_dump(mode="json") for m in models])
        return
    table = Table()
    table.add_column("ID", overflow="fold")
    for col in ("Origin", "Format", "Quant", "Params", "Context", "License"):
        table.add_column(col, overflow="fold")
    for m in models:
        table.add_row(
            escape(m.id),
            m.origin,
            m.weight_format or "?",
            m.quantization or "?",
            fmt_count(m.parameter_count),
            f"{m.context_length_max:,}" if m.context_length_max else "?",
            m.license or "?",
        )
    console.print(table)


@models_app.command("inspect")
def models_inspect(
    target: Annotated[str, typer.Argument(help="Registry id, .gguf file or HF directory.")],
    config: ConfigOption = None,
    as_json: JsonOption = False,
) -> None:
    """Show a model's metadata. Paths are read without registering them."""
    path = Path(target).expanduser()
    try:
        if path.exists():
            model = services.inspect_local_model(path)
        else:
            ctx = open_context(config)
            try:
                model = services.get_model(ctx, target)
            finally:
                ctx.close()
    except (ModelInspectionError, ModelNotFoundError) as exc:
        err_console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    if as_json:
        print_json(model.model_dump(mode="json"))
        return
    console.print(f"[bold]{escape(model.display_name)}[/]  [dim]{escape(model.id)}[/]")
    rows = {
        "Format": f"{model.weight_format or '?'} ({model.quantization or '?'})",
        "Parameters": fmt_count(model.parameter_count),
        "Weights on disk": fmt_bytes(model.weight_bytes),
        "Max context": f"{model.context_length_max:,}" if model.context_length_max else "?",
        "License": model.license or "unknown - check the model card",
    }
    if a := model.architecture:
        rows["Architecture"] = (
            f"{a.family or '?'}: {a.num_layers} layers, hidden {a.hidden_size}, "
            f"{a.num_attention_heads} heads ({a.num_kv_heads} KV) x {a.head_dim}"
        )
    for k, v in rows.items():
        console.print(f"  {k + ':':<17}{escape(v)}")
    for note in model.notes:
        console.print(f"[yellow]note:[/] {escape(note)}")


@models_app.command("add")
def models_add(
    path: Annotated[Path, typer.Argument(exists=True, help=".gguf file or HF directory.")],
    model_id: Annotated[str | None, typer.Option("--id", help="Registry id.")] = None,
    config: ConfigOption = None,
) -> None:
    """Register a local model (headers only are read; nothing is copied)."""
    ctx = open_context(config)
    try:
        model = services.add_local_model(ctx, path, model_id)
    except (ModelInspectionError, ModelExistsError) as exc:
        err_console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    finally:
        ctx.close()
    console.print(f"Added [bold]{escape(model.id)}[/]")


@models_app.command("remove")
def models_remove(model_id: str, config: ConfigOption = None) -> None:
    """Remove a user-added model from the registry (files on disk are not touched)."""
    ctx = open_context(config)
    try:
        services.remove_model(ctx, model_id)
    except (ModelNotFoundError, ModelExistsError) as exc:
        err_console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    finally:
        ctx.close()
    console.print(f"Removed {escape(model_id)}")


def _print_preflight(report: PreflightReport) -> None:
    style = _VERDICT_STYLE[report.verdict]
    console.print(f"[bold]{escape(report.model_id)}[/]: [{style}]{report.verdict}[/]")
    if est := report.estimate:
        console.print(
            f"Memory estimate: [bold]{fmt_bytes(est.expected_bytes)}[/] "
            f"(range {fmt_bytes(est.low_bytes)} - {fmt_bytes(est.high_bytes)})"
        )
        for c in est.components:
            console.print(f"  {c.name:<17}{fmt_bytes(c.bytes):>12}  [dim]{escape(c.basis)}[/]")
    table = Table(title="Devices")
    for col in ("Device", "Verdict", "Reason"):
        table.add_column(col)
    for d in report.devices:
        v = _VERDICT_STYLE[d.verdict]
        table.add_row(f"{d.device_id} {escape(d.device_name)}", f"[{v}]{d.verdict}[/]", d.reason)
    console.print(table)
    for r in report.runtimes:
        mark = "[green]ok[/]" if r.compatible else "[red]no[/]"
        console.print(f"runtime {r.runtime}: {mark} {escape('; '.join(r.reasons))}")
    for m in report.messages:
        console.print(f"[yellow]note:[/] {escape(m)}")
    if est := report.estimate:
        for a in est.assumptions:
            console.print(f"[dim]assumption: {escape(a)}[/]")


def preflight(
    model_id: Annotated[str, typer.Argument(help="Registry id (see `models list`).")],
    context: Annotated[int, typer.Option(help="Context tokens per sequence.")] = 4096,
    concurrency: Annotated[int, typer.Option(help="Concurrent sequences.")] = 1,
    kv_dtype: Annotated[
        str, typer.Option(help="KV cache dtype: f32, f16, bf16, q8_0, q4_0, fp8.")
    ] = "f16",
    batch_tokens: Annotated[int, typer.Option(help="Tokens per forward pass.")] = 512,
    bits: Annotated[float | None, typer.Option(help="Override bits per weight.")] = None,
    no_flash_attn: Annotated[bool, typer.Option("--no-flash-attn")] = False,
    runtime: Annotated[list[str] | None, typer.Option(help="Limit to runtime(s).")] = None,
    rescan: Annotated[bool, typer.Option(help="Re-scan hardware first.")] = False,
    config: ConfigOption = None,
    as_json: JsonOption = False,
) -> None:
    """Estimate memory and check whether a model fits this machine (no download)."""
    try:
        inputs = EstimateInputs.model_validate(
            {
                "context_tokens": context,
                "concurrency": concurrency,
                "kv_cache_dtype": kv_dtype,
                "batch_tokens": batch_tokens,
                "flash_attention": not no_flash_attn,
                "weight_bits_per_param": bits,
            }
        )
    except ValueError as exc:
        err_console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc
    ctx = open_context(config)
    try:
        report = services.run_preflight(ctx, model_id, inputs, runtime, rescan=rescan)
    except ModelNotFoundError as exc:
        err_console.print(f"[red]model {escape(model_id)!r} not found[/]")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        err_console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc
    finally:
        ctx.close()
    if as_json:
        print_json(report.model_dump(mode="json"))
    else:
        _print_preflight(report)
    if report.verdict == "does_not_fit":
        raise typer.Exit(1)

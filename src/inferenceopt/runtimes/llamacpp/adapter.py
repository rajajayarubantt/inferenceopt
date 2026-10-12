# SPDX-License-Identifier: Apache-2.0
"""llama.cpp ``llama-server`` adapter. Support level: tested."""

from __future__ import annotations

import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from inferenceopt.runtimes.base import (
    Installation,
    LaunchSpec,
    ParamSpec,
    RuntimeAdapter,
    RuntimeSettings,
)
from inferenceopt.runtimes.docker import (
    CONTAINER_BIND_ALL,
    docker_executable,
    docker_run_argv,
    image_info,
)
from inferenceopt.runtimes.llamacpp.install import installed_servers
from inferenceopt.runtimes.process import CODE_INTEGRITY_HINT, is_code_integrity_block
from inferenceopt.schemas import (
    DeviceBackend,
    HardwareProfile,
    ModelReference,
    SupportLevel,
)

CACHE_TYPES = ["f16", "f32", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
_UNQUANTIZED = {"f16", "f32", "bf16"}
LOAD_MODES = ["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"]
# Older builds (before --load-mode) only had these switches.
_LEGACY_LOAD_FLAGS = {
    "auto": [], "mmap": [], "none": ["--no-mmap"], "mlock": ["--no-mmap", "--mlock"],
    "mmap+mlock": ["--mlock"],
}  # fmt: skip


DEFAULT_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"
CONTAINER_PORT = 8080


def _run(argv: tuple[str, ...], *args: str, timeout: float = 60) -> tuple[int | None, str]:
    try:
        r = subprocess.run(  # noqa: S603 - fixed argv
            [*argv, *args], capture_output=True, text=True, timeout=timeout, check=False,
            encoding="utf-8", errors="replace",
        )  # fmt: skip
    except OSError as exc:
        return getattr(exc, "winerror", None) or -1, str(exc)
    except subprocess.TimeoutExpired:
        return None, ""
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def parse_version(text: str) -> str | None:
    """``version: 0.6.0-dev (build 11429, commit ...)`` or older ``version: 6000 (hash)``.

    Returns the build number as ``b11429`` when present, since builds are what releases
    and images are tagged with."""
    if build := re.search(r"\(build\s+(\d+)", text):
        return f"b{build.group(1)}"
    m = re.search(r"version:\s*(\S+)", text)
    if not m:
        return None
    return f"b{m.group(1)}" if m.group(1).isdigit() else m.group(1)


def parse_devices(text: str) -> list[str]:
    devices = []
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z]+\d+):\s*(.+)", line)
        if m:
            devices.append(f"{m.group(1)}: {m.group(2).strip()}")
    return devices


def parse_help_features(help_text: str) -> dict[str, Any]:
    return {
        "flash_attn_value": bool(
            re.search(r"--flash-attn\s+\[?on\|off\|auto|-fa,\s*--flash-attn\s+FA", help_text)
        ),
        "no_webui": "--no-webui" in help_text,
        "metrics": "--metrics" in help_text,
        "alias": "--alias" in help_text,
        "load_mode": "--load-mode" in help_text,
        "help_available": bool(help_text.strip()),
    }


@lru_cache(maxsize=16)
def _probe(
    argv: tuple[str, ...], cache_key: str
) -> tuple[str | None, tuple[str, ...], tuple[tuple[str, Any], ...]]:
    """Version, devices and help-derived features. ``argv`` runs llama-server (natively or
    via ``docker run``); results are cached per ``cache_key`` (binary mtime or image)."""
    code, version_out = _run(argv, "--version")
    blocked = code is not None and is_code_integrity_block(code)
    devices = [] if blocked else parse_devices(_run(argv, "--list-devices")[1])
    features = parse_help_features("" if blocked else _run(argv, "--help")[1])
    features["blocked_by_os_policy"] = blocked
    return parse_version(version_out), tuple(devices), tuple(features.items())


class LlamaCppAdapter(RuntimeAdapter):
    name = "llamacpp"
    display_name = "llama.cpp"
    support_level = SupportLevel.TESTED
    platforms = ("Windows", "Linux", "Darwin")
    weight_formats = ("gguf",)
    homepage = "https://github.com/ggml-org/llama.cpp"
    executable_names = ("llama-server", "llama-server.exe")

    # --- discovery ---------------------------------------------------------------------

    def detect(self, settings: RuntimeSettings) -> tuple[Installation | None, list[str]]:
        if settings.mode == "docker":
            return self._detect_docker(settings)
        notes: list[str] = []
        candidates: list[tuple[str, str]] = []
        if settings.executable:
            if Path(settings.executable).is_file():
                candidates.append((settings.executable, "settings"))
            else:
                notes.append(f"Configured executable not found: {settings.executable}")
        candidates += [(str(p), "managed") for p in installed_servers()]
        if on_path := shutil.which("llama-server"):
            candidates.append((on_path, "path"))
        if not candidates:
            notes.append(
                "llama-server not found. Install it with `inferenceopt runtimes install llamacpp`, "
                "set its path, or use docker mode."
            )
            return None, notes

        exe, source = candidates[0]
        version, devices, features = _probe((exe,), str(Path(exe).stat().st_mtime))
        feature_map = dict(features)
        if feature_map.get("blocked_by_os_policy"):
            notes.append(CODE_INTEGRITY_HINT)
        elif not feature_map.get("help_available"):
            notes.append(f"{exe} did not respond to --help; it may be broken or blocked.")
        elif not devices:
            notes.append("This build reports no GPU devices; it will run on the CPU only.")
        return (
            Installation(
                executable=exe,
                version=version,
                source=source,
                devices=list(devices),
                features=feature_map,
            ),
            notes,
        )

    def _detect_docker(self, settings: RuntimeSettings) -> tuple[Installation | None, list[str]]:
        docker = docker_executable()
        if docker is None:
            return None, ["Docker mode selected, but docker is not on PATH."]
        image = settings.image or DEFAULT_IMAGE
        notes = [] if settings.image else [f"No image set; using {DEFAULT_IMAGE}. Pin a build tag."]
        present, _ = image_info(image)
        if not present:
            return None, [*notes, f"Image {image} is not available locally (docker pull {image})."]
        probe_argv = (docker, "run", "--rm", "--gpus", "all", image)
        version, devices, features = _probe(probe_argv, image)
        if not devices:
            notes.append(
                "The container reports no GPU devices. Check that Docker Desktop has GPU "
                "support (WSL2 backend) and that the NVIDIA driver is installed."
            )
        return (
            Installation(
                executable=image,
                version=version,
                mode="docker",
                source="docker",
                devices=list(devices),
                features=dict(features),
            ),
            notes,
        )

    # --- parameters ----------------------------------------------------------------------

    def parameters(self, install: Installation | None) -> list[ParamSpec]:
        has_gpu = bool(install and install.devices)
        return [
            ParamSpec(name="ctx_size", type="int", minimum=256, maximum=1_048_576, default=4096,
                      description="Total context window (-c). Shared by all parallel slots."),
            ParamSpec(name="n_gpu_layers", type="int", minimum=0, maximum=999,
                      default=999 if has_gpu else 0,
                      description="Layers offloaded to the GPU (-ngl). 999 = all, 0 = CPU only."),
            ParamSpec(name="flash_attn", type="choice", choices=["auto", "on", "off"],
                      default="auto", description="Flash attention (-fa)."),
            ParamSpec(name="cache_type_k", type="choice", choices=CACHE_TYPES, default="f16",
                      description="KV cache type for K (-ctk)."),
            ParamSpec(name="cache_type_v", type="choice", choices=CACHE_TYPES, default="f16",
                      description="KV cache type for V (-ctv). Quantized V needs flash attention."),
            ParamSpec(name="batch_size", type="int", minimum=1, maximum=65536, default=2048,
                      description="Logical batch size (-b)."),
            ParamSpec(name="ubatch_size", type="int", minimum=1, maximum=65536, default=512,
                      description="Physical batch size (-ub); sizes compute buffers."),
            ParamSpec(name="parallel", type="int", minimum=1, maximum=256, default=1,
                      description="Concurrent request slots (-np). Each gets ctx_size/parallel."),
            ParamSpec(name="threads", type="int", minimum=1, maximum=1024,
                      description="CPU threads (-t). Default: runtime decides."),
            ParamSpec(name="cont_batching", type="bool", default=True,
                      description="Continuous batching."),
            ParamSpec(name="load_mode", type="choice", choices=LOAD_MODES, default="auto",
                      description="How weights are loaded (--load-mode; older builds: "
                                  "--no-mmap/--mlock)."),
        ]  # fmt: skip

    def cross_validate(
        self,
        config: dict[str, Any],
        model: ModelReference,
        hardware: HardwareProfile | None,
        install: Installation | None,
    ) -> list[str]:
        cfg = self.resolve(config, install)
        errors = []
        if cfg["cache_type_v"] not in _UNQUANTIZED and cfg["flash_attn"] == "off":
            errors.append("a quantized V cache (cache_type_v) requires flash_attn auto or on")
        if cfg["ctx_size"] % cfg["parallel"]:
            errors.append("ctx_size must be divisible by parallel (each slot gets an equal share)")
        if cfg["ubatch_size"] > cfg["batch_size"]:
            errors.append("ubatch_size cannot exceed batch_size")
        if (
            model.context_length_max
            and cfg["ctx_size"] // cfg["parallel"] > model.context_length_max
        ):
            errors.append(
                f"per-slot context {cfg['ctx_size'] // cfg['parallel']:,} exceeds the model's "
                f"trained maximum of {model.context_length_max:,}"
            )
        if cfg["n_gpu_layers"] > 0 and install is not None and not install.devices:
            errors.append("n_gpu_layers > 0 but this llama.cpp build has no GPU backend")
        if (
            cfg["n_gpu_layers"] > 0
            and hardware is not None
            and not any(d.backend != DeviceBackend.CPU for d in hardware.devices)
        ):
            errors.append("n_gpu_layers > 0 but no GPU was detected on this machine")
        if not model.path:
            errors.append("llama.cpp needs a local GGUF file (model has no path)")
        if (
            install is not None
            and install.features.get("help_available")
            and not install.features.get("load_mode")
            and cfg["load_mode"] not in _LEGACY_LOAD_FLAGS
        ):
            errors.append(f"load_mode {cfg['load_mode']!r} needs a newer llama.cpp build")
        return errors

    # --- launch --------------------------------------------------------------------------

    def build_launch(
        self,
        install: Installation,
        model: ModelReference,
        config: dict[str, Any],
        *,
        port: int,
        host: str = "127.0.0.1",
    ) -> LaunchSpec:
        cfg = self.resolve(config, install)
        features = install.features
        served = re.sub(r"[^A-Za-z0-9._-]+", "-", model.id)
        container = None
        if install.mode == "docker":
            prefix, container, model_arg = docker_run_argv(
                image=install.executable,
                host_port=port,
                container_port=CONTAINER_PORT,
                model_path=Path(str(model.path)),
                gpus="all" if cfg["n_gpu_layers"] > 0 else None,
            )
            argv = [
                *prefix,
                "--model",
                model_arg,
                "--host",
                CONTAINER_BIND_ALL,
                "--port",
                str(CONTAINER_PORT),
            ]
        else:
            argv = [
                install.executable,
                "--model",
                str(model.path),
                "--host",
                host,
                "--port",
                str(port),
            ]
        argv += [
            "--ctx-size", str(cfg["ctx_size"]),
            "--n-gpu-layers", str(cfg["n_gpu_layers"]),
            "--cache-type-k", cfg["cache_type_k"],
            "--cache-type-v", cfg["cache_type_v"],
            "--batch-size", str(cfg["batch_size"]),
            "--ubatch-size", str(cfg["ubatch_size"]),
            "--parallel", str(cfg["parallel"]),
        ]  # fmt: skip
        if features.get("flash_attn_value", True):
            argv += ["--flash-attn", cfg["flash_attn"]]
        elif cfg["flash_attn"] == "on":
            argv.append("--flash-attn")  # older builds: boolean flag
        if cfg.get("threads"):
            argv += ["--threads", str(cfg["threads"])]
        argv.append("--cont-batching" if cfg["cont_batching"] else "--no-cont-batching")
        if cfg["load_mode"] != "auto":
            if features.get("load_mode", True):
                argv += ["--load-mode", cfg["load_mode"]]
            else:
                argv += _LEGACY_LOAD_FLAGS[cfg["load_mode"]]
        if features.get("alias", True):
            argv += ["--alias", served]
        if features.get("metrics", True):
            argv.append("--metrics")
        if features.get("no_webui", True):
            argv.append("--no-webui")
        return LaunchSpec(
            argv=argv,
            host=host,
            port=port,
            health_path="/health",
            api_base=f"http://{host}:{port}/v1",
            served_model_name=served,
            startup_timeout_s=180,
            container_name=container,
        )

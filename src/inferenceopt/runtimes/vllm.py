# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter (``vllm serve``). Support level: experimental (not yet validated on hardware).

Native mode needs Linux (or WSL2) with vLLM installed in a Python environment. Docker mode
uses an image such as ``vllm/vllm-openai:<version>`` and also works from Docker Desktop on
Windows with GPU support enabled.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
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
from inferenceopt.schemas import DeviceBackend, HardwareProfile, ModelReference, SupportLevel

CONTAINER_PORT = 8000


def _venv_executable(env: Path) -> Path:
    return env / ("Scripts/vllm.exe" if platform.system() == "Windows" else "bin/vllm")


class VLLMAdapter(RuntimeAdapter):
    name = "vllm"
    display_name = "vLLM"
    support_level = SupportLevel.EXPERIMENTAL
    platforms = ("Linux",)
    weight_formats = ("safetensors", "pytorch")
    homepage = "https://github.com/vllm-project/vllm"
    executable_names = ("vllm", "vllm.exe")

    def detect(self, settings: RuntimeSettings) -> tuple[Installation | None, list[str]]:
        notes = [
            "Experimental: launch arguments are contract-tested but not yet validated on a GPU."
        ]
        if settings.mode == "docker":
            if docker_executable() is None:
                return None, [*notes, "Docker mode selected, but docker is not on PATH."]
            image = settings.image or "vllm/vllm-openai:latest"
            if settings.image is None:
                notes.append("No image set; using vllm/vllm-openai:latest. Pin a version tag.")
            present, version = image_info(image)
            if not present:
                notes.append(f"Image {image} is not pulled yet (docker pull {image}).")
            return Installation(
                executable=image, version=version, mode="docker", source="docker"
            ), notes

        if platform.system() not in self.platforms:
            notes.append(
                f"vLLM does not run natively on {platform.system()}; use WSL2 or docker mode."
            )
        exe: str | None = None
        if settings.executable:
            exe = settings.executable if Path(settings.executable).is_file() else None
        elif settings.python_env:
            candidate = _venv_executable(Path(settings.python_env))
            exe = str(candidate) if candidate.is_file() else None
        else:
            exe = shutil.which("vllm")
        if exe is None:
            return None, [*notes, "vllm executable not found; set python_env or use docker mode."]
        version = None
        try:
            out = subprocess.run(  # noqa: S603
                [exe, "--version"], capture_output=True, text=True, timeout=60, check=False
            ).stdout
            m = re.search(r"\d+\.\d+\.\d+\S*", out)
            version = m.group(0) if m else None
        except (OSError, subprocess.TimeoutExpired):
            notes.append("`vllm --version` failed.")
        source = "settings" if settings.executable or settings.python_env else "path"
        return Installation(executable=exe, version=version, source=source), notes

    def parameters(self, install: Installation | None) -> list[ParamSpec]:
        return [
            ParamSpec(name="max_model_len", type="int", minimum=128, maximum=10_000_000,
                      description="Maximum context length. Default: the model's maximum."),
            ParamSpec(name="gpu_memory_utilization", type="float", minimum=0.1, maximum=0.99,
                      default=0.9, description="Fraction of GPU memory vLLM may use."),
            ParamSpec(name="max_num_seqs", type="int", minimum=1, maximum=4096,
                      description="Maximum concurrent sequences."),
            ParamSpec(name="max_num_batched_tokens", type="int", minimum=16, maximum=1_048_576,
                      description="Tokens per scheduler step."),
            ParamSpec(name="kv_cache_dtype", type="choice", default="auto",
                      choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
                      description="KV cache dtype."),
            ParamSpec(name="dtype", type="choice", default="auto",
                      choices=["auto", "half", "float16", "bfloat16", "float32"],
                      description="Weight/activation dtype."),
            ParamSpec(name="quantization", type="str",
                      description="Quantization method (awq, gptq, fp8, ...). Default: model."),
            ParamSpec(name="enable_prefix_caching", type="bool",
                      description="Automatic prefix caching. Default: runtime decides."),
            ParamSpec(name="tensor_parallel_size", type="int", minimum=1, maximum=64, default=1,
                      description="GPUs to shard the model across."),
            ParamSpec(name="enforce_eager", type="bool", default=False,
                      description="Disable CUDA graphs (less memory, usually slower)."),
            ParamSpec(name="trust_remote_code", type="bool", default=False, tunable=False,
                      description="Run the model repository's custom code. Review it first."),
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
        if not model.path:
            errors.append("vLLM adapter needs a local model directory (downloads are not enabled)")
        if model.requires_custom_code and not cfg.get("trust_remote_code"):
            errors.append("model requires custom code; set trust_remote_code after reviewing it")
        if hardware is not None:
            gpus = [
                d for d in hardware.devices if d.backend in (DeviceBackend.CUDA, DeviceBackend.ROCM)
            ]
            if not gpus:
                errors.append("vLLM needs an NVIDIA or AMD GPU; none detected")
            elif cfg["tensor_parallel_size"] > len(gpus):
                errors.append(
                    f"tensor_parallel_size {cfg['tensor_parallel_size']} > {len(gpus)} GPU(s)"
                )
        if (
            cfg.get("max_model_len")
            and model.context_length_max
            and cfg["max_model_len"] > model.context_length_max
        ):
            errors.append("max_model_len exceeds the model's trained maximum")
        return errors

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
        served = re.sub(r"[^A-Za-z0-9._-]+", "-", model.id)
        container = None
        if install.mode == "docker":
            prefix, container, model_arg = docker_run_argv(
                image=install.executable,
                host_port=port,
                container_port=CONTAINER_PORT,
                model_path=Path(str(model.path)),
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
                "serve",
                str(model.path),
                "--host",
                host,
                "--port",
                str(port),
            ]
        argv += ["--served-model-name", served]
        flags = {
            "max_model_len": "--max-model-len",
            "gpu_memory_utilization": "--gpu-memory-utilization",
            "max_num_seqs": "--max-num-seqs",
            "max_num_batched_tokens": "--max-num-batched-tokens",
            "kv_cache_dtype": "--kv-cache-dtype",
            "dtype": "--dtype",
            "quantization": "--quantization",
            "tensor_parallel_size": "--tensor-parallel-size",
        }
        for key, flag in flags.items():
            if cfg.get(key) is not None:
                argv += [flag, str(cfg[key])]
        if cfg.get("enable_prefix_caching") is True:
            argv.append("--enable-prefix-caching")
        elif cfg.get("enable_prefix_caching") is False:
            argv.append("--no-enable-prefix-caching")
        if cfg.get("enforce_eager"):
            argv.append("--enforce-eager")
        if cfg.get("trust_remote_code"):
            argv.append("--trust-remote-code")
        return LaunchSpec(
            argv=argv,
            host=host,
            port=port,
            health_path="/health",
            api_base=f"http://{host}:{port}/v1",
            served_model_name=served,
            startup_timeout_s=900,  # first start compiles kernels and captures CUDA graphs
            container_name=container,
        )

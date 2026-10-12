# SPDX-License-Identifier: Apache-2.0
"""TensorRT-LLM adapter (``trtllm-serve``). Support level: experimental.

Native mode needs Linux with TensorRT-LLM installed; docker mode needs an NVIDIA
TensorRT-LLM release image that you choose and pin (no default is assumed).
"""

from __future__ import annotations

import platform
import re
import shutil
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


class TensorRTLLMAdapter(RuntimeAdapter):
    name = "tensorrt_llm"
    display_name = "TensorRT-LLM"
    support_level = SupportLevel.EXPERIMENTAL
    platforms = ("Linux",)
    weight_formats = ("safetensors", "tensorrt_engine")
    homepage = "https://github.com/NVIDIA/TensorRT-LLM"
    executable_names = ("trtllm-serve",)

    def detect(self, settings: RuntimeSettings) -> tuple[Installation | None, list[str]]:
        notes = [
            "Experimental: launch arguments are contract-tested but not yet validated on a GPU."
        ]
        if settings.mode == "docker":
            if docker_executable() is None:
                return None, [*notes, "Docker mode selected, but docker is not on PATH."]
            if not settings.image:
                return None, [*notes, "Set a TensorRT-LLM release image (e.g. from nvcr.io)."]
            present, version = image_info(settings.image)
            if not present:
                notes.append(f"Image {settings.image} is not pulled yet.")
            return (
                Installation(
                    executable=settings.image, version=version, mode="docker", source="docker"
                ),
                notes,
            )
        if platform.system() not in self.platforms:
            notes.append(
                f"TensorRT-LLM does not run natively on {platform.system()}; use docker mode."
            )
        exe = (
            settings.executable
            if settings.executable and Path(settings.executable).is_file()
            else None
        )
        exe = exe or shutil.which("trtllm-serve")
        if exe is None:
            return None, [*notes, "trtllm-serve not found; set its path or use docker mode."]
        return Installation(
            executable=exe, source="settings" if settings.executable else "path"
        ), notes

    def parameters(self, install: Installation | None) -> list[ParamSpec]:
        return [
            ParamSpec(name="max_batch_size", type="int", minimum=1, maximum=4096,
                      description="Maximum requests per batch. Default: runtime decides."),
            ParamSpec(name="max_num_tokens", type="int", minimum=16, maximum=1_048_576,
                      description="Maximum tokens per batch. Default: runtime decides."),
            ParamSpec(name="max_seq_len", type="int", minimum=128, maximum=10_000_000,
                      description="Maximum sequence length. Default: from the model."),
            ParamSpec(name="tp_size", type="int", minimum=1, maximum=64, default=1,
                      description="Tensor parallel size (GPUs)."),
            ParamSpec(name="kv_cache_free_gpu_memory_fraction", type="float", minimum=0.05,
                      maximum=0.99, default=0.9, description="Free GPU memory used for KV cache."),
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
            errors.append("TensorRT-LLM adapter needs a local model directory")
        if model.requires_custom_code and not cfg.get("trust_remote_code"):
            errors.append("model requires custom code; set trust_remote_code after reviewing it")
        if hardware is not None:
            gpus = [d for d in hardware.devices if d.backend == DeviceBackend.CUDA]
            if not gpus:
                errors.append("TensorRT-LLM needs an NVIDIA GPU; none detected")
            elif cfg["tp_size"] > len(gpus):
                errors.append(f"tp_size {cfg['tp_size']} > {len(gpus)} GPU(s)")
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
                "trtllm-serve",
                model_arg,
                "--host",
                CONTAINER_BIND_ALL,
                "--port",
                str(CONTAINER_PORT),
            ]
        else:
            argv = [install.executable, str(model.path), "--host", host, "--port", str(port)]
        for key in ("max_batch_size", "max_num_tokens", "max_seq_len", "tp_size",
                    "kv_cache_free_gpu_memory_fraction"):  # fmt: skip
            if cfg.get(key) is not None:
                argv += [f"--{key}", str(cfg[key])]
        if cfg.get("trust_remote_code"):
            argv.append("--trust_remote_code")
        return LaunchSpec(
            argv=argv,
            host=host,
            port=port,
            health_path="/health",
            api_base=f"http://{host}:{port}/v1",
            served_model_name=served,
            startup_timeout_s=1800,  # engine build on first start can take a long time
            container_name=container,
        )

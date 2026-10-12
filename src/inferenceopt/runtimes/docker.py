# SPDX-License-Identifier: Apache-2.0
"""Helpers for runtimes launched as Docker containers.

Containers publish their port on 127.0.0.1 only, mount the model read-only, and are
removed when stopped. The image is user-chosen; pin it by tag or digest.
"""

from __future__ import annotations

import json
import secrets
import shutil
import subprocess
from pathlib import Path

CONTAINER_MODEL_DIR = "/models/model"
# Inside the container the server must listen on all interfaces; the host side is
# published on 127.0.0.1 only (see docker_run_argv).
CONTAINER_BIND_ALL = "0.0.0.0"  # noqa: S104


def docker_executable() -> str | None:
    return shutil.which("docker")


def image_info(image: str) -> tuple[bool, str | None]:
    """(available locally, version label if any). Never pulls."""
    docker = docker_executable()
    if docker is None:
        return False, None
    try:
        r = subprocess.run(  # noqa: S603
            [docker, "image", "inspect", image], capture_output=True, text=True, timeout=20,
            check=False,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return False, None
    if r.returncode != 0:
        return False, None
    try:
        labels = json.loads(r.stdout)[0].get("Config", {}).get("Labels") or {}
    except (ValueError, IndexError, AttributeError):
        labels = {}
    version = labels.get("org.opencontainers.image.version") or image.rpartition(":")[2] or None
    return True, version


def docker_run_argv(
    *,
    image: str,
    host_port: int,
    container_port: int,
    model_path: Path,
    gpus: str | None = "all",
    env: dict[str, str] | None = None,
    extra_docker_args: list[str] | None = None,
) -> tuple[list[str], str, str]:
    """Return (argv prefix up to and including the image, container name, model path inside)."""
    docker = docker_executable() or "docker"
    name = f"inferenceopt-{secrets.token_hex(4)}"
    mount_src = model_path if model_path.is_dir() else model_path.parent
    inside = (
        CONTAINER_MODEL_DIR if model_path.is_dir() else f"{CONTAINER_MODEL_DIR}/{model_path.name}"
    )
    argv = [
        docker, "run", "--rm", "--name", name,
        "-p", f"127.0.0.1:{host_port}:{container_port}",
        "-v", f"{mount_src}:{CONTAINER_MODEL_DIR}:ro",
        "--ipc=host",
    ]  # fmt: skip
    if gpus:
        argv += ["--gpus", gpus]
    for key, value in (env or {}).items():
        argv += ["-e", f"{key}={value}"]
    argv += extra_docker_args or []
    argv.append(image)
    return argv, name, inside

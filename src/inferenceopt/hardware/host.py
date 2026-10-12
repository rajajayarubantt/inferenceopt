# SPDX-License-Identifier: Apache-2.0
"""Host OS, CPU and RAM detection using only the standard library and psutil."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import psutil

from inferenceopt.hardware.base import DetectionResult
from inferenceopt.schemas import Device, DeviceBackend, HostInfo, Provenance

# CPU features that matter to inference runtimes (SIMD and matrix extensions).
_INTERESTING = {
    "sse3", "ssse3", "sse4_1", "sse4_2", "avx", "avx2", "fma", "f16c",
    "avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512_bf16", "avx_vnni",
    "amx_tile", "amx_int8", "amx_bf16",
    "neon", "dotprod", "i8mm", "bf16", "sve", "sve2", "sme",
}  # fmt: skip
_ALIASES = {
    "pni": "sse3",
    "asimd": "neon",
    "asimddp": "dotprod",
    "avx512vnni": "avx512_vnni",
    "avxvnni": "avx_vnni",
    "fma3": "fma",
    "sse4.1": "sse4_1",
    "sse4.2": "sse4_2",
    "feat_dotprod": "dotprod",
    "feat_i8mm": "i8mm",
    "feat_bf16": "bf16",
    "feat_sme": "sme",
    "adv_simd": "neon",
    "advsimd": "neon",
}

# Windows IsProcessorFeaturePresent() feature ids.
_WIN_FEATURES = {
    13: "sse3",
    36: "ssse3",
    37: "sse4_1",
    38: "sse4_2",
    39: "avx",
    40: "avx2",
    41: "avx512f",
    19: "neon",  # PF_ARM_NEON_INSTRUCTIONS_AVAILABLE
    43: "dotprod",  # PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
}


def normalize_flags(raw: list[str]) -> list[str]:
    out = set()
    for flag in raw:
        name = flag.strip().lower()
        name = _ALIASES.get(name, name)
        if name in _INTERESTING:
            out.add(name)
    return sorted(out)


def _run(*args: str) -> str | None:
    exe = shutil.which(args[0])
    if exe is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 - fixed argument list, no shell
            [exe, *args[1:]], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _linux_cpu() -> tuple[str | None, list[str]]:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, []
    model = re.search(r"^(?:model name|Model)\s*:\s*(.+)$", text, re.M)
    flags = re.search(r"^(?:flags|Features)\s*:\s*(.+)$", text, re.M)
    return (
        model.group(1).strip() if model else None,
        flags.group(1).split() if flags else [],
    )


def _mac_cpu() -> tuple[str | None, list[str]]:
    model = _run("sysctl", "-n", "machdep.cpu.brand_string")
    flags: list[str] = []
    for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
        out = _run("sysctl", "-n", key)
        if out:
            flags += out.split()
    optional = _run("sysctl", "hw.optional")
    if optional:
        for line in optional.splitlines():
            key, _, value = line.partition(":")
            if value.strip() == "1":
                flags.append(key.rsplit(".", 1)[-1])
    if platform.machine() == "arm64":
        flags.append("neon")  # mandatory on Apple Silicon
    return (model.strip() if model else None), flags


def _windows_cpu() -> tuple[str | None, list[str]]:
    model = None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            model = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except OSError:
        pass
    flags: list[str] = []
    try:
        import ctypes

        present = getattr(ctypes, "windll").kernel32.IsProcessorFeaturePresent  # noqa: B009
        flags = [name for fid, name in _WIN_FEATURES.items() if present(fid)]
    except (AttributeError, OSError):
        pass
    return model, flags


def detect_host() -> tuple[HostInfo, DetectionResult]:
    """Detect the host and return it plus a CPU pseudo-device for host-memory runtimes."""
    result = DetectionResult()
    if sys.platform.startswith("linux"):
        cpu_model, raw_flags = _linux_cpu()
    elif sys.platform == "darwin":
        cpu_model, raw_flags = _mac_cpu()
    elif sys.platform == "win32":
        cpu_model, raw_flags = _windows_cpu()
        result.notes.append(
            "CPU flags on Windows come from IsProcessorFeaturePresent, which does not "
            "report FMA, F16C, AVX-VNNI or AMX."
        )
    else:
        cpu_model, raw_flags = None, []
    flags = normalize_flags(raw_flags)
    if not raw_flags:
        result.notes.append("Could not read CPU feature flags on this platform.")

    vm = psutil.virtual_memory()
    host = HostInfo(
        os=platform.system(),
        os_version=platform.release(),
        python_version=platform.python_version(),
        cpu_arch=platform.machine(),
        cpu_model=cpu_model or (platform.processor() or None),
        cpu_flags=flags,
        logical_cores=psutil.cpu_count(logical=True),
        physical_cores=psutil.cpu_count(logical=False),
        ram_total_bytes=vm.total,
        ram_available_bytes=vm.available,
    )
    result.devices.append(
        Device(
            device_id="cpu0",
            backend=DeviceBackend.CPU,
            name=host.cpu_model or host.cpu_arch,
            architecture=host.cpu_arch,
            memory_total_bytes=vm.total,
            memory_free_bytes=vm.available,
            supported_dtypes=["f32", "f16", "bf16", "int8"],
            field_sources={"supported_dtypes": Provenance.INFERRED},
        )
    )
    return host, result

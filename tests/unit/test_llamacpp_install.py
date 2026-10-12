# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from inferenceopt.runtimes.llamacpp.install import (
    Asset,
    InstallError,
    _download,
    extract_archive,
    plan_install,
)

pytestmark = pytest.mark.unit

TAG = "b11429"


def asset(name: str, sha: str | None = "0" * 64) -> Asset:
    return Asset(name=name, url=f"https://example.invalid/{name}", size=100, sha256=sha)


ASSETS = [
    asset(f"llama-{TAG}-bin-win-cpu-x64.zip"),
    asset(f"llama-{TAG}-bin-win-cuda-12.4-x64.zip"),
    asset(f"llama-{TAG}-bin-win-cuda-13.4-x64.zip"),
    asset("cudart-llama-bin-win-cuda-12.4-x64.zip"),
    asset("cudart-llama-bin-win-cuda-13.4-x64.zip"),
    asset(f"llama-{TAG}-bin-win-vulkan-x64.zip"),
    asset(f"llama-{TAG}-bin-ubuntu-x64.tar.gz"),
    asset(f"llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"),
    asset(f"cudart-llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"),
    asset(f"llama-{TAG}-bin-macos-arm64.tar.gz"),
]


@pytest.mark.parametrize(
    ("system", "machine", "driver", "variant", "variant_out", "names"),
    [
        ("Windows", "AMD64", "13.3", "auto", "cuda-12.4",
         [f"llama-{TAG}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]),
        ("Windows", "AMD64", "13.4", "auto", "cuda-13.4",
         [f"llama-{TAG}-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip"]),
        ("Windows", "AMD64", None, "auto", "cpu", [f"llama-{TAG}-bin-win-cpu-x64.zip"]),
        ("Windows", "AMD64", "13.3", "cpu", "cpu", [f"llama-{TAG}-bin-win-cpu-x64.zip"]),
        ("Windows", "AMD64", None, "vulkan", "vulkan", [f"llama-{TAG}-bin-win-vulkan-x64.zip"]),
        ("Linux", "x86_64", "12.9", "auto", "cuda-12.8",
         [f"llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
          f"cudart-llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"]),
        ("Linux", "x86_64", "11.8", "auto", "cpu", [f"llama-{TAG}-bin-ubuntu-x64.tar.gz"]),
        ("Darwin", "arm64", None, "auto", "metal", [f"llama-{TAG}-bin-macos-arm64.tar.gz"]),
    ],
)  # fmt: skip
def test_plan_install(
    system: str, machine: str, driver: str | None, variant: str, variant_out: str, names: list[str]
) -> None:
    plan = plan_install(
        TAG,
        ASSETS,
        system=system,
        machine=machine,
        cuda_driver=driver,
        variant=variant,  # type: ignore[arg-type]
    )
    assert plan.variant == variant_out
    assert [a.name for a in plan.assets] == names


def test_cuda_requested_without_compatible_build() -> None:
    with pytest.raises(InstallError, match="no CUDA build compatible"):
        plan_install(
            TAG, ASSETS, system="Windows", machine="AMD64", cuda_driver="11.0", variant="cuda"
        )
    with pytest.raises(InstallError, match="no NVIDIA driver"):
        plan_install(TAG, ASSETS, system="Windows", machine="AMD64", variant="cuda")


def test_unsupported_platform() -> None:
    with pytest.raises(InstallError):
        plan_install(TAG, ASSETS, system="FreeBSD", machine="x86_64")


def test_download_refuses_assets_without_digest(tmp_path: Path) -> None:
    with pytest.raises(InstallError, match="no published SHA-256"):
        _download(asset("x.zip", sha=None), tmp_path / "x.zip", None)


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def _tar(path: Path, members: list[tarfile.TarInfo], data: bytes = b"x") -> Path:
    with tarfile.open(path, "w:gz") as t:
        for m in members:
            if m.isfile():
                m.size = len(data)
                t.addfile(m, io.BytesIO(data))
            else:
                t.addfile(m)
    return path


def test_extract_zip_ok(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "ok.zip", {"bin/llama-server.exe": b"MZ", "README": b"hi"})
    extract_archive(archive, tmp_path / "out")
    assert (tmp_path / "out" / "bin" / "llama-server.exe").read_bytes() == b"MZ"


@pytest.mark.parametrize("name", ["../evil.dll", "a/../../evil.dll"])
def test_extract_zip_rejects_traversal(tmp_path: Path, name: str) -> None:
    archive = _zip(tmp_path / "bad.zip", {name: b"x"})
    with pytest.raises(InstallError, match="escapes"):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "evil.dll").exists()


def test_extract_tar_rejects_escaping_symlink(tmp_path: Path) -> None:
    link = tarfile.TarInfo("lib/libcuda.so")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../../../etc/passwd"
    archive = _tar(tmp_path / "bad.tar.gz", [link])
    with pytest.raises(InstallError, match="escapes"):
        extract_archive(archive, tmp_path / "out")


def test_extract_tar_rejects_devices(tmp_path: Path) -> None:
    dev = tarfile.TarInfo("dev/sda")
    dev.type = tarfile.CHRTYPE
    archive = _tar(tmp_path / "bad.tar.gz", [dev])
    with pytest.raises(InstallError, match="unsupported"):
        extract_archive(archive, tmp_path / "out")


def test_extract_tar_ok(tmp_path: Path) -> None:
    f = tarfile.TarInfo("build/bin/llama-server")
    archive = _tar(tmp_path / "ok.tar.gz", [f], data=b"ELF")
    extract_archive(archive, tmp_path / "out")
    assert (tmp_path / "out" / "build" / "bin" / "llama-server").read_bytes() == b"ELF"


def test_unknown_archive_type(tmp_path: Path) -> None:
    (tmp_path / "x.rar").write_bytes(b"")
    with pytest.raises(InstallError, match="unknown archive"):
        extract_archive(tmp_path / "x.rar", tmp_path / "out")

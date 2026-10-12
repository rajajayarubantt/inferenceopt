# SPDX-License-Identifier: Apache-2.0
"""Download official llama.cpp release builds (explicit opt-in only).

* Assets come from github.com/ggml-org/llama.cpp releases.
* Every download is verified against the SHA-256 digest GitHub publishes for the asset;
  an asset without a digest is refused.
* Archives are extracted with path-traversal, link and size checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import tarfile
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from inferenceopt.config import data_dir
from inferenceopt.runtimes.base import RuntimeAdapterError
from inferenceopt.schemas.base import utcnow

REPO_API = "https://api.github.com/repos/ggml-org/llama.cpp"
MAX_EXTRACTED_BYTES = 8 * 1024**3
Variant = Literal["auto", "cpu", "cuda", "vulkan"]
Progress = Callable[[str, int, int | None], None]


class InstallError(RuntimeAdapterError):
    pass


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int
    sha256: str | None


@dataclass(frozen=True)
class InstallPlan:
    tag: str
    variant: str
    assets: tuple[Asset, ...]

    @property
    def total_bytes(self) -> int:
        return sum(a.size for a in self.assets)


def managed_root() -> Path:
    return data_dir() / "runtimes" / "llamacpp"


def _client() -> httpx.Client:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "inferenceopt"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(headers=headers, timeout=30, follow_redirects=True)


def _assets(release: dict[str, Any]) -> list[Asset]:
    out = []
    for a in release.get("assets", []):
        digest = a.get("digest") or ""
        out.append(
            Asset(
                name=a["name"],
                url=a["browser_download_url"],
                size=int(a.get("size", 0)),
                sha256=digest.removeprefix("sha256:") if digest.startswith("sha256:") else None,
            )
        )
    return out


def resolve_release(channel: str = "stable") -> tuple[str, list[Asset]]:
    """``stable`` (the latest versioned release's build), ``nightly`` or an exact tag."""
    with _client() as c:
        if channel == "stable":
            latest = c.get(f"{REPO_API}/releases/latest").raise_for_status().json()
            assets = _assets(latest)
            pointer = next((a for a in assets if a.name == "nightly-tag.txt"), None)
            if pointer is None and any(a.name.startswith("llama-") for a in assets):
                return latest["tag_name"], assets
            if pointer is None:
                raise InstallError("latest release has neither binaries nor a build pointer")
            tag = c.get(pointer.url).raise_for_status().text.strip()
        elif channel == "nightly":
            releases = c.get(f"{REPO_API}/releases", params={"per_page": 10}).raise_for_status()
            tag = next(
                r["tag_name"]
                for r in releases.json()
                if any(a["name"].startswith("llama-b") for a in r.get("assets", []))
            )
        else:
            tag = channel
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", tag):
            raise InstallError(f"unexpected release tag {tag!r}")
        release = c.get(f"{REPO_API}/releases/tags/{tag}").raise_for_status().json()
    return tag, _assets(release)


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v))


def plan_install(
    tag: str,
    assets: list[Asset],
    *,
    variant: Variant = "auto",
    system: str | None = None,
    machine: str | None = None,
    cuda_driver: str | None = None,
) -> InstallPlan:
    """Pick the archive(s) for this platform. ``cuda_driver`` is e.g. ``"13.3"``."""
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    by_name = {a.name: a for a in assets}

    if system == "Darwin":
        name = f"llama-{tag}-bin-macos-{arch}.tar.gz"
        if name not in by_name:
            raise InstallError(f"no macOS {arch} build in {tag}")
        return InstallPlan(tag, "metal", (by_name[name],))

    os_key, ext = {"Windows": ("win", "zip"), "Linux": ("ubuntu", "tar.gz")}.get(system, ("", ""))
    if not os_key:
        raise InstallError(f"no official llama.cpp builds for {system}")

    if variant in ("auto", "cuda") and cuda_driver:
        pattern = re.compile(
            rf"llama-{re.escape(tag)}-bin-{os_key}-cuda-([\d.]+)-{arch}\.{re.escape(ext)}"
        )
        driver = _version_tuple(cuda_driver)
        options = sorted(
            ((m.group(1), a) for a in assets if (m := pattern.fullmatch(a.name))),
            key=lambda x: _version_tuple(x[0]),
        )
        usable = [(v, a) for v, a in options if _version_tuple(v)[:2] <= driver[:2]]
        if usable:
            version, main = usable[-1]
            runtime = next(
                (
                    a
                    for a in assets
                    if a.name.startswith("cudart-")
                    and f"-{os_key}-cuda-{version}-{arch}." in a.name
                ),
                None,
            )
            parts = (main,) + ((runtime,) if runtime else ())
            return InstallPlan(tag, f"cuda-{version}", parts)
        if variant == "cuda":
            found = ", ".join(v for v, _ in options) or "none"
            raise InstallError(
                f"no CUDA build compatible with driver CUDA {cuda_driver} (available: {found})"
            )
    elif variant == "cuda":
        raise InstallError("CUDA build requested, but no NVIDIA driver was detected")

    if variant == "vulkan":
        name = f"llama-{tag}-bin-{os_key}-vulkan-{arch}.{ext}"
    elif system == "Windows":
        name = f"llama-{tag}-bin-win-cpu-{arch}.zip"
    else:
        name = f"llama-{tag}-bin-ubuntu-{arch}.tar.gz"
    if name not in by_name:
        raise InstallError(f"{name} is not in release {tag}")
    return InstallPlan(tag, "vulkan" if variant == "vulkan" else "cpu", (by_name[name],))


def _download(asset: Asset, dest: Path, progress: Progress | None) -> None:
    if not asset.sha256:
        raise InstallError(f"{asset.name} has no published SHA-256 digest; refusing to install")
    digest = hashlib.sha256()
    done = 0
    with _client() as c, c.stream("GET", asset.url, timeout=None) as r, dest.open("wb") as f:
        r.raise_for_status()
        for chunk in r.iter_bytes(1024 * 1024):
            f.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if progress:
                progress(asset.name, done, asset.size or None)
    if digest.hexdigest() != asset.sha256:
        raise InstallError(f"{asset.name}: SHA-256 mismatch (download corrupted or tampered)")


def _safe_target(root: Path, member: str) -> Path:
    target = (root / member).resolve()
    if not target.is_relative_to(root.resolve()):
        raise InstallError(f"archive entry escapes the install directory: {member}")
    return target


def extract_archive(archive: Path, dest: Path) -> None:
    """Extract zip or tar.gz, rejecting traversal, absolute paths, links and huge archives."""
    total = 0
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                _safe_target(dest, info.filename)
                total += info.file_size
                if total > MAX_EXTRACTED_BYTES:
                    raise InstallError("archive expands beyond the size limit")
            z.extractall(dest)  # noqa: S202 - every member validated above
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as t:
            members = t.getmembers()
            for m in members:
                _safe_target(dest, m.name)
                if m.issym() or m.islnk():
                    link_target = Path(m.name).parent / m.linkname
                    _safe_target(dest, str(link_target))
                elif not (m.isfile() or m.isdir()):
                    raise InstallError(f"unsupported archive entry type: {m.name}")
                total += m.size
                if total > MAX_EXTRACTED_BYTES:
                    raise InstallError("archive expands beyond the size limit")
            t.extractall(dest, filter="data")
    else:
        raise InstallError(f"unknown archive type: {archive.name}")


def find_server(root: Path) -> Path | None:
    name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    found = sorted(root.rglob(name), key=lambda p: len(p.parts))
    return found[0] if found else None


def install(plan: InstallPlan, progress: Progress | None = None) -> Path:
    """Download, verify and extract. Returns the path to ``llama-server``."""
    root = managed_root()
    dest = root / f"{plan.tag}-{plan.variant}"
    if (existing := find_server(dest)) is not None:
        return existing
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=".download-") as tmp:
        staging = Path(tmp) / "extract"
        staging.mkdir()
        main, *extras = plan.assets
        archive = Path(tmp) / main.name
        _download(main, archive, progress)
        extract_archive(archive, staging)
        server = find_server(staging)
        if server is None:
            raise InstallError(f"{main.name} does not contain llama-server")
        for extra in extras:  # e.g. CUDA runtime libraries: place next to the binary
            extra_archive = Path(tmp) / extra.name
            _download(extra, extra_archive, progress)
            extract_archive(extra_archive, server.parent)
        if platform.system() != "Windows":
            server.chmod(0o755)
        (staging / "inferenceopt-install.json").write_text(
            json.dumps(
                {
                    "tag": plan.tag,
                    "variant": plan.variant,
                    "assets": [{"name": a.name, "sha256": a.sha256} for a in plan.assets],
                    "installed_at": utcnow().isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        shutil.move(str(staging), str(dest))
    result = find_server(dest)
    assert result is not None
    return result


def installed_servers() -> list[Path]:
    root = managed_root()
    if not root.is_dir():
        return []
    servers = [
        s
        for d in sorted(root.iterdir(), reverse=True)
        if d.is_dir() and not d.name.startswith(".") and (s := find_server(d))
    ]
    return servers

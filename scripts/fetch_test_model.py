# SPDX-License-Identifier: Apache-2.0
"""Download the pinned tiny GGUF model used by llama.cpp integration tests.

    uv run python scripts/fetch_test_model.py [DEST_DIR]

Prints the file path. The revision and SHA-256 are pinned; a mismatch is an error.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import httpx

REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"  # Apache-2.0 (HuggingFaceTB/SmolLM2-135M-Instruct)
REVISION = "09816acd5d99df7be770d85ea30822623dab342c"
FILENAME = "SmolLM2-135M-Instruct-Q4_K_M.gguf"
SHA256 = "2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d"
DEFAULT_DIR = Path.home() / ".cache" / "inferenceopt-test"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(dest_dir: Path = DEFAULT_DIR) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / FILENAME
    if path.is_file() and sha256_of(path) == SHA256:
        return path
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}"
    partial = path.with_suffix(".part")
    timeout = httpx.Timeout(30, read=300)
    with (
        httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r,
        partial.open("wb") as f,
    ):
        r.raise_for_status()
        for chunk in r.iter_bytes(1024 * 1024):
            f.write(chunk)
    if sha256_of(partial) != SHA256:
        partial.unlink()
        raise SystemExit(f"SHA-256 mismatch for {FILENAME}; refusing to use it")
    partial.replace(path)
    return path


if __name__ == "__main__":
    print(fetch(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR))

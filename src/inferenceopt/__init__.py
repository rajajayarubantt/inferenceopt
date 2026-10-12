# SPDX-License-Identifier: Apache-2.0
"""InferenceOpt: hardware-aware, measurement-driven LLM inference optimization."""

from __future__ import annotations

try:
    from inferenceopt._version import __version__
except ImportError:  # pragma: no cover - only when running from a tree without a build
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("inferenceopt")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = ["__version__"]

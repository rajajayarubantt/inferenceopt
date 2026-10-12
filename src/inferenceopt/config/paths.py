# SPDX-License-Identifier: Apache-2.0
"""Per-user filesystem locations.

Defaults follow each OS's conventions (via platformdirs). Setting ``INFERENCEOPT_HOME``
puts everything under one directory, which tests and portable installs rely on.
"""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import PlatformDirs

APP_NAME = "inferenceopt"
HOME_ENV = "INFERENCEOPT_HOME"

_dirs = PlatformDirs(APP_NAME, appauthor=False)


def _home_override() -> Path | None:
    value = os.environ.get(HOME_ENV)
    return Path(value).expanduser() if value else None


def config_dir() -> Path:
    home = _home_override()
    return home / "config" if home else Path(_dirs.user_config_dir)


def data_dir() -> Path:
    home = _home_override()
    return home / "data" if home else Path(_dirs.user_data_dir)


def default_config_file() -> Path:
    return config_dir() / "config.yaml"


def default_database_path() -> Path:
    return data_dir() / "inferenceopt.sqlite"

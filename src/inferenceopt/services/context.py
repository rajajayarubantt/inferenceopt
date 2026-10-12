# SPDX-License-Identifier: Apache-2.0
"""The application context passed to every service function."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inferenceopt.config import AppConfig, default_config_file, default_database_path, load_config
from inferenceopt.config.redaction import register_secret
from inferenceopt.logs import configure_logging
from inferenceopt.storage import Database


@dataclass
class AppContext:
    config: AppConfig
    config_path: Path
    db: Database

    @classmethod
    def open(cls, config_path: Path | None = None, *, migrate: bool = True) -> AppContext:
        config = load_config(config_path)
        configure_logging(config.logging.level, json_format=config.logging.json_format)
        register_secret(config.llm.api_key)
        register_secret(config.server.token)
        db = Database(config.storage.database_path or default_database_path())
        if migrate:
            db.migrate()
        return cls(config=config, config_path=config_path or default_config_file(), db=db)

    def close(self) -> None:
        self.db.dispose()

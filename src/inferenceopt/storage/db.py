# SPDX-License-Identifier: Apache-2.0
"""SQLite database access and automatic schema migration."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")  # readers don't block the job worker
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def _alembic_config() -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    return cfg


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(f"sqlite:///{path.as_posix()}")
        event.listen(self.engine, "connect", _sqlite_pragmas)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def migrate(self) -> None:
        """Upgrade the schema to the latest revision. Idempotent."""
        cfg = _alembic_config()
        with self.engine.begin() as conn:
            cfg.attributes["connection"] = conn
            command.upgrade(cfg, "head")
        log.debug("database at %s migrated to %s", self.path, self.current_revision())

    def current_revision(self) -> str | None:
        with self.engine.connect() as conn:
            return MigrationContext.configure(conn).get_current_revision()

    @staticmethod
    def head_revision() -> str | None:
        return ScriptDirectory.from_config(_alembic_config()).get_current_head()

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._sessions() as session, session.begin():
            yield session

    def dispose(self) -> None:
        self.engine.dispose()

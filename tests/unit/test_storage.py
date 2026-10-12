# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect, text

from inferenceopt.schemas.base import utcnow
from inferenceopt.storage import Database
from inferenceopt.storage.orm import JobRow

pytestmark = pytest.mark.unit


def test_migrate_creates_tables_and_is_idempotent(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.sqlite")
    db.migrate()
    db.migrate()
    assert db.current_revision() == Database.head_revision()
    tables = set(inspect(db.engine).get_table_names())
    assert {"jobs", "trials", "hardware_profiles", "planner_calls"} <= tables
    db.dispose()


def test_wal_mode_enabled(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.sqlite")
    with db.engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
    db.dispose()


def test_orm_matches_migration(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.sqlite")
    db.migrate()
    now = utcnow()
    with db.session() as s:
        s.add(
            JobRow(
                id="job_1",
                name="n",
                status="pending",
                created_at=now,
                updated_at=now,
                payload={"a": 1},
            )
        )
    with db.session() as s:
        row = s.get(JobRow, "job_1")
        assert row is not None
        assert row.payload == {"a": 1}
    db.dispose()

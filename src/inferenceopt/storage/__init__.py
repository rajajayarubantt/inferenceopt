# SPDX-License-Identifier: Apache-2.0
"""Persistent storage: SQLite (WAL) via SQLAlchemy, migrated with Alembic."""

from inferenceopt.storage.db import Database

__all__ = ["Database"]

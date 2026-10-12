# SPDX-License-Identifier: Apache-2.0
"""Alembic environment. Migrations always run on a connection passed in by Database."""

from alembic import context

from inferenceopt.storage.orm import Base

connection = context.config.attributes.get("connection")
if connection is None:
    raise RuntimeError("run migrations through inferenceopt.storage.Database.migrate()")

context.configure(
    connection=connection,
    target_metadata=Base.metadata,
    render_as_batch=True,  # SQLite cannot ALTER most things in place
)
with context.begin_transaction():
    context.run_migrations()

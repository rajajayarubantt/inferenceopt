# SPDX-License-Identifier: Apache-2.0
"""Per-runtime settings (executable, python env, container image).

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-13
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runtime_settings",
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("runtime_settings")

# SPDX-License-Identifier: Apache-2.0
"""SQLAlchemy tables.

Each row keeps a few indexed columns for querying plus the full Pydantic document as JSON
in ``payload``. The schema version lives inside the payload, so documents can evolve
without a table migration for every new field.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import JSON, DateTime, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {
        dict[str, Any]: JSON,
        datetime: DateTime(timezone=True),
    }


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    payload: Mapped[dict[str, Any]]


class TrialRow(Base):
    __tablename__ = "trials"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime]
    payload: Mapped[dict[str, Any]]


class HardwareProfileRow(Base):
    __tablename__ = "hardware_profiles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    captured_at: Mapped[datetime] = mapped_column(index=True)
    payload: Mapped[dict[str, Any]]


class PlannerCallRow(Base):
    __tablename__ = "planner_calls"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime]
    payload: Mapped[dict[str, Any]]


class ModelRow(Base):
    """User-added models. Curated models ship as package data and are not stored."""

    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(200), primary_key=True)
    origin: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime]
    payload: Mapped[dict[str, Any]]


class RuntimeSettingsRow(Base):
    """Where to find each runtime (path, python env or image), edited from CLI or UI."""

    __tablename__ = "runtime_settings"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    updated_at: Mapped[datetime]
    payload: Mapped[dict[str, Any]]

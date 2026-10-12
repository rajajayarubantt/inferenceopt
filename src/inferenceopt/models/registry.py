# SPDX-License-Identifier: Apache-2.0
"""Model registry: curated entries shipped with the package plus user-added models."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

from inferenceopt.models.local import inspect_path
from inferenceopt.schemas import ModelReference, Provenance
from inferenceopt.schemas.base import utcnow
from inferenceopt.storage import Database
from inferenceopt.storage.orm import ModelRow

CURATED_FILE = Path(__file__).with_name("curated.yaml")


class ModelNotFoundError(KeyError):
    pass


class ModelExistsError(ValueError):
    pass


@cache
def curated_models() -> tuple[ModelReference, ...]:
    entries: list[dict[str, Any]] = yaml.safe_load(CURATED_FILE.read_text(encoding="utf-8"))
    curated_sources = {
        "architecture": Provenance.USER_SPECIFIED,
        "parameter_count": Provenance.USER_SPECIFIED,
        "context_length_max": Provenance.USER_SPECIFIED,
    }
    return tuple(
        ModelReference(
            source="huggingface",
            origin="curated",
            weight_format="safetensors",
            quantization="bf16",
            supported_runtimes=["vllm", "tensorrt_llm"],
            field_sources=curated_sources,
            **entry,
        )
        for entry in entries
    )


class ModelRegistry:
    def __init__(self, db: Database) -> None:
        self.db = db

    def _user_models(self) -> list[ModelReference]:
        with self.db.session() as s:
            rows = s.query(ModelRow).order_by(ModelRow.created_at).all()
            return [ModelReference.model_validate(r.payload) for r in rows]

    def list(self) -> list[ModelReference]:
        return [*curated_models(), *self._user_models()]

    def get(self, model_id: str) -> ModelReference:
        for model in curated_models():
            if model.id == model_id:
                return model
        with self.db.session() as s:
            row = s.get(ModelRow, model_id)
            if row is None:
                raise ModelNotFoundError(model_id)
            return ModelReference.model_validate(row.payload)

    def add(self, model: ModelReference) -> ModelReference:
        if any(m.id == model.id for m in curated_models()):
            raise ModelExistsError(f"{model.id} is a curated model id")
        with self.db.session() as s:
            if s.get(ModelRow, model.id) is not None:
                raise ModelExistsError(f"a model with id {model.id!r} already exists")
            s.add(
                ModelRow(
                    id=model.id,
                    origin=model.origin,
                    created_at=utcnow(),
                    payload=model.model_dump(mode="json"),
                )
            )
        return model

    def add_local(self, path: Path, model_id: str | None = None) -> ModelReference:
        return self.add(inspect_path(path, model_id))

    def remove(self, model_id: str) -> None:
        if any(m.id == model_id for m in curated_models()):
            raise ModelExistsError("curated models cannot be removed")
        with self.db.session() as s:
            row = s.get(ModelRow, model_id)
            if row is None:
                raise ModelNotFoundError(model_id)
            s.delete(row)

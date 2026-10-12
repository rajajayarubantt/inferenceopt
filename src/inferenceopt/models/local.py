# SPDX-License-Identifier: Apache-2.0
"""Inspect local model files (GGUF file or Hugging Face-style directory) without loading
weights. Only headers and small JSON/YAML files are read."""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any

import yaml

from inferenceopt.models.gguf import ArrayInfo, GGUFError, read_gguf
from inferenceopt.schemas import ModelArchitecture, ModelReference, Provenance
from inferenceopt.schemas.model import WeightFormat

MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024
MAX_JSON_FILE = 16 * 1024 * 1024


class ModelInspectionError(ValueError):
    """The path is not a model InferenceOpt can read."""


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.").lower()[:120] or "model"


def default_local_id(path: Path) -> str:
    return f"local:{_slug(path.stem if path.is_file() else path.name)}"


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, list) and value and all(isinstance(v, int) for v in value):
        return int(max(value))  # per-layer values (e.g. varying KV heads): take the largest
    return None


# --- GGUF -----------------------------------------------------------------------------


def inspect_gguf(path: Path, model_id: str | None = None) -> ModelReference:
    try:
        info = read_gguf(path)
    except (GGUFError, OSError) as exc:
        raise ModelInspectionError(str(exc)) from exc

    arch_name = info.architecture
    notes: list[str] = []
    architecture = None
    context = None
    if arch_name:

        def key(name: str) -> Any:
            return info.get(f"{arch_name}.{name}")

        layers = _int(key("block_count"))
        hidden = _int(key("embedding_length"))
        heads = _int(key("attention.head_count"))
        kv_heads = _int(key("attention.head_count_kv")) or heads
        head_dim = _int(key("attention.key_length")) or (
            hidden // heads if hidden and heads else None
        )
        vocab = _int(key("vocab_size"))
        tokens = info.get("tokenizer.ggml.tokens")
        if vocab is None and isinstance(tokens, ArrayInfo):
            vocab = tokens.length
        elif vocab is None and isinstance(tokens, list):
            vocab = len(tokens)
        context = _int(key("context_length"))
        if layers and hidden and heads and kv_heads and head_dim:
            architecture = ModelArchitecture(
                family=arch_name,
                num_layers=layers,
                hidden_size=hidden,
                num_attention_heads=heads,
                num_kv_heads=kv_heads,
                head_dim=head_dim,
                vocab_size=vocab,
                sliding_window=_int(key("attention.sliding_window")),
            )
        else:
            notes.append(f"GGUF metadata lacks the shape keys for architecture {arch_name!r}.")
        if _int(key("attention.value_length")) not in (None, head_dim):
            notes.append("Key and value head sizes differ; the KV estimate uses the key size.")

    name = info.get("general.name")
    return ModelReference(
        id=model_id or default_local_id(path),
        display_name=name if isinstance(name, str) and name else path.stem,
        source="local",
        origin="user",
        path=str(path.resolve()),
        weight_format="gguf",
        quantization=info.quantization,
        parameter_count=info.parameter_count or None,
        weight_bytes=info.file_size,
        architecture=architecture,
        context_length_max=context,
        license=info.get("general.license")
        if isinstance(info.get("general.license"), str)
        else None,
        supported_runtimes=["llamacpp"],
        field_sources={
            "parameter_count": Provenance.DETECTED,
            "weight_bytes": Provenance.DETECTED,
            "architecture": Provenance.DETECTED,
        },
        notes=notes,
    )


# --- Hugging Face directory -----------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_JSON_FILE:
        raise ModelInspectionError(f"{path.name} is too large to be a config file")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ModelInspectionError(f"{path.name} is not a JSON object")
    return data


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ModelInspectionError(f"{path.name}: truncated safetensors file")
        (n,) = struct.unpack("<Q", raw)
        if n > MAX_SAFETENSORS_HEADER or n > path.stat().st_size - 8:
            raise ModelInspectionError(f"{path.name}: invalid safetensors header length")
        header = json.loads(f.read(n))
    if not isinstance(header, dict):
        raise ModelInspectionError(f"{path.name}: invalid safetensors header")
    return header


def _readme_license(directory: Path) -> str | None:
    readme = directory / "README.md"
    if not readme.is_file() or readme.stat().st_size > MAX_JSON_FILE:
        return None
    text = readme.read_text(encoding="utf-8", errors="replace")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not match:
        return None
    try:
        front = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    lic = front.get("license") if isinstance(front, dict) else None
    return lic if isinstance(lic, str) else None


def inspect_hf_dir(directory: Path, model_id: str | None = None) -> ModelReference:
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise ModelInspectionError(f"{directory} has no config.json")
    config = _read_json(config_path)
    text_cfg = config.get("text_config")
    cfg = text_cfg if isinstance(text_cfg, dict) and "num_hidden_layers" not in config else config

    notes: list[str] = []
    layers = _int(cfg.get("num_hidden_layers"))
    hidden = _int(cfg.get("hidden_size"))
    heads = _int(cfg.get("num_attention_heads"))
    kv_heads = _int(cfg.get("num_key_value_heads")) or heads
    head_dim = _int(cfg.get("head_dim")) or (hidden // heads if hidden and heads else None)
    sliding = _int(cfg.get("sliding_window"))
    if cfg.get("use_sliding_window") is False:
        sliding = None
    architecture = None
    if layers and hidden and heads and kv_heads and head_dim:
        architecture = ModelArchitecture(
            family=cfg.get("model_type") or config.get("model_type"),
            num_layers=layers,
            hidden_size=hidden,
            num_attention_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            vocab_size=_int(cfg.get("vocab_size")),
            sliding_window=sliding,
        )
    else:
        notes.append("config.json lacks the shape keys needed for memory estimation.")

    shards = sorted(directory.glob("*.safetensors"))
    params = 0
    dtypes: set[str] = set()
    for shard in shards:
        for name, tensor in read_safetensors_header(shard).items():
            if name == "__metadata__" or not isinstance(tensor, dict):
                continue
            count = 1
            for dim in tensor.get("shape", []):
                count *= int(dim)
            params += count
            dtypes.add(str(tensor.get("dtype", "")).lower())
    if not shards:
        notes.append("No .safetensors files found; parameter count is unknown.")

    weight_format: WeightFormat | None = (
        "safetensors" if shards else ("pytorch" if any(directory.glob("*.bin")) else None)
    )
    dtype = config.get("torch_dtype") or config.get("dtype")
    quant = config.get("quantization_config")
    quantization = (
        str(quant.get("quant_method"))
        if isinstance(quant, dict) and quant.get("quant_method")
        else (str(dtype) if dtype else (next(iter(dtypes)) if len(dtypes) == 1 else None))
    )

    return ModelReference(
        id=model_id or default_local_id(directory),
        display_name=directory.name,
        source="local",
        origin="user",
        path=str(directory.resolve()),
        weight_format=weight_format,
        quantization=quantization,
        parameter_count=params or None,
        weight_bytes=sum(s.stat().st_size for s in shards) or None,
        architecture=architecture,
        context_length_max=_int(cfg.get("max_position_embeddings")),
        license=_readme_license(directory),
        requires_custom_code="auto_map" in config,
        supported_runtimes=["vllm", "tensorrt_llm"],
        field_sources={
            "parameter_count": Provenance.DETECTED,
            "weight_bytes": Provenance.DETECTED,
            "architecture": Provenance.DETECTED,
        },
        notes=notes,
    )


def inspect_path(path: Path, model_id: str | None = None) -> ModelReference:
    path = path.expanduser()
    if path.is_file() and path.suffix.lower() == ".gguf":
        return inspect_gguf(path, model_id)
    if path.is_dir():
        return inspect_hf_dir(path, model_id)
    if not path.exists():
        raise ModelInspectionError(f"{path} does not exist")
    raise ModelInspectionError(f"{path} is neither a .gguf file nor a directory with config.json")

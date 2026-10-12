# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferenceopt.models import ModelInspectionError, inspect_path
from inferenceopt.models.gguf import ArrayInfo, GGUFError, read_gguf
from tests.helpers import STRING, U32, llama_gguf, write_gguf, write_safetensors

pytestmark = pytest.mark.unit


def test_gguf_metadata_and_params(tmp_path: Path) -> None:
    info = read_gguf(llama_gguf(tmp_path / "m.gguf", vocab=300))
    assert info.version == 3
    assert info.architecture == "llama"
    assert info.quantization == "Q4_K_M"
    assert info.parameter_count == 256 * 300 + 256 * 256
    tokens = info.get("tokenizer.ggml.tokens")
    assert isinstance(tokens, ArrayInfo) and tokens.length == 300  # large array skipped


def test_gguf_small_arrays_kept(tmp_path: Path) -> None:
    info = read_gguf(llama_gguf(tmp_path / "m.gguf", vocab=5))
    assert info.get("tokenizer.ggml.tokens") == [f"tok{i}" for i in range(5)]


def test_gguf_to_model_reference(tmp_path: Path) -> None:
    model = inspect_path(llama_gguf(tmp_path / "tiny-q4.gguf"))
    assert model.id == "local:tiny-q4"
    assert model.display_name == "Tiny Llama Test"
    assert model.weight_format == "gguf"
    assert model.license == "apache-2.0"
    assert model.context_length_max == 2048
    arch = model.architecture
    assert arch is not None
    assert (arch.num_layers, arch.hidden_size, arch.num_kv_heads, arch.head_dim) == (4, 256, 2, 32)
    assert arch.vocab_size == 300
    assert model.weight_bytes == (tmp_path / "tiny-q4.gguf").stat().st_size


def test_gguf_per_layer_kv_heads_uses_max(tmp_path: Path) -> None:
    path = write_gguf(
        tmp_path / "m.gguf",
        {
            "general.architecture": (STRING, "x"),
            "x.block_count": (U32, 2),
            "x.embedding_length": (U32, 64),
            "x.attention.head_count": (U32, 4),
            "x.attention.head_count_kv": (9, (U32, [1, 4])),
        },
        [],
    )
    arch = inspect_path(path).architecture
    assert arch is not None and arch.num_kv_heads == 4


@pytest.mark.parametrize(
    "content",
    [b"", b"NOPE" + b"\0" * 32, b"GGUF" + (1).to_bytes(4, "little") + b"\0" * 16],
)
def test_gguf_rejects_bad_files(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "bad.gguf"
    path.write_bytes(content)
    with pytest.raises(GGUFError):
        read_gguf(path)


def test_gguf_truncated_file(tmp_path: Path) -> None:
    good = llama_gguf(tmp_path / "m.gguf").read_bytes()
    bad = tmp_path / "trunc.gguf"
    bad.write_bytes(good[:200])
    with pytest.raises(GGUFError, match="end of file"):
        read_gguf(bad)


def test_gguf_huge_counts_rejected_without_allocation(tmp_path: Path) -> None:
    path = tmp_path / "huge.gguf"
    path.write_bytes(b"GGUF" + (3).to_bytes(4, "little") + (2**60).to_bytes(8, "little") * 2)
    with pytest.raises(GGUFError, match="implausible"):
        read_gguf(path)


def _hf_dir(tmp_path: Path, **overrides: object) -> Path:
    d = tmp_path / "TinyModel"
    d.mkdir()
    config = {
        "model_type": "qwen2",
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 1000,
        "max_position_embeddings": 4096,
        "torch_dtype": "bfloat16",
        "sliding_window": 1024,
        "use_sliding_window": False,
        **overrides,
    }
    (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (d / "README.md").write_text("---\nlicense: mit\n---\n# Tiny\n", encoding="utf-8")
    write_safetensors(
        d / "model.safetensors",
        {"embed": ("BF16", [1000, 64]), "layer.w": ("BF16", [64, 64])},
    )
    return d


def test_hf_directory(tmp_path: Path) -> None:
    model = inspect_path(_hf_dir(tmp_path))
    assert model.weight_format == "safetensors"
    assert model.quantization == "bfloat16"
    assert model.parameter_count == 1000 * 64 + 64 * 64
    assert model.license == "mit"
    assert model.context_length_max == 4096
    arch = model.architecture
    assert arch is not None
    assert arch.head_dim == 16
    assert arch.sliding_window is None  # use_sliding_window: false
    assert not model.requires_custom_code


def test_hf_custom_code_flagged(tmp_path: Path) -> None:
    model = inspect_path(_hf_dir(tmp_path, auto_map={"AutoModel": "x.Y"}))
    assert model.requires_custom_code


def test_inspect_rejects_other_paths(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(ModelInspectionError):
        inspect_path(tmp_path / "x.txt")
    with pytest.raises(ModelInspectionError, match="does not exist"):
        inspect_path(tmp_path / "missing.gguf")
    with pytest.raises(ModelInspectionError, match=r"config.json"):
        inspect_path(tmp_path)

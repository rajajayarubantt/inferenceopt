# SPDX-License-Identifier: Apache-2.0
"""Builders for synthetic model files and hardware profiles used in tests."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from inferenceopt.schemas import Device, DeviceBackend, HardwareProfile, HostInfo

GIB = 1024**3

# GGUF value type ids
U32, I32, F32, BOOL, STRING, ARRAY, U64 = 4, 5, 6, 7, 8, 9, 10


def _gguf_str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _gguf_value(vtype: int, value: Any) -> bytes:
    if vtype == U32:
        return struct.pack("<I", value)
    if vtype == I32:
        return struct.pack("<i", value)
    if vtype == F32:
        return struct.pack("<f", value)
    if vtype == BOOL:
        return struct.pack("<?", value)
    if vtype == U64:
        return struct.pack("<Q", value)
    if vtype == STRING:
        return _gguf_str(value)
    if vtype == ARRAY:
        item_type, items = value
        out = struct.pack("<IQ", item_type, len(items))
        return out + b"".join(_gguf_value(item_type, v) for v in items)
    raise ValueError(vtype)


def write_gguf(
    path: Path,
    metadata: dict[str, tuple[int, Any]],
    tensors: list[tuple[str, list[int], int]],
    *,
    version: int = 3,
    padding: int = 0,
) -> Path:
    """Write a GGUF header. ``tensors`` holds (name, dims, ggml_type); no tensor data."""
    out = bytearray(b"GGUF")
    out += struct.pack("<IQQ", version, len(tensors), len(metadata))
    for key, (vtype, value) in metadata.items():
        out += _gguf_str(key) + struct.pack("<I", vtype) + _gguf_value(vtype, value)
    offset = 0
    for name, dims, ttype in tensors:
        out += _gguf_str(name) + struct.pack("<I", len(dims))
        out += b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", ttype, offset)
        offset += 32
    out += b"\0" * padding
    path.write_bytes(bytes(out))
    return path


def llama_gguf(path: Path, *, vocab: int = 300) -> Path:
    return write_gguf(
        path,
        {
            "general.architecture": (STRING, "llama"),
            "general.name": (STRING, "Tiny Llama Test"),
            "general.file_type": (U32, 15),  # Q4_K_M
            "general.license": (STRING, "apache-2.0"),
            "llama.block_count": (U32, 4),
            "llama.embedding_length": (U32, 256),
            "llama.attention.head_count": (U32, 8),
            "llama.attention.head_count_kv": (U32, 2),
            "llama.context_length": (U32, 2048),
            "tokenizer.ggml.tokens": (ARRAY, (STRING, [f"tok{i}" for i in range(vocab)])),
        },
        [("token_embd.weight", [256, vocab], 12), ("blk.0.attn_q.weight", [256, 256], 12)],
        padding=1024,
    )


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> Path:
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        n = 2
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + n]}
        offset += n
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)
    return path


def hardware_profile(
    *,
    os: str = "Linux",
    gpu_free_gib: float | None = 20,
    gpu_total_gib: float = 24,
    ram_free_gib: float = 30,
    ram_total_gib: float = 64,
) -> HardwareProfile:
    devices = [
        Device(
            device_id="cpu0",
            backend=DeviceBackend.CPU,
            name="Test CPU",
            memory_total_bytes=int(ram_total_gib * GIB),
            memory_free_bytes=int(ram_free_gib * GIB),
        )
    ]
    if gpu_free_gib is not None:
        devices.append(
            Device(
                device_id="gpu0",
                backend=DeviceBackend.CUDA,
                name="Test GPU",
                compute_capability="8.6",
                memory_total_bytes=int(gpu_total_gib * GIB),
                memory_free_bytes=int(gpu_free_gib * GIB),
            )
        )
    return HardwareProfile(
        host=HostInfo(
            os=os,
            os_version="test",
            python_version="3.12",
            cpu_arch="x86_64",
            ram_total_bytes=int(ram_total_gib * GIB),
            ram_available_bytes=int(ram_free_gib * GIB),
        ),
        devices=devices,
    )

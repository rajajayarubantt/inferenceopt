# SPDX-License-Identifier: Apache-2.0
"""Minimal, bounds-checked GGUF metadata reader (v2 and v3).

Reads only the header, key/value metadata and tensor descriptors, never tensor data.
Large arrays (e.g. the tokenizer vocabulary) are skipped and only their length is kept.
Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"GGUF"
MAX_ITEMS = 1_000_000
MAX_STRING = 64 * 1024 * 1024
KEEP_ARRAY_MAX = 256  # arrays longer than this are summarized as ArrayInfo

# GGUF metadata value types.
_SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_STRING, _ARRAY = 8, 9

# llama.cpp ``general.file_type`` (llama_ftype) to quantization name.
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0",
}  # fmt: skip


class GGUFError(ValueError):
    """The file is not a readable GGUF file."""


@dataclass(frozen=True)
class ArrayInfo:
    item_type: int
    length: int


@dataclass
class GGUFInfo:
    version: int
    metadata: dict[str, Any]
    tensor_count: int
    parameter_count: int
    file_size: int
    tensor_types: dict[int, int] = field(default_factory=dict)
    """ggml tensor type id to number of tensors of that type."""

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    @property
    def architecture(self) -> str | None:
        arch = self.metadata.get("general.architecture")
        return arch if isinstance(arch, str) else None

    @property
    def quantization(self) -> str | None:
        ftype = self.metadata.get("general.file_type")
        if isinstance(ftype, int):
            return FILE_TYPES.get(ftype, f"ftype_{ftype}")
        return None


class _Reader:
    def __init__(self, f: BinaryIO, size: int) -> None:
        self.f = f
        self.size = size

    def _read(self, n: int) -> bytes:
        if n < 0 or self.f.tell() + n > self.size:
            raise GGUFError("unexpected end of file")
        return self.f.read(n)

    def _skip(self, n: int) -> None:
        if n < 0 or self.f.tell() + n > self.size:
            raise GGUFError("unexpected end of file")
        self.f.seek(n, 1)

    def scalar(self, fmt: str) -> Any:
        return struct.unpack(fmt, self._read(struct.calcsize(fmt)))[0]

    def u32(self) -> int:
        return int(self.scalar("<I"))

    def u64(self) -> int:
        return int(self.scalar("<Q"))

    def string(self, keep: bool = True) -> str:
        n = self.u64()
        if n > MAX_STRING:
            raise GGUFError(f"string too long ({n} bytes)")
        if not keep:
            self._skip(n)
            return ""
        return self._read(n).decode("utf-8", errors="replace")

    def value(self, vtype: int) -> Any:
        if vtype in _SCALARS:
            return self.scalar(_SCALARS[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            item_type, length = self.u32(), self.u64()
            if length > 100 * MAX_ITEMS:
                raise GGUFError(f"array too long ({length})")
            if length > KEEP_ARRAY_MAX:
                self.skip_array(item_type, length)
                return ArrayInfo(item_type, length)
            return [self.value(item_type) for _ in range(length)]
        raise GGUFError(f"unknown metadata value type {vtype}")

    def skip_array(self, item_type: int, length: int) -> None:
        if item_type in _SCALARS:
            self._skip(struct.calcsize(_SCALARS[item_type]) * length)
        elif item_type == _STRING:
            for _ in range(length):
                self.string(keep=False)
        else:
            raise GGUFError(f"unsupported array item type {item_type}")


def read_gguf(path: Path) -> GGUFInfo:
    size = path.stat().st_size
    with path.open("rb") as f:
        r = _Reader(f, size)
        if r._read(4) != MAGIC:
            raise GGUFError(f"{path.name}: not a GGUF file")
        version = r.u32()
        if version not in (2, 3):
            raise GGUFError(f"{path.name}: unsupported GGUF version {version}")
        tensor_count, kv_count = r.u64(), r.u64()
        if tensor_count > MAX_ITEMS or kv_count > MAX_ITEMS:
            raise GGUFError(f"{path.name}: implausible header counts")

        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = r.string()
            metadata[key] = r.value(r.u32())

        params = 0
        tensor_types: dict[int, int] = {}
        for _ in range(tensor_count):
            r.string(keep=False)  # name
            n_dims = r.u32()
            if n_dims > 8:
                raise GGUFError(f"{path.name}: tensor with {n_dims} dimensions")
            elements = 1
            for _ in range(n_dims):
                elements *= r.u64()
            ttype = r.u32()
            r.u64()  # data offset
            params += elements
            tensor_types[ttype] = tensor_types.get(ttype, 0) + 1

    return GGUFInfo(
        version=version,
        metadata=metadata,
        tensor_count=tensor_count,
        parameter_count=params,
        file_size=size,
        tensor_types=tensor_types,
    )

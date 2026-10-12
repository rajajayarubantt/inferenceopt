# SPDX-License-Identifier: Apache-2.0
"""Adapter contracts: exact launch commands and validation rules, no runtime needed.

These are the main safety net for the experimental adapters (vLLM, TensorRT-LLM) until
they are validated on real hardware.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inferenceopt.runtimes.base import Installation, ParamSpec, RuntimeSettings
from inferenceopt.runtimes.llamacpp.adapter import (
    LlamaCppAdapter,
    parse_devices,
    parse_help_features,
    parse_version,
)
from inferenceopt.runtimes.registry import adapters, get_adapter
from inferenceopt.runtimes.tensorrt_llm import TensorRTLLMAdapter
from inferenceopt.runtimes.vllm import VLLMAdapter
from inferenceopt.schemas import ModelArchitecture, ModelReference, SupportLevel
from tests.helpers import hardware_profile

pytestmark = pytest.mark.contract

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ARCH = ModelArchitecture(
    num_layers=24, hidden_size=896, num_attention_heads=14, num_kv_heads=2, head_dim=64
)


def gguf(tmp_path: Path) -> ModelReference:
    path = tmp_path / "m.gguf"
    path.write_bytes(b"GGUF")
    return ModelReference(
        id="local:m", display_name="m", source="local", weight_format="gguf",
        path=str(path), architecture=ARCH, context_length_max=32768,
    )  # fmt: skip


def hf_dir(tmp_path: Path, **kw: Any) -> ModelReference:
    d = tmp_path / "hf"
    d.mkdir(exist_ok=True)
    return ModelReference(
        id="local:hf", display_name="hf", source="local", weight_format="safetensors",
        path=str(d), architecture=ARCH, context_length_max=32768, **kw,
    )  # fmt: skip


NATIVE_LLAMA = Installation(
    executable="/opt/llama/llama-server",
    version="b11429",
    source="path",
    devices=["CUDA0: NVIDIA GeForce RTX 3050"],
    features={"flash_attn_value": True, "no_webui": True, "metrics": True, "alias": True},
)


# --- registry ----------------------------------------------------------------------------


def test_registry_has_all_builtins_with_expected_support_levels() -> None:
    found = adapters()
    assert found["llamacpp"].support_level == SupportLevel.TESTED
    assert found["vllm"].support_level == SupportLevel.EXPERIMENTAL
    assert found["tensorrt_llm"].support_level == SupportLevel.EXPERIMENTAL
    with pytest.raises(KeyError):
        get_adapter("nope")


@pytest.mark.parametrize("name", ["llamacpp", "vllm", "tensorrt_llm"])
def test_parameter_defaults_satisfy_their_own_specs(name: str) -> None:
    for spec in get_adapter(name).parameters(None):
        assert isinstance(spec, ParamSpec)
        if spec.default is not None:
            assert spec.check(spec.default) is None, spec.name


# --- llama.cpp ---------------------------------------------------------------------------


def test_llamacpp_native_launch(tmp_path: Path) -> None:
    model = gguf(tmp_path)
    spec = LlamaCppAdapter().build_launch(NATIVE_LLAMA, model, {"ctx_size": 8192}, port=8123)
    assert spec.argv == [
        "/opt/llama/llama-server",
        "--model", str(model.path), "--host", "127.0.0.1", "--port", "8123",
        "--ctx-size", "8192", "--n-gpu-layers", "999",
        "--cache-type-k", "f16", "--cache-type-v", "f16",
        "--batch-size", "2048", "--ubatch-size", "512", "--parallel", "1",
        "--flash-attn", "auto", "--cont-batching",
        "--alias", "local-m", "--metrics", "--no-webui",
    ]  # fmt: skip
    assert spec.api_base == "http://127.0.0.1:8123/v1"
    assert spec.served_model_name == "local-m"
    assert spec.container_name is None


def test_llamacpp_load_mode_new_and_legacy(tmp_path: Path) -> None:
    adapter, model = LlamaCppAdapter(), gguf(tmp_path)
    new = NATIVE_LLAMA.model_copy(update={"features": {**NATIVE_LLAMA.features, "load_mode": True}})
    argv = adapter.build_launch(new, model, {"load_mode": "mlock"}, port=1).argv
    assert argv[argv.index("--load-mode") + 1] == "mlock"
    old = NATIVE_LLAMA.model_copy(
        update={"features": {**NATIVE_LLAMA.features, "load_mode": False, "help_available": True}}
    )
    argv = adapter.build_launch(old, model, {"load_mode": "none"}, port=1).argv
    assert "--no-mmap" in argv and "--load-mode" not in argv
    errors = adapter.validate({"load_mode": "dio"}, model, None, old)
    assert any("newer llama.cpp" in e for e in errors)


def test_llamacpp_old_flash_attn_flag_style(tmp_path: Path) -> None:
    old = NATIVE_LLAMA.model_copy(update={"features": {"flash_attn_value": False}})
    argv = LlamaCppAdapter().build_launch(old, gguf(tmp_path), {"flash_attn": "on"}, port=1).argv
    assert "--flash-attn" in argv and "on" not in argv


def test_llamacpp_docker_launch(tmp_path: Path) -> None:
    model = gguf(tmp_path)
    image = Installation(
        executable="ghcr.io/ggml-org/llama.cpp:server-cuda-b11429", mode="docker",
        source="docker", devices=["CUDA0: x"], features=NATIVE_LLAMA.features,
    )  # fmt: skip
    spec = LlamaCppAdapter().build_launch(image, model, {}, port=9000)
    argv = spec.argv
    assert argv[1:4] == ["run", "--rm", "--name"]
    assert spec.container_name and spec.container_name.startswith("inferenceopt-")
    assert "127.0.0.1:9000:8080" in argv  # published on loopback only
    assert f"{tmp_path}:/models/model:ro" in argv  # read-only model mount
    assert argv[argv.index("--gpus") + 1] == "all"
    i = argv.index("ghcr.io/ggml-org/llama.cpp:server-cuda-b11429")
    assert argv[i + 1 : i + 3] == ["--model", "/models/model/m.gguf"]
    assert argv[argv.index("--port") + 1] == "8080"


def test_llamacpp_cpu_only_docker_has_no_gpus_flag(tmp_path: Path) -> None:
    image = Installation(executable="img", mode="docker", source="docker")
    argv = LlamaCppAdapter().build_launch(image, gguf(tmp_path), {"n_gpu_layers": 0}, port=1).argv
    assert "--gpus" not in argv


@pytest.mark.parametrize(
    ("config", "fragment"),
    [
        ({"cache_type_v": "q8_0", "flash_attn": "off"}, "requires flash_attn"),
        ({"ctx_size": 4097, "parallel": 2}, "divisible"),
        ({"ubatch_size": 4096, "batch_size": 1024}, "ubatch_size"),
        ({"ctx_size": 65536}, "exceeds the model"),
        ({"bogus": 1}, "unknown parameter"),
        ({"ctx_size": "big"}, "integer"),
        ({"flash_attn": "maybe"}, "one of"),
    ],
)
def test_llamacpp_validation(tmp_path: Path, config: dict[str, Any], fragment: str) -> None:
    errors = LlamaCppAdapter().validate(config, gguf(tmp_path), hardware_profile(), NATIVE_LLAMA)
    assert any(fragment in e for e in errors), errors


def test_llamacpp_gpu_layers_need_a_gpu(tmp_path: Path) -> None:
    cpu_build = NATIVE_LLAMA.model_copy(update={"devices": []})
    errors = LlamaCppAdapter().validate({"n_gpu_layers": 10}, gguf(tmp_path), None, cpu_build)
    assert any("no GPU backend" in e for e in errors)
    assert LlamaCppAdapter().defaults(cpu_build)["n_gpu_layers"] == 0
    no_gpu_host = hardware_profile(gpu_free_gib=None)
    errors = LlamaCppAdapter().validate({}, gguf(tmp_path), no_gpu_host, NATIVE_LLAMA)
    assert any("no GPU was detected" in e for e in errors)


def test_llamacpp_rejects_non_gguf(tmp_path: Path) -> None:
    errors = LlamaCppAdapter().validate({}, hf_dir(tmp_path), hardware_profile(), NATIVE_LLAMA)
    assert any("cannot load safetensors" in e for e in errors)


def test_llamacpp_help_fixture_parsing() -> None:
    help_text = (FIXTURES / "llama-server-b11429-help.txt").read_text(encoding="utf-8")
    features = parse_help_features(help_text)
    assert features == {
        "flash_attn_value": True,
        "no_webui": True,
        "metrics": True,
        "alias": True,
        "load_mode": True,
        "help_available": True,
    }
    for flag in ("--ctx-size", "--n-gpu-layers", "--cache-type-k", "--cache-type-v",
                 "--batch-size", "--ubatch-size", "--parallel", "--threads",
                 "--cont-batching", "--no-cont-batching", "--load-mode"):  # fmt: skip
        assert flag in help_text, f"{flag} missing from llama-server --help"


def test_llamacpp_version_and_devices_parsing() -> None:
    assert parse_version("version: 11429 (abc1234)\nbuilt with ...") == "b11429"
    assert parse_version("garbage") is None
    device = "CUDA0: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6143 MiB, 5110 MiB free)"
    assert parse_devices(f"Available devices:\n  {device}\n") == [device]


# --- vLLM -------------------------------------------------------------------------------


def test_vllm_native_launch(tmp_path: Path) -> None:
    model = hf_dir(tmp_path)
    inst = Installation(executable="/env/bin/vllm", version="0.11.0", source="settings")
    spec = VLLMAdapter().build_launch(
        inst, model,
        {"max_model_len": 8192, "gpu_memory_utilization": 0.85, "kv_cache_dtype": "fp8",
         "enable_prefix_caching": False, "enforce_eager": True},
        port=8200,
    )  # fmt: skip
    assert spec.argv == [
        "/env/bin/vllm", "serve", str(model.path), "--host", "127.0.0.1", "--port", "8200",
        "--served-model-name", "local-hf",
        "--max-model-len", "8192", "--gpu-memory-utilization", "0.85",
        "--kv-cache-dtype", "fp8", "--dtype", "auto", "--tensor-parallel-size", "1",
        "--no-enable-prefix-caching", "--enforce-eager",
    ]  # fmt: skip
    assert spec.startup_timeout_s >= 600


def test_vllm_docker_launch(tmp_path: Path) -> None:
    inst = Installation(executable="vllm/vllm-openai:v0.11.0", mode="docker", source="docker")
    spec = VLLMAdapter().build_launch(inst, hf_dir(tmp_path), {}, port=8300)
    argv = spec.argv
    i = argv.index("vllm/vllm-openai:v0.11.0")
    assert argv[i + 1 : i + 3] == ["--model", "/models/model"]
    assert "127.0.0.1:8300:8000" in argv and spec.container_name


def test_vllm_validation(tmp_path: Path) -> None:
    adapter = VLLMAdapter()
    linux = hardware_profile()
    assert adapter.validate({}, hf_dir(tmp_path), linux, None) == []
    assert any("tensor_parallel_size" in e for e in adapter.validate(
        {"tensor_parallel_size": 2}, hf_dir(tmp_path), linux, None))  # fmt: skip
    assert any("needs an NVIDIA" in e for e in adapter.validate(
        {}, hf_dir(tmp_path), hardware_profile(gpu_free_gib=None), None))  # fmt: skip
    assert any("cannot load gguf" in e for e in adapter.validate({}, gguf(tmp_path), linux, None))
    custom = hf_dir(tmp_path, requires_custom_code=True)
    assert any("custom code" in e for e in adapter.validate({}, custom, linux, None))
    assert adapter.validate({"trust_remote_code": True}, custom, linux, None) == []


# --- TensorRT-LLM ------------------------------------------------------------------------


def test_trtllm_native_launch(tmp_path: Path) -> None:
    model = hf_dir(tmp_path)
    inst = Installation(executable="/usr/local/bin/trtllm-serve", source="path")
    spec = TensorRTLLMAdapter().build_launch(inst, model, {"max_batch_size": 8}, port=8400)
    assert spec.argv == [
        "/usr/local/bin/trtllm-serve", str(model.path), "--host", "127.0.0.1", "--port", "8400",
        "--max_batch_size", "8", "--tp_size", "1", "--kv_cache_free_gpu_memory_fraction", "0.9",
    ]  # fmt: skip


def test_trtllm_requires_an_image_in_docker_mode() -> None:
    install, notes = TensorRTLLMAdapter().detect(RuntimeSettings(mode="docker"))
    assert install is None
    assert any("image" in n.lower() for n in notes)


# --- settings safety ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("settings", "fragment"),
    [
        (RuntimeSettings(executable="C:/Windows/System32/cmd.exe"), "must be named"),
        (RuntimeSettings(executable="/bin/sh"), "must be named"),
        (RuntimeSettings(executable="llama-server"), "absolute path"),
        (RuntimeSettings(image="img; rm -rf /"), "image"),
        (RuntimeSettings(extra_env={"LD_PRELOAD": "/x.so"}), "not allowed"),
        (RuntimeSettings(extra_env={"PATH": "/evil"}), "not allowed"),
    ],
)
def test_settings_cannot_point_at_arbitrary_programs(
    settings: RuntimeSettings, fragment: str
) -> None:
    errors = LlamaCppAdapter().validate_settings(settings)
    assert any(fragment in e for e in errors), errors


def test_valid_settings(tmp_path: Path) -> None:
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"")
    ok = RuntimeSettings(
        executable=str(exe),
        image="ghcr.io/ggml-org/llama.cpp:server-cuda-b11429@sha256:" + "a" * 64,
        extra_env={"CUDA_VISIBLE_DEVICES": "0", "GGML_CUDA_NO_PINNED": "1"},
    )
    assert LlamaCppAdapter().validate_settings(ok) == []

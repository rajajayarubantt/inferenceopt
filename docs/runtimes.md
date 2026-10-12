# Runtimes

InferenceOpt never links an inference engine into its own process. Each runtime runs as a
separate process or container, and InferenceOpt talks to its OpenAI-compatible HTTP API.

| Runtime | Support level | Native on | Docker |
|---|---|---|---|
| llama.cpp (`llama-server`) | **tested** (CI on Linux CPU; maintainer on an RTX 3050 via Docker) | Windows, Linux, macOS | `ghcr.io/ggml-org/llama.cpp:server-cuda-<build>` |
| vLLM (`vllm serve`) | **experimental**: launch commands are contract-tested, not yet validated on a GPU | Linux / WSL2 | `vllm/vllm-openai:<version>` |
| TensorRT-LLM (`trtllm-serve`) | **experimental**: same as vLLM | Linux | an NVIDIA TensorRT-LLM release image you choose |

```bash
inferenceopt runtimes list                       # what was found, versions, devices
inferenceopt runtimes inspect llamacpp           # tunable parameters with ranges
inferenceopt runtimes launch-preview llamacpp MODEL --set ctx_size=8192   # exact command
inferenceopt runtimes smoke-test llamacpp MODEL  # start, generate a few tokens, stop
```

The same actions are on the UI's **Runtimes** page.

## llama.cpp

### Native install

```bash
inferenceopt runtimes install llamacpp            # asks before downloading
inferenceopt runtimes install llamacpp --variant cpu
```

- The **stable** channel follows llama.cpp's latest versioned release (which points to a
  numbered build); `--channel nightly` takes the newest build.
- For NVIDIA GPUs the installer picks the newest CUDA build your driver supports.
- Every file is verified against the SHA-256 digest GitHub publishes. Archives are
  extracted with path-traversal and link checks. Builds go to the per-user data directory.

### Windows with Smart App Control

Smart App Control (and other App Control policies) blocks unsigned executables, which
includes official llama.cpp builds. InferenceOpt detects this and says so. Options:

1. **Docker (recommended)**. With Docker Desktop's WSL2 backend and an NVIDIA driver:

    ```bash
    docker pull ghcr.io/ggml-org/llama.cpp:server-cuda-b11429
    inferenceopt runtimes configure llamacpp --mode docker \
      --image ghcr.io/ggml-org/llama.cpp:server-cuda-b11429
    ```

2. Run InferenceOpt inside WSL2 and use the Linux build.

Turning Smart App Control off also works but lowers your machine's protection, so it is
not recommended.

### Docker mode details

- The container's port is published on `127.0.0.1` only.
- The model's directory is mounted **read-only**.
- `--gpus all` is passed only when layers are offloaded (`n_gpu_layers > 0`).
- Containers are named `inferenceopt-*`, removed on exit, and stopped by the supervisor.

## vLLM and TensorRT-LLM (experimental)

These adapters build complete launch commands and pass contract tests, but they have not
been run on real hardware yet. Configure one of:

```bash
inferenceopt runtimes configure vllm --python-env /path/to/venv      # Linux/WSL2
inferenceopt runtimes configure vllm --mode docker --image vllm/vllm-openai:<version>
inferenceopt runtimes configure tensorrt_llm --mode docker --image <trt-llm release image>
```

Both currently need a **local model directory** (Hugging Face format); downloads are not
enabled yet. Models that ship custom code are refused until you set
`trust_remote_code` after reviewing that code.

## Safety

- Launch commands are argument lists, never shell strings.
- A configured executable must carry the runtime's own binary name
  (`llama-server`, `vllm`, `trtllm-serve`), so settings can't point at arbitrary programs.
- Only runtime-tuning environment variables can be added (no `PATH`, `LD_PRELOAD`, ...).
- Starting a runtime or downloading a build through the API requires `confirm: true`.
- On Windows each runtime runs inside a Job Object, so its whole process tree is killed
  with it; on Linux/macOS it gets its own process group.

## Writing an adapter

Subclass `inferenceopt.runtimes.base.RuntimeAdapter` and register it under the
`inferenceopt.runtimes` entry-point group in your package's `pyproject.toml`. Start with
`support_level = experimental` and contract tests for your launch commands.

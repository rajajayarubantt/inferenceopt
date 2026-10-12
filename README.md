# InferenceOpt

**Find the best way to run an LLM on the hardware you have, and show the evidence.**

InferenceOpt inventories your hardware, checks model/runtime compatibility, estimates
memory, runs controlled benchmark trials through established runtimes (llama.cpp, vLLM,
TensorRT-LLM), enforces quality and resource constraints, and produces a reproducible
deployment bundle. Everything is available from a **web UI**, a **CLI** and a **Python API**.
An LLM planner (any provider via LiteLLM) is optional; every claim is backed by
measurements, never by the LLM.

> **Status: pre-alpha (Phase 3 of the [implementation plan](docs/design/implementation-plan.md)).**
> Hardware scanning, the model registry, memory/fit estimates and runtime adapters work
> (start, health-check and smoke-test a model). Benchmarking and optimization come next.

## Install

```bash
pip install inferenceopt          # core + CLI + web UI
pip install "inferenceopt[llm]"   # + LiteLLM planner
pip install "inferenceopt[all]"
```

Inference runtimes are **not** pip dependencies. InferenceOpt launches them as separate
processes, so install the ones you want (for example `llama-server` from
[llama.cpp](https://github.com/ggml-org/llama.cpp)) and run `inferenceopt doctor`.

## Quick start

```bash
inferenceopt doctor                        # check the environment
inferenceopt ui                            # open the web UI (prints a private URL)
inferenceopt hardware scan                 # CPUs, GPUs, memory (read-only)
inferenceopt models add ./model-q4_k_m.gguf # register a local model (headers only)
inferenceopt preflight local:model-q4_k_m --context 8192   # will it fit?
inferenceopt job validate configs/example-job.yaml
```

`inferenceopt` is also available as the shorter `iopt`.

The UI listens on `127.0.0.1` only and requires the session token printed in the launch
URL. See [docs/safety.md](docs/safety.md) before exposing it to a network.

## Runtime support

| Runtime | Status |
|---|---|
| llama.cpp | **tested** (native or Docker; see [docs/runtimes.md](docs/runtimes.md)) |
| vLLM | **experimental** (contract-tested, not yet validated on hardware) |
| TensorRT-LLM | **experimental** (contract-tested, not yet validated on hardware) |

## Development

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), Node.js 22+ and pnpm
(only for working on the UI).

```bash
uv sync                                   # Python env with dev tools
uv run pre-commit install
uv run pytest                             # tests
uv run ruff check . && uv run mypy        # lint + types

cd ui
pnpm install
pnpm dev                                  # UI dev server with hot reload, proxies /api
pnpm build                                # writes src/inferenceopt/_ui/ for packaging
```

For UI development, run the API with a fixed token so the dev server can use it:

```bash
INFERENCEOPT_SERVER__TOKEN=dev-token uv run inferenceopt ui --no-browser
# then open http://localhost:5173/?token=dev-token
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

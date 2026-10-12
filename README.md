# InferenceOpt

**Find the best way to run an LLM on the hardware you have, and show the evidence.**

InferenceOpt inventories your hardware, checks model/runtime compatibility, estimates
memory, runs controlled benchmark trials through established runtimes (llama.cpp, vLLM,
TensorRT-LLM), enforces quality and resource constraints, and produces a reproducible
deployment bundle. Everything is available from a **web UI**, a **CLI** and a **Python API**.
An LLM planner (any provider via LiteLLM) is optional; every claim is backed by
measurements, never by the LLM.

> **Status: pre-alpha (Phase 1 of the [implementation plan](docs/design/implementation-plan.md)).**
> The project skeleton, configuration, storage, CLI and UI shell exist. Hardware scanning,
> runtimes, benchmarking and optimization are not implemented yet.

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
inferenceopt job validate configs/example-job.yaml
```

`inferenceopt` is also available as the shorter `iopt`.

The UI listens on `127.0.0.1` only and requires the session token printed in the launch
URL. See [docs/safety.md](docs/safety.md) before exposing it to a network.

## Runtime support

| Runtime | Status |
|---|---|
| llama.cpp | Planned for Phase 3, tested in CI |
| vLLM | Planned for Phase 3, **experimental** (not yet validated on hardware) |
| TensorRT-LLM | Planned for Phase 3, **experimental** (not yet validated on hardware) |

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

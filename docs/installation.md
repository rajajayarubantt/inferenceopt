# Installation

Requires Python 3.11 or newer.

```bash
pip install inferenceopt            # core, CLI and web UI
pip install "inferenceopt[llm]"     # + LiteLLM planner
pip install "inferenceopt[optimize]" # + Optuna Bayesian search
pip install "inferenceopt[all]"
```

The web UI is pre-built and included in the package; you do not need Node.js.

## Inference runtimes

Runtimes are not installed by pip. InferenceOpt starts them as separate processes, which
keeps your Python environment free of large, conflicting CUDA/PyTorch dependencies.

| Runtime | How InferenceOpt finds it |
|---|---|
| llama.cpp | `llama-server` on `PATH`, or a path set in Settings |
| vLLM (experimental) | A Python environment path or Docker image (Linux / WSL2 only) |
| TensorRT-LLM (experimental) | A Docker image (Linux, NVIDIA) |

Run `inferenceopt doctor` to see what was found.

## Where data is stored

| What | Default location |
|---|---|
| Config | per-user config dir, e.g. `%LOCALAPPDATA%\inferenceopt\` on Windows, `~/.config/inferenceopt/` on Linux |
| Database and results | per-user data dir |

Set `INFERENCEOPT_HOME` to keep everything under one directory instead.

# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Runtime adapters behind one interface: llama.cpp (**tested**), vLLM and TensorRT-LLM
  (**experimental**: contract-tested, not yet validated on hardware). Native and Docker
  modes; parameters, validation rules and capability probes per runtime and version.
- Process supervisor: no shell, Windows Job Objects / POSIX process groups so the whole
  tree dies with the runtime, health-based readiness, and classified failures
  (OOM, bad config, crash, timeout, blocked by Windows App Control).
- Streaming OpenAI-compatible client measuring TTFT and inter-chunk latency.
- llama.cpp installer: stable/nightly builds from GitHub, CUDA build chosen by driver
  version, SHA-256 verified downloads and safe extraction.
- Smoke tests (start, generate, stop) and launch previews from CLI, API and UI.
- CLI: `runtimes list|inspect|configure|install|launch-preview|smoke-test`.
- UI: Runtimes page with settings, consented installs and a smoke-test panel.
- Hardware discovery: CPU model, cores and SIMD features, RAM, NVIDIA GPUs via NVML
  (compute capability, memory, driver/CUDA versions), Apple Silicon, and a plugin entry
  point (`inferenceopt.hardware`). Every value is tagged detected / inferred / user.
- Model registry: curated models plus local GGUF files and Hugging Face directories,
  read from headers only (no weights loaded).
- Memory estimator (weights, KV cache, compute buffers, overhead) with a low/expected/high
  range and explicit assumptions; preflight fit checks per device and static runtime
  compatibility rules.
- CLI: `hardware scan|show`, `models list|inspect|add|remove`, `preflight`.
- API and UI: Hardware page, Models page with "Will it fit?" memory chart.
- Project skeleton: `src/` layout, Apache-2.0 license, community files, CI and release
  workflows publishing to TestPyPI via Trusted Publishing.
- Versioned Pydantic schemas for hardware, models, runtimes, jobs, candidates, trials,
  artifacts, deployment manifests and planner calls.
- Configuration from YAML and `INFERENCEOPT_*` environment variables with secret
  redaction in logs.
- SQLite storage (WAL) with automatic Alembic migrations.
- CLI: `version`, `doctor`, `config show`, `job validate`, `ui` (also available as `iopt`).
- Local web server with token authentication, Host/Origin checks and the bundled
  React UI shell (Dashboard, Doctor and Job validation pages).

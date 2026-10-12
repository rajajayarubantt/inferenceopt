# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

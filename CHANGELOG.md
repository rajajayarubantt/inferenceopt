# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

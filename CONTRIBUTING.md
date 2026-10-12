# Contributing to InferenceOpt

Thanks for helping! This document covers the development workflow and the rules every
change must follow.

## Ground rules

1. **Measure, don't guess.** No performance or hardware-support claims without
   measurements recorded by the benchmark harness.
2. **Feature parity.** New capabilities go into `src/inferenceopt/services/` first, then
   get a CLI command, an API route and a UI surface. If one of those is missing, open a
   tracked issue in the same PR.
3. **No secrets** in code, tests, fixtures, logs or generated artifacts.
4. **Safe by default.** Nothing executes generated code or shell strings on the host.
   Runtime launches use structured argument lists from adapter allowlists.

## Setup

```bash
uv sync
uv run pre-commit install
cd ui && pnpm install
```

## Checks (CI runs all of these)

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy   # CI also runs --platform linux, darwin and win32
uv run pytest
cd ui && pnpm lint && pnpm typecheck && pnpm test && pnpm build
```

If you change an API route or response model, regenerate the TypeScript client and
commit the result:

```bash
uv run python scripts/gen_api_client.py
```

## Tests

Use markers from `pyproject.toml`. Tests that need real hardware or runtimes
(`hardware_*`, `integration_llamacpp`) must be marked so public CI can skip them.

## Commits and sign-off (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/) instead
of a CLA. Sign off every commit:

```bash
git commit -s -m "Add hardware scan endpoint"
```

This adds `Signed-off-by: Your Name <you@example.com>`, certifying that you wrote the
change or have the right to submit it under the Apache-2.0 license.

## Source headers

Every source file starts with an SPDX header (enforced by pre-commit):

```python
# SPDX-License-Identifier: Apache-2.0
```

```ts
// SPDX-License-Identifier: Apache-2.0
```

## Changelog

Add an entry under `## [Unreleased]` in `CHANGELOG.md` for any user-visible change.

## Architecture decisions

Significant design decisions are recorded in `docs/adr/`. Propose changes to them with a
new ADR rather than editing an accepted one.

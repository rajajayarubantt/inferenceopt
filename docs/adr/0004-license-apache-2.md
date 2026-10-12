# 0004: Apache-2.0 license

**Status:** Accepted (2026-10-12)

## Context

The project targets both research and production use and may include performance and
kernel techniques that companies adopt. Key dependencies are MIT (LiteLLM, llama.cpp)
or Apache-2.0 (vLLM, TensorRT-LLM).

## Decision

- License: Apache-2.0, for its explicit patent grant and patent-retaliation clause.
- Copyright line: "Copyright 2026 The InferenceOpt Authors", kept in `NOTICE`.
- Contributions use DCO sign-off (`git commit -s`), not a CLA.
- Source files carry `SPDX-License-Identifier: Apache-2.0`, enforced by pre-commit/CI.
- `pyproject.toml` uses the PEP 639 SPDX expression and ships `LICENSE` and `NOTICE`.

## Consequences

- Compatible with MIT/BSD/Apache dependencies. GPL-licensed code cannot be bundled.
- Bundled UI dependencies must be license-checked; their notices ship in the wheel.

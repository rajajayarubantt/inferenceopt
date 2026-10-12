# 0003: One services layer

**Status:** Accepted (2026-10-12)

## Context

The project has three interfaces: web UI (via the HTTP API), CLI and Python API. The
original plan treated the API as optional and later, but the UI must control everything.

## Decision

All application logic lives in `inferenceopt.services`. The CLI commands and API routes
are thin adapters that parse input, call a service function and format the result. The
API is a core component from Phase 1.

## Consequences

- Feature parity is structural: a capability added to services is one wrapper away from
  each interface. PRs must add all three or open a follow-up issue.
- Service functions return Pydantic models, which double as API response schemas and
  feed the generated TypeScript client.

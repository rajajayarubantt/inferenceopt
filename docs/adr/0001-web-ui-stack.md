# 0001: Web UI stack

**Status:** Accepted (2026-10-12)

## Context

Everything must be controllable from a UI, and the project ships as a pip package.
Users must not need Node.js. The UI needs live job progress, interactive charts (Pareto
frontiers) and multi-step approval flows.

## Decision

- React + TypeScript + Vite in a top-level `ui/` directory, with Tailwind CSS, TanStack
  Router and TanStack Query.
- `pnpm build` writes to `src/inferenceopt/_ui/`; hatch includes it in the wheel and
  sdist as a build artifact. FastAPI serves it.
- The TypeScript API client is generated from the OpenAPI schema (`openapi-typescript`
  + `openapi-fetch`); CI fails on drift.
- ESLint + Prettier for linting and formatting. (Biome was tried first but its native
  binary is blocked by Windows Smart App Control on some developer machines.)
- TypeScript is pinned to 5.x: TypeScript 7's native compiler has no JavaScript API,
  which `openapi-typescript` requires.

## Consequences

- Contributors working on the UI need Node 22+ and pnpm; users and Python-only
  contributors don't.
- The wheel stays pure-Python (`py3-none-any`).
- Bundled JavaScript is redistributed, so `THIRD_PARTY_LICENSES.txt` is generated at
  build time and shipped with it.

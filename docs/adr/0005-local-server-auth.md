# 0005: Local server authentication

**Status:** Accepted (2026-10-12)

## Context

The UI server will start processes and run workloads on the host. A localhost server
without auth can be reached by other local users and, through DNS rebinding or CSRF, by
any website the user visits.

## Decision

- Bind to `127.0.0.1` by default; refuse non-loopback addresses unless
  `--allow-remote` is given.
- Generate a random token per launch (or use `server.token`) and print it in the launch
  URL. The SPA posts it to `/api/v1/auth/session`, which sets an HttpOnly,
  SameSite=Strict cookie, then removes the token from the address bar. Scripts may use
  `Authorization: Bearer`.
- Every `/api/*` route except `/api/v1/health` and the session exchange requires auth,
  including the OpenAPI schema.
- Reject requests whose Host header is not a loopback name or an explicitly allowed host.
- Reject state-changing requests whose Origin doesn't match the Host.
- Redact the token from all logs, including uvicorn access logs.

## Consequences

- Same model as Jupyter: familiar to users, works without extra setup.
- Remote use needs `allowed_hosts` plus a TLS reverse proxy (documented in Safety).

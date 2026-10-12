# Architecture Decision Records

Each ADR records one significant decision, its context and its consequences. Accepted
ADRs are not edited; a new ADR supersedes an old one.

| # | Decision | Status |
|---|---|---|
| [0001](0001-web-ui-stack.md) | React + Vite UI in `ui/`, served by FastAPI, shipped in the wheel | Accepted |
| [0002](0002-runtimes-as-external-processes.md) | Inference runtimes run as external processes, not pip dependencies | Accepted |
| [0003](0003-services-layer.md) | One services layer behind CLI, API and Python API | Accepted |
| [0004](0004-license-apache-2.md) | Apache-2.0 license with DCO sign-off | Accepted |
| [0005](0005-local-server-auth.md) | Token auth and Host/Origin checks for the local server | Accepted |

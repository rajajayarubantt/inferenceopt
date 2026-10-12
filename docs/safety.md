# Safety

InferenceOpt can start inference servers and run workloads on your machine. Its web
server is therefore treated as a privileged local service.

## Web UI and API protections (Phase 1)

| Protection | What it prevents |
|---|---|
| Binds to `127.0.0.1` by default; other addresses need `--allow-remote` | Accidental exposure on a network |
| Per-launch session token, exchanged for an HttpOnly, SameSite=Strict cookie | Other local users or web pages using the API |
| Host-header allowlist | DNS-rebinding attacks from malicious websites |
| Same-origin check on state-changing requests | Cross-site request forgery |
| Secrets are write-only in the API | Leaking API keys to the browser |
| Token redaction in logs, including access logs | Leaking the session token through log files |
| No endpoint runs arbitrary commands | Turning the UI into a remote shell |

## Exposing the UI on a network

Not recommended before Phase 8. If you must:

```bash
inferenceopt ui --host 0.0.0.0 --allow-remote
```

Then add the hostname clients will use to `server.allowed_hosts`, and put the server
behind a TLS-terminating reverse proxy. Token authentication stays on.

## Experiment isolation

Phase 8 adds container-isolated workers with resource limits and no network access. On
hosts without Docker, a reduced-isolation mode is used and clearly labelled, and
high-risk candidates (such as generated kernels) are refused.

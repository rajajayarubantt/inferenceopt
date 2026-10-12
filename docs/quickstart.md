# Quick start

## 1. Check the environment

```bash
inferenceopt doctor
```

Fix anything marked `fail`. `info` items are optional extras.

## 2. Open the web UI

```bash
inferenceopt ui
```

The command prints a URL like `http://127.0.0.1:8765/?token=...` and opens it in your
browser. The token signs the browser in; it changes every time the server starts. Keep
the URL private.

Useful options: `--port 9000`, `--no-browser`.

## 3. Write and validate a job

A job says what to optimize, under which constraints, on what workload. Start from
`configs/example-job.yaml` in the repository:

```yaml
job:
  name: local-chat-latency
  model: hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF
  runtimes: [llamacpp]
  objective:
    metric: latency_ms_p95
    direction: minimize
  constraints:
    peak_memory_gib_max: 8
```

Validate it from the CLI or on the UI's **Jobs** page:

```bash
inferenceopt job validate my-job.yaml
```

Running jobs arrives in Phase 4.

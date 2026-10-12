# Configuration

InferenceOpt reads, in order of precedence:

1. `INFERENCEOPT_*` environment variables (nested keys use `__`)
2. The YAML config file (default: `config.yaml` in the per-user config directory, or
   `--config PATH`)
3. Built-in defaults

Show the effective configuration (secrets are never printed):

```bash
inferenceopt config show
```

## Example

See `configs/example-config.yaml`. Typical overrides:

```bash
INFERENCEOPT_SERVER__PORT=9000
INFERENCEOPT_LLM__ENABLED=true
INFERENCEOPT_LLM__PLANNER_MODEL=openai/gpt-4.1
```

Unknown keys are rejected, so typos fail loudly instead of being ignored.

## Secrets

Never put API keys in the config file. Use the environment variable your LLM provider
expects (for example `OPENAI_API_KEY`). Secrets are redacted from logs, and the API
reports only whether a secret is set, never its value.

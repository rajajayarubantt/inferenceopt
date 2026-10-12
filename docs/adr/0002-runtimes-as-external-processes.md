# 0002: Runtimes as external processes

**Status:** Accepted (2026-10-12)

## Context

vLLM and TensorRT-LLM pin specific PyTorch and CUDA versions that conflict with each
other and with users' environments. vLLM does not run natively on Windows. All three
target runtimes (llama.cpp's `llama-server`, `vllm serve`, `trtllm-serve`) expose an
OpenAI-compatible HTTP server.

## Decision

- No runtime is a pip dependency or extra of `inferenceopt`.
- Each adapter locates its runtime by executable path, Python environment path or Docker
  image, launches it with a structured argument list (never a shell string), and talks
  to it over HTTP.
- Benchmarking uses one shared OpenAI-compatible client for every runtime.
- Each adapter declares a support level: `tested`, `experimental` or `unsupported`.
  Initially only llama.cpp is `tested`.

## Consequences

- `pip install inferenceopt` stays small and conflict-free.
- Process isolation (Plan §11) is natural: runtimes already live in separate processes
  or containers.
- Users install runtimes themselves; the UI will guide this and may offer a consented,
  checksum-verified llama.cpp download.

# InferenceOpt

**Given a model, a hardware target and explicit performance/resource constraints,
InferenceOpt finds and validates the best configuration it can within a declared search
budget, and shows the evidence.**

!!! warning "Pre-alpha"
    This is Phase 1 of the [implementation plan](design/implementation-plan.md): the
    project skeleton, configuration, storage, CLI and web UI shell. Hardware scanning,
    runtime adapters, benchmarking and optimization arrive in later phases.

## Principles

- **Measure, don't guess.** Benchmarks decide; an LLM planner may only propose.
- **Constraints before scores.** Configurations that break memory, quality or latency
  limits are rejected, however fast they are.
- **Use existing runtimes.** llama.cpp, vLLM and TensorRT-LLM do the inference.
- **Safe by default.** Nothing generated runs unrestricted on your machine.
- **Reproducible.** Every result records the environment, versions, workload and method.

## Three ways to use it

| Interface | Start with |
|---|---|
| Web UI | `inferenceopt ui` |
| CLI | `inferenceopt --help` |
| Python | `import inferenceopt.services` |

All three call the same service layer, so they always offer the same capabilities.

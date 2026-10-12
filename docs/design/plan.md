# InferOpt: Complete Implementation Plan

## Production-Ready, Autonomous LLM Inference Optimization Platform

**Document type:** Engineering roadmap and research execution plan\
**Version:** 1.0\
**Date:** October 2026\
**Working project name:** InferOpt (rename after checking existing
projects/trademarks)\
**Primary goals:** Open-source software, reproducible research, and
production deployment artifacts

------------------------------------------------------------------------

## 1. Executive Summary

InferOpt is a hardware-aware, goal-driven system that helps a user run
an LLM on available infrastructure. It inventories the hardware, checks
model/runtime compatibility, estimates resource needs, proposes
candidate configurations, executes controlled experiments, measures
performance and quality, selects a feasible configuration, and produces
deployment artifacts with an auditable report.

The system will use **LiteLLM as its configurable LLM integration
layer**. Users can configure supported providers and self-hosted
endpoints through LiteLLM, including model/provider settings, API base
URLs, credentials, timeouts, retries, and routing options. The LLM is a
planner and hypothesis generator---not the source of truth for
performance or safety. All candidate results must be established by
deterministic checks and actual measurements.

### Product principles

1.  **Measure, do not guess:** benchmark measurements determine whether
    an optimization is accepted.
2.  **Constraints before scores:** reject configurations that violate
    memory, quality, latency, compatibility, or safety limits.
3.  **Use existing inference engines first:** integrate established
    runtimes instead of building a complete inference engine in the MVP.
4.  **Safe-by-default execution:** generated code and user-supplied
    artifacts never run unrestricted on the host.
5.  **Reproducible outcomes:** every result records the environment,
    model revision, runtime version, workload, configuration, and
    benchmark method.
6.  **Backend-neutral core:** hardware and runtime capabilities are
    expressed through adapters.
7.  **Research-grade comparisons:** retain baselines, failed trials,
    uncertainty, ablations, and complete evaluation protocols.
8.  **Graceful operation without an LLM:** core profiling, feasibility
    checks, and deterministic tuning should work without configured LLM
    credentials.

------------------------------------------------------------------------

## 2. Product Scope and Non-Goals

### 2.1 Intended user workflow

A user should be able to:

1.  Install InferOpt and run a hardware scan.
2.  Select a model or ask InferOpt to recommend compatible models.
3.  Define the target workload and optimization goal.
4.  Select allowed runtimes, precisions, memory limits, and safety
    policies.
5.  Preview a proposed experiment plan and approve execution.
6.  Run benchmarks and optimization trials.
7.  Inspect the Pareto frontier and why candidates were accepted or
    rejected.
8.  Select a validated candidate.
9.  Generate deployment files, a model artifact when permitted, and a
    reproducible report.
10. Re-run the experiment on another machine and compare results.

### 2.2 Optimization objectives

Support one primary objective and multiple hard constraints. Examples:

-   Maximize aggregate output tokens/second.
-   Minimize time to first token (TTFT).
-   Minimize inter-token latency (ITL).
-   Minimize p95 end-to-end latency.
-   Maximize concurrent requests while meeting an SLO.
-   Maximize supported context length.
-   Minimize peak VRAM/RAM.
-   Minimize energy per generated token when trustworthy energy
    telemetry is available.
-   Minimize tuning time or number of experiments.
-   Maximize quality subject to memory, latency, and throughput
    constraints.

Do not imply that every objective can be optimized simultaneously.
Report trade-offs explicitly.

### 2.3 In scope

-   Hardware discovery and capability reporting.
-   Model metadata and compatibility registry.
-   Memory and feasibility estimation.
-   Runtime adapters.
-   Workload definitions and benchmark harness.
-   Constrained optimization/search.
-   Optional LiteLLM-powered planning.
-   Configuration-level optimization.
-   Optional compiler/kernel experimentation in a later phase.
-   Quality and correctness gates.
-   Safe experiment execution.
-   Deployment artifact generation.
-   Results database, reports, CLI, documentation, and research tooling.

### 2.4 Out of scope for the initial release

-   A new general-purpose LLM inference runtime.
-   Guaranteed support for every accelerator or mobile device.
-   Automatic arbitrary kernel changes in production without review.
-   Automatic model-weight redistribution without license checks.
-   Claims of universal speedups.
-   Training or fine-tuning foundation models as a core feature.
-   Executing arbitrary generated shell scripts directly on the host.

------------------------------------------------------------------------

## 3. System Architecture

``` text
                         User / CI / Python API
                                  |
                         CLI / REST API (later)
                                  |
                       Goal + Workload Specification
                                  |
                    Policy, Constraints, and Validation
                                  |
            +---------------------+---------------------+
            |                     |                     |
      Hardware Registry     Model Registry        LiteLLM Gateway
            |                     |              Planner / Hypotheses
            +---------------------+---------------------+
                                  |
                      Feasibility and Cost Estimator
                                  |
                       Candidate Generation Engine
                                  |
                    Experiment Scheduler / Trial Store
                                  |
                     Isolated Worker / Runtime Adapter
                                  |
                   Benchmark + Quality + Safety Gates
                                  |
                       Results and Pareto Analysis
                                  |
                  Candidate Selection / Human Approval
                                  |
               Artifact Builder + Deployment Generator
                                  |
                   Report, Manifest, and Reproduction
```

### 3.1 Core components

1.  **CLI and API:** user-facing entry points.
2.  **Hardware profiler:** discovers devices and supported features.
3.  **Model registry:** records model metadata, revisions, formats, and
    licenses.
4.  **Runtime registry:** records runtime versions, capabilities, and
    compatibility.
5.  **Goal specification:** formal representation of objectives,
    constraints, and workload.
6.  **Feasibility estimator:** filters impossible or unsafe
    configurations before expensive trials.
7.  **Optimization planner:** combines deterministic search with
    optional LLM-proposed hypotheses.
8.  **Experiment manager:** schedules, isolates, retries, records, and
    stops trials.
9.  **Benchmark harness:** measures throughput, latency, memory, and
    optional energy.
10. **Quality evaluator:** checks task quality and output degradation.
11. **Artifact builder:** packages only validated candidates.
12. **Deployment generator:** produces configuration, container files,
    startup scripts, health checks, and rollback instructions.
13. **Results store:** persists trial metadata, raw measurements, and
    artifacts.
14. **Research module:** runs baselines, ablations, statistical
    comparisons, and exportable tables.

------------------------------------------------------------------------

## 4. Proposed Technology Stack

Keep dependencies modular. Optional hardware backends must not be
mandatory for users who do not have that hardware.

  -----------------------------------------------------------------------
  Layer                   Starting choice         Notes
  ----------------------- ----------------------- -----------------------
  Language                Python 3.11+ or a       Pin and test supported
                          version validated       versions
                          against selected        
                          runtimes                

  Packaging               `uv`, `pyproject.toml`  Lock dependencies for
                                                  reproducibility

  CLI                     Typer + Rich            Human-friendly commands
                                                  and JSON output

  Data validation         Pydantic                Versioned configuration
                                                  schemas

  Internal LLM            **LiteLLM**             Provider-neutral calls,
  integration                                     model aliases, optional
                                                  routing

  Search/tuning           Start with grid/random  Keep algorithm
                          search; add Optuna or   interface replaceable
                          custom constrained      
                          search                  

  GPU runtime             vLLM adapter first      Do not make it the only
                          where compatible        backend

  Local/CPU runtime       llama.cpp adapter       Validate quantization
                                                  and supported formats

  Additional GPU runtime  Optional TensorRT-LLM   NVIDIA-specific
                          adapter                 optional dependency

  Kernel research         Triton first; CUDA/C++  Later-stage feature
                          when justified          

  GPU profiling           PyTorch Profiler;       Tool availability
                          optional NVIDIA Nsight  varies by host
                          tools                   

  Benchmarking            Internal harness plus   Define workload
                          established benchmark   semantics carefully
                          tools where appropriate 

  Quality evaluation      Task-specific test      Avoid a single
                          suites; optional        universal quality score
                          lm-evaluation-harness   
                          integration             

  Metadata store          SQLite for local MVP;   Store structured trials
                          PostgreSQL optional for and manifests
                          service mode            

  Artifact storage        Local filesystem        Content hashes and
                          initially               immutable trial
                                                  directories

  API                     FastAPI, optional after Keep core callable
                          CLI                     without API

  Packaging/deployment    Docker where supported; Containers are not
                          generated native        universally suitable
                          scripts                 for mobile

  Testing                 pytest, property tests, Hardware tests marked
                          integration tests       separately

  Documentation           MkDocs or equivalent    Tutorials, API docs,
                                                  architecture records

  CI                      GitHub Actions plus     Public CI cannot
                          optional self-hosted    validate every GPU path
                          hardware runners        
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 5. LiteLLM Integration Design

### 5.1 Purpose

LiteLLM provides a consistent interface to supported hosted and
self-hosted LLM endpoints. InferOpt should not hard-code a single model
provider. The user chooses which model is used for planning, and the
system remains useful without any LLM configured.

### 5.2 Configuration

Provide a user configuration file such as
`~/.config/inferopt/config.yaml`, plus environment variables for
secrets. Example:

``` yaml
llm:
  enabled: true
  planner_model: "openai/gpt-4.1"
  # Replace with any supported LiteLLM model/provider string or a user-defined alias.
  api_base: null
  temperature: 0.1
  timeout_seconds: 60
  max_retries: 2
  max_planning_tokens: 4000
  allow_external_data: false
  send_hardware_metadata: true
  send_model_names: true
  send_benchmark_traces: false

  # Optional LiteLLM router/deployment configuration can be added here.
  # API keys should be supplied through environment variables or a secret manager,
  # never committed to source control.
```

Example environment variables (names depend on the configured provider):

``` bash
# Example only: set the credential required by your chosen provider.
export OPENAI_API_KEY="..."
```

For a self-hosted OpenAI-compatible endpoint, configure the appropriate
LiteLLM model string and `api_base` according to the provider's current
documentation.

### 5.3 LLM responsibilities

The planner may:

-   Propose candidate runtime configurations from an allowed schema.
-   Explain why a candidate may improve a specific objective.
-   Suggest additional experiments based on measured bottlenecks.
-   Propose a kernel optimization hypothesis in a later research mode.
-   Summarize benchmark results and identify possible anomalies.
-   Generate human-readable deployment documentation from validated
    metadata.

The planner must not:

-   Invent measured throughput, latency, memory use, or correctness.
-   bypass hard constraints or safety policies.
-   execute raw shell code or arbitrary Python directly.
-   claim a kernel is correct without test results.
-   silently download or upload model data.
-   change production deployment without explicit approval.
-   choose an unsupported runtime feature.
-   expose API keys, private prompts, customer data, or benchmark
    payloads to a remote provider without consent.

### 5.4 Structured planner contract

Ask the model for structured candidate proposals validated by Pydantic.
Example schema:

``` json
{
  "hypothesis": "A smaller KV-cache dtype may reduce memory pressure.",
  "candidate_changes": [
    {
      "parameter": "kv_cache_dtype",
      "value": "supported_value_from_runtime_capabilities"
    }
  ],
  "expected_effect": ["lower_memory"],
  "possible_regressions": ["quality_or_latency_change"],
  "required_checks": ["runtime_support", "quality_test", "benchmark"]
}
```

The value above is illustrative, not a universal runtime setting. Actual
choices must come from the selected adapter's capability schema.

### 5.5 Planner workflow

1.  Read the goal, capabilities, policy, and existing results.
2.  Send only the minimum necessary metadata to LiteLLM.
3.  Require structured output matching a strict schema.
4.  Validate all proposed parameters against runtime capabilities and
    allowed search bounds.
5.  Reject invalid proposals; record the reason.
6.  Run the experiment in isolation.
7.  Accept or reject based on actual metrics, quality gates, and hard
    constraints.
8.  Return the measured outcome to the planner for a possible next
    proposal.
9.  Record provider/model, prompt template version, response hash, and
    usage/cost metadata when available.

### 5.6 Failure handling

-   If the LLM is unavailable, continue deterministic search.
-   Apply timeouts, bounded retries, and rate limits.
-   Do not let repeated LLM failures block already scheduled benchmark
    trials.
-   Cache planner results only when the input fingerprint and policy
    version match.
-   Treat all LLM output as untrusted data.
-   Offer local-only mode, with no external API calls.
-   Redact credentials and sensitive paths from logs.

### 5.7 Privacy and cost controls

Provide settings for: - Which metadata can leave the machine. - Whether
raw prompts/completions are ever sent to the planner. - Whether traces
are uploaded (default: no). - Maximum planning calls and estimated
budget per optimization job. - Approved providers and model aliases. -
Local-only execution. - Audit logging of external LLM calls.

------------------------------------------------------------------------

## 6. Hardware Discovery and Capability Model

### 6.1 Discovery targets

Start with: - NVIDIA GPUs: device name, compute capability, VRAM,
driver/runtime availability, supported precision features where
detectable. - CPU: architecture, instruction-set features,
logical/physical cores when available. - Host memory: total and
available RAM. - Operating system, Python version, relevant runtime
versions. - Optional accelerators only through separate adapters.

Later add: - AMD GPUs. - Apple Silicon. - ARM CPUs. - Mobile GPU/NPU
backends through supported runtimes. - Other accelerators as
community-contributed plugins.

### 6.2 Capability schema

Represent capabilities explicitly, for example:

``` yaml
device_id: gpu0
backend: cuda
architecture: detected_value
memory_bytes: 0
supported_dtypes: []
supported_kernels: []
runtime_versions: {}
capability_source: "detected"
confidence: "measured_or_reported"
```

The values are placeholders. Do not fabricate support claims when a
device query is unavailable.

### 6.3 Design requirements

-   Distinguish detected, inferred, and user-specified properties.
-   Support multiple devices and multiple-device configurations.
-   Report free memory separately from total memory.
-   Account for other processes and memory safety margins.
-   Make hardware discovery read-only by default.
-   Do not require administrator privileges for normal profiling.
-   Explain missing capabilities and how to enable optional adapters.

------------------------------------------------------------------------

## 7. Model Registry and Memory Estimation

### 7.1 Model metadata

For each model, record: - Repository/source and immutable revision where
available. - Architecture and parameter count. - Weight format and
precision. - Quantization method and group/block metadata when
applicable. - Tokenizer and chat template. - Maximum supported context
and model-specific constraints. - License and artifact provenance. -
Required custom code or dependencies. - Supported runtime adapters. -
Known quality-evaluation results and hardware benchmarks, with
environment metadata.

Never infer that a model's advertised context length is feasible on the
current machine.

### 7.2 Memory estimation

Estimate total memory as a sum of: - Model weights. - KV cache. -
Runtime/workspace allocations. - Activations and temporary buffers. -
CUDA graphs or compiler/runtime overhead where relevant. - Allocator
fragmentation and safety margin.

A simplified KV-cache estimate for a standard decoder-only transformer
is:

`KV bytes ≈ 2 × layers × KV heads × head dimension × cached tokens × bytes per element × active sequences`

This estimate must be adapted for the architecture, sliding-window
attention, cache dtype, hybrid layers, shared prefixes, quantization
metadata, and runtime implementation. It is a feasibility estimate, not
a guarantee. Actual memory must be measured.

### 7.3 Feasibility stages

1.  Static compatibility checks.
2.  Memory estimate with uncertainty margin.
3.  Small smoke test.
4.  Warm-up and representative workload test.
5.  Full benchmark only if previous stages pass.

Avoid full model downloads or large allocations until the user approves
the candidate and required storage is available.

------------------------------------------------------------------------

## 8. Goals, Workloads, and Constraints

Define a versioned `OptimizationJob` schema containing:

-   Model selection: exact model, candidate list, or allowed registry
    query.
-   Hardware scope.
-   Runtime allowlist.
-   Workload profile: prompt length distribution, generation length,
    concurrency, request rate, batch policy.
-   Primary objective.
-   Hard constraints.
-   Quality test and minimum quality threshold.
-   Trial/time/compute budget.
-   Network/download permissions.
-   Safety policy.
-   Artifact and deployment requirements.
-   Whether LLM planning is enabled.
-   User approval requirements.

Example:

``` yaml
job:
  name: "local-chat-latency"
  model: "user-selected-model"
  hardware: ["gpu0"]
  objective:
    metric: "p95_latency_ms"
    direction: "minimize"
  constraints:
    peak_memory_gib_max: 20
    context_tokens_min: 8192
    output_tokens_per_second_min: null
    quality_regression_max_percent: 1.0
  workload:
    concurrency: [1, 4]
    prompt_tokens: [512, 4096]
    generation_tokens: [128, 512]
  search:
    max_trials: 40
    max_wall_time_minutes: 180
    strategy: "constrained_search"
  llm_planner:
    enabled: true
  deployment:
    generate_container: true
    export_model_artifact: false
```

This is an example schema to refine during implementation. It must be
validated against the supported metric definitions. Do not treat the
example values as universally feasible.

------------------------------------------------------------------------

## 9. Optimization Engine

### 9.1 Candidate parameters

Begin with safe, adapter-supported parameters: - Model and model
format. - Runtime. - Precision/quantization variant. - Context and batch
limits. - Concurrency and batching policy. - KV-cache configuration. -
Supported attention/kernel settings. - Prefix caching and speculative
decoding settings when available. - CPU thread count and affinity where
safe. - Memory offload options where supported.

### 9.2 Search stages

**Stage A --- deterministic feasibility filtering** - Remove
incompatible combinations. - Reject candidates whose conservative memory
estimate exceeds the budget. - Enforce the user's allowlist and safety
constraints.

**Stage B --- baseline and sensitivity tests** - Run the default runtime
configuration. - Change one parameter or a small controlled group. -
Identify the dominant bottleneck.

**Stage C --- search** - Start with grid/random search for transparent
baselines. - Add Bayesian or multi-objective optimization when it is
justified. - Use early stopping only when the stopping rule is
documented and safe. - Prioritize trials by expected information gain or
likely objective improvement.

**Stage D --- LLM-guided proposals** - Use LiteLLM to suggest hypotheses
and valid candidate settings. - Deduplicate proposals against prior
trials. - Validate proposals with the same deterministic gate as other
candidates.

**Stage E --- refinement** - Search around promising configurations. -
Re-run top candidates to assess measurement variance. - Test realistic
concurrency and long-context workloads.

**Stage F --- final validation** - Rebuild from a clean environment. -
Repeat the benchmark and quality suite. - Generate artifacts only after
passing all release gates.

### 9.3 Candidate acceptance

A candidate is deployable only when: - Runtime compatibility is
confirmed. - Build and load tests pass. - Correctness and task-quality
thresholds pass. - Memory and operational constraints pass. -
Performance measurements are valid and reproducible enough for the
selected policy. - Security and artifact provenance checks pass. -
Required human approval has been obtained.

Keep rejected candidates and reasons in the trial record; do not delete
failures that are relevant to research.

------------------------------------------------------------------------

## 10. Benchmarking and Quality Evaluation

### 10.1 Performance metrics

Record as appropriate: - Model load time. - Time to first token
(TTFT). - Inter-token latency (ITL). - End-to-end request latency:
median, p95, p99 where sample size supports it. - Input/prefill tokens
per second. - Output/decode tokens per second. - Aggregate throughput
under concurrency. - Request success/error rate. - Peak GPU memory and
host RAM. - Power/energy only when the telemetry source is
trustworthy. - Optimization time and number of trials.

Define tokens/second carefully: distinguish per-request decode speed
from aggregate output throughput across all concurrent requests.

### 10.2 Benchmark protocol

1.  Capture machine state and versions.
2.  Use fixed, versioned workload traces.
3.  Separate cold-start and warm steady-state results.
4.  Warm up the runtime consistently.
5.  Repeat runs and retain raw measurements.
6.  Record concurrency, input/output lengths, sampling parameters, and
    stop conditions.
7.  Detect OOM, timeouts, throttling, and invalid outputs.
8.  Avoid running competing GPU workloads during controlled trials.
9.  Report uncertainty and measurement noise.
10. Compare the same workload and hardware across baselines.

### 10.3 Quality gates

Use task-appropriate tests: - Exact match or unit tests for
deterministic structured outputs. - Task accuracy for a fixed, versioned
evaluation set. - Schema validity and tool-call correctness for
structured generation. - Output comparisons for numerical or token-level
equivalence where appropriate. - A quality regression threshold with an
explicit evaluation method. - Optional judge-based evaluation, with
documented judge model, prompt, and limitations.

A judge model must not be the sole gate for safety-critical or
high-stakes correctness. Do not equate lower perplexity or faster
generation with preserved task quality.

### 10.4 Research baselines

At minimum compare: 1. Runtime default configuration. 2. Expert/manual
tuning under a declared time budget. 3. Random search or another
transparent search baseline. 4. A standard tuner such as Bayesian
optimization, if applicable. 5. InferOpt without LLM planning. 6.
InferOpt with LLM planning. 7. For kernel work, existing
compiler/runtime autotuning and kernels.

Report tuning budget fairly. If InferOpt consumes more trials or compute
than a baseline, report it.

------------------------------------------------------------------------

## 11. Safe Experiment Execution

Generated code and candidate builds are untrusted.

### Required controls

-   Run trials in isolated containers or disposable workers where
    supported.
-   Use non-root execution, resource limits, bounded wall time, and disk
    quotas.
-   Mount only required input and output directories.
-   Default to no network in the execution worker after required
    artifacts are present.
-   Require explicit permission for downloads and external endpoints.
-   Keep API credentials outside the worker and never inject secrets
    into generated code.
-   Validate generated configuration against strict schemas.
-   Allowlist executable commands and runtime entry points.
-   Prevent path traversal and unsafe artifact extraction.
-   Record image digests, package versions, and build logs.
-   Provide cancellation, cleanup, and recovery after worker failure.
-   Require explicit approval before deploying or replacing a production
    service.
-   Keep deployment changes reversible.

Containers are not a complete security boundary for every threat model.
For high-risk untrusted code, use a stronger isolated environment or do
not run the candidate.

------------------------------------------------------------------------

## 12. Kernel Optimization Research Track

This is a later phase, not a prerequisite for the first release.

### 12.1 Workflow

1.  Identify a measured bottleneck using a profiler.
2.  Select a small, supported kernel family and bounded tensor shapes.
3.  Establish a reference implementation and correctness suite.
4.  Generate a constrained candidate or use an existing autotuner.
5.  Compile in an isolated worker.
6.  Run unit, numerical-tolerance, memory-safety, and shape-boundary
    tests.
7.  Profile the candidate and reference on the target hardware.
8.  Repeat measurements and assess variance.
9.  Reject candidates that fail correctness, quality, or performance
    criteria.
10. Save source, compiler version, build flags, hardware, and raw
    results.
11. Require human review before shipping generated kernel source in a
    release.

### 12.2 Potential targets

-   Small shape-specialized operations.
-   Supported attention sub-operations.
-   Normalization and activation fusion.
-   Matrix-operation configuration.
-   Data movement and memory-layout tuning.
-   Workload-specific fused kernels.

Do not begin by rewriting a complete attention implementation. Start
with a narrow kernel where the reference, correctness properties, and
performance bottleneck are well understood.

### 12.3 Novelty bar

A kernel paper needs more than code generated by an LLM. It should
establish a measurable contribution, such as: - better search sample
efficiency, - reliable optimization from compiler/profiler feedback, -
cross-shape or cross-device transfer, - improved performance on
workloads where baselines leave headroom, - a clear analysis of failure
cases and overhead.

------------------------------------------------------------------------

## 13. Deployment Artifact Generation

The final bundle should be explicit, versioned, and auditable. It may
contain:

``` text
deployment/
  README.md
  manifest.json
  model-reference.yaml
  runtime-config.yaml
  Dockerfile                 # when supported
  compose.yaml               # optional service composition
  start.sh                   # or platform-specific equivalent
  healthcheck.sh
  benchmark-report.json
  benchmark-report.md
  environment.lock
  SBOM or dependency report  # where supported
  rollback.md
```

### Artifact rules

-   Do not copy model weights by default; allow users to choose
    references or export a permitted artifact.
-   Verify model license and source before redistribution.
-   Store checksums for model files and generated artifacts.
-   Pin runtime and dependency versions where possible.
-   Include the exact tested configuration and benchmark command.
-   Mark any untested deployment target as unvalidated.
-   Generate health checks, startup failure diagnostics, and rollback
    instructions.
-   Keep secrets in environment variables or a secret manager.
-   Separate development and production configurations.
-   Never claim production readiness from a successful microbenchmark
    alone.

------------------------------------------------------------------------

## 14. CLI and API Design

Initial CLI proposal:

``` bash
inferopt doctor
inferopt hardware scan
inferopt models list
inferopt models inspect MODEL
inferopt runtimes list
inferopt llm test
inferopt job validate job.yaml
inferopt optimize --job job.yaml
inferopt trials list --job JOB_ID
inferopt results show TRIAL_ID
inferopt report export JOB_ID
inferopt deploy generate --trial TRIAL_ID --output ./deployment
inferopt reproduce --manifest ./deployment/manifest.json
```

Command names are proposals; finalize them through the first CLI
prototype.

### API principles

-   CLI-first MVP; API is optional.
-   Use versioned request/response schemas.
-   Support job cancellation and progress reporting if a service API is
    added.
-   Keep authentication and authorization explicit in service mode.
-   Never expose arbitrary host command execution through the API.
-   Return machine-readable metrics as well as human-readable reports.

------------------------------------------------------------------------

## 15. Data Model and Experiment Tracking

Persist these entities:

-   `HardwareProfile`
-   `ModelReference`
-   `RuntimeProfile`
-   `OptimizationJob`
-   `CandidateConfiguration`
-   `Trial`
-   `BenchmarkRun`
-   `QualityRun`
-   `Artifact`
-   `DeploymentManifest`
-   `PlannerCall`
-   `FailureRecord`

Each trial should include: - Unique ID and parent job. - Hardware and
software environment fingerprints. - Model revision and artifact
hashes. - Runtime and configuration. - Workload ID and seed where
applicable. - Start/end time and exit status. - Raw metric samples and
summary statistics. - Quality results. - Logs and profiler traces where
allowed. - Rejection/acceptance reason. - LLM planner metadata if
used. - Artifact links and checksums.

Do not store secrets or private prompts in the general trial database.
Add retention controls for logs and traces.

------------------------------------------------------------------------

## 16. Repository Layout

``` text
inferopt/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
├── docs/
│   ├── architecture.md
│   ├── installation.md
│   ├── configuration.md
│   ├── llm-providers.md
│   ├── benchmarks.md
│   ├── safety.md
│   ├── supported-hardware.md
│   └── research-protocol.md
├── configs/
│   ├── example-job.yaml
│   └── example-config.yaml
├── src/inferopt/
│   ├── cli/
│   ├── api/
│   ├── config/
│   ├── schemas/
│   ├── hardware/
│   ├── models/
│   ├── runtimes/
│   ├── llm/
│   ├── planner/
│   ├── constraints/
│   ├── optimizer/
│   ├── experiments/
│   ├── benchmarks/
│   ├── quality/
│   ├── artifacts/
│   ├── deployment/
│   ├── storage/
│   ├── reporting/
│   └── security/
├── tests/
│   ├── unit/
│   ├── property/
│   ├── integration/
│   └── hardware/
├── examples/
├── benchmarks/
│   ├── workloads/
│   ├── baselines/
│   └── schemas/
├── research/
│   ├── hypotheses.md
│   ├── experiment-matrix.csv
│   ├── analysis/
│   ├── figures/
│   └── paper/
└── scripts/
```

------------------------------------------------------------------------

## 17. Phased Implementation Roadmap

This plan assumes one primary developer working steadily and reusing
established runtimes. It is an estimate; GPU debugging, hardware access,
kernel work, and paper evaluation can add significant time.

### Phase 0 --- Literature review and requirements (Weeks 1--2)

**Tasks** - Review vLLM, llama.cpp, TensorRT-LLM, TVM and related
tuning/serving work. - Search recent research on LLM inference
autotuning, kernel generation, constrained optimization, and
cross-device transfer. - Write a related-work matrix and identify an
experimentally testable gap. - Choose the first GPU and CPU targets
based on hardware actually available. - Define the model set, workloads,
metrics, quality gates, baselines, and trial budget. - Specify LiteLLM
privacy and provider configuration. - Decide what the first release will
explicitly not support.

**Exit criteria** - Written system requirements. - Reproducible
benchmark protocol. - Initial paper hypothesis and baseline matrix. - No
unsupported claim of novelty.

### Phase 1 --- Project skeleton and schemas (Weeks 2--3)

**Tasks** - Initialize Python package, `uv`, linting, type checks, and
CI. - Create Pydantic schemas for hardware, models, jobs, candidates,
trials, and metrics. - Build configuration loading and secret
redaction. - Implement CLI skeleton and structured logging. - Set up
SQLite storage and schema migrations. - Add a no-LLM mode.

**Exit criteria** - Package installs cleanly. - Example configuration
validates. - Unit tests run on CPU-only CI. - No API secrets appear in
logs.

### Phase 2 --- Hardware and model inventory (Weeks 3--5)

**Tasks** - Implement CPU, RAM, OS, and optional NVIDIA GPU detection. -
Record available memory and runtime/tool versions. - Build the model
metadata schema and initial curated registry. - Implement compatibility
queries. - Add model provenance and license metadata. - Add memory
estimator and uncertainty/safety margin. - Implement preflight reports
and smoke tests.

**Exit criteria** - Clear hardware inventory on supported machines. -
Estimates are labelled as estimates. - Unsupported configurations are
rejected before expensive trials.

### Phase 3 --- Runtime adapters and baseline serving (Weeks 5--8)

**Tasks** - Define a runtime adapter interface: inspect capabilities,
prepare, start, health-check, benchmark, stop, and export
configuration. - Implement the first runtime adapter. - Implement the
second runtime adapter only for a clearly defined target. - Pin versions
and capture runtime metadata. - Normalize tokenization, request, and
output measurement semantics. - Run a fixed baseline workload suite.

**Exit criteria** - Same workload can be run through the adapter
interface. - Baseline metrics and environment metadata are persisted. -
Load failures and OOMs are recorded correctly.

### Phase 4 --- Benchmark and quality harness (Weeks 7--9)

**Tasks** - Implement warm-up, repeated trials, concurrency profiles,
and cold/warm modes. - Measure TTFT, ITL, aggregate throughput,
per-request output rate, memory, and errors. - Add task-specific quality
tests. - Save raw samples and summary statistics. - Add benchmark report
export. - Validate metric definitions with manual spot checks.

**Exit criteria** - Repeated runs are comparable. - Raw results can
regenerate report tables. - Quality regression is a deployment gate, not
merely a note.

### Phase 5 --- Deterministic optimizer (Weeks 9--12)

**Tasks** - Implement goal/constraint evaluation. - Add candidate
generation and capability-aware filtering. - Start with transparent
random/grid search. - Add memory-feasibility screening. - Add early
stopping with explicit rules. - Store every candidate, trial, metric,
and rejection reason. - Produce a Pareto frontier for multiple
objectives. - Re-run top candidates to estimate variance.

**Exit criteria** - Optimizer beats or matches the default on at least
some declared workloads without violating constraints. - Baselines and
search budget are comparable. - Failed trials are handled without
corrupting the job.

### Phase 6 --- LiteLLM planning integration (Weeks 11--13)

**Tasks** - Implement configurable LiteLLM model aliases and provider
settings. - Support optional API base, timeouts, retries, and bounded
planning budget. - Add local-only mode and privacy controls. - Define
structured proposal schemas. - Validate proposals against adapter
capabilities. - Add deduplication, audit records, and failure
fallback. - Compare LLM-guided search against the same optimizer without
LLM planning.

**Exit criteria** - Users can configure their own supported LiteLLM
provider/model. - Invalid proposals are safely rejected. - The system
continues when the provider is unavailable. - Experiments demonstrate
whether the LLM helps, rather than assuming it does.

### Phase 7 --- Deployment generation (Weeks 13--16)

**Tasks** - Generate runtime configuration and launch scripts. -
Generate container artifacts where supported. - Include model/runtime
hashes and tested environment details. - Add health checks, resource
limits, logs, and rollback instructions. - Rebuild and re-run the chosen
candidate from a clean environment. - Add artifact provenance and
license checks.

**Exit criteria** - A user can reproduce a validated deployment using
the generated bundle. - The bundle distinguishes tested and untested
properties. - Secrets are external to artifacts.

### Phase 8 --- Safety and production hardening (Weeks 15--18)

**Tasks** - Isolate workers and apply resource quotas. - Add
cancellation, cleanup, timeout handling, and crash recovery. - Validate
archives and prevent path traversal. - Enforce download/network
policies. - Add dependency scanning, SBOM where practical, and security
documentation. - Add regression tests and hardware-specific CI runners
where available. - Test concurrent jobs and storage recovery.

**Exit criteria** - No generated candidate runs unrestricted on the
host. - Failures are observable and recoverable. - Production deployment
requires explicit approval.

### Phase 9 --- Kernel research extension (Weeks 18--24+, optional)

**Tasks** - Select a measured kernel bottleneck. - Build a reference and
correctness tests. - Integrate Triton or another suitable
compiler/autotuning path. - Run candidate generation in isolation. -
Profile and compare against existing kernels/autotuning. - Validate
numerical tolerances, memory safety, and shape boundaries. - Measure
whether the method improves search efficiency or final performance.

**Exit criteria** - Reproducible improvement on a defined workload, or a
clear negative result. - Correctness and performance claims are
supported by published artifacts. - No claim of generality beyond tested
hardware/shapes.

### Phase 10 --- Research evaluation and release (Weeks 20--28+, overlapping)

**Tasks** - Freeze model and workload revisions. - Run all baselines and
ablations. - Evaluate multiple seeds/runs and report uncertainty. -
Analyze optimization time and compute overhead. - Evaluate unseen
workloads or a second hardware path. - Publish code, configs, benchmark
data and scripts. - Write the paper and target an appropriate venue. -
Publish a versioned release and reproducibility guide.

**Exit criteria** - Results can be reproduced from the repository. -
Research claims match the evidence. - Limitations and negative results
are documented.

------------------------------------------------------------------------

## 18. Research Plan

### Candidate research question

Can a capability-aware, constraint-driven optimizer using measured
feedback and optional LLM-generated hypotheses discover feasible LLM
inference configurations with fewer experiments than strong tuning
baselines while preserving task quality?

### Hypotheses

-   **H1:** Capability and memory prefiltering reduce wasted trials
    without excluding the best feasible candidates too often.
-   **H2:** Feedback-guided search finds a high-performing feasible
    configuration in fewer trials than random search under the same
    budget.
-   **H3:** LiteLLM-powered hypotheses improve search efficiency on some
    workloads; this must be measured, and the effect may be neutral or
    negative.
-   **H4:** Optimization strategies transfer partially between related
    workloads or hardware; transfer limits should be characterized.
-   **H5:** Deployment manifests and clean-room reproduction yield
    results within a declared tolerance of the original run.

### Experimental matrix

Vary: - Hardware family and memory size. - Model architecture and
parameter scale. - Context length and output length. - Concurrency and
request arrival pattern. - Runtime and precision/quantization. - Search
algorithm and trial budget. - LLM planner on/off and different planner
models when useful. - Cold start versus steady state.

### Ablations

-   No feasibility estimator.
-   No historical trial reuse.
-   No LLM planner.
-   No profiling feedback.
-   No quality gate (offline analysis only; never use this configuration
    for deployment).
-   Alternative search strategies.
-   Reduced trial budgets.

### Report

-   Performance gain over runtime default and best baseline.
-   Feasible-candidate success rate.
-   Number of trials and tuning wall time.
-   Quality regressions and constraint violations.
-   Optimization overhead.
-   Variance and confidence intervals where justified.
-   Hardware/workload generalization.
-   Failed cases and limitations.

Do not assert a specific speedup until the experiments support it.

------------------------------------------------------------------------

## 19. Release Milestones

### Milestone A --- Prototype

-   Hardware scan.
-   One model/runtime path.
-   Baseline benchmark.
-   Simple candidate search.
-   Local report.

### Milestone B --- Useful MVP

-   At least two supported runtime/hardware paths where available.
-   Goal and constraint schema.
-   Reproducible trial store.
-   Quality gate.
-   Optional LiteLLM planner.
-   Deployment configuration generation.

### Milestone C --- Research release

-   Strong baselines.
-   Ablations and repeatability checks.
-   Public benchmark suite.
-   Paper-ready plots and tables.
-   Documented limitations.
-   Reproducible release tag.

### Milestone D --- Production-oriented release

-   Worker isolation.
-   Robust recovery.
-   Model provenance and license checks.
-   Versioned deployment manifests.
-   Security documentation.
-   Broader hardware support with real tests.
-   API/service mode only if users need it.

------------------------------------------------------------------------

## 20. Risks and Mitigations

  -----------------------------------------------------------------------
  Risk                                Mitigation
  ----------------------------------- -----------------------------------
  Scope becomes too broad             Start with one GPU path and one
                                      CPU/local path; keep all other
                                      adapters optional

  No genuine research novelty         Review prior work early and define
                                      a measurable algorithmic
                                      contribution

  LLM proposes invalid settings       Strict schemas, adapter capability
                                      checks, sandboxed trials

  Planner adds cost without value     Budget caps, deterministic
                                      fallback, LLM-on/off ablation

  Misleading performance claims       Fixed workloads, repeated runs, raw
                                      measurements, clear metric
                                      definitions

  Model does not fit despite          Conservative memory margin, smoke
  estimates                           tests, measured peak memory

  Quantization degrades output        Task-specific quality gates and
  quality                             explicit trade-off reports

  Generated code is unsafe            Isolation, least privilege, network
                                      controls, human approval

  Hardware cannot be tested in CI     Hardware labels, self-hosted
                                      runners, clearly marked unsupported
                                      paths

  Model/license incompatibility       Track provenance and license; do
                                      not redistribute by default

  Deployment differs from benchmark   Rebuild from manifest and rerun
                                      final validation

  Framework changes break adapters    Versioned capability schemas and
                                      compatibility tests
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 21. Definition of Done

A candidate feature is complete only when it has:

-   A documented interface and supported versions.
-   Unit tests and applicable integration tests.
-   Capability checks and safe failure handling.
-   Structured logs and actionable errors.
-   Benchmark and quality coverage when it affects inference results.
-   Documentation and an example.
-   Reproducible environment information.
-   No hard-coded provider secrets.
-   No unsupported claims about hardware or performance.

The overall project is ready for a research release when the main
experiment can be reproduced by a developer following the documentation,
including the baseline comparisons and the failed/negative cases.

The overall project is ready for production-oriented use only when
deployment, security, reliability, compatibility, and operational
requirements have been tested for the declared supported environment. A
research result or benchmark improvement alone is not sufficient.

------------------------------------------------------------------------

## 22. Immediate First 10 Tasks

1.  Create the repository and write a one-page problem statement.
2.  Build a related-work matrix covering inference runtimes, autotuners,
    compilers, and LLM-based optimization.
3.  Choose the first actual hardware target and a small set of model
    sizes that can realistically fit.
4.  Select one GPU/local runtime and one CPU/local runtime, based on
    hardware access.
5.  Define the versioned optimization-job schema and benchmark metric
    definitions.
6.  Implement `inferopt doctor` and `inferopt hardware scan`.
7.  Implement one baseline benchmark end-to-end and persist raw results.
8.  Implement a runtime adapter interface and a memory feasibility
    estimator.
9.  Implement transparent random/grid search with hard constraints and
    quality gates.
10. Add LiteLLM structured proposals only after deterministic candidate
    validation and benchmark recording work.

**Do not start with automatic kernel generation.** First prove that the
system can reliably measure, compare, and reproduce configurations.
Kernel generation becomes much more useful once the profiling and
validation infrastructure is trustworthy.

------------------------------------------------------------------------

## 23. Final Recommendation

Build InferOpt in layers:

1.  **Reliable profiling and benchmarking.**
2.  **Constraint-aware model/runtime/configuration selection.**
3.  **Reproducible automated search.**
4.  **LiteLLM-powered hypothesis generation with deterministic
    validation.**
5.  **Deployment artifacts and production safeguards.**
6.  **Kernel-level experimentation and additional hardware backends.**
7.  **Rigorous evaluation, open-source release, and paper submission.**

The central product promise should be:

> **Given a model, a hardware target, and explicit performance/resource
> constraints, InferOpt finds and validates the best configuration it
> can within a declared search budget---and shows the evidence.**

That promise is narrower than "automatically invent every possible
optimization," but much more credible, implementable, and valuable to
users. It also gives the research paper a measurable foundation.

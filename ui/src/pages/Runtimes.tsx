// SPDX-License-Identifier: Apache-2.0
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api, unwrap, type Schemas } from "../api/client";
import { Card, ErrorState, Loading, PageHeader } from "../components/ui";
import { fmtBytes } from "../lib/format";

type Runtime = Schemas["RuntimeStatus"];
type Param = Schemas["ParamSpec"];
type Settings = Schemas["RuntimeSettings"];

const field =
  "w-full rounded-md border border-zinc-300 bg-white px-2 py-1 text-sm dark:border-zinc-700 dark:bg-zinc-950";
const btn =
  "rounded-md border border-zinc-300 px-3 py-1.5 text-sm hover:bg-zinc-100 disabled:opacity-50 dark:border-zinc-700 dark:hover:bg-zinc-800";
const primary =
  "rounded-md bg-accent px-4 py-1.5 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50";

function SupportBadge({ level }: { level: Runtime["support_level"] }) {
  const styles = {
    tested: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300",
    experimental: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
    unsupported: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
  } as const;
  const title =
    level === "experimental"
      ? "Implemented and contract-tested, not yet validated on real hardware"
      : undefined;
  return (
    <span title={title} className={`rounded px-2 py-0.5 text-xs font-medium ${styles[level]}`}>
      {level}
    </span>
  );
}

function SettingsForm({ runtime }: { runtime: Runtime }) {
  const qc = useQueryClient();
  const [s, setS] = useState<Settings>(runtime.settings);
  const save = useMutation({
    mutationFn: async () =>
      unwrap(
        await api.PUT("/api/v1/runtimes/{name}/settings", {
          params: { path: { name: runtime.name } },
          body: s,
        }),
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["runtimes"] }),
  });
  const set = (patch: Partial<Settings>) => setS((prev) => ({ ...prev, ...patch }));
  const docker = s.mode === "docker";

  return (
    <div className="space-y-2">
      <div className="grid gap-2 md:grid-cols-[10rem_1fr]">
        <label className="text-xs text-zinc-500">
          Mode
          <select
            className={field}
            value={s.mode ?? "native"}
            onChange={(e) => set({ mode: e.target.value as Settings["mode"] })}
          >
            <option value="native">native</option>
            <option value="docker">docker</option>
          </select>
        </label>
        {docker ? (
          <label className="text-xs text-zinc-500">
            Image (pin a tag or digest)
            <input
              className={field}
              value={s.image ?? ""}
              placeholder="e.g. ghcr.io/ggml-org/llama.cpp:server-cuda-b11429"
              onChange={(e) => set({ image: e.target.value || null })}
            />
          </label>
        ) : runtime.name === "vllm" ? (
          <label className="text-xs text-zinc-500">
            Python environment with vLLM
            <input
              className={field}
              value={s.python_env ?? ""}
              placeholder="/home/me/.venvs/vllm"
              onChange={(e) => set({ python_env: e.target.value || null })}
            />
          </label>
        ) : (
          <label className="text-xs text-zinc-500">
            Executable (absolute path; empty = auto-detect)
            <input
              className={field}
              value={s.executable ?? ""}
              onChange={(e) => set({ executable: e.target.value || null })}
            />
          </label>
        )}
      </div>
      <button type="button" className={btn} disabled={save.isPending} onClick={() => save.mutate()}>
        {save.isPending ? "Saving…" : "Save settings"}
      </button>
      {save.error && <ErrorState error={save.error} />}
    </div>
  );
}

function LlamaInstall() {
  const qc = useQueryClient();
  const [taskId, setTaskId] = useState<string | null>(null);
  const plan = useMutation({
    mutationFn: async () => unwrap(await api.GET("/api/v1/runtimes/llamacpp/install-plan")),
  });
  const start = useMutation({
    mutationFn: async () =>
      unwrap(
        await api.POST("/api/v1/runtimes/llamacpp/install", {
          body: { channel: "stable", variant: "auto", confirm: true },
        }),
      ),
    onSuccess: (t) => setTaskId(t.id),
  });
  const task = useQuery({
    queryKey: ["install", taskId],
    enabled: taskId !== null,
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/runtimes/llamacpp/install/{task_id}", {
          params: { path: { task_id: taskId ?? "" } },
        }),
      ),
    refetchInterval: (q) =>
      q.state.data && ["done", "failed"].includes(q.state.data.status) ? false : 1000,
  });
  const status = task.data?.status;
  useEffect(() => {
    if (status === "done") void qc.invalidateQueries({ queryKey: ["runtimes"] });
  }, [status, qc]);

  return (
    <div className="space-y-2 rounded-md border border-zinc-200 p-3 dark:border-zinc-800">
      <div className="text-sm font-medium">Install an official llama.cpp build</div>
      <p className="text-xs text-zinc-500">
        Downloads from github.com/ggml-org/llama.cpp and verifies every file against GitHub's
        SHA-256 digest. Nothing is downloaded until you confirm.
      </p>
      {!plan.data && (
        <button
          type="button"
          className={btn}
          disabled={plan.isPending}
          onClick={() => plan.mutate()}
        >
          {plan.isPending ? "Checking releases…" : "Check what would be installed"}
        </button>
      )}
      {plan.error && <ErrorState error={plan.error} />}
      {plan.data && !taskId && (
        <div className="space-y-2 text-sm">
          <div>
            Build <strong>{plan.data.tag}</strong> ({plan.data.variant}),{" "}
            {fmtBytes(plan.data.total_bytes)}:
          </div>
          <ul className="list-disc pl-5 font-mono text-xs">
            {plan.data.assets.map((a) => (
              <li key={a}>{a}</li>
            ))}
          </ul>
          <button
            type="button"
            className={primary}
            disabled={start.isPending}
            onClick={() => start.mutate()}
          >
            Download {fmtBytes(plan.data.total_bytes)} and install
          </button>
        </div>
      )}
      {task.data && (
        <div className="space-y-1 text-sm">
          <div>
            {task.data.status === "done"
              ? "Installed."
              : task.data.status === "failed"
                ? "Install failed."
                : `Downloading ${task.data.current_asset ?? ""}…`}
          </div>
          {task.data.total_bytes ? (
            <div
              className="h-2 w-full rounded bg-zinc-200 dark:bg-zinc-800"
              role="progressbar"
              aria-valuenow={task.data.downloaded_bytes ?? 0}
              aria-valuemax={task.data.total_bytes}
            >
              <div
                className="h-full rounded"
                style={{
                  width: `${((task.data.downloaded_bytes ?? 0) / task.data.total_bytes) * 100}%`,
                  background: "var(--series-1)",
                }}
              />
            </div>
          ) : null}
          <div className="text-xs text-zinc-500">
            {fmtBytes(task.data.downloaded_bytes ?? 0)} of {fmtBytes(task.data.total_bytes)}
          </div>
          {task.data.error && <ErrorState error={new Error(task.data.error)} />}
        </div>
      )}
    </div>
  );
}

function ParamInput({
  p,
  value,
  onChange,
}: {
  p: Param;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const placeholder =
    p.default === null || p.default === undefined ? "runtime default" : String(p.default);
  if (p.type === "bool")
    return (
      <select
        className={field}
        value={value === undefined ? "" : String(value)}
        onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value === "true")}
      >
        <option value="">default ({placeholder})</option>
        <option value="true">true</option>
        <option value="false">false</option>
      </select>
    );
  if (p.type === "choice")
    return (
      <select
        className={field}
        value={(value as string | undefined) ?? ""}
        onChange={(e) => onChange(e.target.value || undefined)}
      >
        <option value="">default ({placeholder})</option>
        {p.choices?.map((c) => (
          <option key={c}>{c}</option>
        ))}
      </select>
    );
  return (
    <input
      className={field}
      type={p.type === "str" ? "text" : "number"}
      step={p.type === "float" ? "any" : 1}
      placeholder={placeholder}
      value={value === undefined ? "" : String(value)}
      onChange={(e) => {
        const raw = e.target.value;
        if (raw === "") onChange(undefined);
        else onChange(p.type === "str" ? raw : Number(raw));
      }}
    />
  );
}

function TestPanel({ runtime }: { runtime: Runtime }) {
  const models = useQuery({
    queryKey: ["models"],
    queryFn: async () => unwrap(await api.GET("/api/v1/models")),
  });
  const usable = (models.data ?? []).filter(
    (m) => m.path && runtime.weight_formats.includes(m.weight_format ?? ""),
  );
  const [modelId, setModelId] = useState("");
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const body = { model_id: modelId, config };
  const preview = useMutation({
    mutationFn: async () =>
      unwrap(
        await api.POST("/api/v1/runtimes/{name}/launch-preview", {
          params: { path: { name: runtime.name } },
          body,
        }),
      ),
  });
  const smoke = useMutation({
    mutationFn: async () =>
      unwrap(
        await api.POST("/api/v1/runtimes/{name}/smoke-test", {
          params: { path: { name: runtime.name } },
          body: { ...body, max_tokens: 24, confirm: true },
        }),
      ),
  });
  const r = smoke.data;

  return (
    <div className="space-y-3">
      <label className="block text-xs text-zinc-500">
        Model ({runtime.weight_formats.join(", ")} files registered on the Models page)
        <select className={field} value={modelId} onChange={(e) => setModelId(e.target.value)}>
          <option value="">Choose a model…</option>
          {usable.map((m) => (
            <option key={m.id} value={m.id}>
              {m.display_name} ({m.id})
            </option>
          ))}
        </select>
      </label>
      <details>
        <summary className="cursor-pointer text-sm">
          Parameters ({runtime.parameters.length})
        </summary>
        <div className="mt-2 grid gap-2 md:grid-cols-3">
          {runtime.parameters.map((p) => (
            <label key={p.name} className="text-xs text-zinc-500" title={p.description}>
              {p.name}
              <ParamInput
                p={p}
                value={config[p.name]}
                onChange={(v) =>
                  setConfig((prev) => {
                    const next = { ...prev };
                    if (v === undefined) delete next[p.name];
                    else next[p.name] = v;
                    return next;
                  })
                }
              />
            </label>
          ))}
        </div>
      </details>
      <div className="flex gap-2">
        <button
          type="button"
          className={btn}
          disabled={!modelId || preview.isPending}
          onClick={() => preview.mutate()}
        >
          Preview command
        </button>
        <button
          type="button"
          className={primary}
          disabled={!modelId || smoke.isPending}
          onClick={() => {
            if (
              confirm(
                `Start ${runtime.display_name} with ${modelId}, generate a few tokens, then stop it?`,
              )
            )
              smoke.mutate();
          }}
        >
          {smoke.isPending ? "Running… (loading the model can take a while)" : "Run smoke test"}
        </button>
      </div>
      {(preview.error || smoke.error) && <ErrorState error={preview.error ?? smoke.error} />}
      {preview.data && (
        <div className="space-y-1">
          {preview.data.errors.map((e) => (
            <div key={e} className="text-sm text-red-700 dark:text-red-400">
              ✕ {e}
            </div>
          ))}
          {preview.data.spec && (
            <pre className="overflow-x-auto rounded bg-zinc-100 p-2 font-mono text-xs dark:bg-zinc-950">
              {preview.data.spec.argv.join(" ")}
            </pre>
          )}
        </div>
      )}
      {r && (
        <div className="space-y-2 rounded-md border border-zinc-200 p-3 text-sm dark:border-zinc-800">
          <div
            className={
              r.ok ? "text-emerald-700 dark:text-emerald-400" : "text-red-700 dark:text-red-400"
            }
          >
            {r.ok
              ? "✓ Smoke test passed"
              : `✕ ${r.failure?.kind ?? "failed"}: ${r.failure?.message ?? r.errors?.join("; ")}`}
          </div>
          {r.ok && (
            <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-0.5 text-sm">
              <dt className="text-zinc-500">Load time</dt>
              <dd className="font-mono">{r.load_time_s?.toFixed(2)} s</dd>
              <dt className="text-zinc-500">Time to first token</dt>
              <dd className="font-mono">{((r.ttft_s ?? 0) * 1000).toFixed(0)} ms</dd>
              <dt className="text-zinc-500">Decode speed (1 request)</dt>
              <dd className="font-mono">{r.decode_tokens_per_s?.toFixed(1) ?? "?"} tok/s</dd>
              <dt className="text-zinc-500">Tokens</dt>
              <dd className="font-mono">
                {r.completion_tokens} ({r.token_count_source})
              </dd>
              <dt className="text-zinc-500">Output</dt>
              <dd>{r.output_excerpt}</dd>
            </dl>
          )}
          <p className="text-xs text-zinc-500">
            A single request: a sanity check, not a benchmark.
          </p>
          {r.log_tail && (
            <details>
              <summary className="cursor-pointer text-xs">Runtime log (tail)</summary>
              <pre className="mt-1 max-h-64 overflow-auto rounded bg-zinc-100 p-2 font-mono text-[11px] dark:bg-zinc-950">
                {r.log_tail}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

export function Runtimes() {
  const runtimes = useQuery({
    queryKey: ["runtimes"],
    queryFn: async () => unwrap(await api.GET("/api/v1/runtimes")),
  });

  return (
    <>
      <PageHeader
        title="Runtimes"
        description="Inference engines run as separate processes or containers. Experimental adapters are contract-tested but not yet validated on real hardware."
      />
      {runtimes.isPending && <Loading />}
      {runtimes.error && <ErrorState error={runtimes.error} />}
      <div className="space-y-4">
        {runtimes.data?.map((rt) => (
          <Card key={rt.name}>
            <div className="mb-3 flex flex-wrap items-center gap-3">
              <h2 className="text-lg font-semibold">{rt.display_name}</h2>
              <SupportBadge level={rt.support_level} />
              <span className="text-xs text-zinc-500">
                {rt.weight_formats.join(", ")} · native on {rt.platforms.join(", ")}
              </span>
              <a
                href={rt.homepage}
                target="_blank"
                rel="noreferrer"
                className="ml-auto text-xs text-accent hover:underline"
              >
                project
              </a>
            </div>
            <div className="mb-3 text-sm">
              {rt.installation ? (
                <>
                  <span className="text-emerald-700 dark:text-emerald-400">Found</span>{" "}
                  <span className="font-mono text-xs">{rt.installation.executable}</span> (
                  {rt.installation.mode}, {rt.installation.version ?? "version unknown"})
                  {(rt.installation.devices?.length ?? 0) > 0 && (
                    <div className="text-xs text-zinc-500">
                      Devices: {rt.installation.devices?.join("; ")}
                    </div>
                  )}
                </>
              ) : (
                <span className="text-red-700 dark:text-red-400">Not found</span>
              )}
            </div>
            {rt.notes.length > 0 && (
              <ul className="mb-3 list-disc pl-5 text-sm text-amber-800 dark:text-amber-300">
                {rt.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            )}
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="space-y-3">
                <SettingsForm key={JSON.stringify(rt.settings)} runtime={rt} />
                {rt.name === "llamacpp" && rt.settings.mode !== "docker" && <LlamaInstall />}
              </div>
              <div>{rt.installation && <TestPanel runtime={rt} />}</div>
            </div>
          </Card>
        ))}
      </div>
    </>
  );
}

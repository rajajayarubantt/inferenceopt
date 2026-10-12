// SPDX-License-Identifier: Apache-2.0
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { api, unwrap, type Schemas } from "../api/client";
import { fmtBytes } from "../lib/format";
import { MemoryBar } from "./MemoryBar";
import { ErrorState } from "./ui";

type Inputs = Schemas["EstimateInputs"];
type Verdict = Schemas["PreflightReport"]["verdict"];

const KV_DTYPES = ["f16", "bf16", "f32", "q8_0", "q4_0", "fp8"] as const;
const RUNTIMES = ["llamacpp", "vllm", "tensorrt_llm"] as const;

const VERDICT: Record<Verdict, { label: string; icon: string; cls: string }> = {
  fits: {
    label: "Fits",
    icon: "✓",
    cls: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300",
  },
  tight: {
    label: "Tight",
    icon: "!",
    cls: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
  },
  does_not_fit: {
    label: "Does not fit",
    icon: "✕",
    cls: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
  },
  unknown: {
    label: "Unknown",
    icon: "?",
    cls: "bg-zinc-200 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300",
  },
};

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  const v = VERDICT[verdict];
  return (
    <span
      className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-medium ${v.cls}`}
    >
      <span aria-hidden>{v.icon}</span>
      {v.label}
    </span>
  );
}

const field =
  "w-full rounded-md border border-zinc-300 bg-white px-2 py-1 text-sm dark:border-zinc-700 dark:bg-zinc-950";

export function PreflightPanel({ modelId }: { modelId: string }) {
  const [inputs, setInputs] = useState<Inputs>({
    context_tokens: 4096,
    concurrency: 1,
    kv_cache_dtype: "f16",
    batch_tokens: 512,
    flash_attention: true,
    weight_bits_per_param: null,
  });
  const [runtimes, setRuntimes] = useState<string[]>([...RUNTIMES]);
  const run = useMutation({
    mutationFn: async (rescan: boolean) =>
      unwrap(
        await api.POST("/api/v1/preflight", {
          body: { model_id: modelId, inputs, runtimes, rescan },
        }),
      ),
  });
  const report = run.data;
  const set = <K extends keyof Inputs>(key: K, value: Inputs[K]) =>
    setInputs((prev) => ({ ...prev, [key]: value }));

  const markers =
    report?.devices
      .filter((d) => d.available_bytes !== null)
      .map((d) => ({ label: `${d.device_id} usable`, bytes: d.available_bytes ?? 0 })) ?? [];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
        <label className="text-xs text-zinc-500">
          Context tokens
          <input
            type="number"
            min={1}
            className={field}
            value={inputs.context_tokens}
            onChange={(e) => set("context_tokens", Number(e.target.value))}
          />
        </label>
        <label className="text-xs text-zinc-500">
          Concurrent sequences
          <input
            type="number"
            min={1}
            className={field}
            value={inputs.concurrency}
            onChange={(e) => set("concurrency", Number(e.target.value))}
          />
        </label>
        <label className="text-xs text-zinc-500">
          KV cache dtype
          <select
            className={field}
            value={inputs.kv_cache_dtype}
            onChange={(e) => set("kv_cache_dtype", e.target.value as Inputs["kv_cache_dtype"])}
          >
            {KV_DTYPES.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </label>
        <label className="text-xs text-zinc-500">
          Batch tokens
          <input
            type="number"
            min={1}
            className={field}
            value={inputs.batch_tokens}
            onChange={(e) => set("batch_tokens", Number(e.target.value))}
          />
        </label>
        <label className="text-xs text-zinc-500">
          Bits / weight (optional)
          <input
            type="number"
            min={1}
            step={0.01}
            placeholder="from model"
            className={field}
            value={inputs.weight_bits_per_param ?? ""}
            onChange={(e) =>
              set("weight_bits_per_param", e.target.value === "" ? null : Number(e.target.value))
            }
          />
        </label>
        <label className="flex items-end gap-2 pb-1 text-sm">
          <input
            type="checkbox"
            checked={inputs.flash_attention}
            onChange={(e) => set("flash_attention", e.target.checked)}
          />
          Flash attention
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-4 text-sm">
        <span className="text-xs text-zinc-500">Runtimes:</span>
        {RUNTIMES.map((r) => (
          <label key={r} className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={runtimes.includes(r)}
              onChange={(e) =>
                setRuntimes((prev) =>
                  e.target.checked ? [...prev, r] : prev.filter((x) => x !== r),
                )
              }
            />
            {r}
          </label>
        ))}
        <button
          type="button"
          disabled={run.isPending || runtimes.length === 0}
          onClick={() => run.mutate(false)}
          className="ml-auto rounded-md bg-accent px-4 py-1.5 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
        >
          {run.isPending ? "Checking…" : "Check fit"}
        </button>
        <button
          type="button"
          disabled={run.isPending || runtimes.length === 0}
          onClick={() => run.mutate(true)}
          className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm hover:bg-zinc-100 disabled:opacity-50 dark:border-zinc-700 dark:hover:bg-zinc-800"
          title="Re-scan hardware first (free memory changes over time)"
        >
          Re-scan &amp; check
        </button>
      </div>

      {run.error && <ErrorState error={run.error} />}
      {report && (
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <VerdictBadge verdict={report.verdict} />
            <span className="text-xs text-zinc-500">
              Estimate only: real usage is measured in benchmark trials.
            </span>
          </div>
          {report.estimate && <MemoryBar estimate={report.estimate} markers={markers} />}
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-zinc-500">
              <tr>
                <th className="pb-1 font-medium">Device</th>
                <th className="pb-1 font-medium">Usable now</th>
                <th className="pb-1 font-medium">Verdict</th>
                <th className="pb-1 font-medium">Reason</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
              {report.devices.map((d) => (
                <tr key={d.device_id} className="align-top">
                  <td className="py-1.5 pr-3">
                    <span className="font-mono text-xs">{d.device_id}</span> {d.device_name}
                  </td>
                  <td className="py-1.5 pr-3 font-mono">{fmtBytes(d.available_bytes)}</td>
                  <td className="py-1.5 pr-3">
                    <VerdictBadge verdict={d.verdict} />
                  </td>
                  <td className="py-1.5 text-zinc-600 dark:text-zinc-400">{d.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <ul className="space-y-1 text-sm">
            {report.runtimes.map((r) => (
              <li key={r.runtime}>
                <span
                  className={
                    r.compatible
                      ? "text-emerald-700 dark:text-emerald-400"
                      : "text-red-700 dark:text-red-400"
                  }
                >
                  {r.compatible ? "✓" : "✕"} {r.runtime}
                </span>
                {(r.reasons?.length ?? 0) > 0 && (
                  <span className="text-zinc-600 dark:text-zinc-400">
                    {" "}
                    — {r.reasons?.join(" ")}
                  </span>
                )}
              </li>
            ))}
          </ul>
          {(report.messages?.length ?? 0) > 0 && (
            <ul className="list-disc space-y-0.5 pl-5 text-sm text-amber-800 dark:text-amber-300">
              {report.messages?.map((m) => (
                <li key={m}>{m}</li>
              ))}
            </ul>
          )}
          {report.estimate && (
            <details className="text-xs text-zinc-500">
              <summary className="cursor-pointer">Assumptions</summary>
              <ul className="mt-1 list-disc pl-5">
                {report.estimate.assumptions.map((a) => (
                  <li key={a}>{a}</li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

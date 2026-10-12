// SPDX-License-Identifier: Apache-2.0
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, unwrap, type Schemas } from "../api/client";
import { PreflightPanel } from "../components/PreflightPanel";
import { Card, ErrorState, Loading, PageHeader } from "../components/ui";
import { fmtBytes, fmtCount } from "../lib/format";

type Model = Schemas["ModelReference"];

const field =
  "w-full rounded-md border border-zinc-300 bg-white px-3 py-1.5 text-sm dark:border-zinc-700 dark:bg-zinc-950";

function ModelFacts({ model }: { model: Model }) {
  const a = model.architecture;
  const rows: [string, string][] = [
    ["Format", `${model.weight_format ?? "?"} (${model.quantization ?? "?"})`],
    ["Parameters", fmtCount(model.parameter_count)],
    ["Weights on disk", fmtBytes(model.weight_bytes)],
    ["Max context", model.context_length_max?.toLocaleString() ?? "?"],
    ["License", model.license ?? "unknown: check the model card"],
  ];
  if (a)
    rows.push([
      "Architecture",
      `${a.family ?? "?"}: ${a.num_layers} layers, hidden ${a.hidden_size}, ${a.num_attention_heads} heads (${a.num_kv_heads} KV) × ${a.head_dim}`,
    ]);
  if (model.path) rows.push(["Path", model.path]);
  return (
    <>
      <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-zinc-500">{k}</dt>
            <dd className="break-all">{v}</dd>
          </div>
        ))}
      </dl>
      {model.origin === "curated" && (
        <p className="mt-2 text-xs text-zinc-500">
          Curated entry: shape values come from the published config and are not yet verified
          against downloaded files.
        </p>
      )}
      {(model.notes?.length ?? 0) > 0 && (
        <ul className="mt-2 list-disc pl-5 text-sm text-amber-800 dark:text-amber-300">
          {model.notes?.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
    </>
  );
}

function AddLocalModel({ onAdded }: { onAdded: (m: Model) => void }) {
  const [path, setPath] = useState("");
  const [id, setId] = useState("");
  const body = { path: path.trim(), id: id.trim() || null };
  const inspect = useMutation({
    mutationFn: async () => unwrap(await api.POST("/api/v1/models/inspect", { body })),
  });
  const add = useMutation({
    mutationFn: async () => unwrap(await api.POST("/api/v1/models", { body })),
    onSuccess: (m) => {
      setPath("");
      setId("");
      inspect.reset();
      onAdded(m);
    },
  });

  return (
    <Card title="Add a local model">
      <p className="mb-3 text-xs text-zinc-500">
        A <code>.gguf</code> file or a Hugging Face model directory. Only headers are read; nothing
        is copied, downloaded or loaded.
      </p>
      <div className="grid gap-2 md:grid-cols-[1fr_16rem_auto_auto]">
        <input
          aria-label="Absolute path"
          placeholder="Absolute path, e.g. D:\models\qwen2.5-0.5b-instruct-q4_k_m.gguf"
          className={field}
          value={path}
          onChange={(e) => {
            setPath(e.target.value);
            inspect.reset();
          }}
        />
        <input
          aria-label="Registry id (optional)"
          placeholder="id (optional)"
          className={field}
          value={id}
          onChange={(e) => setId(e.target.value)}
        />
        <button
          type="button"
          disabled={!path.trim() || inspect.isPending}
          onClick={() => inspect.mutate()}
          className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm hover:bg-zinc-100 disabled:opacity-50 dark:border-zinc-700 dark:hover:bg-zinc-800"
        >
          Inspect
        </button>
        <button
          type="button"
          disabled={!path.trim() || add.isPending}
          onClick={() => add.mutate()}
          className="rounded-md bg-accent px-4 py-1.5 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
        >
          Add
        </button>
      </div>
      {(inspect.error || add.error) && (
        <div className="mt-3">
          <ErrorState error={inspect.error ?? add.error} />
        </div>
      )}
      {inspect.data && (
        <div className="mt-3 rounded-md border border-zinc-200 p-3 dark:border-zinc-800">
          <div className="mb-2 font-medium">
            {inspect.data.display_name}{" "}
            <span className="font-mono text-xs text-zinc-500">{inspect.data.id}</span>
          </div>
          <ModelFacts model={inspect.data} />
        </div>
      )}
    </Card>
  );
}

export function Models() {
  const qc = useQueryClient();
  const [selected, setSelected] = useState<string | null>(null);
  const models = useQuery({
    queryKey: ["models"],
    queryFn: async () => unwrap(await api.GET("/api/v1/models")),
  });
  const remove = useMutation({
    mutationFn: async (modelId: string) => {
      const res = await api.DELETE("/api/v1/models/{model_id}", {
        params: { path: { model_id: modelId } },
      });
      if (!res.response.ok) unwrap(res);
    },
    onSuccess: () => {
      setSelected(null);
      void qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
  const current = models.data?.find((m) => m.id === selected) ?? null;

  return (
    <>
      <PageHeader
        title="Models"
        description="Curated models plus your own local files. Pick one to estimate memory and check whether it fits."
      />
      <div className="space-y-4">
        <Card>
          {models.isPending && <Loading />}
          {models.error && <ErrorState error={models.error} />}
          {models.data && (
            <table className="w-full text-left text-sm">
              <thead className="text-xs text-zinc-500">
                <tr>
                  <th className="pb-2 font-medium">Model</th>
                  <th className="pb-2 font-medium">Format</th>
                  <th className="pr-3 pb-2 text-right font-medium">Params</th>
                  <th className="pr-3 pb-2 text-right font-medium">Max context</th>
                  <th className="pb-2 font-medium">License</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
                {models.data.map((m) => (
                  <tr
                    key={m.id}
                    onClick={() => setSelected(m.id)}
                    className={`cursor-pointer hover:bg-zinc-50 dark:hover:bg-zinc-800/50 ${
                      m.id === selected ? "bg-accent-soft dark:bg-teal-950" : ""
                    }`}
                  >
                    <td className="py-2 pr-3">
                      {m.display_name}
                      <span
                        className={`ml-2 rounded px-1.5 text-[10px] ${
                          m.origin === "curated"
                            ? "bg-zinc-200 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400"
                            : "bg-sky-100 text-sky-800 dark:bg-sky-900/40 dark:text-sky-300"
                        }`}
                      >
                        {m.origin}
                      </span>
                      <div className="font-mono text-xs text-zinc-500">{m.id}</div>
                    </td>
                    <td className="py-2 pr-3">
                      {m.weight_format ?? "?"}
                      <div className="text-xs text-zinc-500">{m.quantization}</div>
                    </td>
                    <td className="py-2 pr-3 text-right font-mono">
                      {fmtCount(m.parameter_count)}
                    </td>
                    <td className="py-2 pr-3 text-right font-mono">
                      {m.context_length_max?.toLocaleString() ?? "?"}
                    </td>
                    <td className="py-2">{m.license ?? "?"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>

        {current && (
          <Card title={`${current.display_name}`}>
            <div className="mb-4 flex items-start gap-4">
              <div className="flex-1">
                <ModelFacts model={current} />
              </div>
              {current.origin === "user" && (
                <button
                  type="button"
                  onClick={() => {
                    if (confirm(`Remove ${current.id} from the registry? Files are not deleted.`))
                      remove.mutate(current.id);
                  }}
                  className="rounded-md border border-red-300 px-3 py-1 text-sm text-red-700 hover:bg-red-50 dark:border-red-900 dark:text-red-400 dark:hover:bg-red-950"
                >
                  Remove
                </button>
              )}
            </div>
            {remove.error && <ErrorState error={remove.error} />}
            <h3 className="mb-2 text-sm font-medium">Will it fit?</h3>
            <PreflightPanel key={current.id} modelId={current.id} />
          </Card>
        )}

        <AddLocalModel
          onAdded={(m) => {
            void qc.invalidateQueries({ queryKey: ["models"] });
            setSelected(m.id);
          }}
        />
      </div>
    </>
  );
}

// SPDX-License-Identifier: Apache-2.0
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { api, unwrap } from "../api/client";
import { Card, ErrorState, PageHeader } from "../components/ui";

const EXAMPLE = `job:
  name: local-chat-latency
  model: hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF
  runtimes: [llamacpp]
  objective:
    metric: latency_ms_p95
    direction: minimize
  constraints:
    peak_memory_gib_max: 8
  workload:
    concurrency: [1, 4]
    prompt_tokens: [512, 2048]
    generation_tokens: [128]
`;

export function Jobs() {
  const [yaml, setYaml] = useState(EXAMPLE);
  const validate = useMutation({
    mutationFn: async (text: string) =>
      unwrap(await api.POST("/api/v1/jobs/validate", { body: { yaml: text } })),
  });
  const result = validate.data;

  return (
    <>
      <PageHeader
        title="Jobs"
        description="Running optimization jobs arrives in Phase 4. For now, validate job files here."
      />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Job YAML">
          <textarea
            aria-label="Job YAML"
            value={yaml}
            onChange={(e) => setYaml(e.target.value)}
            spellCheck={false}
            className="h-96 w-full resize-y rounded-md border border-zinc-300 bg-zinc-50 p-3 font-mono text-xs dark:border-zinc-700 dark:bg-zinc-950"
          />
          <button
            type="button"
            onClick={() => validate.mutate(yaml)}
            disabled={validate.isPending}
            className="mt-3 rounded-md bg-accent px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
          >
            {validate.isPending ? "Validating…" : "Validate"}
          </button>
        </Card>
        <Card title="Result">
          {validate.error && <ErrorState error={validate.error} />}
          {!result && !validate.error && (
            <p className="text-sm text-zinc-500">Press Validate to check the file.</p>
          )}
          {result?.ok && (
            <p className="text-sm text-emerald-700 dark:text-emerald-400">
              Valid job <strong>{result.job?.name}</strong>: optimize{" "}
              <code>{result.job?.objective.metric}</code> ({result.job?.objective.direction}).
            </p>
          )}
          {result && !result.ok && (
            <ul className="space-y-2 text-sm">
              {result.issues?.map((issue) => (
                <li key={`${issue.location}:${issue.message}`}>
                  <code className="text-red-700 dark:text-red-400">{issue.location}</code>
                  <div className="text-zinc-600 dark:text-zinc-400">{issue.message}</div>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </>
  );
}

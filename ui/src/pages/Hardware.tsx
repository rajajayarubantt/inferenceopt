// SPDX-License-Identifier: Apache-2.0
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError, unwrap, type Schemas } from "../api/client";
import { Card, ErrorState, Loading, PageHeader } from "../components/ui";
import { fmtBytes } from "../lib/format";

type Device = Schemas["Device"];

function SourceTag({ device, field }: { device: Device; field: string }) {
  const source = device.field_sources?.[field];
  if (!source || source === "detected") return null;
  return (
    <span
      className="ml-1 rounded bg-zinc-200 px-1 text-[10px] text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400"
      title={
        source === "inferred" ? "Inferred from the device type, not measured" : "Set by the user"
      }
    >
      {source === "inferred" ? "inferred" : "user"}
    </span>
  );
}

function MemoryCell({ free, total }: { free?: number | null; total?: number | null }) {
  if (!total) return <span className="text-zinc-500">?</span>;
  const used = free === null || free === undefined ? null : 1 - free / total;
  return (
    <div className="min-w-40">
      <div className="font-mono text-xs">
        {fmtBytes(free)} free / {fmtBytes(total)}
      </div>
      {used !== null && (
        <div
          className="mt-1 h-1.5 w-full rounded bg-zinc-200 dark:bg-zinc-800"
          role="img"
          aria-label={`${Math.round(used * 100)}% in use`}
        >
          <div
            className="h-full rounded"
            style={{ width: `${used * 100}%`, background: "var(--series-1)" }}
          />
        </div>
      )}
    </div>
  );
}

export function Hardware() {
  const qc = useQueryClient();
  const profile = useQuery({
    queryKey: ["hardware"],
    queryFn: async () => {
      try {
        return unwrap(await api.GET("/api/v1/hardware"));
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null;
        throw e;
      }
    },
  });
  const scan = useMutation({
    mutationFn: async () => unwrap(await api.POST("/api/v1/hardware/scan")),
    onSuccess: (data) => qc.setQueryData(["hardware"], data),
  });
  const p = profile.data;

  return (
    <>
      <PageHeader
        title="Hardware"
        description="Read-only discovery of CPU, memory and accelerators. Nothing is changed on this machine."
      />
      <div className="mb-4 flex items-center gap-3">
        <button
          type="button"
          onClick={() => scan.mutate()}
          disabled={scan.isPending}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
        >
          {scan.isPending ? "Scanning…" : p ? "Scan again" : "Scan hardware"}
        </button>
        {p && (
          <span className="text-xs text-zinc-500">
            Last scan {new Date(p.captured_at ?? "").toLocaleString()}
          </span>
        )}
      </div>
      {profile.isPending && <Loading />}
      {(profile.error || scan.error) && <ErrorState error={profile.error ?? scan.error} />}
      {p === null && !scan.isPending && (
        <Card>
          <p className="text-sm text-zinc-500">
            No scan yet. Run one to see this machine's devices.
          </p>
        </Card>
      )}
      {p && (
        <div className="space-y-4">
          <Card title="Host">
            <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1 text-sm md:grid-cols-[auto_1fr_auto_1fr]">
              <dt className="text-zinc-500">CPU</dt>
              <dd>{p.host.cpu_model ?? p.host.cpu_arch}</dd>
              <dt className="text-zinc-500">OS</dt>
              <dd>
                {p.host.os} {p.host.os_version} ({p.host.cpu_arch})
              </dd>
              <dt className="text-zinc-500">Cores</dt>
              <dd>
                {p.host.physical_cores ?? "?"} physical / {p.host.logical_cores ?? "?"} logical
              </dd>
              <dt className="text-zinc-500">Python</dt>
              <dd>{p.host.python_version}</dd>
              <dt className="text-zinc-500">CPU features</dt>
              <dd className="font-mono text-xs md:col-span-3">
                {p.host.cpu_flags?.join(" ") || "unknown"}
              </dd>
            </dl>
          </Card>
          <Card title="Devices">
            <table className="w-full text-left text-sm">
              <thead className="text-xs text-zinc-500">
                <tr>
                  <th className="pb-2 font-medium">ID</th>
                  <th className="pb-2 font-medium">Device</th>
                  <th className="pb-2 font-medium">Architecture</th>
                  <th className="pb-2 font-medium">Memory</th>
                  <th className="pb-2 font-medium">Data types</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
                {p.devices?.map((d) => (
                  <tr key={d.device_id} className="align-top">
                    <td className="py-2 pr-3 font-mono text-xs">{d.device_id}</td>
                    <td className="py-2 pr-3">
                      {d.name}
                      <div className="text-xs text-zinc-500">{d.backend}</div>
                    </td>
                    <td className="py-2 pr-3">
                      {d.architecture ?? "?"}
                      <SourceTag device={d} field="architecture" />
                      {d.compute_capability && (
                        <div className="text-xs text-zinc-500">CC {d.compute_capability}</div>
                      )}
                    </td>
                    <td className="py-2 pr-3">
                      <MemoryCell free={d.memory_free_bytes} total={d.memory_total_bytes} />
                    </td>
                    <td className="py-2">
                      <span className="font-mono text-xs">{d.supported_dtypes?.join(", ")}</span>
                      <SourceTag device={d} field="supported_dtypes" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
          {(Object.keys(p.runtime_versions ?? {}).length > 0 || (p.notes?.length ?? 0) > 0) && (
            <Card title="Drivers and notes">
              <dl className="mb-2 grid grid-cols-[auto_1fr] gap-x-6 text-sm">
                {Object.entries(p.runtime_versions ?? {}).map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="text-zinc-500">{k}</dt>
                    <dd className="font-mono">{v}</dd>
                  </div>
                ))}
              </dl>
              <ul className="list-disc space-y-0.5 pl-5 text-sm text-zinc-600 dark:text-zinc-400">
                {p.notes?.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      )}
    </>
  );
}

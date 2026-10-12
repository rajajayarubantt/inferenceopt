// SPDX-License-Identifier: Apache-2.0
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { api, unwrap } from "../api/client";
import { Card, ErrorState, Loading, PageHeader, StatusBadge } from "../components/ui";

export function Dashboard() {
  const version = useQuery({
    queryKey: ["version"],
    queryFn: async () => unwrap(await api.GET("/api/v1/version")),
  });
  const doctor = useQuery({
    queryKey: ["doctor"],
    queryFn: async () => unwrap(await api.GET("/api/v1/doctor")),
  });

  const counts = { ok: 0, info: 0, warn: 0, fail: 0 };
  for (const c of doctor.data?.checks ?? []) counts[c.status] += 1;

  return (
    <>
      <PageHeader
        title="Dashboard"
        description="Find the best way to run an LLM on this machine, and show the evidence."
      />
      <div className="grid gap-4 md:grid-cols-2">
        <Card title="Installation">
          {version.isPending && <Loading />}
          {version.error && <ErrorState error={version.error} />}
          {version.data && (
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
              <dt className="text-zinc-500">Version</dt>
              <dd className="font-mono">{version.data.inferenceopt}</dd>
              <dt className="text-zinc-500">Python</dt>
              <dd className="font-mono">{version.data.python}</dd>
              <dt className="text-zinc-500">Platform</dt>
              <dd>{version.data.platform}</dd>
            </dl>
          )}
        </Card>
        <Card title="Environment checks">
          {doctor.isPending && <Loading />}
          {doctor.error && <ErrorState error={doctor.error} />}
          {doctor.data && (
            <div className="flex flex-wrap items-center gap-3 text-sm">
              {(Object.keys(counts) as (keyof typeof counts)[]).map((s) => (
                <span key={s} className="flex items-center gap-1.5">
                  <StatusBadge status={s} /> {counts[s]}
                </span>
              ))}
              <Link to="/doctor" className="ml-auto text-accent hover:underline">
                Details →
              </Link>
            </div>
          )}
        </Card>
        <Card title="Get started">
          <ol className="list-decimal space-y-1 pl-5 text-sm text-zinc-600 dark:text-zinc-400">
            <li>
              Fix any <strong>fail</strong> or <strong>warn</strong> items on the Doctor page.
            </li>
            <li>
              Write a job file and check it on the{" "}
              <Link to="/jobs" className="text-accent hover:underline">
                Jobs
              </Link>{" "}
              page.
            </li>
            <li>Hardware scan, runtimes and optimization arrive in the next phases.</li>
          </ol>
        </Card>
      </div>
    </>
  );
}

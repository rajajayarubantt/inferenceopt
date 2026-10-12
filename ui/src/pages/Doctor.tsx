// SPDX-License-Identifier: Apache-2.0
import { useQuery } from "@tanstack/react-query";
import { api, unwrap } from "../api/client";
import { Card, ErrorState, Loading, PageHeader, StatusBadge } from "../components/ui";

export function Doctor() {
  const doctor = useQuery({
    queryKey: ["doctor"],
    queryFn: async () => unwrap(await api.GET("/api/v1/doctor")),
  });

  return (
    <>
      <PageHeader
        title="Doctor"
        description="Checks the installation, storage, optional extras and runtimes."
      />
      <Card>
        {doctor.isPending && <Loading />}
        {doctor.error && <ErrorState error={doctor.error} />}
        {doctor.data && (
          <table className="w-full text-left text-sm">
            <thead className="text-zinc-500">
              <tr>
                <th className="pb-2 font-medium">Check</th>
                <th className="pb-2 font-medium">Status</th>
                <th className="pb-2 font-medium">Detail</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
              {doctor.data.checks.map((c) => (
                <tr key={c.name} className="align-top">
                  <td className="py-2 pr-4 font-medium whitespace-nowrap">{c.name}</td>
                  <td className="py-2 pr-4">
                    <StatusBadge status={c.status} />
                  </td>
                  <td className="py-2 break-all">
                    {c.detail}
                    {c.hint && <div className="mt-0.5 text-xs text-zinc-500">{c.hint}</div>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </>
  );
}

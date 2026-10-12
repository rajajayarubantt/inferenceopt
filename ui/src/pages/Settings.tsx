// SPDX-License-Identifier: Apache-2.0
import { useQuery } from "@tanstack/react-query";
import { api, unwrap } from "../api/client";
import { Card, ErrorState, Loading, PageHeader } from "../components/ui";

export function Settings() {
  const config = useQuery({
    queryKey: ["config"],
    queryFn: async () => unwrap(await api.GET("/api/v1/config")),
  });

  return (
    <>
      <PageHeader
        title="Settings"
        description="Effective configuration (read-only for now). Secrets are never sent to the browser; only whether they are set."
      />
      <Card>
        {config.isPending && <Loading />}
        {config.error && <ErrorState error={config.error} />}
        {config.data && (
          <pre className="overflow-x-auto font-mono text-xs">
            {JSON.stringify(config.data, null, 2)}
          </pre>
        )}
      </Card>
    </>
  );
}

// SPDX-License-Identifier: Apache-2.0
import { Card, PageHeader } from "../components/ui";

export function Planned({
  title,
  phase,
  summary,
}: {
  title: string;
  phase: number;
  summary: string;
}) {
  return (
    <>
      <PageHeader title={title} />
      <Card>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          Arrives in <strong>Phase {phase}</strong>. {summary}
        </p>
      </Card>
    </>
  );
}

// SPDX-License-Identifier: Apache-2.0
import type { ReactNode } from "react";

export function PageHeader({ title, description }: { title: string; description?: string }) {
  return (
    <header className="mb-6">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      {description && <p className="mt-1 text-sm text-zinc-500">{description}</p>}
    </header>
  );
}

export function Card({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <section className="rounded-lg border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      {title && <h2 className="mb-3 text-sm font-medium text-zinc-500">{title}</h2>}
      {children}
    </section>
  );
}

const STATUS_STYLES = {
  ok: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300",
  info: "bg-sky-100 text-sky-800 dark:bg-sky-900/40 dark:text-sky-300",
  warn: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
  fail: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
} as const;

export type Status = keyof typeof STATUS_STYLES;

export function StatusBadge({ status }: { status: Status }) {
  return (
    <span
      className={`inline-flex rounded px-2 py-0.5 text-xs font-medium uppercase ${STATUS_STYLES[status]}`}
    >
      {status}
    </span>
  );
}

export function PhaseBadge({ phase }: { phase: number }) {
  return (
    <span className="ml-auto rounded bg-zinc-200 px-1.5 text-[10px] font-medium text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
      P{phase}
    </span>
  );
}

export function ErrorState({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
      {message}
    </div>
  );
}

export function Loading() {
  return <p className="text-sm text-zinc-500">Loading…</p>;
}

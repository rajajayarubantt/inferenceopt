// SPDX-License-Identifier: Apache-2.0
import { Link, Outlet } from "@tanstack/react-router";
import { PhaseBadge } from "../components/ui";

type NavItem = { to: string; label: string; phase?: number };

const NAV: NavItem[] = [
  { to: "/", label: "Dashboard" },
  { to: "/hardware", label: "Hardware", phase: 2 },
  { to: "/models", label: "Models", phase: 2 },
  { to: "/runtimes", label: "Runtimes", phase: 3 },
  { to: "/jobs", label: "Jobs" },
  { to: "/planner", label: "LLM Planner", phase: 6 },
  { to: "/deploy", label: "Deploy", phase: 7 },
  { to: "/doctor", label: "Doctor" },
  { to: "/settings", label: "Settings" },
];

export function Layout() {
  return (
    <div className="flex min-h-screen">
      <nav className="w-56 shrink-0 border-r border-zinc-200 bg-white px-3 py-5 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mb-6 flex items-center gap-2 px-2">
          <img src="/favicon.svg" alt="" className="h-6 w-6" />
          <span className="font-semibold">InferenceOpt</span>
        </div>
        <ul className="space-y-0.5">
          {NAV.map((item) => (
            <li key={item.to}>
              <Link
                to={item.to}
                activeOptions={{ exact: item.to === "/" }}
                className="flex items-center rounded-md px-2 py-1.5 text-sm text-zinc-700 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800"
                activeProps={{
                  className:
                    "bg-accent-soft! text-accent! font-medium dark:bg-teal-950! dark:text-teal-300!",
                }}
              >
                {item.label}
                {item.phase !== undefined && <PhaseBadge phase={item.phase} />}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
      <main className="min-w-0 flex-1 px-8 py-6">
        <Outlet />
      </main>
    </div>
  );
}

// SPDX-License-Identifier: Apache-2.0
import { createRootRoute, createRoute, createRouter } from "@tanstack/react-router";
import { Dashboard } from "../pages/Dashboard";
import { Doctor } from "../pages/Doctor";
import { Jobs } from "../pages/Jobs";
import { Planned } from "../pages/Planned";
import { Settings } from "../pages/Settings";
import { Layout } from "./Layout";

const rootRoute = createRootRoute({ component: Layout });

const page = <P extends string>(path: P, component: () => React.JSX.Element) =>
  createRoute({ getParentRoute: () => rootRoute, path, component });

const planned = <P extends string>(path: P, title: string, phase: number, summary: string) =>
  page(path, () => <Planned title={title} phase={phase} summary={summary} />);

const routeTree = rootRoute.addChildren([
  page("/", Dashboard),
  planned(
    "/hardware",
    "Hardware",
    2,
    "Scan CPUs, GPUs and memory, with detected vs. inferred values.",
  ),
  planned("/models", "Models", 2, "Browse the model registry, add local models, check licenses."),
  planned("/runtimes", "Runtimes", 3, "Configure llama.cpp, vLLM and TensorRT-LLM."),
  page("/jobs", Jobs),
  planned(
    "/planner",
    "LLM Planner",
    6,
    "Configure an optional LiteLLM planner and audit its calls.",
  ),
  planned(
    "/deploy",
    "Deploy",
    7,
    "Generate reproducible deployment bundles from validated trials.",
  ),
  page("/doctor", Doctor),
  page("/settings", Settings),
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

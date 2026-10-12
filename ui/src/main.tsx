// SPDX-License-Identifier: Apache-2.0
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider } from "@tanstack/react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ApiError } from "./api/client";
import { router } from "./app/router";
import { establishSession } from "./lib/auth";
import { NotAuthenticated } from "./pages/NotAuthenticated";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (count, error) => !(error instanceof ApiError && error.status === 401) && count < 2,
      refetchOnWindowFocus: false,
    },
  },
});

async function start() {
  await establishSession();
  const root = createRoot(document.getElementById("root") as HTMLElement);
  const authed = (await fetch("/api/v1/version", { credentials: "same-origin" })).status !== 401;
  root.render(
    <StrictMode>
      {authed ? (
        <QueryClientProvider client={queryClient}>
          <RouterProvider router={router} />
        </QueryClientProvider>
      ) : (
        <NotAuthenticated />
      )}
    </StrictMode>,
  );
}

void start();

// SPDX-License-Identifier: Apache-2.0
import createClient from "openapi-fetch";
import type { components, paths } from "./schema";

export type Schemas = components["schemas"];

/** Typed client for the InferenceOpt API. Authenticates with the session cookie. */
export const api = createClient<paths>({
  baseUrl: window.location.origin,
  credentials: "same-origin",
  // Resolve fetch per call (not at import) so test doubles and polyfills apply.
  fetch: (request) => globalThis.fetch(request),
});

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Turn an openapi-fetch result into data or a thrown ApiError (for TanStack Query). */
export function unwrap<T>(result: { data?: T; error?: unknown; response: Response }): T {
  if (result.data !== undefined) return result.data;
  const detail =
    typeof result.error === "object" && result.error !== null && "detail" in result.error
      ? String((result.error as { detail: unknown }).detail)
      : result.response.statusText;
  throw new ApiError(result.response.status, detail);
}

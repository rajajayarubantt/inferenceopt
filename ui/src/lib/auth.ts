// SPDX-License-Identifier: Apache-2.0
import { api } from "../api/client";

/**
 * `inferenceopt ui` prints a URL with `?token=...`. Exchange it once for an HttpOnly
 * session cookie, then remove it from the address bar and browser history.
 */
export async function establishSession(location: Location = window.location): Promise<void> {
  const url = new URL(location.href);
  const token = url.searchParams.get("token");
  if (!token) return;
  url.searchParams.delete("token");
  window.history.replaceState(null, "", url.pathname + url.search + url.hash);
  await api.POST("/api/v1/auth/session", { body: { token } });
}

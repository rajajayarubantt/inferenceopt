// SPDX-License-Identifier: Apache-2.0
import { afterEach, describe, expect, it, vi } from "vitest";
import { establishSession } from "./auth";

describe("establishSession", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    window.history.replaceState(null, "", "/");
  });

  it("exchanges the token and removes it from the URL", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ authenticated: true }), {
        headers: { "Content-Type": "application/json" },
      }),
    );
    window.history.replaceState(null, "", "/jobs?token=abc123&x=1#top");

    await establishSession();

    expect(window.location.search).toBe("?x=1");
    expect(window.location.hash).toBe("#top");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const request = fetchMock.mock.calls[0]?.[0] as Request;
    expect(request.method).toBe("POST");
    expect(new URL(request.url).pathname).toBe("/api/v1/auth/session");
    expect(await request.json()).toEqual({ token: "abc123" });
  });

  it("does nothing without a token", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    window.history.replaceState(null, "", "/doctor");
    await establishSession();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

// SPDX-License-Identifier: Apache-2.0
export function NotAuthenticated() {
  return (
    <div className="mx-auto mt-24 max-w-lg px-4">
      <h1 className="text-xl font-semibold">Not signed in</h1>
      <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
        Open the full URL printed in the terminal by <code>inferenceopt ui</code>. It contains a
        one-time token that signs this browser in. The token changes every time the server starts.
      </p>
    </div>
  );
}

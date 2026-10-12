// SPDX-License-Identifier: Apache-2.0
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The production build goes straight into the Python package so it ships in the wheel.
const outDir = fileURLToPath(new URL("../src/inferenceopt/_ui", import.meta.url));
const apiTarget = process.env.INFERENCEOPT_API ?? "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir,
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    strictPort: true,
    // Keep the browser's Host header so the API's Host/Origin checks see a same-origin call.
    proxy: { "/api": { target: apiTarget, changeOrigin: false } },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test-setup.ts"],
  },
});

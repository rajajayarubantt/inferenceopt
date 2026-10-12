// SPDX-License-Identifier: Apache-2.0
// Writes THIRD_PARTY_LICENSES.txt next to the built UI. The bundle redistributes these
// packages inside the Python wheel, so their notices must ship with it.
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const outFile = fileURLToPath(
  new URL("../../src/inferenceopt/_ui/THIRD_PARTY_LICENSES.txt", import.meta.url),
);

// npm_execpath is pnpm's entry point: a JS file for npm-installed pnpm, or a native binary.
const execPath = process.env.npm_execpath ?? "pnpm";
const [cmd, prefix] = /\.[cm]?js$/.test(execPath) ? [process.execPath, [execPath]] : [execPath, []];
const raw = execFileSync(cmd, [...prefix, "licenses", "list", "--prod", "--json"], {
  encoding: "utf8",
  maxBuffer: 64 * 1024 * 1024,
  shell: process.platform === "win32" && !/\.(exe|[cm]?js)$/i.test(cmd),
});
const byLicense = JSON.parse(raw);

const packages = [];
for (const [license, entries] of Object.entries(byLicense)) {
  for (const entry of entries) {
    const versions = entry.versions ?? [entry.version];
    const paths = entry.paths ?? [entry.path];
    versions.forEach((version, i) => {
      packages.push({ name: entry.name, version, license, path: paths[i] ?? paths[0] });
    });
  }
}
packages.sort((a, b) => a.name.localeCompare(b.name) || a.version.localeCompare(b.version));

function licenseText(dir) {
  if (!dir || !existsSync(dir)) return null;
  const file = readdirSync(dir).find((f) => /^(licen[cs]e|copying|notice)/i.test(f));
  return file ? readFileSync(join(dir, file), "utf8").trim() : null;
}

const sections = packages.map((p) => {
  const text = licenseText(p.path) ?? `License: ${p.license} (no license file in package)`;
  return `${"=".repeat(78)}\n${p.name}@${p.version} (${p.license})\n${"=".repeat(78)}\n${text}\n`;
});

writeFileSync(
  outFile,
  "Third-party software bundled in the InferenceOpt web UI\n\n" + sections.join("\n"),
);
console.log(`wrote ${packages.length} package notices to ${outFile}`);

// SPDX-License-Identifier: Apache-2.0
import { useState } from "react";
import type { Schemas } from "../api/client";
import { fmtBytes } from "../lib/format";

type Estimate = Schemas["MemoryEstimate"];
type Marker = { label: string; bytes: number };

// Categorical slots in fixed order; identity follows the component, never its size.
const SERIES: Record<string, { color: string; label: string }> = {
  weights: { color: "var(--series-1)", label: "Weights" },
  kv_cache: { color: "var(--series-2)", label: "KV cache" },
  compute_buffers: { color: "var(--series-3)", label: "Compute buffers" },
  runtime_overhead: { color: "var(--series-4)", label: "Runtime overhead" },
};

/**
 * One horizontal stacked bar of the expected memory, with the low-high estimate range
 * and each device's usable memory drawn on the same byte axis.
 */
export function MemoryBar({ estimate, markers }: { estimate: Estimate; markers: Marker[] }) {
  const [hover, setHover] = useState<string | null>(null);
  const max = Math.max(estimate.high_bytes, ...markers.map((m) => m.bytes)) * 1.05;
  const pct = (n: number) => `${(n / max) * 100}%`;
  const hovered = estimate.components.find((c) => c.name === hover);

  return (
    <figure className="space-y-3">
      <div className="h-5 text-xs" aria-live="polite">
        {hovered ? (
          <>
            <strong>{SERIES[hovered.name]?.label ?? hovered.name}</strong>{" "}
            <span className="font-mono">{fmtBytes(hovered.bytes)}</span>{" "}
            <span className="text-zinc-500">{hovered.basis}</span>
          </>
        ) : (
          <span className="text-zinc-500">Hover a segment for details.</span>
        )}
      </div>
      <div
        className="relative pb-7"
        style={{ paddingTop: `${markers.length * 14 + 6}px` }}
        onMouseLeave={() => setHover(null)}
      >
        {/* stacked bar: 2px surface gap between segments, rounded data ends */}
        <div className="flex h-7 gap-[2px]" style={{ width: pct(estimate.expected_bytes) }}>
          {estimate.components.map((c, i, all) => (
            <div
              key={c.name}
              role="img"
              aria-label={`${SERIES[c.name]?.label ?? c.name}: ${fmtBytes(c.bytes)}`}
              onMouseEnter={() => setHover(c.name)}
              className={`h-full min-w-[2px] cursor-default transition-opacity ${
                hover && hover !== c.name ? "opacity-40" : ""
              } ${i === 0 ? "rounded-l" : ""} ${i === all.length - 1 ? "rounded-r" : ""}`}
              style={{
                flexGrow: c.bytes,
                flexBasis: 0,
                background: SERIES[c.name]?.color ?? "gray",
              }}
            />
          ))}
        </div>
        {/* estimate range */}
        <div
          className="absolute bottom-3 h-2 border-x-2 border-zinc-500"
          style={{
            left: pct(estimate.low_bytes),
            width: pct(estimate.high_bytes - estimate.low_bytes),
          }}
          title={`Estimate range ${fmtBytes(estimate.low_bytes)} - ${fmtBytes(estimate.high_bytes)}`}
        >
          <div className="mt-[3px] h-[2px] bg-zinc-500" />
        </div>
        <span
          className="absolute bottom-0 text-[11px] text-zinc-500"
          style={{ left: pct(estimate.low_bytes) }}
        >
          range {fmtBytes(estimate.low_bytes)} – {fmtBytes(estimate.high_bytes)}
        </span>
        {/* device capacity markers */}
        {markers.map((m, i) => {
          const flip = m.bytes / max > 0.7;
          return (
            <div
              key={m.label}
              className="absolute bottom-3 border-l-2 border-dashed border-zinc-800 dark:border-zinc-200"
              style={{ left: pct(m.bytes), top: `${i * 14}px` }}
            >
              <span
                className={`absolute -top-0.5 text-[11px] whitespace-nowrap text-zinc-700 dark:text-zinc-300 ${
                  flip ? "right-1" : "left-1"
                }`}
              >
                {m.label} {fmtBytes(m.bytes)}
              </span>
            </div>
          );
        })}
      </div>
      {/* legend doubles as the table view: every value is readable as text */}
      <table className="w-full text-sm">
        <tbody>
          {estimate.components.map((c) => (
            <tr
              key={c.name}
              onMouseEnter={() => setHover(c.name)}
              onMouseLeave={() => setHover(null)}
            >
              <td className="py-0.5 pr-2">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-sm align-middle"
                  style={{ background: SERIES[c.name]?.color }}
                />
              </td>
              <td className="py-0.5 pr-4">{SERIES[c.name]?.label ?? c.name}</td>
              <td className="py-0.5 pr-4 text-right font-mono">{fmtBytes(c.bytes)}</td>
              <td className="py-0.5 text-xs text-zinc-500">{c.basis}</td>
            </tr>
          ))}
          <tr className="border-t border-zinc-200 font-medium dark:border-zinc-800">
            <td />
            <td className="py-1 pr-4">Expected total</td>
            <td className="py-1 pr-4 text-right font-mono">{fmtBytes(estimate.expected_bytes)}</td>
            <td className="py-1 text-xs text-zinc-500">
              range {fmtBytes(estimate.low_bytes)} – {fmtBytes(estimate.high_bytes)}
            </td>
          </tr>
        </tbody>
      </table>
    </figure>
  );
}

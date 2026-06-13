"use client";
import { useState } from "react";
import type { TraceInfo } from "@/lib/api";

export default function TraceExpander({ trace }: { trace: TraceInfo }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="mt-1">
      <button
        onClick={() => setOpen(!open)}
        className="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1 transition-colors"
      >
        <span>{open ? "▲" : "▼"}</span>
        🔍 How this was answered · {trace.latency_ms}ms
      </button>
      {open && (
        <div className="mt-1 p-3 rounded text-xs font-mono space-y-1" style={{ background: "#1a1d2e", border: "1px solid #2D3250" }}>
          <div><span className="text-gray-400">Path:</span> <span className="text-green-400">{trace.path}</span></div>
          <div><span className="text-gray-400">Query type:</span> {trace.query_type || "—"}</div>
          <div><span className="text-gray-400">Destination:</span> {trace.destination || "—"}</div>
          <div><span className="text-gray-400">Tools called:</span>{" "}
            {trace.tools_called?.length
              ? trace.tools_called.map((t, i) => (
                  <span key={i} className="inline-block bg-blue-900 text-blue-200 px-1.5 py-0.5 rounded mr-1">{t}</span>
                ))
              : "—"}
          </div>
          <div><span className="text-gray-400">Iterations:</span> {trace.iterations ?? "—"}</div>
          <div><span className="text-gray-400">Latency:</span> {trace.latency_ms}ms</div>
          {trace.error && <div><span className="text-red-400">Error:</span> {trace.error}</div>}
        </div>
      )}
    </div>
  );
}

/**
 * Live tail of the api / vllm / megatron service logs. One live stream
 * at a time (the active tab) to keep the number of concurrent SSE
 * connections bounded. An "Expand" button pops the current tab into a
 * full-screen modal so the operator can see more context without
 * leaving the metrics page.
 *
 * Backend contract: GET /v1/health/logs/{service_name} yields NDJSON
 * {"line": "...", "line_number": N}\n — see
 * infra/cray_infra/api/fastapi/routers/health_router.py:29.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";

import { tailServiceLogs, type LogStreamStatus, type ServiceName } from "@/api/logs";
import type { LogLine } from "@/api/training";

const SERVICES: { name: ServiceName; label: string; hint: string }[] = [
  { name: "api", label: "API", hint: "FastAPI control plane" },
  { name: "vllm", label: "vLLM", hint: "Inference engine" },
  { name: "megatron", label: "Megatron", hint: "Training + slurmd" },
];

const MAX_LINES = 10_000;
const ERROR_REGEX = /\berror\b|traceback|cuda error|oom|killed/i;

export function ServiceLogsCard() {
  const [active, setActive] = useState<ServiceName>("api");
  const [expanded, setExpanded] = useState(false);

  // Close the modal on escape so the expanded view behaves like a dialog.
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setExpanded(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  return (
    <>
      <section className="flex flex-col gap-0 rounded-lg border border-border-subtle bg-bg-card">
        <header className="flex flex-wrap items-center gap-2 border-b border-border-subtle px-4 py-3">
          <h2 className="text-sm font-semibold text-fg">Service logs</h2>
          <div
            role="tablist"
            aria-label="Service"
            className="ml-2 flex items-center gap-1"
          >
            {SERVICES.map((svc) => (
              <button
                key={svc.name}
                type="button"
                role="tab"
                aria-selected={active === svc.name}
                onClick={() => setActive(svc.name)}
                title={svc.hint}
                className={clsx(
                  "rounded-md px-3 py-1 text-xs transition-colors",
                  active === svc.name
                    ? "bg-bg-hover text-fg ring-1 ring-border"
                    : "text-fg-muted hover:bg-bg hover:text-fg",
                )}
              >
                {svc.label}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => setExpanded(true)}
            className="ml-auto rounded-md border border-border-subtle bg-bg-card px-3 py-1 text-xs text-fg hover:border-border hover:bg-bg-hover"
            aria-label="Expand log viewer"
          >
            Expand
          </button>
        </header>
        <ServiceLogTail service={active} expanded={false} />
      </section>

      {expanded && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Expanded service logs"
          className="fixed inset-0 z-50 flex flex-col bg-bg/95 backdrop-blur-sm"
        >
          <header className="flex items-center gap-2 border-b border-border-subtle bg-bg-card px-4 py-3">
            <h2 className="text-sm font-semibold text-fg">Service logs</h2>
            <div role="tablist" className="ml-2 flex items-center gap-1">
              {SERVICES.map((svc) => (
                <button
                  key={svc.name}
                  type="button"
                  role="tab"
                  aria-selected={active === svc.name}
                  onClick={() => setActive(svc.name)}
                  className={clsx(
                    "rounded-md px-3 py-1 text-xs transition-colors",
                    active === svc.name
                      ? "bg-bg-hover text-fg ring-1 ring-border"
                      : "text-fg-muted hover:bg-bg hover:text-fg",
                  )}
                >
                  {svc.label}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={() => setExpanded(false)}
              className="ml-auto rounded-md border border-border-subtle bg-bg-card px-3 py-1 text-xs text-fg hover:border-border hover:bg-bg-hover"
              autoFocus
            >
              Close (Esc)
            </button>
          </header>
          <div className="min-h-0 flex-1">
            <ServiceLogTail service={active} expanded={true} />
          </div>
        </div>
      )}
    </>
  );
}

interface ServiceLogTailProps {
  service: ServiceName;
  expanded: boolean;
}

function ServiceLogTail({ service, expanded }: ServiceLogTailProps) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [status, setStatus] = useState<LogStreamStatus>("connecting");
  const [autoScroll, setAutoScroll] = useState(true);
  const [query, setQuery] = useState("");
  const [useRegex, setUseRegex] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLines([]);
    setStatus("connecting");
    tailServiceLogs({
      service,
      signal: controller.signal,
      onLine: (line) =>
        setLines((prev) => {
          const next =
            prev.length >= MAX_LINES
              ? prev.slice(prev.length - MAX_LINES + 1)
              : prev.slice();
          next.push(line);
          return next;
        }),
      onStatus: setStatus,
    });
    return () => controller.abort();
  }, [service]);

  useEffect(() => {
    if (!autoScroll) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [lines, autoScroll]);

  const filter = useMemo(() => {
    if (!query) return null;
    if (useRegex) {
      try {
        return new RegExp(query, "i");
      } catch {
        return null;
      }
    }
    return query.toLowerCase();
  }, [query, useRegex]);

  const filtered = useMemo(() => {
    if (!filter) return lines;
    if (typeof filter === "string") {
      return lines.filter((l) => l.line.toLowerCase().includes(filter));
    }
    return lines.filter((l) => filter.test(l.line));
  }, [lines, filter]);

  const jumpToLastError = () => {
    for (let i = lines.length - 1; i >= 0; i--) {
      if (ERROR_REGEX.test(lines[i].line)) {
        const el = document.querySelector(
          `[data-line="${service}-${lines[i].line_number}"]`,
        );
        el?.scrollIntoView({ block: "center", behavior: "smooth" });
        setAutoScroll(false);
        return;
      }
    }
  };

  const downloadLog = () => {
    const blob = new Blob([lines.map((l) => l.line).join("\n")], {
      type: "text/plain",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${service}.log`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-border-subtle px-4 py-2">
        <StreamStatusDot status={status} />
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={`Search ${service} logs…`}
          className="w-56 rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs text-fg placeholder-fg-subtle focus:border-border focus:outline-none"
        />
        <label className="flex items-center gap-1 text-xs text-fg-muted">
          <input
            type="checkbox"
            checked={useRegex}
            onChange={(e) => setUseRegex(e.target.checked)}
            className="h-3 w-3 accent-accent"
          />
          regex
        </label>
        <span className="text-[10px] text-fg-subtle">
          {filtered.length.toLocaleString()}
          {query ? ` / ${lines.length.toLocaleString()}` : ""} lines
        </span>
        <div className="ml-auto flex items-center gap-1">
          <button
            type="button"
            onClick={jumpToLastError}
            className="rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs text-fg hover:border-border hover:bg-bg-hover"
          >
            Jump to last error
          </button>
          <button
            type="button"
            onClick={() => setAutoScroll((v) => !v)}
            className={clsx(
              "rounded-md border px-2 py-1 text-xs",
              autoScroll
                ? "border-accent/40 bg-accent/10 text-accent"
                : "border-border-subtle bg-bg text-fg hover:border-border hover:bg-bg-hover",
            )}
          >
            {autoScroll ? "Auto-scroll on" : "Auto-scroll off"}
          </button>
          <button
            type="button"
            onClick={downloadLog}
            className="rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs text-fg hover:border-border hover:bg-bg-hover"
          >
            Download
          </button>
        </div>
      </div>
      <div
        ref={scrollRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          const nearBottom =
            el.scrollHeight - el.scrollTop - el.clientHeight < 20;
          if (autoScroll && !nearBottom) setAutoScroll(false);
        }}
        className={clsx(
          "min-h-[240px] overflow-auto px-2 py-1 font-mono text-[11px] leading-relaxed",
          expanded ? "flex-1" : "max-h-[420px]",
        )}
      >
        {filtered.length === 0 ? (
          <div className="px-2 py-3 text-xs text-fg-subtle">
            {lines.length === 0
              ? "Waiting for log output…"
              : "No lines match the current filter."}
          </div>
        ) : (
          filtered.map((l) => (
            <div
              key={l.line_number}
              data-line={`${service}-${l.line_number}`}
              className={clsx(
                "flex gap-3 px-2 py-[1px]",
                ERROR_REGEX.test(l.line) && "bg-danger/10 text-danger",
              )}
            >
              <span className="w-14 shrink-0 select-none text-right text-fg-subtle">
                {l.line_number}
              </span>
              <span className="whitespace-pre-wrap break-all text-fg">
                {l.line}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function StreamStatusDot({ status }: { status: LogStreamStatus }) {
  const map: Record<LogStreamStatus, { color: string; label: string }> = {
    connecting: { color: "bg-fg-subtle", label: "connecting" },
    streaming: { color: "bg-success", label: "streaming" },
    reconnecting: { color: "bg-warning animate-pulse", label: "reconnecting" },
    closed: { color: "bg-fg-subtle", label: "closed" },
  };
  const { color, label } = map[status];
  return (
    <span
      title={label}
      className="flex items-center gap-1.5 rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs text-fg-muted"
    >
      <span className={clsx("h-2 w-2 rounded-full", color)} />
      {label}
    </span>
  );
}

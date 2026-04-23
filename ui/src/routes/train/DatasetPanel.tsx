/**
 * DatasetPanel — paginated, searchable view of a training job's
 * dataset.jsonlines. Fetches GET /v1/megatron/train/{hash}/dataset.
 *
 * Collapsed by default; the underlying query only fires when expanded,
 * so opening the train-detail page costs zero dataset bytes until the
 * user asks for them.
 *
 * See ui/docs/dataset-viewer.md for the design.
 */

import { useEffect, useMemo, useState } from "react";

import { ApiError } from "@/api/client";
import { useTrainingDataset } from "@/api/training";
import { Card } from "@/components/Card";
import { ErrorState } from "@/components/ErrorState";
import { Skeleton } from "@/components/Skeleton";

const PAGE_SIZE = 50;
const SEARCH_DEBOUNCE_MS = 200;

interface DatasetPanelProps {
  jobHash: string;
}

export function DatasetPanel({ jobHash }: DatasetPanelProps) {
  const [open, setOpen] = useState(false);
  const [searchInput, setSearchInput] = useState("");
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);

  // Debounce the search input and reset paging when the query changes.
  useEffect(() => {
    const handle = setTimeout(() => {
      const trimmed = searchInput.trim();
      if (trimmed !== q) {
        setQ(trimmed);
        setOffset(0);
      }
    }, SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(handle);
  }, [searchInput, q]);

  // Gate the network call on `open` — passing undefined disables the hook.
  const { data, error, isPending, isFetching, refetch } = useTrainingDataset(
    open ? jobHash : undefined,
    { offset, limit: PAGE_SIZE, q },
  );

  const total = data?.total ?? 0;
  const matched = data?.matched ?? 0;
  const examples = data?.examples ?? [];

  const pageStart = examples.length > 0 ? offset + 1 : 0;
  const pageEnd = offset + examples.length;
  const pageIndex = Math.floor(offset / PAGE_SIZE) + 1;
  const pageCount = Math.max(1, Math.ceil(matched / PAGE_SIZE));
  const hasNext = offset + PAGE_SIZE < matched;
  const hasPrev = offset > 0;

  const is404 =
    error instanceof ApiError &&
    error.status === 404 &&
    typeof error.body === "object" &&
    error.body !== null &&
    "detail" in error.body &&
    String((error.body as { detail: unknown }).detail) === "dataset not found";

  return (
    <Card
      title="Dataset"
      subtitle={
        !open
          ? "Collapsed — click to load"
          : q
          ? `${matched.toLocaleString()} matches of ${total.toLocaleString()} rows`
          : `${total.toLocaleString()} rows`
      }
      action={
        <div className="flex items-center gap-2">
          {open && (
            <input
              type="search"
              role="searchbox"
              aria-controls="dataset-panel-list"
              placeholder="Search…"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              className="w-64 rounded-md border border-border-subtle bg-bg px-3 py-1 text-sm text-fg placeholder:text-fg-subtle focus:border-accent focus:outline-none"
            />
          )}
          {open && isFetching && !isPending && (
            <span
              className="text-xs text-fg-subtle"
              aria-live="polite"
            >
              …
            </span>
          )}
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            className="rounded-md border border-border-subtle bg-bg-card px-3 py-1 text-xs text-fg hover:border-border hover:bg-bg-hover"
          >
            {open ? "Hide" : "Show"}
          </button>
        </div>
      }
    >
      {!open ? (
        <p className="text-xs text-fg-muted">
          The dataset panel is collapsed to avoid fetching bytes you don't
          need. Click <span className="font-semibold">Show</span> to load
          the first page and enable search.
        </p>
      ) : isPending ? (
        <DatasetSkeleton />
      ) : is404 ? (
        <EmptyNotice
          title="No dataset.jsonlines for this job"
          body="The dataset file is missing from the job directory. This usually means the upload didn't complete."
        />
      ) : error ? (
        <ErrorState error={error} onRetry={refetch} />
      ) : total === 0 ? (
        <EmptyNotice
          title="Empty dataset"
          body="This job's dataset.jsonlines has no rows."
        />
      ) : matched === 0 ? (
        <EmptyNotice
          title={`No rows match "${q}"`}
          body="Try a shorter query or clear the search box."
        />
      ) : (
        <>
          {data?.truncated && (
            <div className="mb-3 rounded-md border border-warning/30 bg-warning/5 px-3 py-2 text-xs text-warning">
              Search stopped after scanning 256 MiB. Narrow the query for
              complete results.
            </div>
          )}
          <ul
            id="dataset-panel-list"
            role="list"
            className="flex max-h-[480px] flex-col divide-y divide-border-subtle overflow-y-auto rounded-md border border-border-subtle"
          >
            {examples.map((ex) => (
              <ExampleRow
                key={ex.index}
                index={ex.index}
                input={ex.input}
                output={ex.output}
                raw={ex.raw}
                truncatedFields={ex.truncated_fields}
              />
            ))}
          </ul>
          <div className="mt-3 flex items-center justify-between text-xs text-fg-muted">
            <span>
              Showing {pageStart.toLocaleString()}–
              {pageEnd.toLocaleString()} of {matched.toLocaleString()}
              {q ? " matches" : " rows"}
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                disabled={!hasPrev || isFetching}
                className="rounded-md border border-border-subtle bg-bg-card px-3 py-1 text-fg hover:border-border hover:bg-bg-hover disabled:cursor-not-allowed disabled:opacity-40"
              >
                ← Prev
              </button>
              <span className="font-mono">
                {pageIndex} / {pageCount}
              </span>
              <button
                type="button"
                onClick={() => setOffset(offset + PAGE_SIZE)}
                disabled={!hasNext || isFetching}
                className="rounded-md border border-border-subtle bg-bg-card px-3 py-1 text-fg hover:border-border hover:bg-bg-hover disabled:cursor-not-allowed disabled:opacity-40"
              >
                Next →
              </button>
            </div>
          </div>
        </>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------

interface ExampleRowProps {
  index: number;
  input?: string;
  output?: string;
  raw: Record<string, unknown>;
  truncatedFields?: string[];
}

function ExampleRow({
  index,
  input,
  output,
  raw,
  truncatedFields,
}: ExampleRowProps) {
  const [expanded, setExpanded] = useState(false);

  // If neither `input` nor `output` is present, fall back to dumping raw JSON
  // so datasets with unusual schemas still render something.
  const hasCanonical =
    typeof input === "string" || typeof output === "string";

  const rawPreview = useMemo(() => {
    try {
      return JSON.stringify(raw, null, 2);
    } catch {
      return String(raw);
    }
  }, [raw]);

  const hasExtraFields = useMemo(() => {
    const keys = Object.keys(raw);
    return keys.some((k) => k !== "input" && k !== "output");
  }, [raw]);

  return (
    <li role="listitem" className="px-3 py-2">
      <div className="flex gap-3">
        <span className="shrink-0 font-mono text-[11px] text-fg-subtle">
          #{index}
        </span>
        <div className="min-w-0 flex-1">
          {hasCanonical ? (
            <div className="grid grid-cols-1 gap-2 text-sm md:grid-cols-2">
              <Cell label="input" value={input} expanded={expanded} />
              <Cell label="output" value={output} expanded={expanded} />
            </div>
          ) : (
            <pre className="whitespace-pre-wrap break-words font-mono text-xs text-fg">
              {rawPreview}
            </pre>
          )}
          {truncatedFields && truncatedFields.length > 0 && (
            <p className="mt-1 text-[11px] text-warning">
              Truncated at 4 KiB: {truncatedFields.join(", ")}
            </p>
          )}
          {hasCanonical && (hasExtraFields || expanded) && (
            <button
              type="button"
              onClick={() => setExpanded((v) => !v)}
              className="mt-1 text-[11px] text-fg-subtle hover:text-fg"
            >
              {expanded ? "hide raw" : "show raw"}
            </button>
          )}
          {hasCanonical && expanded && (
            <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md bg-bg px-2 py-1 font-mono text-[11px] text-fg-muted">
              {rawPreview}
            </pre>
          )}
        </div>
      </div>
    </li>
  );
}

interface CellProps {
  label: string;
  value: string | undefined;
  expanded: boolean;
}

function Cell({ label, value, expanded }: CellProps) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] uppercase tracking-wider text-fg-subtle">
        {label}
      </div>
      <div
        className={
          expanded
            ? "whitespace-pre-wrap break-words font-mono text-xs text-fg"
            : "line-clamp-3 whitespace-pre-wrap break-words font-mono text-xs text-fg"
        }
      >
        {value ?? <span className="text-fg-subtle">—</span>}
      </div>
    </div>
  );
}

function DatasetSkeleton() {
  return (
    <div className="flex flex-col gap-2">
      {Array.from({ length: 5 }).map((_, i) => (
        <Skeleton key={i} className="h-12" />
      ))}
    </div>
  );
}

function EmptyNotice({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-md border border-border-subtle bg-bg px-4 py-6 text-center">
      <p className="text-sm font-medium text-fg">{title}</p>
      <p className="mt-1 text-xs text-fg-muted">{body}</p>
    </div>
  );
}

/**
 * Phase 0 of the Publish-to-HF flow. Renders the form shell so the
 * trigger button is wired and operators can see what's coming, but
 * the submit path is intentionally inert — Phase 1 lands the
 * checkpoint-as-is upload, Phase 2 the merge-then-push.
 *
 * See ui/docs/publish-to-hf.md for the full design.
 */

import { useEffect, useId, useState } from "react";

import { useCheckpoints, type CheckpointEntry } from "@/api/publish";

interface PublishToHFModalProps {
  open: boolean;
  jobHash: string;
  onClose: () => void;
}

type Mode = "checkpoint" | "merge";

// Loose client-side guard: must contain exactly one `/`, both halves
// non-empty, and only word-ish characters. The server forwards to HF
// which is the authority on naming, so we don't try to reproduce
// every quirk of org-vs-repo allowed-character lists here.
const REPO_ID_RE = /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/;

export function PublishToHFModal({ open, jobHash, onClose }: PublishToHFModalProps) {
  const [mode, setMode] = useState<Mode>("checkpoint");
  const [repoId, setRepoId] = useState("");
  const [isPrivate, setIsPrivate] = useState(false);
  const [token, setToken] = useState("");
  const [showToken, setShowToken] = useState(false);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>("");

  const { data: checkpoints, isPending: ckptPending, error: ckptError } =
    useCheckpoints(open ? jobHash : undefined);

  // Default to the latest checkpoint once the list arrives.
  useEffect(() => {
    if (!checkpoints || checkpoints.length === 0) return;
    if (selectedCheckpoint === "") {
      setSelectedCheckpoint(checkpoints[0].name);
    }
  }, [checkpoints, selectedCheckpoint]);

  // Wipe sensitive state when closing.
  useEffect(() => {
    if (!open) {
      setToken("");
      setShowToken(false);
    }
  }, [open]);

  // Esc to close.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const repoIdValid = repoId === "" || REPO_ID_RE.test(repoId);
  const formReady =
    repoId !== "" &&
    repoIdValid &&
    token !== "" &&
    selectedCheckpoint !== "";

  const tokenInputId = useId();
  const repoInputId = useId();
  const ckptInputId = useId();

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="publish-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="flex w-full max-w-lg flex-col gap-4 rounded-lg border border-border bg-bg-card p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-start justify-between gap-4">
          <div>
            <h2
              id="publish-modal-title"
              className="text-sm font-semibold text-fg"
            >
              Publish to HuggingFace
            </h2>
            <p className="mt-0.5 text-xs text-fg-muted">
              Job <code className="font-mono">{jobHash.slice(0, 8)}</code>
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs text-fg-muted hover:border-border hover:bg-bg-hover hover:text-fg"
          >
            Close
          </button>
        </header>

        <ModeSelector mode={mode} setMode={setMode} />

        <div className="flex flex-col gap-1.5">
          <label htmlFor={repoInputId} className="text-xs text-fg-muted">
            Repository
          </label>
          <input
            id={repoInputId}
            type="text"
            placeholder="myorg/my-adapter"
            value={repoId}
            onChange={(e) => setRepoId(e.target.value)}
            className="rounded-md border border-border-subtle bg-bg px-3 py-1.5 font-mono text-sm text-fg focus:border-accent focus:outline-none"
            spellCheck={false}
            autoCapitalize="off"
            autoCorrect="off"
          />
          {!repoIdValid && (
            <p className="text-[11px] text-danger">
              Use the form <code className="font-mono">org/repo-name</code>.
              Letters, digits, <code>.</code>, <code>-</code>, <code>_</code>.
            </p>
          )}
          <label className="flex items-center gap-2 text-xs text-fg-muted">
            <input
              type="checkbox"
              checked={isPrivate}
              onChange={(e) => setIsPrivate(e.target.checked)}
              className="h-3 w-3 accent-accent"
            />
            Create as private (existing repo visibility wins)
          </label>
        </div>

        <div className="flex flex-col gap-1.5">
          <label htmlFor={tokenInputId} className="text-xs text-fg-muted">
            HuggingFace token{" "}
            <span className="text-fg-subtle">(write access)</span>
          </label>
          <div className="flex items-center gap-1.5">
            <input
              id={tokenInputId}
              // Treat the field as opaque text rather than `password`
              // so browser password managers don't latch onto it; per
              // ui/docs/publish-to-hf.md the token is held only in
              // component state, not localStorage.
              type={showToken ? "text" : "password"}
              autoComplete="off"
              autoCorrect="off"
              spellCheck={false}
              data-1p-ignore
              data-lpignore="true"
              placeholder="hf_***"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              className="flex-1 rounded-md border border-border-subtle bg-bg px-3 py-1.5 font-mono text-sm text-fg focus:border-accent focus:outline-none"
            />
            <button
              type="button"
              onClick={() => setShowToken((v) => !v)}
              className="rounded-md border border-border-subtle bg-bg px-2 py-1.5 text-[11px] text-fg-muted hover:border-border hover:text-fg"
            >
              {showToken ? "Hide" : "Show"}
            </button>
          </div>
          <p className="text-[11px] text-fg-subtle">
            Generate at{" "}
            <a
              href="https://huggingface.co/settings/tokens"
              target="_blank"
              rel="noopener noreferrer"
              className="underline-offset-2 hover:underline"
            >
              huggingface.co/settings/tokens
            </a>
            . Held only in this tab; not saved.
          </p>
        </div>

        <div className="flex flex-col gap-1.5">
          <label htmlFor={ckptInputId} className="text-xs text-fg-muted">
            Checkpoint
          </label>
          <select
            id={ckptInputId}
            value={selectedCheckpoint}
            onChange={(e) => setSelectedCheckpoint(e.target.value)}
            disabled={!checkpoints || checkpoints.length === 0}
            className="rounded-md border border-border-subtle bg-bg px-3 py-1.5 text-sm text-fg focus:border-accent focus:outline-none disabled:opacity-50"
          >
            {ckptPending && <option>Loading…</option>}
            {checkpoints &&
              checkpoints.map((c, i) => (
                <option key={c.name} value={c.name}>
                  {formatCheckpointLabel(c, i === 0)}
                </option>
              ))}
            {checkpoints && checkpoints.length === 0 && (
              <option value="">No checkpoints written yet</option>
            )}
          </select>
          {ckptError && (
            <p className="text-[11px] text-danger">
              Couldn't load checkpoints: {String((ckptError as Error).message)}
            </p>
          )}
        </div>

        <Phase0Banner />

        <footer className="flex justify-end gap-2 pt-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-border-subtle bg-bg px-3 py-1.5 text-sm text-fg hover:border-border hover:bg-bg-hover"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled
            title="Phase 0: form preview only — submit lands in Phase 1"
            className="rounded-md bg-accent/40 px-3 py-1.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-60"
          >
            Publish
          </button>
        </footer>

        {/* Visually-hidden status for screen readers + e2e selectors. */}
        <span className="sr-only" aria-live="polite">
          {formReady ? "Form complete" : "Form incomplete"}
        </span>
      </div>
    </div>
  );
}

function ModeSelector({
  mode,
  setMode,
}: {
  mode: Mode;
  setMode: (m: Mode) => void;
}) {
  return (
    <fieldset className="flex flex-col gap-2 rounded-md border border-border-subtle bg-bg/50 p-3">
      <legend className="px-1 text-xs text-fg-muted">Mode</legend>
      <ModeRadio
        value="checkpoint"
        current={mode}
        onSelect={setMode}
        title="Push checkpoint as-is"
        body="Adapter .pt + metadata only. Loadable by ScalarLM and any vLLM with the LoRA adapter resolver."
      />
      <ModeRadio
        value="merge"
        current={mode}
        onSelect={setMode}
        title="Merge LoRA into base, push full model"
        body="Single self-contained model. Larger upload (~30–60 GB for a 31B base). Required for off-platform inference."
      />
    </fieldset>
  );
}

function ModeRadio({
  value,
  current,
  onSelect,
  title,
  body,
}: {
  value: Mode;
  current: Mode;
  onSelect: (m: Mode) => void;
  title: string;
  body: string;
}) {
  const checked = current === value;
  return (
    <label
      className={
        "flex cursor-pointer items-start gap-2 rounded-md border px-3 py-2 text-sm transition-colors " +
        (checked
          ? "border-accent/40 bg-accent/10"
          : "border-border-subtle bg-bg hover:border-border")
      }
    >
      <input
        type="radio"
        name="publish-mode"
        value={value}
        checked={checked}
        onChange={() => onSelect(value)}
        className="mt-0.5 h-3 w-3 accent-accent"
      />
      <span className="flex flex-col">
        <span className="text-fg">{title}</span>
        <span className="text-[11px] text-fg-muted">{body}</span>
      </span>
    </label>
  );
}

function Phase0Banner() {
  return (
    <div
      role="note"
      className="rounded-md border border-warning/30 bg-warning/5 px-3 py-2 text-[11px] text-warning"
    >
      <strong className="font-semibold">Preview:</strong> the publish flow
      ships in phases. The form here is wired and your inputs are valid;
      the upload will be enabled in the next release. See{" "}
      <code className="font-mono">ui/docs/publish-to-hf.md</code>.
    </div>
  );
}

function formatCheckpointLabel(c: CheckpointEntry, isLatest: boolean): string {
  const ts = c.mtime ? new Date(c.mtime * 1000) : null;
  const ago = ts ? formatAgo(Date.now() - ts.getTime()) : "";
  const head = `step ${c.step}`;
  const suffix = isLatest ? "  (latest)" : "";
  return ago ? `${head} · ${ago} ago${suffix}` : `${head}${suffix}`;
}

function formatAgo(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h`;
  return `${Math.floor(h / 24)}d`;
}

// Exported only so vitest can verify the regex without a full DOM mount.
export const __test = { REPO_ID_RE, formatCheckpointLabel };

# Publish to HuggingFace

A modal on the train-detail page that takes a completed training job
and uploads it to a HuggingFace repository. Supports two flows:

1. **Push checkpoint as-is.** Upload the trained adapter (LoRA A/B
   matrices and any base-weight overrides — what's already on disk in
   the job directory) as a stand-alone repo of `.pt` + metadata.
2. **Merge LoRA, push the full model.** Fold the LoRA delta back into
   the base model with `merge_and_unload`, save the merged
   `*ForCausalLM` weights + tokenizer, and upload the resulting
   self-contained model.

Anchors to existing UI conventions in [`docs/ui-design.md`](../../docs/ui-design.md):
modal pattern (§6.1, training submit), destructive-action confirmation
(§4.3), live-progress component (§5.5, log pane), credential handling
(§6.4 — secrets are never persisted to the browser, only forwarded
once per request).

## Motivation

The merge-and-push pipeline already exists as a CLI:
[`ml/adapters/merge_lora_and_push.py`](../../ml/adapters/merge_lora_and_push.py).
The CLI is what the UI wraps; this doc covers only the UI flow and the
two HTTP endpoints needed to drive it. Operators don't want to SSH and
remember argparse flags every time they ship a model.

## Non-goals

- **OAuth-based "log in with HuggingFace."** v1 takes a user-pasted
  access token, holds it in component state, and forwards it to a
  single backend call. We do not persist it to localStorage or
  IndexedDB; we do not write it to the server's disk. (Reference the
  same posture as the Cloudflare tunnel token in
  [`docs/configuration.md`](../../docs/configuration.md) §3 — secrets
  belong in a Secret resource, not in the chart.)
- **Repo browser / dataset card editor.** The HF web UI handles those
  better. We just push files and surface the resulting repo URL.
- **Per-step push.** You can publish at any time; the modal always
  acts on the latest checkpoint by default. Earlier checkpoints can be
  selected via a dropdown but not multi-selected.
- **Asynchronous job tracking outside the modal.** While a push is in
  flight the modal remains open with a progress indicator. Closing it
  cancels the upload (the underlying merge job continues server-side
  and can be re-driven; see "cancellation" below).
- **Mobile-optimized layout.** Same constraint as the rest of the app
  per `ui-design.md` non-goals.

## Surface

### Trigger

The train-detail card grows a single button next to "Open in Chat" and
"Cancel": **Publish to HF**. The button is enabled only when the job
status is `COMPLETED`. For `TRAINING`/`QUEUED` it's disabled with a
title tooltip explaining why; for `FAILED` it stays disabled.

### Modal layout

```
┌─ Publish "first-model" to HuggingFace ──────────────────── [×] ┐
│                                                                │
│  Mode                                                          │
│   ◉ Push checkpoint as-is                                      │
│      Adapter `.pt` + metadata only. Loadable by ScalarLM       │
│      and any vLLM with the LoRA adapter resolver.              │
│   ○ Merge LoRA into base, push full model                      │
│      Single self-contained model. Larger upload (~30–60 GB     │
│      for a 31B base). Required for off-platform inference.     │
│                                                                │
│  Repository                                                    │
│   ┌─────────────────────────────┐                              │
│   │ myorg/my-adapter            │                              │
│   └─────────────────────────────┘                              │
│   [ ] Private                                                  │
│                                                                │
│  HuggingFace token                                             │
│   ┌─────────────────────────────┐  hf_*** (paste; not saved)   │
│   │ ●●●●●●●●●●●●●●●●●●●●●●●●●●  │                              │
│   └─────────────────────────────┘                              │
│   Get one at huggingface.co/settings/tokens · Write access     │
│   required for new repos.                                      │
│                                                                │
│  Checkpoint                                                    │
│   [latest (step 100, 14m ago)        ▾]                        │
│                                                                │
│  ▷ Advanced                                                    │
│       Override lora_alpha · custom commit message ·            │
│       output dir for merge dry-run                             │
│                                                                │
│  ─────────────────────────────────────────────────────────     │
│                                          [ Cancel ] [ Publish ]│
└────────────────────────────────────────────────────────────────┘
```

While publishing, the bottom area is replaced by a progress region
that shows phased status (see "Progress states").

### Inline form rules

- **Mode** is a radio. Default = "Push checkpoint as-is" since that's
  the cheaper, faster path.
- **Repository** is `<org>/<name>` validated client-side: lowercase +
  digits + `-` + `_` per HF naming rules. Inline error if invalid.
- **Private** checkbox; defaults off. Existing repo visibility wins
  on the server side regardless.
- **Token** is `<input type="password">` with a "show" toggle. Stored
  only in component state; cleared on modal close. Pre-fills from a
  hint if the user has previously published in this session
  (sessionStorage-only, not localStorage — wiped on tab close).
- **Checkpoint** dropdown lists every `checkpoint_<step>.pt` with its
  step number and "<n>{m,h,d} ago" timestamp. Defaults to the latest.
- **Advanced** is a collapsed disclosure containing:
  - `lora_alpha` numeric override (skip if mode = checkpoint-as-is).
  - `commit_message` free-form text.
  - `dry_run` checkbox — runs the merge but skips the upload, useful
    to verify the merge succeeds before paying for the bandwidth.

### Progress states

Modal mounts a phase indicator while the request is open. Server emits
SSE-style progress events (see §API). Phases:

| phase | shown as |
|---|---|
| `validating` | "Reading checkpoint…" |
| `loading_base` | "Loading base model `<name>`…" |
| `merging` | "Merging LoRA into base…" (only mode=merge) |
| `saving` | "Saving merged weights…" |
| `uploading` | "Uploading to `<repo>` (`<x.x> GB / <y.y> GB`)…" |
| `done` | "Done. → `huggingface.co/<repo>`" with a copy button |
| `error` | `<phase>` + the server error message |

### Cancellation

The "Cancel" button while in flight aborts the HTTP request via the
`AbortController` already in `apiFetch`. Server-side, the publish job
can keep running (HF upload is one big multi-part request — pulling
the rug from under it half-way leaves a partial commit on the repo).
Document this on the cancel button's title tooltip:

> Cancels the UI request. A merge or upload that's already running
> server-side may still complete; check huggingface.co/<repo>.

A nicer alternative — server-side cancel — is not in scope for v1
because there's no job orchestration around the publish task; it
runs synchronously within the request handler.

## API

Two new POST endpoints, both under `/v1/megatron/train/{job_hash}/`.

### `POST /v1/megatron/train/{job_hash}/publish`

Request body:

```json
{
  "mode": "checkpoint" | "merge",
  "repo_id": "myorg/my-adapter",
  "private": false,
  "hf_token": "hf_xxxxxxxxxxxxxxx",
  "checkpoint": "checkpoint_100.pt",      // optional; defaults to latest
  "lora_alpha": 32,                       // optional; merge mode only
  "commit_message": "...",                // optional
  "dry_run": false                        // optional; merge mode only
}
```

Response: `text/event-stream` with newline-delimited JSON events of
the shape used by the existing log endpoints:

```
{"phase": "validating",   "message": "..."}\n
{"phase": "loading_base", "message": "..."}\n
{"phase": "merging",      "message": "..."}\n
{"phase": "saving",       "message": "..."}\n
{"phase": "uploading",    "loaded": 1234567890, "total": 5234567890}\n
{"phase": "done",          "repo_url": "https://huggingface.co/<repo>"}\n
```

On error:

```
{"phase": "error", "phase_failed": "uploading", "message": "401 Unauthorized: bad token"}\n
```

The handler shells the existing CLI as a subprocess for the merge
mode; for checkpoint-as-is it constructs the upload directly with
`huggingface_hub.HfApi`. Either way, the `hf_token` from the request
body is set as `HF_TOKEN` in the subprocess env (or passed straight
to `HfApi`) and **never logged**. The endpoint must scrub the token
from any error message it relays to the client.

### `GET /v1/megatron/train/{job_hash}/checkpoints`

Returns the list of `checkpoint_<step>.pt` files in the job directory
with their mtimes, used to populate the dropdown.

```json
{
  "checkpoints": [
    {"name": "checkpoint_100.pt", "step": 100, "mtime": 1761369000.123},
    {"name": "checkpoint_50.pt",  "step":  50, "mtime": 1761368400.456}
  ]
}
```

Sorted by step descending. The latest entry is what the dropdown
defaults to.

## UI files

```
ui/src/api/publish.ts                       # request hook + SSE consumer
ui/src/routes/train/PublishToHFModal.tsx    # the modal
ui/src/routes/train/TrainDetail.tsx         # adds the trigger button
```

`PublishToHFModal` consumes the SSE stream via the existing
`streamNdjsonLogOnce` helper from `api/training.ts` (the protocol is
the same shape — `{phase, message, ...}` records terminated by a
final `done` or `error`).

## Security

- The token is in `<input type="password">` and held in React state,
  not localStorage. The modal explicitly avoids the autofill ID/name
  pair that browser password managers latch onto.
- The token is forwarded over `POST` body (not query string) to keep
  it out of access logs. The route handler must call
  `logger.info(scrub(payload))` style logging where `scrub()` masks
  the token field; sample the existing `truncate_dict` helper in
  `create_generate_worker.py`.
- The server side never persists the token to disk, only passes it to
  `huggingface_hub` for the duration of the call. If the operator
  wants per-deployment HF credentials they can set `HF_TOKEN` in the
  deployment's cray-config.yaml, in which case the modal token field
  shows a "(use deployment credentials)" placeholder and may be
  blank.

## States

| state            | trigger                                | message |
|------------------|----------------------------------------|---------|
| disabled trigger | job_status != COMPLETED                | tooltip explains why |
| validation error | bad repo_id, missing token             | inline red text under the field |
| in-flight        | request opened                         | phase indicator + cancel |
| success          | server emits `done`                    | green check + repo URL + close |
| error            | server emits `error`                   | red banner with phase, copy-error button, retry stays |
| network failure  | fetch rejects                          | red banner with offline-style copy |

## Testing

- **Unit (vitest):** the form's validation rules — repo_id regex,
  required fields, token-not-saved-to-localStorage. Cover both modes.
- **Component (vitest + jsdom):** mount with a mock SSE source that
  emits each phase, assert the progress component renders each one
  in order. Assert that closing the modal mid-stream aborts the
  controller.
- **e2e:** out of scope. Real HF uploads can't be hermetic.

## Phased rollout

1. **Phase 0:** ship the `GET /checkpoints` endpoint + the form
   shell + the trigger button (disabled). Just renders, no work.
2. **Phase 1:** wire `mode = "checkpoint"`. Smaller upload, faster
   iteration, exercises the SSE plumbing.
3. **Phase 2:** wire `mode = "merge"`. Larger surface — touches the
   `merge_lora_and_push.py` CLI invocation and needs care around
   stdout-as-event-stream.
4. **Phase 3:** advanced options pane (lora_alpha override, dry_run,
   commit message).

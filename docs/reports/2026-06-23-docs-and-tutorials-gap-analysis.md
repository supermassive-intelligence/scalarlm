# ScalarLM Docs & Tutorials — Gap Analysis

_Survey date: 2026-06-23. Covers `README.md`, `Faq.md`, `CONTEXT.md`, everything
under `docs/` (excluding `docs/superpowers/`), and the external docs at
scalarlm.com that the repo links to. Goal: map what onboarding/tutorial material
exists, who it serves, and where the holes are._

---

## 1. The one-sentence finding

**ScalarLM has excellent _internals_ documentation and almost no _user-facing_
documentation in-repo.** Every genuine tutorial ("run your first inference",
"fine-tune a model", "deploy to Kubernetes", "save to HF") lives off-repo at
scalarlm.com and is reached only by an external link. The single best onboarding
artifact in the tree — a runnable end-to-end fine-tune walkthrough — exists but is
**untracked** and **filed as an investigation report**, not as a tutorial.

This matters more than usual because ScalarLM is **CC-0 and explicitly built to be
forked** ("fork it, publish with it, build on it, and ship it" — README). A fork
carries the code and the internals docs but **zero getting-started content**, because
the tutorials are on a website that doesn't come with the clone.

---

## 2. What exists today (inventory)

### 2.1 Tracked, published-quality docs — all internals/design

| Doc | Lines | Audience | Kind |
|---|---|---|---|
| `architecture.md` | 720 | Contributor/maintainer | System deep-dive |
| `training-lifecycle.md` | 550 | Contributor | Subsystem deep-dive |
| `configuration.md` | 603 | Operator/contributor | Reference |
| `adapters.md` | 524 | Contributor | Subsystem deep-dive |
| `inference-queue.md` | 385 | Contributor | Subsystem deep-dive |
| `openai-chat-completions-queue.md` | 530 | Contributor | **Design proposal** (not shipped) |
| `inference-request-browser.md` | 212 | Contributor | Feature design |
| `ui-design.md` | 637 | Contributor | Design record (Shipped) |
| `gpu-aware-mpi.md` | 623 | Contributor | Subsystem deep-dive |
| `chunked-upload.md` | 60 | Contributor | Protocol reference |
| `sparse-sampling.md` | 33 | Contributor | **Design proposal** (not shipped) |
| `observability/Observability.md` | — | Operator | How-to (good) |
| `test-plan.md` | 804 | Contributor | **Design proposal** (not shipped) |
| `blog/training-for-a-week-...md` | — | General | Blog/narrative |
| `adr/0003-...` | — | Maintainer | ADR (the only _tracked_ ADR) |

Quality of these is high. The problem is the **coverage shape**: they all assume you
already know what ScalarLM is and want to understand or modify how it works.

### 2.2 Untracked working notes (not part of published docs)

- `reports/fine-tuning-a-served-model.md` — **the best onboarding doc in the repo.**
  Has a full conceptual model, a footguns list, a compute/VRAM sizing table for every
  sweep model, and a **runnable end-to-end LoRA worked example**. Untracked, and filed
  under `reports/` as an "investigation report."
- `reports/lora-serving-noop-investigation.md`, `reports/2026-06-18-*`, the
  `2026-06-22-*` sweep reports — diagnostic reports (rightly ephemeral).
- `adr/0001`, `0002`, `0004`, `0005` — **untracked**, while `0003` is tracked. The ADR
  set is committed incoherently.
- `personal/`, `handoffs/`, `self-served-llms-scratchpad.md` — working notes (fine as
  untracked).

### 2.3 External-only (scalarlm.com), nothing in-repo

Linked from README's "Get Started" table and `Faq.md`, but **no in-repo equivalent**:

- Quick Start (`/quick-start/`)
- Custom Training (`/training/`)
- Kubernetes Deployment (`/kubernetes/`)
- Save Fine-tuned Model to Hugging Face (`/save-fine-tuned-model-to-hugging-face/`)
- Architecture overview page (`/architecture/`)

### 2.4 Root-level onboarding

- `README.md` — strong. 4-step Quick Start, CLI table, validated-models table, Docker
  targets. This is the _only_ real in-repo onboarding surface.
- `Faq.md` — good, practical, but scattered (CLI listed here _and_ in README; training
  concepts mixed with caching trivia). Defers heavily to scalarlm.com.

---

## 3. Coverage matrix (audience × journey)

`✓` solid in-repo · `~` partial/scattered · `✗` missing in-repo (may exist on website) · `🔒` exists but untracked

| Journey stage | New user | Operator/deployer | Contributor |
|---|---|---|---|
| What is it / why | ✓ README | ✓ README | ✓ architecture |
| Install / first run | ✓ README QuickStart | ~ README + Docker table | ✗ no dev-env/test setup |
| First inference call | ~ README snippet | — | ✓ inference-queue |
| Chat / streaming / model select | ✗ | ✗ | ~ openai-queue (design) |
| **First fine-tune (the loop)** | 🔒 untracked report | 🔒 | ✓ training-lifecycle |
| LoRA vs Tokenformer choice | 🔒 (sizing table) | 🔒 | ✓ adapters |
| Deploy to Kubernetes | ✗ (external link) | ✗ (external link) | ~ architecture mentions Helm |
| Configuration / tuning | ~ Faq | ✓ configuration | ✓ configuration |
| Monitoring / observability | ~ Faq (plot) | ✓ observability | ✓ |
| Troubleshooting / footguns | ✗ | ✗ | 🔒 (report has the best list) |
| Save / export model | ✗ (external link) | ✗ | 🔒 |
| Contributing / dev workflow | ~ README 5 lines | — | ✗ no CONTRIBUTING.md |

The diagonal that matters for adoption — **new user → first inference → first
fine-tune → deploy** — is the emptiest row band, and it's exactly the closed-loop
story the project leads with.

---

## 4. Prioritized gaps

### P0 — Blocks a forker/new user from succeeding without the website

1. **No in-repo getting-started tutorial beyond the README.** All tutorials are
   external links. A clone/fork has no guided path. _This is the headline gap given the
   CC-0 "fork and ship" positioning._

2. **The closed-loop fine-tune tutorial is untracked and mis-filed.**
   `reports/fine-tuning-a-served-model.md` is tutorial-grade (runnable worked example +
   footguns + sizing) but isn't committed and lives under `reports/`. The project's
   _headline feature_ — query → build signal → post-train → next request picks it up —
   has no committed end-to-end walkthrough. **Lowest-effort, highest-impact fix: promote
   this file.**

3. **No docs index / navigation.** `docs/` is a flat pile of 14+ files with no
   `docs/README.md`, no reading order, no audience tiering. The deep-dives cross-link
   "§3.4 of architecture.md" but nothing tells a reader where to _start_ or which docs
   are user-facing vs internals vs design proposals.

### P1 — Present but fragmented or risky to trust

4. **Shipped vs proposed is not labeled consistently.** `ui-design.md` ("Shipped") and
   `test-plan.md` ("Design") carry a `## Status` header; `openai-chat-completions-queue.md`
   and `sparse-sampling.md` are _design proposals for unbuilt features_ but sit
   alongside shipped docs with no status marker. A reader can't tell what's real.
   `inference-queue.md` (shipped) and `openai-chat-completions-queue.md` (proposed,
   re-routes the same path) are especially easy to conflate.

5. **No user-facing inference guide.** Deep internals on the work queue exist, but
   there's no "how to call the API / SDK for generation, chat, streaming, temperature,
   selecting an adapter by job hash" guide. The pieces are scattered across README,
   Faq, and the queue design docs.

6. **No troubleshooting / footguns doc.** The best failure-mode catalog (dtype fp32 →
   mode collapse, adapter no-op / `NO_MEMORIZATION`, `lora_alpha` mis-scaling, sha256
   job dedup silently returning an old job, `block_size` blow-up) lives only in the
   untracked report and in session memories — invisible to users.

7. **No CONTRIBUTING.md / dev-environment guide.** README has a 5-line blurb;
   `test-plan.md` is a _design_, not runnable instructions. The actual way to run tests
   is non-obvious (requires `PYTHONPATH=infra uv run --with pytest --with torch …`) and
   undocumented. `./scalarlm test` is mentioned but the local Python-test path isn't.

### P2 — Hygiene / consistency

8. **ADRs committed incoherently** — `0003` tracked; `0001/0002/0004/0005` untracked.
   Decide whether ADRs are public record (commit the set) and link them from a docs
   index.

9. **CLI reference duplicated** across README and Faq, with slightly different command
   lists (`llm-logs`/`llm-ls`/`llm-plot`/`llm-squeue` vs `logs/plot/ls/squeue`). One
   canonical CLI reference would remove the drift.

10. **No conceptual "adapters for users" doc.** `adapters.md` is implementation-level.
    The user-level question — "Tokenformer vs LoRA: what, when, and what does it cost in
    VRAM/time?" — is answered only in the untracked report's sizing section.

---

## 5. Recommendations (ordered by impact ÷ effort)

1. **Promote and commit the fine-tune walkthrough.** Move
   `reports/fine-tuning-a-served-model.md` → `docs/tutorials/fine-tuning-a-served-model.md`
   (split the runnable worked example + footguns from the investigation framing), and
   `git add` it. Single highest-leverage action.

2. **Add `docs/README.md` as an index** with three sections — _Tutorials_ (start here),
   _Reference_ (configuration, CLI, API), _Internals & Design_ (the deep-dives, each
   tagged Shipped/Proposed). Gives the flat pile a spine and fixes the "where do I
   start" problem.

3. **Add a `## Status: Shipped | Proposed | Design` header to every design doc** and
   backfill the ones missing it (`openai-chat-completions-queue.md`,
   `sparse-sampling.md`, `inference-request-browser.md`, the queue docs).

4. **Write a short in-repo Quick Start tutorial** that mirrors the website's
   `/quick-start/` so a fork is self-sufficient: up → first inference (curl + SDK) →
   first LoRA fine-tune → see the adapter serve. Can largely assemble from the README
   snippets + the worked example.

5. **Add a Troubleshooting / Footguns doc** seeded from the report's footguns list and
   the no-memorization/adapter-no-op diagnostics already captured.

6. **Add `CONTRIBUTING.md`** with the real local dev + test invocation (the
   `PYTHONPATH=infra uv run …` path), branch/PR flow, and where `ml/` ships-with-job
   means you _don't_ rebuild.

7. **Commit the ADR set** (or explicitly gitignore them as private) so the decision
   record is coherent.

---

## 6. What's genuinely good (keep)

- `architecture.md`, `training-lifecycle.md`, `adapters.md`, `configuration.md` are
  thorough, code-anchored, and cross-linked — strong contributor onboarding once a
  reader knows to look there.
- `observability/Observability.md` is a model how-to: states what you get, then how.
- `CONTEXT.md` is an excellent domain glossary; the ADRs reference it well.
- The README is a strong front door — the gap is everything _behind_ it, not the door
  itself.

---

_Note: the depth picture is skewed by the current branch (`georgi/finetune-sweep`),
which has accumulated many sweep diagnostic reports and handoffs in the working tree.
Those are correctly ephemeral; this analysis treats them as such and focuses on the
durable user/contributor documentation surface._

# ScalarLM — serving & fine-tune sweep

Domain language for how ScalarLM serves models and how the fine-tune sweep
(`test/finetune_sweep/`) drives the train → serve closed loop. This glossary
covers only terms that carry project-specific meaning; it is referenced by the
ADRs in `docs/adr/`.

## Language

**Base Model**:
The model vLLM binds at engine startup, i.e. `config["model"]`. Serving a
different Base Model requires a full server restart (a fresh pod), not an engine
swap — this is why the sweep isolates one model per run.
_Avoid_: served model, current model.

**Adapter**:
A fine-tune artifact (LoRA) hot-loaded onto the running Base Model and addressed
by its `job_hash`. Discovered from the `jobs` volume by globbing `*.pt`; never
requires the trainer to be alive.
_Avoid_: fine-tune, model (when you mean the adapter).

**Closed loop**:
The sweep's per-model cycle: launch the stack → train a tiny LoRA → verify the
checkpoint → hot-load the adapter → check the served output memorized the target.
_Avoid_: pipeline, end-to-end test (when you mean this specific cycle).

**Memorization**:
The pass signal: the adapter reproduces `expected_output` (a random hex string a
base model never emits unattributed) for `golden_prompt`. Its absence is
`NO_MEMORIZATION` — a non-failing outcome, not a hard failure.
_Avoid_: accuracy, correctness.

**Phase-scaled run**:
A k8s sweep variant (`phase_scaled: true`) that runs the closed loop in two
sequential single-GPU phases instead of two always-on GPU pods, so peak GPU
demand is one card.
_Avoid_: single-GPU mode, time-sliced (a different GPU-sharing technique).

**Co-located run**:
The Compose sweep's way of fitting the closed loop on one GPU: a single container
runs the server (vLLM, in-process) and dispatches the trainer (a slurm job in the
same container), so both share the one card *simultaneously* — no phase handoff.
This is what the `cuda-docker` target does; it is the co-location the phase-scaled
k8s path could not reach without chart surgery.
_Avoid_: phase-scaled (the disjoint k8s technique), single-GPU mode.

**Train phase** (phase 1):
The phase-scaled stage where megatron holds the single GPU and vLLM is scaled to
zero; training runs and the checkpoint is read here.
_Avoid_: training step.

**Serve phase** (phase 2):
The phase-scaled stage after the GPU handoff: megatron is scaled to zero, vLLM
holds the GPU, and the baseline + hot-load + memorization check run here.
_Avoid_: inference step.

**GPU handoff**:
The phase-scaled transition that frees the GPU from megatron (`scale --replicas=0`,
wait for the pod to fully delete) before vLLM claims it (`scale --replicas=1`).
The card is briefly unreserved during the gap — "1 GPU at a time," not "1 GPU
reserved throughout."
_Avoid_: GPU swap, failover.

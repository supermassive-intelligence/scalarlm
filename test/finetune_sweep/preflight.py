#!/usr/bin/env python3
"""Offline preflight for the fine-tune sweep — predict the silent LoRA no-op
*before* paying for a model's restart + train + serve.

The check is purely structural: synthesize the trainer's would-be LoRA keys,
push them through the fork's REAL serve-time normalization (a two-pass process,
`normalize_lora_key` + `PTWorkerLoRAManager._renormalize_lora_sd_for_model`),
and ask whether the resulting module paths exist in the served model's vLLM
module tree. Zero overlap predicts the no-op (PRECHECK_NO_OP).

Faithfulness comes from running the model through vLLM's OWN loader: we build
only the `nn.Module` tree on the meta device (no weights, no GPU, no engine) via
`vllm.model_executor.model_loader.utils.initialize_model`, then run the actual
fork normalization against that tree. See
docs/superpowers/specs/2026-06-11-finetune-sweep-dry-test-design.md.

The torch/vLLM work happens inside the cray image via `docker compose run`; this
host module owns only the orchestration and the torch-free seams (key synthesis,
set overlap, output parsing). The in-container script is a module constant so the
host can unit-test the parsing seam without an image.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The standard LoRA target leaves the trainer LoRA-ifies, grouped by their
# parent block. Excludes lm_head / embeddings (the trainer does not touch them).
DEFAULT_LORA_TARGETS: dict[str, tuple[str, ...]] = {
    "self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}


def synthesize_lora_keys(n_layers: int = 1,
                         targets: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """The trainer's would-be LoRA keys, shaped exactly as it exports them:
    `model.layers.{i}.{block}.{leaf}.lora_A.default.weight`. One key per target
    module suffices for the overlap check (lora_A and lora_B collapse to the same
    module path), so we emit only the `.lora_A.` form. `n_layers=1` is enough —
    the per-layer pattern repeats — which keeps the synthesized set small."""
    if targets is None:
        targets = DEFAULT_LORA_TARGETS
    keys: list[str] = []
    for i in range(n_layers):
        for block, leaves in targets.items():
            for leaf in leaves:
                keys.append(f"model.layers.{i}.{block}.{leaf}.lora_A.default.weight")
    return keys


def overlap(base_modules: set[str], lora_module_paths: set[str]) -> int:
    """Count of LoRA module paths present in the base model's module set — the
    same intersection `_warn_on_zero_base_match` computes. The preflight's
    `predicted_ok` is the permissive `overlap(...) > 0`: a single match silences
    the no-op prediction (partial overlap, e.g. from q/k/v fusion or unsupported
    vision keys, is benign; only *zero* overlap reliably predicts the no-op)."""
    return len(base_modules & lora_module_paths)


@dataclass
class PreflightResult:
    model_id: str
    predicted_ok: bool = False
    n_overlap: int = 0
    n_total: int = 0
    sample_adapter_keys: list[str] = field(default_factory=list)  # post-normalize
    sample_base_modules: list[str] = field(default_factory=list)
    error: str = ""  # build/introspection failure -> fail open (run the model)

    @property
    def predicted_noop(self) -> bool:
        """True iff the preflight confidently predicts the silent no-op — zero
        overlap AND no build error. A build error is NOT a no-op prediction
        (fail open: run the model rather than skip it on a preflight crash)."""
        return not self.error and not self.predicted_ok


def parse_preflight_output(model_id: str, stdout: str) -> PreflightResult:
    """Parse the in-container script's JSON (the last JSON-bearing line, after
    any torch/vLLM startup chatter) into a PreflightResult. A `{"error": ...}`
    payload or unparseable output yields a result with `error` set, which fails
    open (`predicted_noop` is False)."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("error"):
            return PreflightResult(model_id=model_id, error=str(data["error"]))
        return PreflightResult(
            model_id=model_id,
            predicted_ok=bool(data.get("predicted_ok", False)),
            n_overlap=int(data.get("n_overlap", 0)),
            n_total=int(data.get("n_total", 0)),
            sample_adapter_keys=list(data.get("sample_adapter_keys", [])),
            sample_base_modules=list(data.get("sample_base_modules", [])),
        )
    return PreflightResult(
        model_id=model_id,
        error=f"could not parse preflight output: {stdout[:200]!r}",
    )


# The in-container introspection script: meta-tree build + the REAL two-pass
# normalization. Runs inside the cray image (has transformers, vLLM, the fork).
# Kept as a module constant so the host can unit-test parse_preflight_output
# without an image; the script body itself is integration-tested on the box.
_INTROSPECT_SCRIPT = r'''
import json, sys

def run_introspection(model_id):
    try:
        import socket
        import torch
        from vllm.distributed import (init_distributed_environment,
                                      initialize_model_parallel)
        from vllm.config import set_current_vllm_config
        from vllm.engine.arg_utils import EngineArgs
        from vllm.model_executor.model_loader.utils import initialize_model
        from vllm.tokenformer.adapter_format import normalize_lora_key
        from vllm.tokenformer.hybrid_adapter_manager import PTWorkerLoRAManager

        # 1. Build only the nn.Module tree on the meta device (no weights, no
        #    GPU, no engine) -- the same loader code the real engine runs.
        #    EngineArgs.create_engine_config assembles the VllmConfig (incl. the
        #    LoadConfig that owns load_format) exactly as the engine does, so we
        #    don't hand-construct ModelConfig (whose fields shift between vLLM
        #    versions). load_format="dummy" skips weight download/read.
        vllm_config = EngineArgs(
            model=model_id, load_format="dummy", enforce_eager=True,
        ).create_engine_config()

        # Both the parallel-group init and model construction read
        # get_current_vllm_config(), so they must run inside the config context.
        # Stand up a single-process group first (model construction reads the PP
        # group / TP world size even at TP=1). backend "gloo" works on both the
        # CPU and GPU images -- we never run a collective, the groups just have
        # to exist. A free ephemeral port avoids clashes between concurrent runs.
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0,
            distributed_init_method="tcp://127.0.0.1:%d" % port, backend="gloo")
        with set_current_vllm_config(vllm_config):
            initialize_model_parallel(tensor_model_parallel_size=1,
                                      pipeline_model_parallel_size=1)
            with torch.device("meta"):
                model = initialize_model(vllm_config)
        base_modules = set(n for n, _ in model.named_modules())

        # 2. Synthesize the trainer's would-be LoRA keys (layer 0 suffices).
        targets = {"self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
                   "mlp": ("gate_proj", "up_proj", "down_proj")}
        synth = []
        for block, leaves in targets.items():
            for leaf in leaves:
                synth.append(
                    "model.layers.0.%s.%s.lora_A.default.weight" % (block, leaf))

        # 3. Run the REAL two-pass normalization.
        #    Pass 1 (static): normalize_lora_key -- what load_adapter_from_pt does.
        pass1 = {normalize_lora_key(k): None for k in synth}
        #    Pass 2 (live tree): run the REAL _renormalize_lora_sd_for_model
        #    (which calls _detect_model_layers_prefix). Both methods touch only
        #    self._adapter_manager.model, so subclass PTWorkerLoRAManager and
        #    override __init__ to set just that, skipping the heavy
        #    LRUCacheWorkerLoRAManager.__init__.
        class _AM:
            def __init__(self, m):
                self.model = m
        class _Stand(PTWorkerLoRAManager):
            def __init__(self, m):
                self._adapter_manager = _AM(m)
        pass2 = _Stand(model)._renormalize_lora_sd_for_model(pass1)

        # 4. Recover module paths (strip the .lora_A/.lora_B weight tail), then
        #    overlap exactly as _warn_on_zero_base_match does.
        def module_path(k):
            for tail in (".lora_A.weight", ".lora_B.weight"):
                if k.endswith(tail):
                    return k[:-len(tail)]
            return k
        lora_module_paths = set(module_path(k) for k in pass2)
        matches = lora_module_paths & base_modules

        return {
            "predicted_ok": len(matches) > 0,
            "n_overlap": len(matches),
            "n_total": len(lora_module_paths),
            "sample_adapter_keys": sorted(lora_module_paths)[:3],
            "sample_base_modules": [
                m for m in sorted(base_modules)
                if "self_attn" in m or "mlp" in m][:3],
        }
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e)}

if __name__ == "__main__":
    print(json.dumps(run_introspection(sys.argv[1])))
'''


def run_preflight(model_ids: list[str], compose_service: str) -> dict[str, PreflightResult]:
    """Run the offline preflight as a filter over `model_ids`. Issues one
    throwaway `docker compose run --rm --no-deps <service>` per model (entirely
    inside the cray image) and parses each JSON result. Compose-only: needs a
    `compose_service`. A subprocess failure is recorded as `error` (fail open)."""
    results: dict[str, PreflightResult] = {}
    for mid in model_ids:
        cmd = [
            "docker", "compose", "-f", "docker-compose.yaml",
            "run", "--rm", "--no-deps", compose_service,
            "python3", "-c", _INTROSPECT_SCRIPT, mid,
        ]
        try:
            res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True,
                                 text=True, timeout=300)
            results[mid] = parse_preflight_output(mid, res.stdout)
        except Exception as e:
            results[mid] = PreflightResult(model_id=mid,
                                           error=f"preflight run failed: {e}")
    return results

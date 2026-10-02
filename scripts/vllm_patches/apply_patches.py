#!/usr/bin/env python3
"""Apply ScalarLM's vLLM-fork patches in-place.

Invoked from both:
  - scripts/build-copy-vllm.sh at Docker build time (production path)
  - Kubernetes pod `command:` at startup (dev/bench iteration)

Adds new patches here. Each patch is a function that takes the vLLM tree
root, reads the target file, asserts the exact anchor it expects, and
writes the transformed file back. The assertions are load-bearing — they
fail loudly when a fork rebase drifts the source rather than silently
producing a mis-patched image.

Usage: apply_patches.py <path/to/vllm/root>
"""

from __future__ import annotations

import sys
from pathlib import Path


def patch_output_handler_metrics_offload(vllm_root: Path) -> None:
    """Phase 6.5: move `logger_ref[0].record(...)` out of async_llm's output
    loop onto a background consumer coroutine fed by a bounded asyncio.Queue.

    Follows the TODO vLLM itself left on `vllm/v1/engine/async_llm.py:700`.
    Profile: the synchronous record call was 20.3 %% of main-thread time at
    N=100 and the consumer-offload pattern recovered +15 %% throughput in
    the pilot A/B. See enhance-openai-api.md § "Phase 6.5".
    """
    target = vllm_root / "vllm" / "v1" / "engine" / "async_llm.py"
    src = target.read_text()

    # --- Anchor 1: the inline record call we're replacing. ---
    anchor_record = (
        "                    # 4) Logging.\n"
        "                    # TODO(rob): make into a coroutine and launch it in\n"
        "                    # background thread once Prometheus overhead is non-trivial.\n"
        "                    if logger_ref[0]:\n"
        "                        logger_ref[0].record(\n"
        "                            engine_idx=outputs.engine_index,\n"
        "                            scheduler_stats=outputs.scheduler_stats,\n"
        "                            iteration_stats=iteration_stats,\n"
        "                            mm_cache_stats=renderer.stat_mm_cache(),\n"
        "                        )\n"
    )
    assert anchor_record in src, (
        "async_llm.py anchor missing: the inline `logger_ref[0].record(...)` "
        "block expected at the end of output_handler has drifted. Rebase of "
        "the vLLM fork requires re-reading scripts/vllm_patches/apply_patches.py "
        "to re-anchor this patch."
    )

    replacement_record = (
        "                    # 4) Logging — offloaded to background consumer.\n"
        "                    #    ScalarLM patch (Phase 6.5). vLLM's own TODO\n"
        "                    #    on this spot suggested a background thread;\n"
        "                    #    an asyncio task + bounded queue yields the\n"
        "                    #    same shape with less machinery.\n"
        "                    if logger_ref[0] is not None:\n"
        "                        try:\n"
        "                            metrics_queue_ref[0].put_nowait((\n"
        "                                outputs.engine_index,\n"
        "                                outputs.scheduler_stats,\n"
        "                                iteration_stats,\n"
        "                                renderer.stat_mm_cache(),\n"
        "                            ))\n"
        "                        except asyncio.QueueFull:\n"
        "                            # Metrics are statistical — drop the\n"
        "                            # oldest rather than block the output\n"
        "                            # loop. The engine must keep pulling.\n"
        "                            try:\n"
        "                                metrics_queue_ref[0].get_nowait()\n"
        "                                metrics_queue_ref[0].put_nowait((\n"
        "                                    outputs.engine_index,\n"
        "                                    outputs.scheduler_stats,\n"
        "                                    iteration_stats,\n"
        "                                    renderer.stat_mm_cache(),\n"
        "                                ))\n"
        "                            except (asyncio.QueueEmpty, asyncio.QueueFull):\n"
        "                                pass\n"
    )

    # --- Anchor 2: the ``async def output_handler()`` signature — we hang
    #     the metrics queue + consumer task off this spot, one level up
    #     from the handler body, so they share closure state. ---
    anchor_handler_def = "        async def output_handler():\n"
    assert src.count(anchor_handler_def) == 1, (
        "async_llm.py anchor missing: expected exactly one "
        "`async def output_handler():` inside `_run_output_handler`. "
        "Fork has drifted."
    )

    consumer_preamble = (
        "        # ScalarLM Phase 6.5: bounded queue + consumer task that\n"
        "        # drains the record() calls off the output_handler hot\n"
        "        # path. 1024 items is ~seconds of engine iterations at our\n"
        "        # scale; metric samples that overflow are dropped (see\n"
        "        # output_handler for the drop-oldest path).\n"
        "        metrics_queue_ref: list = [asyncio.Queue(maxsize=1024)]\n"
        "\n"
        "        async def _metrics_consumer():\n"
        "            q = metrics_queue_ref[0]\n"
        "            while True:\n"
        "                try:\n"
        "                    engine_idx, scheduler_stats, iteration_stats, mm_cache_stats = (\n"
        "                        await q.get()\n"
        "                    )\n"
        "                except asyncio.CancelledError:\n"
        "                    return\n"
        "                try:\n"
        "                    if logger_ref[0] is not None:\n"
        "                        logger_ref[0].record(\n"
        "                            engine_idx=engine_idx,\n"
        "                            scheduler_stats=scheduler_stats,\n"
        "                            iteration_stats=iteration_stats,\n"
        "                            mm_cache_stats=mm_cache_stats,\n"
        "                        )\n"
        "                except Exception:\n"
        "                    logger.exception(\"Background metrics record failed.\")\n"
        "\n"
        "        async def output_handler():\n"
    )

    # --- Anchor 3: the ``self.output_handler = asyncio.create_task(...)``
    #     line where the handler task is actually scheduled. We schedule
    #     the consumer next to it so the two have matching lifecycles. ---
    anchor_schedule = (
        "        self.output_handler = asyncio.create_task(output_handler())\n"
    )
    assert anchor_schedule in src, (
        "async_llm.py anchor missing: expected the "
        "`self.output_handler = asyncio.create_task(output_handler())` line."
    )
    replacement_schedule = (
        "        self.output_handler = asyncio.create_task(output_handler())\n"
        "        self._metrics_consumer_task = asyncio.create_task(_metrics_consumer())\n"
    )

    # Apply in order; each assertion above already guaranteed uniqueness.
    patched = (
        src
        .replace(anchor_record, replacement_record)
        .replace(anchor_handler_def, consumer_preamble)
        .replace(anchor_schedule, replacement_schedule)
    )

    # Sanity: the patch should change the file.
    assert patched != src, "patch produced identical output — something's wrong"
    # And it should still parse.
    compile(patched, str(target), "exec")

    target.write_text(patched)
    print(f"[vllm_patches] Applied output_handler metrics offload to {target}")

TOKENFORMER_KEY_RESOLVER_SRC = '''

def _scalarlm_nearest_live_keys(key, model_state_dict, limit=3):
    """Live parameter names that sit near an unmatched checkpoint key.

    A drop warning that only names the missing key says nothing about WHY
    it missed. The live neighbours do: `...qkv_proj.base_layer.weight`
    next to a dropped `...q_proj.weight` shows both that vLLM fused the
    projection and that a LoRA wrapper nested it -- two separate naming
    problems, visible in one line, without reading fork source.

    Tries every contiguous run of path components, longest first, and
    returns the matches for the most specific one that hits anything.
    Trailing components have to be droppable too, not just leading ones:
    a fused `qkv_proj` means the name `q_proj` appears nowhere at all, so
    only backing off to the parent module (`layers.0.self_attn`) shows
    what the layer actually holds.
    """
    parts = key.split(".")
    candidates = {
        ".".join(parts[i:j])
        for i in range(len(parts))
        for j in range(i + 1, len(parts) + 1)
    }
    for needle in sorted(candidates, key=lambda c: (-len(c), c)):
        matches = [name for name in model_state_dict if needle in name]
        if matches:
            return sorted(matches)[:limit]
    return []


def _scalarlm_resolve_adapter_keys(model, tokenformers, model_state_dict):
    """Map trainer-side checkpoint keys onto the live vLLM namespace.

    ScalarLM's Tokenformer checkpoints are not pure adapters: the trainer
    unfreezes attention projections, norms and embeddings alongside the
    tokenformer_{k,v,p} tensors, so the .pt carries base weights named the
    way the HuggingFace module tree names them. `activate_adapter` merges
    every one of those keys into the model's state dict and calls
    `load_weights`, which requires the names to line up with vLLM's own
    parameter layout.

    For plain decoder-only models the two namespaces already agree, which
    is why this went unnoticed. Multimodal wrappers rearrange them (HF
    `model.language_model.layers.*` vs vLLM `language_model.model.layers.*`)
    and expose some modules not at all, so unmapped keys reach
    `load_weights` and raise KeyError, killing EngineCore.

    Resolution order:
      1. Key already present in the model's state dict: keep it unchanged.
         This is the backward-compatibility guarantee — every key of an
         existing checkpoint takes this path, so nothing is renamed or
         dropped and behavior is identical to before the patch.
      2. The model's own `hf_to_vllm_mapper` rewrites it to a key that is
         present: keep it under the rewritten name. This reuses the exact
         mapping vLLM applies when loading a checkpoint, so it stays
         correct per model family instead of hardcoding prefixes here.
      3. Otherwise: drop it, loudly. These are weights vLLM has no home
         for; injecting them is what crashes the engine. Dropping means
         they are not applied at inference — the warning exists so that
         shows up in the logs rather than as unexplained quality loss.
    """
    # Instance lookup, not `type(model)`: most models declare the mapper as a
    # class attribute, but several (mimo_mtp, ernie_mtp, the transformers
    # backend) build it in __init__, and a class-only lookup silently misses
    # those — every key would fall through to the drop path.
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    resolved = {}
    renamed = 0
    dropped = []

    for key, value in tokenformers.items():
        if key in model_state_dict:
            resolved[key] = value
            continue

        mapped = None
        if mapper is not None:
            try:
                mapped = mapper._map_name(key)
            except Exception:  # mapper API drift must not break serving
                mapped = None

        if mapped is not None and mapped in model_state_dict:
            resolved[mapped] = value
            renamed += 1
            continue

        dropped.append(key)

    if renamed:
        logger.info(
            "Tokenformer adapter: remapped %d checkpoint tensors onto the "
            "live model namespace via hf_to_vllm_mapper.",
            renamed,
        )
    if dropped:
        logger.warning(
            "Tokenformer adapter: %d of %d checkpoint tensors have no "
            "counterpart in the live model and will NOT be applied "
            "(first few: %s). These are base weights the trainer saved "
            "under names vLLM does not expose; the adapter still applies "
            "everything that did resolve.",
            len(dropped),
            len(tokenformers),
            dropped[:5],
        )
        # Name the live parameters nearest each drop. Without this the
        # warning says only that a name was missing, and diagnosing why
        # means reading fork source and guessing -- which is how the
        # `.base_layer.` nesting and the fused `qkv_proj` each cost a
        # 30-minute rebuild to discover.
        for key in dropped[:3]:
            logger.warning(
                "Tokenformer adapter:   %s -> no match; live names nearby: %s",
                key,
                _scalarlm_nearest_live_keys(key, model_state_dict) or "(none)",
            )

    return resolved
'''


def patch_tokenformer_adapter_key_resolution(vllm_root: Path) -> None:
    """Resolve adapter checkpoint keys against the live model before use.

    Without this, serving a ScalarLM Tokenformer checkpoint on a
    multimodal model (diffusion-gemma / gemma4) dies with
    `KeyError: 'layers.0.self_attn.q_proj.weight'` inside
    `gemma4.load_weights`, taking EngineCore down with it.

    Resolution happens once in `add_adapter`, so both `activate_adapter`
    and `deactivate_adapter` see keys that exist in the model — patching
    only the activate path would leave deactivate to raise on the same
    keys.

    See `_scalarlm_resolve_adapter_keys` for the rules and why existing
    checkpoints are unaffected.
    """
    target = (
        vllm_root / "vllm" / "tokenformer" / "tokenformer_model_manager.py"
    )
    if not target.exists():
        print(
            f"[vllm_patches] {target} not found; skipping tokenformer "
            f"key-resolution patch"
        )
        return

    src = target.read_text()

    if "_scalarlm_resolve_adapter_keys" in src:
        print(
            "[vllm_patches] tokenformer key resolution already present; "
            "skipping"
        )
        return

    anchor_logger = "logger = init_logger(__name__)\n"
    anchor_register = (
        "        self._registered_adapters[request.adapter_id] = tokenformer\n"
    )

    assert anchor_logger in src, (
        "tokenformer_model_manager.py: `logger = init_logger(__name__)` not "
        "found — cannot place the key resolver. Re-anchor this patch."
    )
    assert anchor_register in src, (
        "tokenformer_model_manager.py: the `self._registered_adapters"
        "[request.adapter_id] = tokenformer` assignment in add_adapter has "
        "drifted. Re-anchor this patch."
    )

    patched = src.replace(
        anchor_logger,
        anchor_logger + TOKENFORMER_KEY_RESOLVER_SRC,
        1,
    ).replace(
        anchor_register,
        "        tokenformer.tokenformers = _scalarlm_resolve_adapter_keys(\n"
        "            self.model, tokenformer.tokenformers, self.model.state_dict()\n"
        "        )\n" + anchor_register,
        1,
    )

    assert patched != src, "patch produced identical output — something's wrong"
    compile(patched, str(target), "exec")

    target.write_text(patched)
    print(f"[vllm_patches] Applied tokenformer key resolution to {target}")


STATE_DICT_EXPORT_SRC = '''    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # ScalarLM trainer contract: expose packed projections under their
        # unpacked HF names. qwen2, qwen3, qwen3_moe and gemma3 implement
        # this; Gemma4 was missed, so `qkv_proj` stayed fused and the
        # Tokenformer manager found no q/k/v to match a ScalarLM
        # checkpoint against. Every trained attention projection was then
        # dropped at adapter-activation time and the served model ran
        # random-init attention -- training converges, the checkpoint is
        # correct, and generation is still garbage.
        #
        # Unlike gemma3, qkv cannot be delegated to
        # `unpack_packed_modules_state_dict`: it derives one split size
        # from the model config, while Gemma4 alternates head_dim per
        # layer (32 on some, 64 on others). That mismatch raises
        # "split_with_sizes expects split_sizes to sum exactly to 1024".
        # Each attention module knows its own TP-local q_size/kv_size,
        # which is what the packed tensor here actually holds, so read
        # the geometry off the module instead of the config.
        #
        # Scoped to `prefix` so recursion from the multimodal wrapper
        # (Gemma4ForConditionalGeneration) can't rewrite sibling modules'
        # keys in the shared destination dict.
        state_dict = super().state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        if not scalarlm_state_dict_export_enabled():
            return state_dict

        layers = getattr(getattr(self, "model", None), "layers", None) or []
        for index, layer in enumerate(layers):
            # PPMissingLayer placeholders have no attention module.
            attn = getattr(layer, "self_attn", None)
            if attn is None or not hasattr(attn, "q_size"):
                continue
            q_size, kv_size = attn.q_size, attn.kv_size
            base = f"{prefix}model.layers.{index}.self_attn"
            for suffix in ("weight", "bias"):
                packed_key = f"{base}.qkv_proj.{suffix}"
                packed = state_dict.get(packed_key)
                if packed is None:
                    continue
                if packed.shape[0] != q_size + 2 * kv_size:
                    # Geometry we don't model. Leaving it packed loses the
                    # tensor for adapter matching, which is recoverable;
                    # splitting it wrongly corrupts weights silently.
                    continue
                del state_dict[packed_key]
                state_dict[f"{base}.q_proj.{suffix}"] = packed[:q_size]
                state_dict[f"{base}.k_proj.{suffix}"] = packed[
                    q_size : q_size + kv_size
                ]
                state_dict[f"{base}.v_proj.{suffix}"] = packed[q_size + kv_size :]

        # qkv is handled above; the shared helper still unpacks
        # gate_up_proj and drops fused-expert / scale keys that have no HF
        # counterpart.
        return unpack_packed_modules_state_dict(
            state_dict,
            prefix=prefix,
            packed_modules_mapping={
                key: value
                for key, value in self.packed_modules_mapping.items()
                if key != "qkv_proj"
            },
            config=self.config,
        )

'''


LATEST_CHECKPOINT_SRC = '''

def _scalarlm_latest_checkpoint(files):
    """Pick the highest-step checkpoint, not the alphabetically first.

    Both call sites used `sorted(...)[0]`, which orders lexically:
    checkpoint_1000.pt sorts before checkpoint_1999.pt, and
    checkpoint_500.pt sorts after both. A job that saved more than one
    checkpoint therefore serves an arbitrary earlier one -- an
    undertrained model that loads cleanly, reports every tensor matched,
    and answers with garbage. Observed on a 2000-step run whose final
    checkpoint was correct under HuggingFace while vLLM served an
    earlier one.

    `max` over the already-sorted list keeps ties deterministic, so the
    loader and the adapter_format classifier still agree on the same
    file -- which the comments at both sites require.
    """

    def step_of(path):
        stem = path.name.rsplit(".", 1)[0]
        tail = stem.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    return max(sorted(files), key=step_of)
'''


def patch_diffusion_gemma_sc_embeds_dtype(vllm_root: Path) -> None:
    """Cast the self-conditioning soft embeds to the buffer's dtype.

    `sc_embeds` is float32; `soft_embeds` inherits embed_weight's dtype
    (bf16), and index_put_ refuses to cast, so DiffusionGemma dies during
    warmup on B200:

        RuntimeError: Index put requires the source and destination dtypes
        match, got Float for the destination and BFloat16 for the source.

    The consumer already does `soft.to(inputs_embeds.dtype)` on read, so
    casting on write matches the existing convention. bf16 -> fp32 is
    lossless.

    Belongs upstream in vllm-fork; carried here so DiffusionGemma can be
    exercised without waiting on that.
    """
    target = (
        vllm_root / "vllm" / "model_executor" / "models" / "diffusion_gemma.py"
    )
    if not target.exists():
        print(
            f"[vllm_patches] {target} not found; skipping diffusion_gemma "
            f"dtype patch"
        )
        return

    src = target.read_text()
    anchor = "    sc_embeds[decode_slots] = soft_embeds * sc_keep\n"

    if "(soft_embeds * sc_keep).to(sc_embeds.dtype)" in src:
        print("[vllm_patches] diffusion_gemma dtype cast already present; skipping")
        return

    assert src.count(anchor) == 1, (
        "diffusion_gemma.py: expected exactly one "
        "`sc_embeds[decode_slots] = soft_embeds * sc_keep`. Re-anchor this patch."
    )

    patched = src.replace(
        anchor,
        "    # ScalarLM patch: sc_embeds is fp32, soft_embeds is bf16, and\n"
        "    # index_put_ will not cast. The consumer casts on read too.\n"
        "    sc_embeds[decode_slots] = (soft_embeds * sc_keep).to(sc_embeds.dtype)\n",
        1,
    )

    assert patched != src, "patch produced identical output — something's wrong"
    compile(patched, str(target), "exec")

    target.write_text(patched)
    print(f"[vllm_patches] Applied diffusion_gemma sc_embeds dtype cast to {target}")


def patch_latest_checkpoint_selection(vllm_root: Path) -> None:
    """Select adapter checkpoints by step number rather than filename.

    Applies to both places that resolve a job directory to a single
    `.pt`: the Tokenformer loader and the adapter_format classifier.
    They are commented as needing to agree, so they are patched together
    or not at all.
    """
    manager = vllm_root / "vllm" / "tokenformer" / "tokenformer_model_manager.py"
    fmt = vllm_root / "vllm" / "tokenformer" / "adapter_format.py"

    for target in (manager, fmt):
        if not target.exists():
            print(
                f"[vllm_patches] {target} not found; skipping "
                f"latest-checkpoint patch"
            )
            return

    manager_src = manager.read_text()
    fmt_src = fmt.read_text()

    if "_scalarlm_latest_checkpoint" in manager_src:
        print("[vllm_patches] latest-checkpoint selection already present; skipping")
        return

    manager_anchor = (
        "        files = sorted(Path(model_dir).glob(\"*.pt\"))\n"
        "\n"
        "        if len(files) == 0:\n"
        "            raise FileNotFoundError(f\"No .pt file found in {model_dir}\")\n"
        "\n"
        "        checkpoint_file = files[0]\n"
    )
    fmt_anchor = (
        "    files = sorted(model_dir.glob(\"*.pt\"))\n"
        "    if not files:\n"
        "        raise FileNotFoundError(f\"No .pt file found in {model_dir}\")\n"
        "    checkpoint_file = files[0]\n"
    )
    manager_logger = "logger = init_logger(__name__)\n"
    fmt_imports = "from typing import Any, Literal, TypeAlias\n"

    assert manager_anchor in manager_src, (
        "tokenformer_model_manager.py: the checkpoint-selection block in "
        "from_local_checkpoint has drifted. Re-anchor this patch."
    )
    assert fmt_anchor in fmt_src, (
        "adapter_format.py: the checkpoint-selection block in "
        "_load_adapter_checkpoint has drifted. Re-anchor this patch."
    )
    assert manager_logger in manager_src and fmt_imports in fmt_src, (
        "tokenformer files: module-level insertion anchors have drifted. "
        "Re-anchor this patch."
    )

    manager_patched = manager_src.replace(
        manager_logger, manager_logger + LATEST_CHECKPOINT_SRC, 1
    ).replace(
        manager_anchor,
        manager_anchor.replace(
            "        checkpoint_file = files[0]\n",
            "        checkpoint_file = _scalarlm_latest_checkpoint(files)\n",
        ),
        1,
    )
    fmt_patched = fmt_src.replace(
        fmt_imports, fmt_imports + LATEST_CHECKPOINT_SRC, 1
    ).replace(
        fmt_anchor,
        fmt_anchor.replace(
            "    checkpoint_file = files[0]\n",
            "    checkpoint_file = _scalarlm_latest_checkpoint(files)\n",
        ),
        1,
    )

    for target, patched, original in (
        (manager, manager_patched, manager_src),
        (fmt, fmt_patched, fmt_src),
    ):
        assert patched != original, f"{target}: patch produced identical output"
        compile(patched, str(target), "exec")
        target.write_text(patched)

    print("[vllm_patches] Applied latest-checkpoint selection to both call sites")


def patch_llama_scalarlm_state_dict_export(vllm_root: Path) -> None:
    """Give LlamaForCausalLM the ScalarLM `state_dict` export override.

    Same defect as gemma4 had: llama fuses q/k/v into `qkv_proj` and never
    exposes the unpacked names, so every trained attention projection is
    dropped when a Tokenformer adapter is applied. Simpler fix than
    gemma4's because llama's head_dim does not vary per layer.
    """
    target = vllm_root / "vllm" / "model_executor" / "models" / "llama.py"
    if not target.exists():
        print(f"[vllm_patches] {target} not found; skipping llama state_dict patch")
        return

    src = target.read_text()

    if "scalarlm_state_dict_export_enabled" in src:
        print("[vllm_patches] llama state_dict export already present; skipping")
        return

    anchor_imports = (
        "from .utils import (\n"
        "    AutoWeightsLoader,\n"
        "    PPMissingLayer,\n"
        "    WeightsMapper,\n"
        "    extract_layer_index,\n"
        "    make_empty_intermediate_tensors_factory,\n"
        "    make_layers,\n"
        "    maybe_prefix,\n"
        ")\n"
    )
    anchor_class = (
        "class LlamaForCausalLM(\n"
        "    LocalArgmaxMixin,\n"
        "    nn.Module,\n"
        "    SupportsLoRA,\n"
        "    SupportsPP,\n"
        "    SupportsEagle,\n"
        "    SupportsEagle3,\n"
        "    SupportsQuant,\n"
        "    SupportsTokenformer,\n"
        "):\n"
    )

    assert anchor_imports in src, (
        "llama.py: the `from .utils import (...)` block has drifted; cannot "
        "add the state_dict export helpers. Re-anchor this patch."
    )
    assert anchor_class in src, (
        "llama.py: the LlamaForCausalLM class header has drifted; cannot "
        "place the state_dict override. Re-anchor this patch."
    )

    patched = src.replace(
        anchor_imports,
        anchor_imports.replace(
            "    maybe_prefix,\n",
            "    maybe_prefix,\n"
            "    scalarlm_state_dict_export_enabled,\n"
            "    unpack_packed_modules_state_dict,\n",
        ),
        1,
    ).replace(anchor_class, anchor_class + STATE_DICT_EXPORT_SRC, 1)

    assert patched != src, "patch produced identical output — something's wrong"
    compile(patched, str(target), "exec")

    target.write_text(patched)
    print(f"[vllm_patches] Applied llama state_dict export to {target}")


def patch_gemma4_scalarlm_state_dict_export(vllm_root: Path) -> None:
    """Give Gemma4ForCausalLM the ScalarLM `state_dict` export override.

    Five model files implement it (qwen2, qwen3, qwen3_moe, gemma3, plus
    the helper in utils.py); gemma4 does not. The consequence is specific
    and quiet: serving a ScalarLM Tokenformer checkpoint on Gemma4 drops
    every `q_proj`/`k_proj`/`v_proj` tensor the trainer produced, because
    vLLM only exposes them fused as `qkv_proj`. Training converges, the
    checkpoint is correct, and the served model still answers with
    garbage because its attention is at initialization.

    Verified with tiny-random/gemma-4-dense: 10 of 52 checkpoint tensors
    unmatched without this, all of them attention projections.
    """
    target = vllm_root / "vllm" / "model_executor" / "models" / "gemma4.py"
    if not target.exists():
        print(f"[vllm_patches] {target} not found; skipping gemma4 state_dict patch")
        return

    src = target.read_text()

    if "scalarlm_state_dict_export_enabled" in src:
        print("[vllm_patches] gemma4 state_dict export already present; skipping")
        return

    anchor_imports = (
        "from .utils import (\n"
        "    AutoWeightsLoader,\n"
        "    WeightsMapper,\n"
        "    extract_layer_index,\n"
        "    is_pp_missing_parameter,\n"
        "    make_layers,\n"
        "    maybe_prefix,\n"
        ")\n"
    )
    anchor_class = (
        "class Gemma4ForCausalLM(\n"
        "    nn.Module, SupportsLoRA, SupportsPP, MixtureOfExperts, SupportsEagle3\n"
        "):\n"
    )

    assert anchor_imports in src, (
        "gemma4.py: the `from .utils import (...)` block has drifted; cannot "
        "add the state_dict export helpers. Re-anchor this patch."
    )
    assert anchor_class in src, (
        "gemma4.py: the Gemma4ForCausalLM class header has drifted; cannot "
        "place the state_dict override. Re-anchor this patch."
    )

    patched = src.replace(
        anchor_imports,
        anchor_imports.replace(
            "    maybe_prefix,\n",
            "    maybe_prefix,\n"
            "    scalarlm_state_dict_export_enabled,\n"
            "    unpack_packed_modules_state_dict,\n",
        ),
        1,
    ).replace(
        anchor_class,
        anchor_class + STATE_DICT_EXPORT_SRC,
        1,
    )

    assert patched != src, "patch produced identical output — something's wrong"
    compile(patched, str(target), "exec")

    target.write_text(patched)
    print(f"[vllm_patches] Applied gemma4 state_dict export to {target}")


FUSED_DIFFUSION_SAMPLER_SRC = '''# SPDX-License-Identifier: Apache-2.0
"""Fused vocab-side sampling for DiffusionGemma (inference, no logprobs).

`_compiled_sample_step` in diffusion_gemma.py works on the full fp32 ``[num_decode * CL,
vocab]`` logits with several whole-tensor passes: temperature scaling, a Gumbel noise
tensor the size of the logits, two argmaxes, log_softmax, exp, the entropy product and a
bf16 cast of the probabilities for the self-conditioning matmul. At 32 decoding requests
that is ~8.6 GB per fp32 pass, and in production profiles those passes cost about as much
GPU time as the two vocab GEMMs.

Here the same quantities come from two streaming kernels:

  _row_stats   one read of each row: online max / sum-exp / sum(exp * s) (entropy),
               argmax(s) and argmax(s + gumbel) with the noise generated in-kernel
  _row_probs   one read of each row: p = exp(s - logZ), written in bf16 -- the dtype
               the original casts probs to before the self-conditioning matmul

then the matmul and the small per-canvas logic (entropy-bound mask, history,
convergence), which is copied unchanged from `_compiled_sample_step`.

Same math up to floating-point summation order; the Gumbel noise is a different random
stream from the same distribution as the original's `torch.rand_like` in fp32
(u = k / 2^24 clamped at 1e-20, g = -log(-log u); see `_uniform24`).
Requests that ask for logprobs keep using the original path (it needs the scaled logits).
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _uniform24(seed, offset):
    """u = k / 2^24 with k uniform in [0, 2^24): the distribution of torch.rand in fp32.

    `tl.rand` rounds to nearest instead, which gives the top level (u = 1 - 2^-24) half
    the probability and thins the far upper tail of the Gumbel noise, so low-probability
    tokens are sampled slightly less often than with the original formula."""
    r = tl.randint(seed, offset)
    return ((r >> 8) & 0xFFFFFF).to(tl.float32) * 5.9604644775390625e-08


@triton.jit
def _row_stats(
    x_ptr, inv_t_ptr, seed,
    m_ptr, l_ptr, a_ptr, am_ptr, gm_ptr,
    V, stride_x, CL, cap,
    BLOCK: tl.constexpr, SOFTCAP: tl.constexpr,
):
    row = tl.program_id(0)
    inv_t = tl.load(inv_t_ptr + row // CL)
    base = x_ptr + row.to(tl.int64) * stride_x
    offs = tl.arange(0, BLOCK)
    m = tl.full([BLOCK], float("-inf"), tl.float32)   # per-lane running max of s
    l = tl.zeros([BLOCK], tl.float32)                 # sum exp(s - m)
    a = tl.zeros([BLOCK], tl.float32)                 # sum exp(s - m) * (m - s) >= 0: entropy = log l + a / l
    best = tl.full([BLOCK], float("-inf"), tl.float32)
    best_i = tl.zeros([BLOCK], tl.int32)
    gbest = tl.full([BLOCK], float("-inf"), tl.float32)
    gbest_i = tl.zeros([BLOCK], tl.int32)
    for start in range(0, V, BLOCK):
        idx = start + offs
        mask = idx < V
        x = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
        if SOFTCAP:   # raw LM-head logits: fp32 tanh softcap as compute_logits does, -inf kept
            x = tl.where(x == float("-inf"), x, libdevice.tanh(x / cap) * cap)
        s = x * inv_t
        # argmax(s): strict > keeps the first index per lane; lanes are merged below
        upd = s > best
        best = tl.where(upd, s, best)
        best_i = tl.where(upd, idx, best_i)
        # Gumbel-max sample
        u = _uniform24(seed, row.to(tl.int64) * V + idx)
        u = tl.maximum(u, 1e-20)
        g = s - tl.log(-tl.log(u))
        gupd = g > gbest
        gbest = tl.where(gupd, g, gbest)
        gbest_i = tl.where(gupd, idx, gbest_i)
        # online softmax statistics, per lane
        # a is kept relative to the running max so every term is >= 0 (no cancellation
        # near zero entropy, where the 0.005 confidence threshold sits):
        #   sum_old e^(s-m')(m'-s) = alpha * (a + (m'-m) * l),  alpha = e^(m-m')
        m_new = tl.maximum(m, s)
        alpha = tl.where(m_new == float("-inf"), 0.0, tl.exp(m - m_new))
        dm = tl.where(m == float("-inf"), 0.0, m_new - m)
        e = tl.where(s == float("-inf"), 0.0, tl.exp(s - m_new))
        es = tl.where(s == float("-inf"), 0.0, e * (m_new - s))
        a = alpha * (a + dm * l) + es
        l = l * alpha + e
        m = m_new
    # merge lanes
    M = tl.max(m, 0)
    scale = tl.where(m == float("-inf"), 0.0, tl.exp(m - M))
    L = tl.sum(l * scale, 0)
    A = tl.sum(scale * (a + tl.where(m == float("-inf"), 0.0, M - m) * l), 0)
    bmax = tl.max(best, 0)
    am = tl.min(tl.where(best == bmax, best_i, 2147483647), 0)     # first index of the max
    gmax = tl.max(gbest, 0)
    gm = tl.min(tl.where(gbest == gmax, gbest_i, 2147483647), 0)
    tl.store(m_ptr + row, M)
    tl.store(l_ptr + row, L)
    tl.store(a_ptr + row, A)
    tl.store(am_ptr + row, am)
    tl.store(gm_ptr + row, gm)


@triton.jit
def _row_probs(
    x_ptr, inv_t_ptr, logz_ptr, p_ptr,
    V, stride_x, stride_p, CL, v_start, v_len, cap,
    BLOCK: tl.constexpr, SOFTCAP: tl.constexpr,
):
    row = tl.program_id(0)
    col0 = tl.program_id(1) * BLOCK
    inv_t = tl.load(inv_t_ptr + row // CL)
    logz = tl.load(logz_ptr + row)
    idx = col0 + tl.arange(0, BLOCK)
    mask = idx < v_len
    x = tl.load(x_ptr + row.to(tl.int64) * stride_x + v_start + idx, mask=mask, other=float("-inf")).to(tl.float32)
    if SOFTCAP:
        x = tl.where(x == float("-inf"), x, libdevice.tanh(x / cap) * cap)
    p = tl.exp(x * inv_t - logz)
    tl.store(p_ptr + row.to(tl.int64) * stride_p + idx, p.to(tl.bfloat16), mask=mask)


def vocab_stats_and_probs(logits, temp, CL, sc_vocab_start, sc_vocab_end, block=2048, softcap=0.0):
    """logits [n*CL, V] (fp32 softcapped, or raw with softcap=cap to apply it in-kernel), temp [n] -> argmax, gumbel-argmax, entropy [n*CL],
    probs bf16 [n*CL, sc_vocab_end - sc_vocab_start]."""
    N, V = logits.shape
    dev = logits.device
    inv_t = (1.0 / temp.float().clamp(min=1e-10)).contiguous()
    m = torch.empty(N, device=dev, dtype=torch.float32)
    l = torch.empty_like(m)
    a = torch.empty_like(m)
    am = torch.empty(N, device=dev, dtype=torch.int32)
    gm = torch.empty_like(am)
    seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    _row_stats[(N,)](logits, inv_t, seed, m, l, a, am, gm, V, logits.stride(0), CL, float(softcap or 1.0),
                     BLOCK=block, SOFTCAP=bool(softcap), num_warps=8)
    logz = m + torch.log(l)
    entropy = torch.log(l) + a / l
    v_len = sc_vocab_end - sc_vocab_start
    probs = torch.empty(N, v_len, device=dev, dtype=torch.bfloat16)
    _row_probs[(N, triton.cdiv(v_len, block))](logits, inv_t, logz, probs, V, logits.stride(0),
                                               probs.stride(0), CL, sc_vocab_start, v_len, float(softcap or 1.0),
                                               BLOCK=block, SOFTCAP=bool(softcap), num_warps=8)
    return am.long(), gm.long(), entropy, probs


@torch.compile(dynamic=True)
def _post_sample(
    new_tokens, argmax_tokens, token_entropy, soft_embeds,
    decode_slots, decode_idx, all_slots, valid_canvas_len,
    canvas, argmax_canvas, step_tensor, is_encoder_phase, confident_tensor, sc_embeds,
    history, history_len_tensor, sampled, num_sampled, draft_tokens,
    max_denoising_steps: float, confidence_threshold: float, vocab_size: int,
    CL: int, ST: int, entropy_bound: float,
):
    """Phases 3b-7 of `_compiled_sample_step`, unchanged, on the fused kernels' outputs."""
    num_decode = decode_slots.shape[0]
    device = decode_slots.device
    new_tokens = new_tokens.view(num_decode, CL)
    argmax_tokens = argmax_tokens.view(num_decode, CL)
    token_entropy = token_entropy.view(num_decode, CL)

    mean_entropy = token_entropy.mean(dim=-1)
    confident_tensor[decode_slots] = mean_entropy < confidence_threshold

    sorted_ent, sorted_idx = torch.sort(token_entropy, dim=-1)
    cumsum_ent = torch.cumsum(sorted_ent, dim=-1)
    cummax_ent = torch.cummax(sorted_ent, dim=-1).values
    sorted_mask = (cumsum_ent - cummax_ent) <= entropy_bound
    eb_mask = torch.zeros_like(sorted_mask)
    eb_mask.scatter_(1, sorted_idx, sorted_mask)

    is_commit = is_encoder_phase[decode_slots]
    is_denoise = ~is_commit
    cur_step = step_tensor[decode_slots].float()
    new_step_val = torch.where(
        is_denoise, (cur_step + 1).to(step_tensor.dtype), step_tensor.new_zeros(num_decode))
    step_tensor[decode_slots] = new_step_val

    random_tokens = torch.randint(0, vocab_size, (num_decode, CL), device=device, dtype=canvas.dtype)
    denoise_canvas = torch.where(eb_mask, new_tokens.to(canvas.dtype), random_tokens)
    canvas[decode_slots] = torch.where(is_commit.unsqueeze(1), random_tokens, denoise_canvas)

    hist_len = history_len_tensor[decode_slots]
    write_pos = hist_len % ST
    for i in range(ST):
        write_here = ((write_pos == i) & is_denoise).unsqueeze(1)
        history[decode_slots, i] = torch.where(
            write_here, argmax_tokens.to(history.dtype), history[decode_slots, i])

    argmax_canvas[decode_slots] = torch.where(
        is_denoise.unsqueeze(1), argmax_tokens.to(argmax_canvas.dtype), argmax_canvas[decode_slots])
    new_hist_len = torch.where(is_denoise, hist_len + 1, hist_len.new_zeros(num_decode))
    history_len_tensor[decode_slots] = new_hist_len

    sampled[decode_idx] = argmax_canvas[decode_slots].to(sampled.dtype) * is_commit.unsqueeze(1).to(sampled.dtype)
    num_sampled[decode_idx] = is_commit.to(num_sampled.dtype) * valid_canvas_len.to(num_sampled.dtype)

    ref = history[decode_slots, 0]
    mismatch = torch.zeros(num_decode, device=device, dtype=torch.int32)
    for h in range(1, ST):
        mismatch = mismatch + (ref != history[decode_slots, h]).sum(dim=-1).int()
    stable = mismatch == 0
    step_after = step_tensor[decode_slots]
    converged = (stable & confident_tensor[decode_slots] & (new_hist_len >= ST)) | (
        step_after >= max_denoising_steps)
    is_encoder_phase[decode_slots] = torch.where(is_commit, is_commit.new_zeros(num_decode), converged)

    sc_keep = (is_denoise & ~is_encoder_phase[decode_slots])[:, None, None]
    sc_embeds[decode_slots] = (soft_embeds.view(num_decode, CL, -1) * sc_keep).to(sc_embeds.dtype)

    newly_converged = (converged & is_denoise).unsqueeze(1)
    canvas[decode_slots] = torch.where(newly_converged, argmax_canvas[decode_slots], canvas[decode_slots])
    draft_tokens[all_slots, :CL] = canvas[all_slots]


def fused_sample_step(
    logits, decode_slots, decode_idx, all_slots, valid_canvas_len,
    canvas, argmax_canvas, step_tensor, is_encoder_phase, confident_tensor, sc_embeds,
    embed_weight, normalizer, history, history_len_tensor, sampled, num_sampled, draft_tokens,
    max_denoising_steps, t_min, t_max, confidence_threshold, vocab_size, CL, ST, entropy_bound,
    sc_vocab_start, sc_vocab_end, tp_size, tp_group_name, softcap=0.0,
):
    """Drop-in for `_compiled_sample_step` when no logprobs are needed (returns None).
    softcap > 0: `logits` are the raw LM-head output and the softcap is applied in-kernel."""
    steps_f = step_tensor[decode_slots].float()
    remaining = (max_denoising_steps - steps_f).clamp(min=1.0)
    temp = t_min + (t_max - t_min) * (remaining / max_denoising_steps)
    argmax_tokens, new_tokens, token_entropy, probs = vocab_stats_and_probs(
        logits, temp, CL, sc_vocab_start, sc_vocab_end, softcap=softcap)
    if not bool((temp > 0).all()):          # temp == 0 means greedy in the original
        greedy = (temp <= 0).repeat_interleave(CL)
        new_tokens = torch.where(greedy, argmax_tokens, new_tokens)
    soft_embeds = torch.matmul(probs, embed_weight[: sc_vocab_end - sc_vocab_start])
    if tp_size > 1:
        soft_embeds = torch.ops.vllm.all_reduce(soft_embeds, group_name=tp_group_name)
    soft_embeds = soft_embeds * normalizer
    _post_sample(
        new_tokens, argmax_tokens, token_entropy, soft_embeds,
        decode_slots, decode_idx, all_slots, valid_canvas_len,
        canvas, argmax_canvas, step_tensor, is_encoder_phase, confident_tensor, sc_embeds,
        history, history_len_tensor, sampled, num_sampled, draft_tokens,
        max_denoising_steps, confidence_threshold, vocab_size, CL, ST, entropy_bound)
    return None
'''


def patch_diffusion_gemma_fused_sampler(vllm_root: Path) -> None:
    """Fused vocab-side sampling for DiffusionGemma's denoising steps.

    `_compiled_sample_step` runs several whole-tensor passes over the fp32
    ``[num_decode * CL, vocab]`` logits: temperature scaling, a Gumbel noise tensor
    the size of the logits, two argmaxes, log_softmax, exp, the entropy product and a
    bf16 cast of the probabilities. At 32 decoding requests one fp32 copy is ~8.6 GB,
    and on RTX PRO 6000 those passes cost about as much GPU time as the two
    vocab-sized GEMMs.

    This patch adds `scalarlm_fused_sampler.py`, two streaming Triton kernels plus the
    unchanged per-canvas logic, and routes decode steps through it when no logprobs
    are requested. Requests that ask for logprobs keep the original path.

    The final-logit softcap moves into the same kernels: the sampling path gets the
    raw LM-head logits (tagged with the softcap value, see
    `patch_model_runner_fused_softcap`) instead of a separate fp32 softcapped copy --
    in eager mode that copy costs four more whole-tensor passes. Requests that ask for
    logprobs get the softcap applied exactly as before and use the original path.

    Results match the original: argmax and all per-request state are identical;
    entropy agrees to ~1e-6 near zero; the Gumbel noise is a different random stream
    from the same distribution. Sampler step 2.1-2.25x faster.

    Measured on one RTX PRO 6000 Max-Q, production requests, 20 in flight, eager, with
    cap 32 and the sm120 attention tiling: 510 -> 815 tok/s (production settings: 328).
    Quality: 1,000 paired nano-rl tasks, mean reward 0.496 and 0.473 in two runs vs
    0.499 for production settings; two identical runs differ by ~0.023, so no
    detectable quality loss (any real effect <= ~0.02-0.03).

    Opt out with SCALARLM_FUSED_DIFFUSION_SAMPLER=0 (disables both patches' effect).
    """
    models = vllm_root / "vllm" / "model_executor" / "models"
    target = models / "diffusion_gemma.py"
    if not target.exists():
        print(f"[vllm_patches] {target} not found; skipping fused diffusion sampler")
        return

    src = target.read_text()
    if "scalarlm_fused_sampler" in src:
        print("[vllm_patches] fused diffusion sampler already present; skipping")
        return

    anchor_budget = (
        "        group = max(num_decode, 1)\n"
        "        if num_decode > 0:\n"
        "            free, _ = current_platform.mem_get_info()\n"
        "            # ~10 transient fp32 copies of [group * CL, vocab] inside the step\n"
        "            # (eager peaks at ~8; pad for allocator overhead and small tensors).\n"
        "            bytes_per_req = CL * self.vocab_size * 4 * 10\n"
    )
    anchor_call = (
        "            scaled = _compiled_sample_step(\n"
        "                logits[start_req * CL : end_req * CL],\n"
    )
    anchor_logits = (
        "    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:\n"
    )
    anchor_prefill = (
        "        if input_batch.num_draft_tokens == 0:\n"
        "            return self._handle_prefill(input_batch, device)\n"
    )
    anchor_call_end = (
        "                tp_group_name=self.tp_group_name,\n"
        "            )\n"
    )
    for name, anchor in (("budget", anchor_budget), ("call", anchor_call), ("logits", anchor_logits),
                         ("prefill", anchor_prefill), ("call_end", anchor_call_end)):
        assert src.count(anchor) == 1, (
            f"diffusion_gemma.py: expected exactly one {name} anchor. Re-anchor this patch."
        )

    patched = src.replace(
        anchor_budget,
        "        # ScalarLM patch: fused vocab-side sampling (scalarlm_fused_sampler.py) when\n"
        "        # no logprobs are needed. It keeps one bf16 [group * CL, vocab] probs buffer\n"
        "        # instead of ~10 fp32 copies of the logits.\n"
        "        import os as _os\n"
        "        use_fused = (\n"
        "            _os.environ.get(\"SCALARLM_FUSED_DIFFUSION_SAMPLER\", \"1\") != \"0\"\n"
        "            and max_num_logprobs < 0\n"
        "        )\n"
        "        if use_fused:\n"
        "            from vllm.model_executor.models.scalarlm_fused_sampler import (\n"
        "                fused_sample_step,\n"
        "            )\n"
        "        elif raw_softcap:\n"
        "            # raw logits but the original path: softcap exactly as compute_logits does\n"
        "            logits = _softcap_logits(logits, raw_softcap)\n"
        "            raw_softcap = 0.0\n"
        "        step_kwargs = {\"softcap\": raw_softcap} if use_fused else {}\n"
        "        group = max(num_decode, 1)\n"
        "        if num_decode > 0:\n"
        "            free, _ = current_platform.mem_get_info()\n"
        "            # ~10 transient fp32 copies of [group * CL, vocab] inside the step\n"
        "            # (eager peaks at ~8; pad for allocator overhead and small tensors).\n"
        "            bytes_per_req = CL * self.vocab_size * (2 * 2 if use_fused else 4 * 10)\n",
        1,
    )
    patched = patched.replace(
        anchor_call,
        "            scaled = (fused_sample_step if use_fused else _compiled_sample_step)(\n"
        "                logits[start_req * CL : end_req * CL],\n",
        1,
    )
    patched = patched.replace(
        anchor_logits,
        "    def scalarlm_compute_sample_logits(\n"
        "        self, hidden_states: torch.Tensor\n"
        "    ) -> torch.Tensor | None:\n"
        "        # ScalarLM patch: the sampling path's logits WITHOUT the separate fp32\n"
        "        # softcap copy. The tensor is tagged so DiffusionSampler applies the\n"
        "        # softcap itself (in-kernel when fused, else exactly as before).\n"
        "        logits = self.logits_processor(self.lm_head, hidden_states)\n"
        "        if logits is not None and self.final_logit_softcapping is not None:\n"
        "            logits._scalarlm_softcap = float(self.final_logit_softcapping)\n"
        "        return logits\n"
        "\n" + anchor_logits,
        1,
    )
    patched = patched.replace(
        anchor_prefill,
        anchor_prefill
        + "        # ScalarLM patch: raw (not yet softcapped) logits from\n"
        "        # scalarlm_compute_sample_logits carry their softcap value.\n"
        "        raw_softcap = getattr(logits, \"_scalarlm_softcap\", 0.0)\n",
        1,
    )
    patched = patched.replace(
        anchor_call_end,
        "                tp_group_name=self.tp_group_name,\n"
        "                **step_kwargs,\n"
        "            )\n",
        1,
    )
    assert patched != src, "patch produced identical output — something's wrong"
    compile(patched, str(target), "exec")
    compile(FUSED_DIFFUSION_SAMPLER_SRC, "scalarlm_fused_sampler.py", "exec")

    (models / "scalarlm_fused_sampler.py").write_text(FUSED_DIFFUSION_SAMPLER_SRC)
    target.write_text(patched)
    print(f"[vllm_patches] Applied fused diffusion sampler to {target}")


def patch_model_runner_fused_softcap(vllm_root: Path) -> None:
    """Sampling path: let DiffusionGemma's fused sampler apply the final-logit softcap.

    `model_runner.sample()` calls `compute_logits`, which for DiffusionGemma writes an
    fp32 softcapped copy of the ``[num_decode * CL, vocab]`` logits. When the model
    provides `scalarlm_compute_sample_logits` (added by
    `patch_diffusion_gemma_fused_sampler`), use it instead: it returns the raw LM-head
    logits tagged with the softcap value, and the sampler applies the softcap inside
    its kernels. Other models, the warm-up run and prompt logprobs keep calling
    `compute_logits` unchanged. Grammar bitmasks still work: a masked -inf stays -inf
    through the in-kernel softcap.
    """
    target = vllm_root / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"
    if not target.exists():
        print(f"[vllm_patches] {target} not found; skipping fused softcap")
        return

    src = target.read_text()
    if "scalarlm_compute_sample_logits" in src:
        print("[vllm_patches] fused softcap already present; skipping")
        return

    anchor = "        logits = self.model.compute_logits(sample_hidden_states)\n"
    assert src.count(anchor) == 1, (
        "model_runner.py: expected exactly one sample() compute_logits call. Re-anchor this patch."
    )
    patched = src.replace(
        anchor,
        "        # ScalarLM patch: DiffusionGemma's fused sampler applies the softcap itself.\n"
        "        _sample_logits = getattr(self.model, \"scalarlm_compute_sample_logits\", None)\n"
        "        if _sample_logits is not None and __import__(\"os\").environ.get(\n"
        "            \"SCALARLM_FUSED_DIFFUSION_SAMPLER\", \"1\"\n"
        "        ) != \"0\":\n"
        "            logits = _sample_logits(sample_hidden_states)\n"
        "        else:\n"
        "            logits = self.model.compute_logits(sample_hidden_states)\n",
        1,
    )
    assert patched != src, "patch produced identical output — something's wrong"
    compile(patched, str(target), "exec")
    target.write_text(patched)
    print(f"[vllm_patches] Applied fused softcap to {target}")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm-root>", file=sys.stderr)
        return 2
    vllm_root = Path(sys.argv[1]).resolve()
    if not (vllm_root / "vllm" / "v1" / "engine" / "async_llm.py").exists():
        print(f"[vllm_patches] async_llm.py not found under {vllm_root}", file=sys.stderr)
        return 3

    patch_output_handler_metrics_offload(vllm_root)
    patch_tokenformer_adapter_key_resolution(vllm_root)
    patch_gemma4_scalarlm_state_dict_export(vllm_root)
    patch_llama_scalarlm_state_dict_export(vllm_root)
    patch_latest_checkpoint_selection(vllm_root)
    patch_diffusion_gemma_sc_embeds_dtype(vllm_root)
    patch_diffusion_gemma_fused_sampler(vllm_root)
    patch_model_runner_fused_softcap(vllm_root)
    print("[vllm_patches] All patches applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

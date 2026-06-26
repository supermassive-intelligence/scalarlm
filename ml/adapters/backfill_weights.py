"""
Helper to backfill missing weights into a merged model's safetensors.

This corrects for truncated-architecture bugs where the transformers version
used during merge materialised fewer layers than the canonical checkpoint
on HuggingFace (e.g. AutoModelForCausalLM loading an older Gemma4 variant with
15 instead of 35 language layers).
"""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import get_safetensors_metadata, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def backfill_missing_weights(output_dir: Path, base_model_name: str) -> None:
    """
    Compare the safetensors written to output_dir against the canonical weight
    set on HuggingFace and copy any missing tensors from the HF base model.

    The HF base weights are used verbatim for those layers — LoRA deltas were
    not applied to them — which is still far better than an incomplete file
    that fails to load at all.
    """
    # Read the canonical key list from HF (just the header, no weight download).
    try:
        hf_meta = get_safetensors_metadata(base_model_name)
        hf_weight_map: dict[str, str] = hf_meta.weight_map  # key -> shard filename
    except Exception as e:
        logger.warning(
            "Could not fetch safetensors metadata for %s: %s. "
            "Skipping missing-weight backfill — the merged model may be incomplete.",
            base_model_name,
            e,
        )
        return

    # Read the keys already present in output_dir.
    saved_files = sorted(output_dir.glob("*.safetensors"))
    if not saved_files:
        logger.warning("No safetensors files found in %s; skipping backfill.", output_dir)
        return

    saved_keys: set[str] = set()
    for sf in saved_files:
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            saved_keys.update(f.keys())

    missing_keys = sorted(set(hf_weight_map.keys()) - saved_keys)
    if not missing_keys:
        logger.info("Merged model has all %d canonical weights — no backfill needed.", len(saved_keys))
        return

    logger.warning(
        "Merged model is missing %d weight(s) that exist in the HF base "
        "(e.g. %s). This usually means the local transformers loaded a "
        "truncated architecture. Backfilling from HF base weights — LoRA "
        "deltas were NOT applied to these layers.",
        len(missing_keys),
        missing_keys[:3],
    )

    # Collect the missing tensors from the base model shards (cached by
    # from_pretrained, so typically no extra download is needed).
    needed_shards: dict[str, list[str]] = {}
    for key in missing_keys:
        shard = hf_weight_map[key]
        needed_shards.setdefault(shard, []).append(key)

    missing_tensors: dict[str, "torch.Tensor"] = {}
    for shard_filename, keys in needed_shards.items():
        shard_path = hf_hub_download(base_model_name, shard_filename)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                missing_tensors[key] = f.get_tensor(key)
        logger.info("Loaded %d missing tensor(s) from %s", len(keys), shard_filename)

    # Append the missing tensors to the (single) output safetensors.  For
    # sharded outputs we add them to the first shard; a proper index is not
    # rebuilt, but vLLM and transformers both tolerate shard imbalance.
    target_shard = saved_files[0]
    all_tensors: dict[str, "torch.Tensor"] = {}
    with safe_open(str(target_shard), framework="pt", device="cpu") as f:
        for key in f.keys():
            all_tensors[key] = f.get_tensor(key)
    all_tensors.update(missing_tensors)

    save_file(all_tensors, str(target_shard))
    logger.info(
        "Backfill complete: added %d tensor(s) to %s (%d total).",
        len(missing_tensors),
        target_shard.name,
        len(all_tensors),
    )

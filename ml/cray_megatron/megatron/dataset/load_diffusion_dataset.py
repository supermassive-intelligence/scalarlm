"""Dataset loader for DiffusionGemma (`DiffusionGemmaForBlockDiffusion`).

Unlike the causal-LM loader (``load_language_model_dataset.py``, which packs many
documents into one block and builds document-masked position_ids), DiffusionGemma
trains one example per fixed-size **canvas**: the ``input`` is encoded by the
prompt encoder, and the ``output`` is the clean canvas the bidirectional decoder
learns to denoise. There is no packing — a canvas is a fixed ``canvas_length``
block, padded or truncated per example.

This loader yields **clean** canvas tokens plus canvas labels. The live
uniform-vocabulary corruption (sample ``t ~ U(eps, 1)`` per example, replace
supervised canvas positions with a random vocab token w.p. ``t``) happens per
training step in ``training_loop.training_step_accumulate`` — NOT here — so the
corruption pattern is resampled fresh every step rather than frozen for a whole
epoch (see the design spec §3).

Fields produced per example:
- ``encoder_input_ids`` / ``encoder_attention_mask`` — the prompt for the encoder.
- ``canvas_input_ids`` — clean canvas tokens, length ``canvas_length``, padded
  with the pad token (a valid embedding index; the decoder processes every canvas
  slot).
- ``canvas_labels`` — the clean token at every supervised canvas position, ``-100``
  at padding (so padded slots are excluded from the cross-entropy loss).

Batch size 1 is assumed (the finetune sweep's default): encoder prompts are
variable length, which the default torch collate only stacks cleanly one row at a
time. The canvas is always ``canvas_length``, so it stacks for any batch size.
"""

from cray_infra.util.get_job_config import get_job_config

from cray_megatron.collectives.data_parallelism import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
)
from cray_megatron.megatron.dataset.diffusion_canvas import (
    anchor_token_id,
    tokenize_canvas_batch,
)

import datasets
import jsonlines

import logging

logger = logging.getLogger(__name__)

# Fallback canvas size if a job somehow omits the diffusion block. Matches the
# DiffusionConfig default and the model's own config.canvas_length default.
_DEFAULT_CANVAS_LENGTH = 256


def load_diffusion_dataset(model, tokenizer, epoch):
    canvas_length = _get_canvas_length()
    anchor_id = _resolve_anchor_id(tokenizer)

    hf_dataset = datasets.IterableDataset.from_generator(
        make_dataset_generator(),
        features=datasets.Features(
            {
                "input": datasets.Value(dtype="string"),
                "output": datasets.Value(dtype="string"),
            }
        ),
    )
    shuffled_dataset = hf_dataset.shuffle(seed=42 + epoch, buffer_size=256)
    split_dataset = split_dataset_by_node(shuffled_dataset)

    tokenized_dataset = split_dataset.map(
        get_canvas_tokenize_function(tokenizer, canvas_length, anchor_id),
        batched=True,
        remove_columns=["input", "output"],
    )

    torch_dataset = tokenized_dataset.with_format("torch")

    return torch_dataset


def make_dataset_generator():
    def read_dataset():
        dataset_path = get_dataset_path()
        with open(dataset_path) as dataset_file:
            reader = jsonlines.Reader(dataset_file)
            for obj in reader:
                yield obj

    return read_dataset


def get_dataset_path():
    job_config = get_job_config()
    return job_config["training_data_path"]


def split_dataset_by_node(dataset):
    data_parallel_rank = get_data_parallel_rank()
    data_parallel_world_size = get_data_parallel_world_size()

    filtered_dataset = dataset.filter(
        lambda example, idx: idx % data_parallel_world_size == data_parallel_rank,
        with_indices=True,
    )

    return filtered_dataset


def _get_canvas_length():
    job_config = get_job_config()
    diffusion = job_config.get("diffusion") or {}
    # job_config is a plain dict here; the nested block may be a dict or a
    # pydantic model depending on the caller, so tolerate both.
    if hasattr(diffusion, "canvas_length"):
        return diffusion.canvas_length
    return diffusion.get("canvas_length", _DEFAULT_CANVAS_LENGTH)


def _anchor_enabled():
    """Whether the Tier-2 anchor token is requested via the diffusion job config."""
    job_config = get_job_config()
    diffusion = job_config.get("diffusion") or {}
    if hasattr(diffusion, "anchor_token"):
        return bool(diffusion.anchor_token)
    return bool(diffusion.get("anchor_token", False))


def _resolve_anchor_id(tokenizer):
    """Resolve the anchor token id when enabled, else None. Warns and disables the
    anchor if the tokenizer has no BOS rather than inventing a spurious id."""
    if not _anchor_enabled():
        return None
    anchor_id = anchor_token_id(tokenizer)
    if anchor_id is None:
        logger.warning(
            "diffusion.anchor_token is set but the tokenizer has no bos_token_id; "
            "training without a canvas anchor."
        )
    return anchor_id


def get_canvas_tokenize_function(tokenizer, canvas_length, anchor_id=None):
    def tokenize(dataset):
        return tokenize_canvas_batch(
            tokenizer, canvas_length, dataset["input"], dataset["output"], anchor_id
        )

    return tokenize

from cray_infra.util.get_job_config import get_job_config

from cray_megatron.collectives.data_parallelism import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
)

import datasets
import jsonlines
import random

import logging

logger = logging.getLogger(__name__)


def load_language_model_dataset(model, tokenizer, epoch):
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
        get_tokenize_function(model, tokenizer),
        batched=True,
        remove_columns=[
            "input",
            "output",
        ],
    )

    packed_dataset = tokenized_dataset.map(
        get_pack_function(model),
        batched=True,
    )

    torch_dataset = packed_dataset.with_format("torch")

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

    num_shards = data_parallel_world_size
    index = data_parallel_rank

    filtered_dataset = dataset.filter(
        lambda example, idx: idx % data_parallel_world_size == data_parallel_rank,
        with_indices=True,
    )

    return filtered_dataset


def get_tokenize_function(model, tokenizer):

    def tokenize(dataset):
        text = [
            input_text + output_text
            for input_text, output_text in zip(dataset["input"], dataset["output"])
        ]

        tokens = tokenizer(text)

        tokens = add_eos_token(tokens, model, tokenizer)

        # Get the length of the input sequence in tokens
        input_text_lengths = [
            len(tokenizer(input_text)["input_ids"]) for input_text in dataset["input"]
        ]

        # labels are -100 for the input_text and input_ids for the output_text
        tokens["labels"] = [
            [-100] * input_text_length + input_ids[input_text_length:]
            for input_text_length, input_ids in zip(
                input_text_lengths, tokens["input_ids"]
            )
        ]

        return tokens

    return tokenize


def add_eos_token(tokens, model, tokenizer):
    # add stop token to the end of the sequence
    if model.generation_config is None:
        eos_token = tokenizer.eos_token_id
    elif model.generation_config.eos_token_id is None:
        eos_token = tokenizer.eos_token_id
    else:
        if isinstance(model.generation_config.eos_token_id, list):
            eos_token = model.generation_config.eos_token_id[-1]
        else:
            eos_token = model.generation_config.eos_token_id

    tokens["input_ids"] = [input_ids + [eos_token] for input_ids in tokens["input_ids"]]
    tokens["attention_mask"] = [
        attention_mask + [1] for attention_mask in tokens["attention_mask"]
    ]

    return tokens


def get_pack_function(model):
    job_config = get_job_config()

    # Cap block size at the model's max_position_embeddings so sampled position_ids
    # are always in-bounds for RoPE. This is also the effective upper bound on S,
    # which keeps the attention matrix memory at O(block_size^2).
    budget = min(
        get_max_position_embeddings(model.config), job_config["max_token_block_size"]
    )

    # One RNG per pack function instantiation, seeded per job. This gives
    # deterministic sampling within a job without tying the seed to document
    # position (which would repeat seeds across batches).
    sampling_seed = job_config.get("sampling_seed", 42)
    rng = random.Random(sampling_seed)

    def pack(dataset):
        # Per-doc lengths (taken from input_ids before concatenation) drive
        # two auxiliary streams that the trainer uses to break the default
        # "every position attends to every prior position in the block"
        # behavior of plain packed sequences:
        #
        #   - position_ids: counts up within each document and resets to 0
        #     at every boundary. Carries the right token-within-doc index
        #     into RoPE so a doc that lands mid-block doesn't pretend it
        #     started at position 6144 (or wherever).
        #   - document_ids: monotonically increasing per-doc tag. The
        #     trainer turns this into a block-diagonal additive attention
        #     mask so a query in doc N cannot attend to keys from docs
        #     0..N-1 in the same packed block. Without this, packed
        #     blocks make the model attend across unrelated documents,
        #     which corrupts the loss signal and (for Gemma's
        #     larger-head_dim layers) is a plausible NaN source.

        compressed_input_ids = []
        compressed_labels = []
        compressed_position_ids = []
        compressed_document_ids = []

        doc_idx = 0
        for i in range(len(dataset["input_ids"])):
            ids = dataset["input_ids"][i]
            labels = dataset["labels"][i]
            doc_len = len(ids)

            if doc_len > budget:
                # Sample budget tokens from the document, preserving original order.
                # Original indices are kept as position_ids so RoPE sees the correct
                # global distance between tokens even though the sequence is sparse.
                indices = sorted(rng.sample(range(doc_len), budget))

                compressed_input_ids.extend([ids[idx] for idx in indices])
                compressed_labels.extend([labels[idx] for idx in indices])
                compressed_position_ids.extend(indices)
                current_doc_len = budget
            else:
                compressed_input_ids.extend(ids)
                compressed_labels.extend(labels)
                compressed_position_ids.extend(range(doc_len))
                current_doc_len = doc_len

            compressed_document_ids.extend([doc_idx] * current_doc_len)
            doc_idx += 1

        # Handle any other keys (e.g. attention_mask). We re-derive the indices
        # from the same rng state, which is why we compute them in a second pass
        # using a fresh rng seeded identically.
        concatenated_dataset = {
            "input_ids": compressed_input_ids,
            "labels": compressed_labels,
            "position_ids": compressed_position_ids,
            "document_ids": compressed_document_ids,
        }
        extra_keys = [k for k in dataset.keys() if k not in concatenated_dataset]
        if extra_keys:
            replay_rng = random.Random(sampling_seed)
            for i in range(len(dataset["input_ids"])):
                doc_len = len(dataset["input_ids"][i])
                if doc_len > budget:
                    indices = sorted(replay_rng.sample(range(doc_len), budget))
                    for k in extra_keys:
                        val = dataset[k][i]
                        if k not in concatenated_dataset:
                            concatenated_dataset[k] = []
                        concatenated_dataset[k].extend([val[idx] for idx in indices])
                else:
                    for k in extra_keys:
                        val = dataset[k][i]
                        if k not in concatenated_dataset:
                            concatenated_dataset[k] = []
                        concatenated_dataset[k].extend(val)

        total_length = len(concatenated_dataset["input_ids"])
        if total_length >= budget:
            total_length = (total_length // budget) * budget

        result = {
            k: [t[i : i + budget] for i in range(0, total_length, budget)]
            for k, t in concatenated_dataset.items()
        }

        return result


    return pack

def get_max_position_embeddings(config):
    """Get max_position_embeddings from config, handling nested configs like Gemma3."""
    if hasattr(config, 'max_position_embeddings'):
        return config.max_position_embeddings
    if hasattr(config, 'text_config') and hasattr(config.text_config, 'max_position_embeddings'):
        return config.text_config.max_position_embeddings
    if hasattr(config, 'n_positions'):
        return config.n_positions
    raise AttributeError(f"Cannot find max_position_embeddings in {type(config)}")


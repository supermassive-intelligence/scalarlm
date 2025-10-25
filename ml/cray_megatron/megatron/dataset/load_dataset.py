from cray_infra.util.get_job_config import get_job_config

from cray_megatron.collectives.data_parallelism import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
)

import datasets
import jsonlines

import logging

logger = logging.getLogger(__name__)


def load_dataset(model, tokenizer, epoch):
    hf_dataset = datasets.IterableDataset.from_generator(
        make_dataset_generator(),
        features=datasets.Features(
            {
                "sentence1": datasets.Value(dtype="string"),
                "sentence2": datasets.Value(dtype="string"),
                "score": datasets.Value(dtype="float"),
            }
        ),
    )
    shuffled_dataset = hf_dataset.shuffle(seed=42 + epoch, buffer_size=256)
    split_dataset = split_dataset_by_node(shuffled_dataset)

    tokenized_dataset = split_dataset.map(
        get_tokenize_function(model, tokenizer),
        batched=True,
        remove_columns=[
            "sentence1",
            "sentence2",
            "score",
        ],
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

    num_shards = data_parallel_world_size
    index = data_parallel_rank

    filtered_dataset = dataset.filter(
        lambda example, idx: idx % data_parallel_world_size == data_parallel_rank,
        with_indices=True,
    )

    return filtered_dataset

def get_tokenize_function(model, tokenizer):

    def tokenize(dataset):

        sentence1_tokens = tokenizer(dataset["sentence1"])
        sentence2_tokens = tokenizer(dataset["sentence2"])

        job_config = get_job_config()

        max_length = job_config["max_token_block_size"]

        tokens = {"sentence1_input_ids": [], "sentence2_input_ids": []}

        for s1_ids, s2_ids in zip(
            sentence1_tokens["input_ids"], sentence2_tokens["input_ids"]
        ):
            padding_length_s1 = max_length - len(s1_ids)
            padding_length_s2 = max_length - len(s2_ids)

            s1_ids = s1_ids + ([tokenizer.pad_token_id] * padding_length_s1)
            s2_ids = s2_ids + ([tokenizer.pad_token_id] * padding_length_s2)

            tokens["sentence1_input_ids"].append(s1_ids)
            tokens["sentence2_input_ids"].append(s2_ids)

        tokens["labels"] = dataset["score"]

        return tokens

    return tokenize



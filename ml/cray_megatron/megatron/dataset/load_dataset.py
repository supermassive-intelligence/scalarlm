from cray_infra.util.get_job_config import get_job_config

from cray_megatron.megatron.dataset.load_embedding_dataset import load_embedding_dataset
from cray_megatron.megatron.dataset.load_language_model_dataset import load_language_model_dataset

def load_dataset(model, tokenizer, epoch):
    """Load dataset for either language model or embedding model training."""
    job_config = get_job_config()
    training_mode = job_config["training_mode"]

    if training_mode == "embedding":
        return load_embedding_dataset(model, tokenizer, epoch)
    else:
        return load_language_model_dataset(model, tokenizer, epoch)

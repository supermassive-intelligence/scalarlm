from cray_infra.util.get_job_config import get_job_config

from cray_megatron.megatron.doc_mask import is_diffusion
from cray_megatron.megatron.dataset.load_embedding_dataset import load_embedding_dataset
from cray_megatron.megatron.dataset.load_diffusion_dataset import load_diffusion_dataset
from cray_megatron.megatron.dataset.load_language_model_dataset import load_language_model_dataset

def load_dataset(model, tokenizer, epoch):
    """Load dataset for language-model, diffusion, or embedding training.

    DiffusionGemma is auto-detected from the model's HF config (is_diffusion),
    not the training_mode string — the model class is the source of truth, and a
    diffusion job needs the canvas loader regardless of what training_mode says.
    The existing `model.config` access in the LM loader confirms the (wrapped)
    model exposes `.config` here."""
    job_config = get_job_config()
    training_mode = job_config["training_mode"]

    if is_diffusion(getattr(model, "config", None)):
        return load_diffusion_dataset(model, tokenizer, epoch)
    elif training_mode == "embedding":
        return load_embedding_dataset(model, tokenizer, epoch)
    else:
        return load_language_model_dataset(model, tokenizer, epoch)

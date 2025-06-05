import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('tokenformer').setLevel(logging.ERROR)
logging.getLogger('tokenformer.llama_tokenformer_model').setLevel(logging.ERROR)
logging.getLogger('tokenformer.tokenformer_surgeon').setLevel(logging.ERROR)

from cray_megatron.huggingface.download_model import download_model
from cray_megatron.megatron.distribution.apply_distribution_strategy import (
    apply_distribution_strategy,
)

from tokenformer.llama_tokenformer_model import create_llama_tokenformer_model

from cray_infra.util.get_job_config import get_job_config
from cray_infra.util.get_config import get_config

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import torch

logger = logging.getLogger(__name__)


def load_tokenformer_model():
    import time
    total_start = time.time()
    
    model_info = load_model_config()
    model_info = apply_tokenformer_adapter(model_info)
    model_info = apply_distribution_strategy(model_info)
    model_info = materialize_model(model_info)
    model_info = load_checkpoint_weights_if_exist(model_info)
    
    total_time = time.time() - total_start
    logger.info(f"🚀 Total model loading time: {total_time:.2f}s ({total_time/60:.1f} minutes)")  # Add this
    
    return model_info

def load_model_config():
    job_config = get_job_config()

    model_name = job_config["llm_name"]

    model_config = AutoConfig.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_info = {
        "model_name": model_name,
        "model_config": model_config,
        "tokenizer": tokenizer,
    }

    return model_info


def apply_tokenformer_adapter(model_info):
    return model_info


def materialize_model(model_info):
    download_model(model_info["model_name"])
    model_info["model"] = AutoModelForCausalLM.from_pretrained(model_info["model_name"])
    model_info["model"] = create_llama_tokenformer_model(
        model_info["model"], model_info["distribution_strategy"]["device"]
    )

    config = get_config()
    config_dtype = config["dtype"]
    dtype = (
        torch.float16
        if config_dtype == "float16"
        else torch.float32 if config_dtype == "float32" else torch.bfloat16
    )
    logger.info(f"Converting model to {dtype}...")
    model_info["model"] = model_info["model"].to(dtype=dtype)

    if (
        "distribution_strategy" in model_info
        and "strategy" in model_info["distribution_strategy"]
    ):
        model_info["model"] = model_info["distribution_strategy"]["strategy"](
            model_info["model"]
        )
    logger.info(f"Model: {model_info['model']}")
    model_info["model"].to(model_info["distribution_strategy"]["device"])
    return model_info


def load_checkpoint_weights_if_exist(model_info):
    return model_info

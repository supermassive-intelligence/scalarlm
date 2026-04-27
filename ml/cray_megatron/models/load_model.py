from cray_megatron.huggingface.download_model import download_model
from cray_megatron.megatron.distribution.apply_distribution_strategy import (
    apply_distribution_strategy,
)
from cray_megatron.collectives.main_rank_only import is_main_rank

from gpu_aware_mpi import get_size, get_rank, allgather

from adapters.add_adapters_to_model import add_adapters_to_model

from cray_infra.util.get_job_config import get_job_config
from cray_infra.util.get_config import get_config

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import torch

import logging
import time

logger = logging.getLogger(__name__)


def pick_attention_backend() -> tuple[str, str | None]:
    """
    Choose the best attention backend HF transformers can serve given
    what's installed in this process. Returns `(impl, warning)` where
    `impl` is the string passed to `from_pretrained(attn_implementation=)`
    and `warning` is an operator-facing message about a missed
    speed-up — None when the chosen backend is already fast.

    Preference: flash_attention_3 > flash_attention_2 > sdpa > eager.

      - flash_attention_3 is Blackwell-only at the time of writing
        and lives behind a recent transformers + a `flash-attn 3.x`
        wheel.
      - flash_attention_2 works on Ampere / Hopper / Blackwell and
        ships in the upstream `flash-attn` package.
      - sdpa is PyTorch's built-in fused attention; no extra
        dependency, slower than flash on long sequences but well
        ahead of the eager Python reference path.

    The `is_flash_attn_*_available` helpers from transformers do the
    runtime probing for us — they return False when the wheel isn't
    present or when CUDA isn't usable, so we don't have to gate on
    `torch.cuda.is_available()` ourselves for the boolean result.
    The warning is suppressed on CPU-only hosts (sdpa is the right
    answer there too, and there's no flash-attn build for CPU).
    """
    # The probes are gated through importerror because old
    # transformers versions don't ship is_flash_attn_3_available().
    try:
        from transformers.utils import is_flash_attn_3_available

        if is_flash_attn_3_available():
            return "flash_attention_3", None
    except ImportError:
        pass

    try:
        from transformers.utils import is_flash_attn_2_available

        if is_flash_attn_2_available():
            return "flash_attention_2", None
    except ImportError:
        pass

    cuda_available = False
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    if cuda_available:
        warning = (
            "Flash attention is not available; falling back to PyTorch "
            "SDPA. Install flash-attn (>=2.7.0) in the training image "
            "for a ~2-4x speedup on Ampere/Hopper/Blackwell — typical "
            "command: `pip install flash-attn --no-build-isolation`. "
            "On Blackwell, flash-attn 3.x via `pip install flash-attn==3.*` "
            "is faster still."
        )
        return "sdpa", warning

    # CPU-only host: sdpa is fine and there's no flash-attn build to
    # install, so don't nag the operator.
    return "sdpa", None


def load_model():
    start_time = time.time()
    model_info = load_model_config()

    model_info = apply_distribution_strategy(model_info)

    model_info = materialize_model(model_info)

    total_time = time.time() - start_time
    logger.info(
        f"Total model loading time: {total_time:.2f}s ({total_time/60:.1f} minutes)"
    )
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


def materialize_model(model_info):
    download_model(model_info["model_name"])

    attn_impl, warning = pick_attention_backend()
    if warning:
        logger.warning(warning)
    logger.info("Loading model with attn_implementation=%s", attn_impl)

    start_time = time.time()
    try:
        model_info["model"] = AutoModelForCausalLM.from_pretrained(
            model_info["model_name"],
            torch_dtype="auto",  # Use model's native dtype
            attn_implementation=attn_impl,
            # device_map="auto",            # Enable Big Model Inference
            # low_cpu_mem_usage=True,       # Reduce CPU memory usage
            # _fast_init=True               # Skip weight initialization (default True)
        )
    except (ValueError, ImportError, RuntimeError) as e:
        # Some model configs (older architectures, multimodal heads with
        # custom attention) reject the chosen backend at load time.
        # Drop to sdpa — PyTorch's built-in fused attention — which
        # every modern transformers config supports. Eager is the
        # last resort; we don't try it explicitly because if sdpa
        # also fails the original exception is more informative.
        if attn_impl != "sdpa":
            logger.warning(
                "from_pretrained refused attn_implementation=%s (%s); "
                "falling back to sdpa.",
                attn_impl,
                e,
            )
            model_info["model"] = AutoModelForCausalLM.from_pretrained(
                model_info["model_name"],
                torch_dtype="auto",
                attn_implementation="sdpa",
            )
        else:
            raise

    total_time = time.time() - start_time
    logger.info(
        f"from_pretrained latency: {total_time:.2f}s ({total_time/60:.1f} minutes)"
    )

    start_time = time.time()
    model_info["model"] = add_adapters_to_model(
        model=model_info["model"], device=model_info["distribution_strategy"]["device"]
    )
    total_time = time.time() - start_time

    logger.info(
        f"create_tokenformer_model latency: {total_time:.2f}s ({total_time/60:.1f} minutes)"
    )
    start_time = time.time()
    config = get_config()
    config_dtype = config["dtype"]

    if config_dtype != "auto":
        dtype = (
            torch.float16
            if config_dtype == "float16"
            else torch.float32 if config_dtype == "float32" else torch.bfloat16
        )
        logger.info(f"Converting model to {dtype}...")

        model_info["model"] = model_info["model"].to(dtype=dtype)
    else:
        logger.info("Using model's native dtype, no conversion needed.")

    total_time = time.time() - start_time
    logger.info(
        f"model dtype conversion latency: {total_time:.2f}s ({total_time/60:.1f} minutes)"
    )

    model_info["model"] = model_info["distribution_strategy"]["strategy"](
        model_info["model"]
    )

    if is_main_rank():
        logger.info(f"Model: {model_info['model']}")

    logger.info(
        f"Moving model to device: {model_info['distribution_strategy']['device']}..."
    )

    model_info["model"].to(model_info["distribution_strategy"]["device"])

    return model_info

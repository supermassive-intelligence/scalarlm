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
    job_config = get_job_config()
    download_model(model_info["model_name"])

    attn_impl, warning = pick_attention_backend(model_info["model_config"])
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

    if job_config.get("gradient_checkpointing", False):
        # Order matters: enable on the bare HF model before PEFT
        # wraps it. `enable_input_require_grads` is the PEFT-specific
        # incantation that lets gradients flow back through a frozen
        # base into LoRA adapters when the base is checkpointed —
        # without it the adapter params get zero grad. `use_cache`
        # has to be off because checkpointing recomputes activations
        # in backward and the kv cache assumes they were retained.
        logger.info("Enabling gradient checkpointing")
        model_info["model"].config.use_cache = False
        model_info["model"].gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model_info["model"], "enable_input_require_grads"):
            model_info["model"].enable_input_require_grads()

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


# flash-attn's CUDA kernels hard-cap head_dim at 256. Models like
# Gemma-4 mix layers with head_dim > 256, which makes flash unusable
# even though the wheel is installed and the GPU is fine. The picker
# probes the model config and rolls back to sdpa when any layer
# exceeds this bound. fa3 inherits the same cap for our purposes —
# we'd rather take the safe sdpa path than crash mid-step.
_FLASH_MAX_HEAD_DIM = 256


def _max_head_dim(config: object) -> int | None:
    """
    Largest head_dim across `config` and its nested HF sub-configs
    (`text_config`, `vision_config`, `audio_config`, …). Returns None
    when nothing useful is discoverable — caller should treat that as
    "no information, don't gate".

    Each visited config contributes either its explicit `head_dim`
    attribute, or `hidden_size // num_attention_heads` when that pair
    is present. Multimodal HF configs (Gemma4, Llama4) put the real
    transformer params inside `text_config` rather than at the top
    level, so we have to walk one level down.
    """
    if config is None:
        return None

    seen: set[int] = set()
    best: int | None = None

    def visit(cfg: object) -> None:
        nonlocal best
        if cfg is None or id(cfg) in seen:
            return
        seen.add(id(cfg))

        head_dim = getattr(cfg, "head_dim", None)
        if isinstance(head_dim, int) and head_dim > 0:
            best = head_dim if best is None else max(best, head_dim)
        else:
            hidden = getattr(cfg, "hidden_size", None)
            heads = getattr(cfg, "num_attention_heads", None)
            if isinstance(hidden, int) and isinstance(heads, int) and heads > 0:
                derived = hidden // heads
                if derived > 0:
                    best = derived if best is None else max(best, derived)

        for sub in ("text_config", "vision_config", "audio_config"):
            visit(getattr(cfg, sub, None))

    visit(config)
    return best


def pick_attention_backend(
    model_config: object | None = None,
) -> tuple[str, str | None]:
    """
    Choose the best attention backend HF transformers can serve given
    what's installed in this process AND what the model needs. Returns
    `(impl, warning)` where `impl` is the string passed to
    `from_pretrained(attn_implementation=)` and `warning` is an
    operator-facing message about a missed speed-up or a forced
    fallback — None when the chosen backend is already fast.

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

    `model_config`, when supplied, is the HF AutoConfig the caller is
    about to pass to `from_pretrained`. We inspect it (and any nested
    sub-configs) for `head_dim > 256` and disqualify flash variants in
    that case — flash-attn's kernels hard-cap at 256 and would crash
    mid-step otherwise.
    """
    head_dim = _max_head_dim(model_config)
    flash_blocked_by_head_dim = (
        head_dim is not None and head_dim > _FLASH_MAX_HEAD_DIM
    )

    # The probes are gated through importerror because old
    # transformers versions don't ship is_flash_attn_3_available().
    if not flash_blocked_by_head_dim:
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

    if flash_blocked_by_head_dim:
        warning = (
            f"Model has head_dim={head_dim} which exceeds flash-attn's "
            f"hard cap of {_FLASH_MAX_HEAD_DIM}; falling back to PyTorch "
            "SDPA. This is expected for Gemma-4 and other architectures "
            "with oversize attention heads — flash-attn would crash "
            "mid-step on these layers."
        )
        return "sdpa", warning

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

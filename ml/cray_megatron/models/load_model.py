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
from transformers import AutoModel
from transformers import AutoModelForCausalLM
from transformers import AutoModelForImageTextToText
from transformers import PreTrainedModel

from cray_megatron.megatron.doc_mask import is_multimodal

import torch

import logging
import time

logger = logging.getLogger(__name__)

# transformers 5.x sets `self.all_tied_weights_keys` (a {target: source} dict)
# inside PreTrainedModel.tie_weights(), and several from_pretrained paths then
# read it unguarded: _move_missing_keys_from_meta_to_device (the low_cpu_mem_usage
# meta-load finalizer) and caching_allocator_warmup (the device_map path). Hub
# custom-code models built for older transformers (ChatGLM, Molmo, ...) override
# tie_weights and never set it, so those loaders crash with
# "'XForCausalLM' object has no attribute 'all_tied_weights_keys'". Provide a
# class-level empty default: native models shadow it with their real per-instance
# dict (a plain class attr is not a data descriptor, so the instance attr wins);
# the vendor models that skip tie_weights fall back to {} on the read-only paths.
if "all_tied_weights_keys" not in PreTrainedModel.__dict__:
    PreTrainedModel.all_tied_weights_keys = {}


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

    # Opt-in per job (train_args: {trust_remote_code: true}). Some HF repos
    # (InternVL3, Molmo, ...) ship custom modeling/config code that AutoConfig
    # refuses to execute unless explicitly trusted — without this they raise
    # "contains custom code which must be executed ... pass trust_remote_code=
    # True" right here at load and TRAIN_FAILED. Defaults False so arbitrary
    # remote code only runs for models the yaml explicitly marks. (vLLM serve
    # already passes it, which is why these models serve but wouldn't train.)
    trust_remote_code = job_config.get("trust_remote_code", False)

    model_config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    # Compatibility shim for hub custom modeling code written against older
    # transformers. ChatGLM's modeling_chatglm.py __init__ reads
    # `config.max_length`, an attribute transformers 5.x no longer sets on the
    # config (ChatGLMConfig exposes `seq_length` instead) -> AttributeError at
    # model init and TRAIN_FAILED. Backfill it from seq_length when the repo
    # ships custom code and the attribute is absent. Harmless for models that
    # already define max_length. materialize_model passes this same config
    # object to from_pretrained (config=...) so the backfill actually reaches
    # the model constructor.
    if (
        trust_remote_code
        and not hasattr(model_config, "max_length")
        and hasattr(model_config, "seq_length")
    ):
        model_config.max_length = model_config.seq_length

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    model_info = {
        "model_name": model_name,
        "model_config": model_config,
        "tokenizer": tokenizer,
    }

    return model_info


def materialize_model(model_info):
    job_config = get_job_config()
    download_model(model_info["model_name"])

    # SDPA (PyTorch's built-in fused attention) is the default and only
    # auto-selected backend. Flash-attention support was removed: the
    # head_dim/per-layer gating and the Qwen3 family bugs made it more
    # trouble than it was worth for training, and it bloated the image
    # build. A job can still force another HF backend (e.g. "eager") via
    # attn_implementation; "auto" (the default) means SDPA.
    override = job_config.get("attn_implementation", "auto")
    attn_impl = override if (override and override != "auto") else "sdpa"

    # Multimodal wrappers (vision_config present) need the image-text-to-text
    # AutoModel: AutoModelForCausalLM rejects e.g. Qwen2VLConfig at load
    # ("Unrecognized configuration class ... for AutoModelForCausalLM"), which
    # is the Qwen2-VL TRAIN_FAILED in the cuda-spark sweep. AutoModelForImage-
    # TextToText loads both Qwen2-VL and Gemma3 conditional-generation models;
    # the text-only training forward works on them (LoRA is confined to the
    # language tower by resolve_target_modules).
    #
    # But custom-code VLMs may not register AutoModelForImageTextToText in their
    # auto_map: InternVL exposes only AutoModel/AutoModelForCausalLM -> forcing
    # AutoModelForImageTextToText raises "Unrecognized configuration class ...
    # InternVLChatConfig for AutoModelForImageTextToText". Honor the auto_map:
    # for such a model use the (causal) class the repo actually declares.
    config = model_info["model_config"]
    auto_map = getattr(config, "auto_map", None) or {}
    if is_multimodal(config):
        if "AutoModelForImageTextToText" not in auto_map and "AutoModelForCausalLM" in auto_map:
            model_cls = AutoModelForCausalLM
        elif "AutoModelForImageTextToText" not in auto_map and "AutoModel" in auto_map:
            model_cls = AutoModel
        else:
            model_cls = AutoModelForImageTextToText
    else:
        model_cls = AutoModelForCausalLM
    logger.info(
        "Loading model with %s, attn_implementation=%s",
        model_cls.__name__,
        attn_impl,
    )

    # Load weights straight onto the target GPU (Big Model Inference) instead of
    # onto CPU and then .to(device). On a unified-memory pool (the GB10/DGX Spark,
    # 128GiB shared by CPU+GPU) the CPU-resident copy and the .to(device) GPU copy
    # coexist transiently at ~2x the weights (~120GiB for a 30B bf16 MoE) and OOM
    # before step 0 — even though the SAME model serves fine (vLLM loads it once).
    # device_map={"": device} materializes shards directly on-device (~1x peak).
    # Only for a real GPU device: the cpu target keeps the plain CPU load.
    device = model_info["distribution_strategy"]["device"]
    on_gpu = isinstance(device, int) or (
        isinstance(device, torch.device) and device.type == "cuda"
    )
    trust_remote_code = job_config.get("trust_remote_code", False)
    # device_map={"": device} does the on-GPU "Big Model Inference" load. But
    # transformers 5.x's device_map warmup (caching_allocator_warmup ->
    # get_total_byte_count -> model.all_tied_weights_keys) assumes an API that
    # hub custom-code models built for older transformers don't expose: Molmo's
    # MolmoForCausalLM raises "no attribute 'all_tied_weights_keys'", and ChatGLM
    # would hit the same path. Custom-code models here are <=9B and load fine
    # plain, so skip device_map for them and move to device explicitly below.
    # (Native large archs like the 30B MoE — not trust_remote_code — keep
    # device_map, which is what the OOM fix was for.)
    use_device_map = on_gpu and not trust_remote_code
    load_kwargs = {"torch_dtype": "auto", "low_cpu_mem_usage": True}
    if use_device_map:
        load_kwargs["device_map"] = {"": device}
    # Same opt-in as load_model_config: custom-code repos also need it on the
    # weight load, not just AutoConfig/AutoTokenizer. Reuse the config we already
    # loaded (config=...) so load_model_config's compatibility shims (e.g. the
    # ChatGLM max_length backfill) reach the model constructor. Scoped to
    # trust_remote_code so the load path for the many native, already-validated
    # models is byte-for-byte unchanged.
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True
        load_kwargs["config"] = model_info["model_config"]

    start_time = time.time()
    try:
        model_info["model"] = model_cls.from_pretrained(
            model_info["model_name"],
            attn_implementation=attn_impl,
            **load_kwargs,  # torch_dtype="auto" + on-device load (see above)
        )
    except (ValueError, ImportError, RuntimeError) as e:
        # Some model configs (older architectures, custom attention heads)
        # reject SDPA at load time. eager is the universal pure-Python
        # reference path that every transformers config supports — the
        # last resort.
        if attn_impl != "eager":
            logger.warning(
                "from_pretrained refused attn_implementation=%s (%s); "
                "falling back to eager.",
                attn_impl,
                e,
            )
            attn_impl = "eager"
            model_info["model"] = model_cls.from_pretrained(
                model_info["model_name"],
                attn_implementation="eager",
                **load_kwargs,
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
    # Per-job dtype (train_args["dtype"]) wins over global cray-config.yaml.
    # Both default to "auto", which means "use the model's native dtype".
    # The job-level knob is what the FAQ documents and is the right place
    # to put a per-run dtype override (e.g. fp32 on Apple Silicon CPU where
    # bf16 matmuls SIGILL).
    job_dtype = job_config.get("dtype", "auto")
    config_dtype = job_dtype if job_dtype != "auto" else get_config()["dtype"]

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

    if use_device_map:
        # Already materialized on-device by device_map at load; a subsequent
        # .to(device) on a dispatched model is redundant (and reintroduces the
        # CPU->GPU peak this fix avoids). Custom-code models that skipped
        # device_map fall through to the explicit .to(device) below.
        logger.info(
            f"Model already on device via device_map: "
            f"{model_info['distribution_strategy']['device']}"
        )
    else:
        logger.info(
            f"Moving model to device: {model_info['distribution_strategy']['device']}..."
        )
        model_info["model"].to(model_info["distribution_strategy"]["device"])

    return model_info

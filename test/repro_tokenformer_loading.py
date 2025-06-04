#!/usr/bin/env python3
"""
Direct fix for your 10-minute Tokenformer loading issue.
This script reproduces your exact scenario and provides the fix.
"""

import os
import time
import torch
import logging
from pathlib import Path

# CRITICAL: Configure logging BEFORE importing transformers
logging.basicConfig(
    level=logging.INFO,  # Not DEBUG!
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Suppress excessive logging from libraries
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('tokenformer').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

def reproduce_slow_loading():
    """Reproduce your current slow loading approach"""
    logger.info("🐌 REPRODUCING SLOW LOADING (Your Current Approach)")
    logger.info("="*80)

    model_name = "meta-llama/Llama-3.3-70B-Instruct"

    # Your current approach
    start = time.time()

    # This is what your code does:
    model = AutoModelForCausalLM.from_pretrained(model_name)  # No optimizations!

    # Then converts dtype
    model = model.to(dtype=torch.bfloat16)

    # Then moves to device
    model.to("cuda")

    load_time = time.time() - start
    logger.info(f"❌ Slow loading took: {load_time:.2f}s")

    return model

def fast_loading_from_cache():
    """Optimized loading that uses cache properly"""
    logger.info("\n⚡ FAST LOADING FROM CACHE (Optimized)")
    logger.info("="*80)

    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    cache_dir = Path.home() / ".cache" / "huggingface"

    # Verify cache
    logger.info("Checking cache status...")
    hub_dir = cache_dir / "hub"
    model_cache_pattern = f"models--{model_name.replace('/', '--')}"

    model_in_cache = False
    for item in hub_dir.iterdir() if hub_dir.exists() else []:
        if model_cache_pattern in str(item):
            model_in_cache = True
            # Get size
            size_bytes = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            logger.info(f"✅ Model found in cache: {size_bytes / (1024**3):.1f}GB")
            break

    if not model_in_cache:
        logger.warning("⚠️  Model not in cache, will be downloaded")

    start = time.time()

    # Step 1: Load config (fast)
    config_time = time.time()
    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    logger.info(f"Config loaded in {time.time() - config_time:.2f}s")

    # Step 2: Load tokenizer (fast)
    tokenizer_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    logger.info(f"Tokenizer loaded in {time.time() - tokenizer_time:.2f}s")

    # Step 3: Load model with ALL optimizations
    model_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # Critical optimizations:
        torch_dtype=torch.bfloat16,     # Load directly in target dtype
        low_cpu_mem_usage=True,         # CRITICAL for 70B model
        device_map="cpu",               # Load to CPU first
        # Cache settings:
        cache_dir=cache_dir,            # Explicit cache directory
        local_files_only=False,         # Use cache, download if needed
        resume_download=True,           # Resume if interrupted
        # Additional optimizations:
        use_safetensors=True,          # Faster format if available
        _fast_init=True,               # Skip weight initialization
    )
    logger.info(f"Model loaded in {time.time() - model_time:.2f}s")

    # Step 4: Move to GPU efficiently
    if torch.cuda.is_available():
        gpu_time = time.time()
        model = model.to("cuda")
        torch.cuda.synchronize()
        logger.info(f"Moved to GPU in {time.time() - gpu_time:.2f}s")

    total_time = time.time() - start
    logger.info(f"✅ Fast loading took: {total_time:.2f}s")

    return model, config, tokenizer

def apply_tokenformer_efficiently(model):
    """Apply tokenformer modifications efficiently"""
    logger.info("\n🔧 APPLYING TOKENFORMER EFFICIENTLY")
    logger.info("="*80)

    start = time.time()

    # Suppress debug logging
    import logging as py_logging
    tokenformer_logger = py_logging.getLogger('tokenformer')
    original_level = tokenformer_logger.level
    tokenformer_logger.setLevel(py_logging.WARNING)

    try:
        # Import here to control logging
        from tokenformer.llama_tokenformer_model import create_llama_tokenformer_model

        # Apply modifications
        device = next(model.parameters()).device
        model = create_llama_tokenformer_model(model, device)

    finally:
        # Restore logging
        tokenformer_logger.setLevel(original_level)

    apply_time = time.time() - start
    logger.info(f"Tokenformer applied in {apply_time:.2f}s")

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model

def complete_optimized_pipeline():
    """Complete optimized pipeline matching your workflow"""
    logger.info("\n🚀 COMPLETE OPTIMIZED PIPELINE")
    logger.info("="*80)

    total_start = time.time()

    # Your workflow with optimizations
    job_config = {"llm_name": "meta-llama/Llama-3.3-70B-Instruct"}
    dtype_config = {"dtype": "bfloat16"}

    model_info = {
        "model_name": job_config["llm_name"],
        "distribution_strategy": {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }
    }

    # Step 1: Load config and tokenizer (fast)
    config_start = time.time()
    model_info["model_config"] = AutoConfig.from_pretrained(
        model_info["model_name"],
        cache_dir=os.path.expanduser("~/.cache/huggingface")
    )
    model_info["tokenizer"] = AutoTokenizer.from_pretrained(
        model_info["model_name"],
        cache_dir=os.path.expanduser("~/.cache/huggingface")
    )
    logger.info(f"Config/tokenizer loaded in {time.time() - config_start:.2f}s")

    # Step 2: Load model optimally
    model_start = time.time()
    model_info["model"] = AutoModelForCausalLM.from_pretrained(
        model_info["model_name"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        cache_dir=os.path.expanduser("~/.cache/huggingface"),
        local_files_only=False,
        resume_download=True,
        use_safetensors=True,
    )
    logger.info(f"Model loaded in {time.time() - model_start:.2f}s")

    # Step 3: Apply tokenformer
    tokenformer_start = time.time()

    # Suppress logging
    logging.getLogger('tokenformer').setLevel(logging.WARNING)

    from tokenformer.llama_tokenformer_model import create_llama_tokenformer_model
    model_info["model"] = create_llama_tokenformer_model(
        model_info["model"], model_info["distribution_strategy"]["device"]
    )

    logger.info(f"Tokenformer applied in {time.time() - tokenformer_start:.2f}s")

    # Step 4: Move to device
    device_start = time.time()
    model_info["model"].to(model_info["distribution_strategy"]["device"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.info(f"Moved to device in {time.time() - device_start:.2f}s")

    total_time = time.time() - total_start
    logger.info(f"\n✅ TOTAL PIPELINE TIME: {total_time:.2f}s")

    return model_info

def diagnose_and_fix():
    """Main diagnostic and fix routine"""
    logger.info("🔍 TOKENFORMER LOADING DIAGNOSTIC & FIX")
    logger.info("="*80)

    # System info
    logger.info("\n📊 System Information:")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    # Check cache
    cache_dir = Path.home() / ".cache" / "huggingface"
    if cache_dir.exists():
        cache_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
        logger.info(f"HF Cache size: {cache_size / 1024**3:.1f}GB")

    # Check logging levels (common issue)
    logger.info("\n⚠️  Checking logging levels:")
    loggers_to_check = ['transformers', 'tokenformer', 'urllib3', 'torch']
    for logger_name in loggers_to_check:
        level = logging.getLogger(logger_name).level
        level_name = logging.getLevelName(level)
        if level <= logging.DEBUG:
            logger.warning(f"  {logger_name}: {level_name} - TOO VERBOSE!")
        else:
            logger.info(f"  {logger_name}: {level_name} ✓")

    # Run the optimized pipeline
    try:
        model_info = complete_optimized_pipeline()
        logger.info("\n✅ SUCCESS! Model loaded efficiently")

        # Provide the fix
        logger.info("\n💡 TO FIX YOUR CODE:")
        logger.info("="*80)
        logger.info("1. Add at the top of your script:")
        logger.info("   logging.getLogger('tokenformer').setLevel(logging.WARNING)")
        logger.info("   logging.getLogger('transformers').setLevel(logging.WARNING)")
        logger.info("")
        logger.info("2. Update your materialize_model function to include:")
        logger.info("   - torch_dtype=torch.bfloat16")
        logger.info("   - low_cpu_mem_usage=True")
        logger.info("   - cache_dir=os.path.expanduser('~/.cache/huggingface')")
        logger.info("")
        logger.info("3. Remove the separate dtype conversion - load directly in bfloat16")
        logger.info("")
        logger.info("These changes should reduce loading from 10 minutes to ~1-2 minutes")

        return model_info

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def create_fixed_materialize_function():
    """Generate the fixed materialize_model function"""

    fixed_code = '''
def materialize_model(model_info):
    """Fixed version - loads in ~1-2 minutes instead of 10"""

    # Set optimal logging levels
    import logging
    logging.getLogger('tokenformer').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)

    # Download model (should be cached)
    download_model(model_info["model_name"])

    # Get config
    config = get_config()
    config_dtype = config["dtype"]
    dtype = (
        torch.float16 if config_dtype == "float16"
        else torch.float32 if config_dtype == "float32"
        else torch.bfloat16
    )

    # Load model WITH OPTIMIZATIONS
    cache_dir = os.path.expanduser("~/.cache/huggingface")

    model_info["model"] = AutoModelForCausalLM.from_pretrained(
        model_info["model_name"],
        torch_dtype=dtype,           # Load directly in target dtype
        low_cpu_mem_usage=True,      # Critical for 70B models
        device_map="cpu",            # Load to CPU first
        cache_dir=cache_dir,         # Use cache explicitly
        local_files_only=False,      # Use cache but allow download
        use_safetensors=True,        # Faster format
    )

    # Apply tokenformer (with logging suppressed)
    model_info["model"] = create_llama_tokenformer_model(
        model_info["model"], model_info["distribution_strategy"]["device"]
    )

    # Apply distribution strategy if exists
    if (
        "distribution_strategy" in model_info
        and "strategy" in model_info["distribution_strategy"]
    ):
        model_info["model"] = model_info["distribution_strategy"]["strategy"](
            model_info["model"]
        )

    # Log summary only
    logger.info(f"Model: {model_info['model'].__class__.__name__}")

    # Move to device
    model_info["model"].to(model_info["distribution_strategy"]["device"])

    return model_info
'''

    logger.info("\n📝 FIXED materialize_model FUNCTION:")
    logger.info("="*80)
    print(fixed_code)
    logger.info("="*80)

if __name__ == "__main__":
    # Run diagnostic and fix
    diagnose_and_fix()

    # Show the fixed function
    create_fixed_materialize_function()

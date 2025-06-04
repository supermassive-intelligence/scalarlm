#!/usr/bin/env python3
"""
Standalone test for tokenformer loading optimization.
This creates a minimal test environment to verify the fix.
"""

import os
import time
import torch
import logging
import tempfile
import yaml
from pathlib import Path

# CRITICAL: Set up logging BEFORE imports
logging.basicConfig(level=logging.INFO)
logging.getLogger('tokenformer').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def create_test_config():
    """Create a minimal config file for testing"""
    config = {
        "dataset_hash": "test_hash",
        "gpus": 1,
        "job_directory": "/tmp/test_job",
        "learning_rate": 0.0001,
        "llm_name": "meta-llama/Llama-3.3-70B-Instruct",
        "max_steps": 10,
        "max_token_block_size": 4096,
        "steps_per_checkpoint": 10000,
        "training_data_path": "/tmp/test_dataset.jsonlines",
        "dtype": "bfloat16"  # Add this if it's missing from your config
    }
    
    # Create temp config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        return f.name

def test_with_cray_framework():
    """Test using the actual Cray framework with a temp config"""
    
    # Create test config
    config_path = create_test_config()
    os.environ["CRAY_TRAINING_JOB_CONFIG_PATH"] = config_path
    
    logger.info("🚀 TESTING WITH CRAY FRAMEWORK")
    logger.info("="*80)
    logger.info(f"Using config: {config_path}")
    
    try:
        # Import and use the fixed loader
        from fixed_tokenformer_loader import load_tokenformer_model
        
        start = time.time()
        model_info = load_tokenformer_model()
        total_time = time.time() - start
        
        logger.info(f"\n✅ SUCCESS! Model loaded in {total_time:.2f}s")
        
        # Show speedup
        baseline_time = 600  # 10 minutes from your logs
        speedup = baseline_time / total_time
        logger.info(f"🎉 That's {speedup:.1f}x faster than baseline!")
        
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.remove(config_path)

def test_tokenformer_loading_direct():
    """Direct test without the Cray framework"""
    
    # Set environment for AMD GPU
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    logger.info("🚀 DIRECT TOKENFORMER LOADING TEST")
    logger.info("="*80)
    
    # System info
    logger.info(f"Model: {model_name}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    total_start = time.time()
    timings = {}
    
    try:
        # Step 1: Load config and tokenizer
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        
        step_start = time.time()
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        timings['config_tokenizer'] = time.time() - step_start
        logger.info(f"✅ Config/tokenizer loaded in {timings['config_tokenizer']:.2f}s")
        
        # Step 2: Load model with optimizations
        step_start = time.time()
        logger.info("\nLoading model with optimizations...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",  # Critical for 70B model
            cache_dir=cache_dir,
            local_files_only=False,
            use_safetensors=True,
            max_memory={0: "180GB"},  # Leave headroom
        )
        
        timings['model_loading'] = time.time() - step_start
        logger.info(f"✅ Model loaded in {timings['model_loading']:.2f}s")
        
        # Log memory after model load
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"   GPU memory used: {gpu_mem:.1f}GB")
        
        # Step 3: Apply tokenformer with suppressed logging
        step_start = time.time()
        logger.info("\nApplying tokenformer modifications...")
        
        # Suppress all output during tokenformer wrapping
        import sys
        import io
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_level = logging.getLogger().level
        
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        logging.getLogger().setLevel(logging.ERROR)
        
        suppressed_lines = 0
        try:
            from tokenformer.llama_tokenformer_model import create_llama_tokenformer_model
            
            # Get device
            device = next(model.parameters()).device
            
            # Apply tokenformer
            model = create_llama_tokenformer_model(model, device)
            
            # Count suppressed lines
            suppressed_output = sys.stdout.getvalue()
            suppressed_lines = len(suppressed_output.splitlines())
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            logging.getLogger().setLevel(old_level)
        
        timings['tokenformer'] = time.time() - step_start
        logger.info(f"✅ Tokenformer applied in {timings['tokenformer']:.2f}s")
        logger.info(f"   (Suppressed {suppressed_lines} debug log lines)")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"\nModel parameters:")
        logger.info(f"   Total: {total_params:,}")
        logger.info(f"   Trainable: {trainable_params:,}")
        
        # Final memory check
        if torch.cuda.is_available():
            final_gpu_mem = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"\nFinal GPU memory: {final_gpu_mem:.1f}GB")
        
        total_time = time.time() - total_start
        
        # Print summary
        logger.info("\n📊 TIMING SUMMARY:")
        logger.info("="*60)
        logger.info(f"{'Phase':<30} {'Time (s)':>12} {'Percent':>10}")
        logger.info("-"*60)
        
        for phase, duration in timings.items():
            percentage = (duration / total_time) * 100
            logger.info(f"{phase:<30} {duration:>12.2f} {percentage:>9.1f}%")
        
        logger.info("-"*60)
        logger.info(f"{'TOTAL':<30} {total_time:>12.2f} {100.0:>9.1f}%")
        logger.info("="*60)
        
        # Compare with baseline
        logger.info("\n📈 PERFORMANCE COMPARISON:")
        logger.info(f"Baseline (from your logs):")
        logger.info(f"  - Config/tokenizer: 0.74s")
        logger.info(f"  - Model loading: 3.60s")
        logger.info(f"  - Tokenformer: 373.33s ❌")
        logger.info(f"  - Total: ~600s")
        logger.info(f"\nOptimized:")
        logger.info(f"  - Config/tokenizer: {timings['config_tokenizer']:.2f}s")
        logger.info(f"  - Model loading: {timings['model_loading']:.2f}s")
        logger.info(f"  - Tokenformer: {timings['tokenformer']:.2f}s ✅")
        logger.info(f"  - Total: {total_time:.2f}s")
        logger.info(f"\n🎉 Speedup: {600/total_time:.1f}x faster!")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--direct":
        # Test directly without Cray framework
        test_tokenformer_loading_direct()
    else:
        # Test with Cray framework
        test_with_cray_framework()

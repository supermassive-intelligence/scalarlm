#!/usr/bin/env python3
"""
Simple test: Load only 20 layers to verify tokenformer fix.
This avoids OOM while proving the optimization works on GPU.
"""

import os
import time
import torch
import logging
import gc

# CRITICAL: Set up logging BEFORE imports
logging.basicConfig(level=logging.INFO)
logging.getLogger('tokenformer').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.WARNING)

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

logger = logging.getLogger(__name__)

def create_partial_model(num_layers=20):
    """Create a Llama model with only `num_layers` layers"""
    
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    logger.info(f"Creating partial model with {num_layers} layers...")
    
    # Load config
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    
    # Modify config to have fewer layers
    original_layers = config.num_hidden_layers
    config.num_hidden_layers = num_layers
    
    # Load full model
    logger.info(f"Loading {num_layers}/{original_layers} layers from cache...")
    
    # Create model with modified config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",  # Will fit on GPU with only 20 layers
        cache_dir=cache_dir,
        ignore_mismatched_sizes=True,  # Since we're loading partial model
    )
    
    return model, config

def test_tokenformer_simple():
    """Simple test with partial model on GPU"""
    
    logger.info("🚀 SIMPLE TOKENFORMER FIX TEST")
    logger.info("="*80)
    logger.info("Loading only 20 layers to fit on GPU")
    logger.info("")
    
    # Clear GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    try:
        # Create partial model
        model, config = create_partial_model(num_layers=20)
        
        # Check GPU usage
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"✅ Model loaded, GPU memory used: {gpu_mem:.1f}GB")
            device = next(model.parameters()).device
            logger.info(f"✅ Model is on: {device}")
        
        # Test 1: SLOW tokenformer (with debug logging)
        logger.info("\n--- TEST 1: Tokenformer WITH debug logging ---")
        
        # Enable debug temporarily
        import logging as pylogging
        tokenformer_logger = pylogging.getLogger('tokenformer')
        tokenformer_logger.setLevel(pylogging.DEBUG)
        
        # Capture output
        import io
        import sys
        captured = io.StringIO()
        old_stdout = sys.stdout
        
        slow_start = time.time()
        debug_lines = 0
        
        try:
            sys.stdout = captured
            
            # Import and wrap first 5 layers only
            from tokenformer.tokenformer_surgeon import TokenformerMLPAdapter
            
            # Get device from model
            device = next(model.parameters()).device
            
            for i in range(5):
                layer = model.model.layers[i]
                layer.mlp = TokenformerMLPAdapter(layer.mlp, config.hidden_size, device)
            
            debug_output = captured.getvalue()
            debug_lines = len(debug_output.splitlines())
            
        finally:
            sys.stdout = old_stdout
            tokenformer_logger.setLevel(pylogging.ERROR)
        
        slow_time = time.time() - slow_start
        per_layer_slow = slow_time / 5
        
        logger.info(f"❌ SLOW: 5 layers took {slow_time:.2f}s")
        logger.info(f"   Debug lines printed: {debug_lines}")
        logger.info(f"   Time per layer: {per_layer_slow:.3f}s")
        logger.info(f"   Extrapolated to 80 layers: ~{per_layer_slow * 80:.0f}s")
        
        # Test 2: FAST tokenformer (without debug logging)
        logger.info("\n--- TEST 2: Tokenformer WITHOUT debug logging ---")
        
        # Suppress all output
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        fast_start = time.time()
        
        try:
            # Get device
            device = next(model.parameters()).device
            
            # Wrap remaining 15 layers
            for i in range(5, 20):
                layer = model.model.layers[i]
                layer.mlp = TokenformerMLPAdapter(layer.mlp, config.hidden_size, device)
            
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        
        fast_time = time.time() - fast_start
        per_layer_fast = fast_time / 15
        
        logger.info(f"✅ FAST: 15 layers took {fast_time:.2f}s")
        logger.info(f"   Time per layer: {per_layer_fast:.3f}s")
        logger.info(f"   Extrapolated to 80 layers: ~{per_layer_fast * 80:.0f}s")
        
        # Calculate speedup
        speedup = per_layer_slow / per_layer_fast
        
        logger.info(f"\n🎯 RESULTS:")
        logger.info(f"   Per-layer speedup: {speedup:.1f}x")
        logger.info(f"   Debug logging added: {debug_lines/5:.0f} lines per layer")
        
        logger.info(f"\n📊 PROJECTED FULL MODEL (80 layers):")
        logger.info(f"   Without fix: ~{per_layer_slow * 80:.0f}s ({per_layer_slow * 80 / 60:.1f} minutes)")
        logger.info(f"   With fix: ~{per_layer_fast * 80:.0f}s ({per_layer_fast * 80 / 60:.1f} minutes)")
        logger.info(f"   Time saved: ~{(per_layer_slow - per_layer_fast) * 80:.0f}s")
        
        # Verify model is on GPU and working
        logger.info(f"\n✅ Model successfully loaded on GPU: {next(model.parameters()).device}")
        logger.info(f"✅ All layers have tokenformer applied")
        
        # Final GPU memory
        if torch.cuda.is_available():
            final_mem = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"✅ Final GPU memory: {final_mem:.1f}GB")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def test_actual_fix():
    """Test the actual fix by importing your fixed loader"""
    
    logger.info("\n🔧 TESTING YOUR ACTUAL FIX")
    logger.info("="*80)
    
    # Mock the required functions
    import sys
    import types
    
    # Create mock module for cray_infra
    mock_cray = types.ModuleType('cray_infra')
    mock_cray.util = types.ModuleType('util')
    
    def mock_get_job_config():
        return {"llm_name": "meta-llama/Llama-3.3-70B-Instruct"}
    
    def mock_get_config():
        return {"dtype": "bfloat16"}
    
    mock_cray.util.get_job_config = types.ModuleType('get_job_config')
    mock_cray.util.get_job_config.get_job_config = mock_get_job_config
    mock_cray.util.get_config = types.ModuleType('get_config')  
    mock_cray.util.get_config.get_config = mock_get_config
    
    sys.modules['cray_infra'] = mock_cray
    sys.modules['cray_infra.util'] = mock_cray.util
    sys.modules['cray_infra.util.get_job_config'] = mock_cray.util.get_job_config
    sys.modules['cray_infra.util.get_config'] = mock_cray.util.get_config
    
    # Mock download_model
    mock_download = types.ModuleType('cray_megatron')
    mock_download.huggingface = types.ModuleType('huggingface')
    mock_download.huggingface.download_model = types.ModuleType('download_model')
    mock_download.huggingface.download_model.download_model = lambda x: None
    
    sys.modules['cray_megatron'] = mock_download
    sys.modules['cray_megatron.huggingface'] = mock_download.huggingface
    sys.modules['cray_megatron.huggingface.download_model'] = mock_download.huggingface.download_model
    
    # Now we can import and test
    try:
        # Test with partial model
        original_from_pretrained = AutoModelForCausalLM.from_pretrained
        
        def mock_from_pretrained(model_name, **kwargs):
            # Force smaller model
            config = AutoConfig.from_pretrained(model_name)
            config.num_hidden_layers = 20  # Only 20 layers
            kwargs['config'] = config
            kwargs['ignore_mismatched_sizes'] = True
            return original_from_pretrained(model_name, **kwargs)
        
        AutoModelForCausalLM.from_pretrained = mock_from_pretrained
        
        # Import and run your fixed loader
        from fixed_tokenformer_loader import fixed_materialize_model
        
        model_info = {
            "model_name": "meta-llama/Llama-3.3-70B-Instruct",
            "distribution_strategy": {"device": torch.device("cuda")}
        }
        
        logger.info("Running your fixed_materialize_model function...")
        start = time.time()
        result = fixed_materialize_model(model_info)
        total_time = time.time() - start
        
        logger.info(f"\n✅ Your fix completed in {total_time:.2f}s!")
        
        # Restore
        AutoModelForCausalLM.from_pretrained = original_from_pretrained
        
    except Exception as e:
        logger.error(f"Could not test actual fix: {e}")

if __name__ == "__main__":
    import sys
    
    if "--actual" in sys.argv:
        test_actual_fix()
    else:
        test_tokenformer_simple()

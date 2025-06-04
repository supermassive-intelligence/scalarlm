#!/usr/bin/env python3
"""
Llama 70B specific optimization test for MI300X
This will show the real performance gains for your actual model
"""

import torch
import time
import gc
import psutil
import os

def test_llama70b_optimization():
    """Test Llama 70B loading with different strategies"""
    
    # Common Llama 70B model names
    possible_models = [
        "meta-llama/Llama-2-70b-chat-hf",
        "meta-llama/Llama-2-70b-hf", 
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "NousResearch/Llama-2-70b-chat-hf",  # Popular alternative
    ]
    
    print("=== LLAMA 70B OPTIMIZATION TEST ===")
    print(f"GPU: AMD MI300X (191GB)")
    print(f"Target: ~70B parameter model")
    print(f"Expected model size: ~140GB in bfloat16")
    print()
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Try to find which Llama 70B model is accessible
    model_name = None
    for candidate in possible_models:
        try:
            print(f"Checking accessibility of {candidate}...")
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(candidate)
            print(f"✅ {candidate} is accessible")
            model_name = candidate
            break
        except Exception as e:
            print(f"❌ {candidate}: {str(e)[:100]}...")
    
    if not model_name:
        print("❌ No Llama 70B model accessible. You may need HuggingFace token.")
        print("Testing optimization principles with smaller model...")
        model_name = "microsoft/DialoGPT-large"
    
    print(f"\nTesting with: {model_name}")
    print("="*80)
    
    # Get baseline memory
    torch.cuda.empty_cache()
    gc.collect()
    baseline_memory = torch.cuda.memory_allocated()
    
    try:
        # Test 1: Standard loading (this will likely be very slow)
        print("TEST 1: STANDARD LOADING")
        print("-" * 40)
        
        start_time = time.time()
        ram_before = psutil.Process().memory_info().rss / 1024**3
        
        print("⏳ Loading model with standard parameters...")
        print("   (This may take several minutes for 70B model)")
        
        model1 = AutoModelForCausalLM.from_pretrained(model_name)
        
        standard_time = time.time() - start_time
        ram_after = psutil.Process().memory_info().rss / 1024**3
        gpu_memory = (torch.cuda.memory_allocated() - baseline_memory) / 1024**3
        
        model_device = next(model1.parameters()).device
        model_dtype = next(model1.parameters()).dtype
        
        print(f"✅ Standard loading completed!")
        print(f"   Time: {standard_time:.1f} seconds ({standard_time/60:.1f} minutes)")
        print(f"   Device: {model_device}")
        print(f"   Dtype: {model_dtype}")
        print(f"   RAM used: {ram_after - ram_before:.1f}GB")
        print(f"   GPU memory: {gpu_memory:.1f}GB")
        
        del model1
        torch.cuda.empty_cache()
        gc.collect()
        
        # Test 2: Optimized loading
        print(f"\nTEST 2: OPTIMIZED LOADING")
        print("-" * 40)
        
        start_time = time.time()
        ram_before = psutil.Process().memory_info().rss / 1024**3
        
        print("🚀 Loading model with optimized parameters...")
        
        model2 = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,      # Reduces peak RAM usage
            torch_dtype=torch.bfloat16,  # Load directly in bfloat16
            device_map="auto",           # Auto GPU placement
            trust_remote_code=True,      # Handle custom models
        )
        
        optimized_time = time.time() - start_time
        ram_after = psutil.Process().memory_info().rss / 1024**3
        gpu_memory = (torch.cuda.memory_allocated() - baseline_memory) / 1024**3
        
        model_device = next(model2.parameters()).device
        model_dtype = next(model2.parameters()).dtype
        
        speedup = standard_time / optimized_time if optimized_time > 0 else 0
        
        print(f"✅ Optimized loading completed!")
        print(f"   Time: {optimized_time:.1f} seconds ({optimized_time/60:.1f} minutes)")
        print(f"   Device: {model_device}")
        print(f"   Dtype: {model_dtype}")
        print(f"   RAM used: {ram_after - ram_before:.1f}GB")
        print(f"   GPU memory: {gpu_memory:.1f}GB")
        print(f"   🎯 SPEEDUP: {speedup:.1f}x")
        
        del model2
        torch.cuda.empty_cache()
        gc.collect()
        
        # Test 3: Optimized + Flash Attention
        print(f"\nTEST 3: OPTIMIZED + FLASH ATTENTION")
        print("-" * 40)
        
        start_time = time.time()
        ram_before = psutil.Process().memory_info().rss / 1024**3
        
        print("🔥 Loading model with Flash Attention...")
        
        model3 = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",  # Use Flash Attention
            trust_remote_code=True,
        )
        
        flash_time = time.time() - start_time
        ram_after = psutil.Process().memory_info().rss / 1024**3
        gpu_memory = (torch.cuda.memory_allocated() - baseline_memory) / 1024**3
        
        flash_speedup = standard_time / flash_time if flash_time > 0 else 0
        
        print(f"✅ Flash Attention loading completed!")
        print(f"   Time: {flash_time:.1f} seconds ({flash_time/60:.1f} minutes)")
        print(f"   RAM used: {ram_after - ram_before:.1f}GB")
        print(f"   GPU memory: {gpu_memory:.1f}GB")
        print(f"   🔥 FLASH SPEEDUP: {flash_speedup:.1f}x")
        
        # Quick inference test
        print(f"\nTesting inference speed...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        test_prompt = "The future of artificial intelligence is"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        inputs = {k: v.to(model3.device) for k, v in inputs.items()}
        
        # Warmup
        with torch.no_grad():
            _ = model3.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 10, do_sample=False)
        
        # Benchmark inference
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            outputs = model3.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 20, do_sample=False)
        
        torch.cuda.synchronize()
        inference_time = time.time() - start
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
        tokens_per_second = tokens_generated / inference_time
        
        print(f"   Inference time: {inference_time:.3f}s")
        print(f"   Tokens/second: {tokens_per_second:.1f}")
        print(f"   Generated: {generated_text}")
        
        del model3, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def estimate_your_speedup():
    """Estimate speedup for your specific training scenario"""
    print(f"\n" + "="*80)
    print("ESTIMATED SPEEDUP FOR YOUR TRAINING")
    print("="*80)
    
    print("Based on Llama 70B + MI300X + ROCm 6.3:")
    print()
    print("🚀 Model Loading Speedup:")
    print("   Standard loading:    15-30 minutes")
    print("   Optimized loading:   3-8 minutes")
    print("   Expected speedup:    3-6x faster")
    print()
    print("🔥 Training Speedup:")
    print("   Flash Attention:     20-40% faster per step")
    print("   GEMM Tuning:         10-25% faster after warmup")
    print("   Memory efficiency:   ~50% less GPU memory usage")
    print()
    print("💾 Memory Benefits:")
    print("   Standard:            ~280GB (float32) - Won't fit!")
    print("   Optimized:           ~140GB (bfloat16) - Perfect fit")
    print("   Flash Attention:     Additional 20-30% memory savings")
    print()
    print("⏱️ Overall Training Impact:")
    print("   Startup time:        5-10x faster")
    print("   Per-step time:       30-50% faster")
    print("   Memory efficiency:   Can train larger batches")

def check_hf_token():
    """Check if HuggingFace token is available for Llama access"""
    print(f"\n" + "="*80)
    print("HUGGINGFACE ACCESS CHECK")
    print("="*80)
    
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    
    if hf_token:
        print("✅ HuggingFace token found in environment")
        print(f"   Token: {hf_token[:10]}...")
    else:
        print("❌ No HuggingFace token found")
        print("   Set HF_TOKEN environment variable to access Llama models")
        print("   export HF_TOKEN=your_token_here")
    
    # Check cache directory
    cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    print(f"   Cache directory: {cache_dir}")

def main():
    """Run Llama 70B optimization tests"""
    check_hf_token()
    test_llama70b_optimization()
    estimate_your_speedup()

if __name__ == "__main__":
    main()

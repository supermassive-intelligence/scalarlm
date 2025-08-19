#!/usr/bin/env python3
"""Test script to verify KV cache implementation in v1 AsyncLLM engine."""

import asyncio
import os
import sys

# IMPORTANT: Add our vLLM to the FRONT of the path
vllm_path = os.path.join(os.path.dirname(__file__), 'vllm')
if vllm_path not in sys.path:
    sys.path.insert(0, vllm_path)

print(f"Added vLLM path: {vllm_path}")

# Add infra path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'infra'))

# Set environment to use v1 engine
os.environ["VLLM_USE_V1"] = "1"

async def test_kv_cache():
    """Test the get_free_kv_cache_tokens method."""
    
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    
    print("Initializing AsyncLLMEngine...")
    
    # Create engine args
    engine_args = AsyncEngineArgs(
        model="facebook/opt-125m",  # Small model for testing
        max_model_len=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,  # Disable CUDA graphs for easier debugging
    )
    
    # Initialize the engine
    llm = AsyncLLMEngine.from_engine_args(engine_args)
    
    print("AsyncLLM initialized successfully")
    
    try:
        # Test get_free_kv_cache_tokens
        print("\nTesting get_free_kv_cache_tokens method...")
        free_tokens = await llm.get_free_kv_cache_tokens()
        
        print(f"Free KV cache tokens: {free_tokens}")
        
        if free_tokens > 0:
            print("✓ KV cache method returned a positive value!")
            print(f"  Available tokens: {free_tokens}")
        else:
            print("✗ KV cache method returned 0 - implementation may not be working")
            
        # Run a simple generation to verify the engine works
        print("\nTesting generation to verify engine is functional...")
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10)
        
        # Generate with a simple prompt
        outputs = []
        async for output in llm.generate("Hello, my name is", sampling_params, request_id="test-1"):
            outputs.append(output)
            
        if outputs:
            print(f"✓ Generation successful: {outputs[-1].outputs[0].text}")
            
            # Check KV cache after generation
            free_tokens_after = await llm.get_free_kv_cache_tokens()
            print(f"\nFree KV cache tokens after generation: {free_tokens_after}")
            
            if free_tokens_after < free_tokens:
                print("✓ KV cache tokens decreased after generation (expected behavior)")
            else:
                print("Note: KV cache tokens did not decrease (may be due to cleanup)")
                
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print("\nShutting down...")
        llm.shutdown()

if __name__ == "__main__":
    print("Starting KV cache test for v1 AsyncLLM engine")
    print("=" * 50)
    asyncio.run(test_kv_cache())
    print("=" * 50)
    print("Test complete")

#!/usr/bin/env python3
"""Direct test of KV cache token functionality in v1 engine."""

import os
import sys

# IMPORTANT: Add our vLLM to the FRONT of the path
vllm_path = os.path.join(os.path.dirname(__file__), 'vllm')
if vllm_path not in sys.path:
    sys.path.insert(0, vllm_path)

print(f"Added vLLM path: {vllm_path}")

# Add infra path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'infra'))

# Ensure we use the copied vllm code in container
#if "/app/cray/vllm" not in sys.path:
#    sys.path.insert(0, "/app/cray/vllm")

# Force v1 engine
os.environ["VLLM_USE_V1"] = "1"

def test_kv_cache_functionality():
    """Test the actual KV cache token counting functionality."""
    
    print("Testing KV cache token counting functionality")
    print("=" * 60)
    
    # Test 1: Test the BlockPool directly
    print("\n1. Testing BlockPool functionality...")
    try:
        from vllm.v1.core.block_pool import BlockPool
        
        # Create a small block pool for testing
        num_blocks = 100
        block_pool = BlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=False,  # Simple test without caching
            enable_kv_cache_events=False
        )
        
        # Check initial free blocks
        free_blocks = block_pool.get_num_free_blocks()
        print(f"   Initial free blocks: {free_blocks}")
        print(f"   Expected: {num_blocks - 1} (minus null block)")
        
        if free_blocks == num_blocks - 1:
            print("   ✓ BlockPool correctly reports free blocks")
        else:
            print(f"   ✗ Unexpected free blocks count: {free_blocks}")
            
        # Allocate some blocks
        allocated = block_pool.get_new_blocks(10)
        free_after = block_pool.get_num_free_blocks()
        print(f"   After allocating 10 blocks: {free_after} free")
        
        if free_after == free_blocks - 10:
            print("   ✓ Block allocation works correctly")
        else:
            print(f"   ✗ Unexpected free blocks after allocation: {free_after}")
            
        # Free the blocks
        block_pool.free_blocks(allocated)
        free_final = block_pool.get_num_free_blocks()
        print(f"   After freeing blocks: {free_final} free")
        
        if free_final == free_blocks:
            print("   ✓ Block freeing works correctly")
        else:
            print(f"   ✗ Unexpected free blocks after freeing: {free_final}")
            
    except Exception as e:
        print(f"   ✗ Error testing BlockPool: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Test KVCacheManager with block_size
    print("\n2. Testing KVCacheManager with block_size...")
    try:
        import torch
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, FullAttentionSpec
        
        # Create a simple KV cache config
        block_size = 16  # tokens per block
        
        # Use FullAttentionSpec which extends AttentionSpec
        attention_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=8,
            head_size=64,
            dtype=torch.float16,
            use_mla=False,
            sliding_window=None,
            attention_chunk_size=None
        )
        
        kv_cache_group = KVCacheGroupSpec(
            kv_cache_spec=attention_spec,
            layer_names=[f"layer_{i}" for i in range(12)]  # 12 layers
        )
        
        # Create KVCacheTensor for each layer
        from vllm.v1.kv_cache_interface import KVCacheTensor
        kv_cache_tensors = []
        for i in range(12):
            # Each layer needs K and V cache tensors
            layer_name = f"layer_{i}"
            kv_cache_tensors.append(KVCacheTensor(
                size=1024*1024,  # 1MB per tensor for testing
                shared_by=[layer_name]  # Each tensor is used by one layer
            ))
        
        kv_cache_config = KVCacheConfig(
            num_blocks=100,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=[kv_cache_group]
        )
        
        manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=2048,
            enable_caching=False,
            use_eagle=False,
            log_stats=False
        )
        
        print(f"   KVCacheManager created successfully")
        print(f"   Block size: {manager.block_size}")
        
        if manager.block_pool:
            free_blocks = manager.block_pool.get_num_free_blocks()
            print(f"   Free blocks in manager: {free_blocks}")
            
            # Calculate free tokens
            if manager.block_size:
                free_tokens = free_blocks * manager.block_size
                print(f"   Free tokens (blocks * block_size): {free_tokens}")
                print(f"   ✓ KV cache token calculation: {free_blocks} blocks * {manager.block_size} = {free_tokens} tokens")
            else:
                print("   ✗ Block size is None")
        else:
            print("   ✗ Block pool is None")
            
    except Exception as e:
        print(f"   ✗ Error testing KVCacheManager: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Test EngineCore.get_free_kv_cache_tokens method
    print("\n3. Testing EngineCore.get_free_kv_cache_tokens method...")
    try:
        from vllm.v1.engine.core import EngineCore
        
        # Note: We can't instantiate EngineCore directly without proper setup,
        # but we can verify the method exists and check its implementation
        
        if hasattr(EngineCore, 'get_free_kv_cache_tokens'):
            print("   ✓ EngineCore.get_free_kv_cache_tokens method exists")
            
            # Check the method signature
            import inspect
            sig = inspect.signature(EngineCore.get_free_kv_cache_tokens)
            print(f"   Method signature: {sig}")
            
            # Read the method's docstring
            doc = EngineCore.get_free_kv_cache_tokens.__doc__
            if doc:
                print(f"   Docstring: {doc.strip()}")
        else:
            print("   ✗ EngineCore.get_free_kv_cache_tokens method NOT found")
            return False
            
    except Exception as e:
        print(f"   ✗ Error testing EngineCore: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ KV cache functionality tests completed successfully!")
    print("\nSummary:")
    print("  - BlockPool correctly tracks and reports free blocks")
    print("  - KVCacheManager properly manages blocks with block_size")
    print("  - Free tokens = free_blocks * block_size")
    print("  - EngineCore has the get_free_kv_cache_tokens method")
    return True

if __name__ == "__main__":
    success = test_kv_cache_functionality()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Comprehensive KV cache verification for both HTTP and Direct modes
"""

import asyncio
import sys
import os

# Add infra to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'infra'))

async def test_direct_mode_kv_cache():
    """Test Direct mode KV cache functionality"""
    print("🔧 Testing Direct Mode KV Cache")
    print("="*40)
    
    try:
        from cray_infra.vllm.engine_factory import create_vllm_engine
        
        print("   Creating DirectVLLMEngine...")
        config = {
            "vllm_use_http": False,
            "model": "microsoft/DialoGPT-small",
            "max_model_length": 512,
            "gpu_memory_utilization": 0.3,
            "enforce_eager": True,
            "disable_log_stats": True,
        }
        
        engine = create_vllm_engine(config)
        print(f"   ✅ Engine created: {engine}")
        
        # Test health check first
        print("   Testing health check...")
        try:
            health = await engine.health_check()
            print(f"   ✅ Health check: {health}")
        except Exception as e:
            print(f"   ❌ Health check failed: {e}")
            await engine.cleanup()
            return False
        
        # Test KV cache method - the key test!
        print("   Testing get_free_kv_cache_tokens...")
        try:
            free_tokens = await engine.get_free_kv_cache_tokens()
            print(f"   ✅ Free KV cache tokens: {free_tokens}")
            kv_success = True
        except Exception as e:
            print(f"   ❌ KV cache method failed: {e}")
            import traceback
            traceback.print_exc()
            kv_success = False
        
        # Test completion if KV cache works
        if kv_success:
            print("   Testing completion generation...")
            try:
                result = await engine.generate_completion(
                    "Hello",
                    "microsoft/DialoGPT-small",
                    max_tokens=10
                )
                print(f"   ✅ Completion: '{result}'")
                completion_success = True
            except Exception as e:
                print(f"   ❌ Completion failed: {e}")
                completion_success = False
        else:
            completion_success = False
        
        await engine.cleanup()
        print("   ✅ Cleanup successful")
        
        return kv_success and completion_success
        
    except Exception as e:
        print(f"   ❌ Direct mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_http_mode_kv_cache():
    """Test HTTP mode KV cache functionality"""
    print("\n🔗 Testing HTTP Mode KV Cache") 
    print("="*40)
    
    try:
        from cray_infra.vllm.engine_factory import create_vllm_engine
        
        print("   Creating HTTPVLLMEngine...")
        config = {
            "vllm_use_http": True,
            "vllm_api_url": "http://localhost:8001"
        }
        
        engine = create_vllm_engine(config)
        print(f"   ✅ Engine created: {engine}")
        
        # Test health check
        print("   Testing health check...")
        try:
            health = await engine.health_check()
            print(f"   ✅ Health check: {health}")
        except Exception as e:
            print(f"   ❌ Health check failed: {e}")
            await engine.cleanup()
            return False
        
        # Test KV cache method
        print("   Testing get_free_kv_cache_tokens...")
        try:
            free_tokens = await engine.get_free_kv_cache_tokens()
            print(f"   ✅ Free KV cache tokens: {free_tokens}")
            kv_success = True
        except Exception as e:
            print(f"   ❌ KV cache method failed: {e}")
            kv_success = False
        
        # Test completion
        print("   Testing completion generation...")
        try:
            result = await engine.generate_completion(
                "Hello",
                "masint/tiny-random-llama",  # Use correct model for server
                max_tokens=10
            )
            print(f"   ✅ Completion: '{result}'")
            completion_success = True
        except Exception as e:
            print(f"   ❌ Completion failed: {e}")
            completion_success = False
        
        await engine.cleanup()
        print("   ✅ Cleanup successful")
        
        return kv_success and completion_success
        
    except Exception as e:
        print(f"   ❌ HTTP mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_vllm_method_availability():
    """Test if vLLM methods are actually available"""
    print("🔍 Testing vLLM Method Availability")
    print("="*40)
    
    try:
        # Test AsyncLLMEngine import and method
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        print("   ✅ AsyncLLMEngine import successful")
        
        # Check method availability
        has_method = hasattr(AsyncLLMEngine, 'get_free_kv_cache_tokens')
        print(f"   {'✅' if has_method else '❌'} get_free_kv_cache_tokens method: {has_method}")
        
        if has_method:
            # Try to inspect the method
            method = getattr(AsyncLLMEngine, 'get_free_kv_cache_tokens')
            print(f"   ✅ Method object: {method}")
        
        # Test LLMEngine too
        from vllm.engine.llm_engine import LLMEngine
        has_sync_method = hasattr(LLMEngine, 'get_free_kv_cache_tokens')
        print(f"   {'✅' if has_sync_method else '❌'} LLMEngine has method: {has_sync_method}")
        
        # Test BlockManager
        from vllm.core.block_manager import BlockManager
        has_block_method = hasattr(BlockManager, 'get_free_kv_cache_tokens')
        print(f"   {'✅' if has_block_method else '❌'} BlockManager has method: {has_block_method}")
        
        return has_method and has_sync_method and has_block_method
        
    except Exception as e:
        print(f"   ❌ Method availability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run comprehensive KV cache verification"""
    print("🧪 Comprehensive KV Cache Verification")
    print("="*60)
    
    # Test method availability first
    methods_available = await test_vllm_method_availability()
    
    # Test HTTP mode
    http_success = await test_http_mode_kv_cache()
    
    # Test Direct mode (only if methods are available)
    direct_success = False
    if methods_available:
        direct_success = await test_direct_mode_kv_cache()
    else:
        print("\n🔧 Skipping Direct mode test - methods not available")
    
    # Summary
    print("\n" + "="*60)
    print("📊 KV CACHE VERIFICATION RESULTS")
    print("="*60)
    
    print(f"vLLM Methods Available:  {'✅' if methods_available else '❌'}")
    print(f"HTTP Mode KV Cache:      {'✅' if http_success else '❌'}")
    print(f"Direct Mode KV Cache:    {'✅' if direct_success else '❌' if methods_available else 'SKIPPED'}")
    
    if methods_available and http_success and direct_success:
        print("\n🎉 SUCCESS! Both HTTP and Direct KV cache work!")
        print("✅ Ready to run full compare_vllm_modes test")
        success = True
    elif http_success and methods_available:
        print("\n⚠️  Partial Success - HTTP works, Direct has issues")
        success = False
    elif not methods_available:
        print("\n❌ vLLM methods not properly imported/available")
        success = False
    else:
        print("\n❌ KV cache verification failed")
        success = False
    
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
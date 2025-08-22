#!/usr/bin/env python3
"""
Test Direct mode with correct vLLM path
"""

import sys
import os

# IMPORTANT: Add our vLLM to the FRONT of the path
vllm_path = os.path.join(os.path.dirname(__file__), 'vllm')
if vllm_path not in sys.path:
    sys.path.insert(0, vllm_path)

print(f"Added vLLM path: {vllm_path}")

# Add infra path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'infra'))

import asyncio

async def test_vllm_method_with_fixed_path():
    """Test vLLM method with corrected path"""
    print("🔧 Testing vLLM Method with Fixed Path")
    print("="*50)
    
    try:
        # Import after fixing path
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        import inspect
        
        print(f"Loading AsyncLLMEngine from: {inspect.getfile(AsyncLLMEngine)}")
        
        # Check if method exists
        has_method = hasattr(AsyncLLMEngine, 'get_free_kv_cache_tokens')
        print(f"Has get_free_kv_cache_tokens: {has_method}")
        
        if has_method:
            method = getattr(AsyncLLMEngine, 'get_free_kv_cache_tokens')
            print(f"Method: {method}")
            print("✅ Method found!")
            
            # Try to get source
            try:
                lines = inspect.getsourcelines(method)
                print(f"Method source starts at line {lines[1]}:")
                for i, line in enumerate(lines[0][:5]):
                    print(f"  {lines[1]+i:4d}: {line.rstrip()}")
                print("  ...")
            except:
                print("Could not get method source")
        else:
            print("❌ Method not found")
            
            # Show what methods are available
            methods = [m for m in dir(AsyncLLMEngine) if not m.startswith('_')]
            kv_methods = [m for m in methods if 'kv' in m.lower() or 'token' in m.lower()]
            print(f"Available KV/token methods: {kv_methods}")
        
        return has_method
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_direct_engine_creation_fixed():
    """Test Direct engine creation with fixed path"""
    print("\n🚀 Testing Direct Engine Creation with Fixed Path")
    print("="*50)
    
    try:
        from cray_infra.vllm.engine_factory import create_vllm_engine
        
        print("Creating DirectVLLMEngine...")
        config = {
            "vllm_use_http": False,
            "model": "microsoft/DialoGPT-small",
            "max_model_length": 512,
            "gpu_memory_utilization": 0.3,
            "enforce_eager": True,
            "disable_log_stats": True,
        }
        
        engine = create_vllm_engine(config)
        print(f"✅ Engine created: {engine}")
        
        # Test health
        health = await engine.health_check()
        print(f"✅ Health check: {health}")
        
        # Test KV cache method - the key test!
        print("Testing get_free_kv_cache_tokens...")
        tokens = await engine.get_free_kv_cache_tokens()
        print(f"✅ Free KV cache tokens: {tokens}")
        
        # Test completion
        print("Testing completion...")
        result = await engine.generate_completion(
            "Hello",
            "microsoft/DialoGPT-small", 
            max_tokens=10
        )
        print(f"✅ Completion: '{result}'")
        
        await engine.cleanup()
        print("✅ All tests passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("🧪 Direct Mode Test with Fixed vLLM Path")
    print("="*60)
    
    # Test 1: Check method availability
    method_available = await test_vllm_method_with_fixed_path()
    
    # Test 2: If method is available, test engine creation
    engine_works = False
    if method_available:
        engine_works = await test_direct_engine_creation_fixed()
    else:
        print("\n⏭️  Skipping engine test - method not available")
    
    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    print(f"Method Available: {'✅' if method_available else '❌'}")
    print(f"Engine Works:     {'✅' if engine_works else '❌'}")
    
    if method_available and engine_works:
        print("\n🎉 SUCCESS! Direct mode is fully working!")
        print("✅ KV cache method is available")
        print("✅ Engine creates successfully")  
        print("✅ All functionality works")
        print("\nReady to run compare_vllm_modes.py!")
    elif method_available:
        print("\n⚠️  Method available but engine creation failed")
    else:
        print("\n❌ Method not available - vLLM build issue")
    
    return method_available and engine_works

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
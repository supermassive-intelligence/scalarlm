#!/usr/bin/env python3
"""
Simple focused test of KV cache functionality
"""

import asyncio
import sys
import os
import aiohttp

# Add infra to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'infra'))

async def test_http_kv_cache_direct():
    """Test KV cache via direct HTTP call"""
    print("🔗 Testing KV Cache via Direct HTTP")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8001/v1/kv_cache/free_tokens") as response:
                status = response.status
                text = await response.text()
                
                print(f"   Status: {status}")
                print(f"   Response: {text}")
                
                if status == 200:
                    try:
                        import json
                        data = json.loads(text)
                        print(f"   ✅ KV Cache tokens: {data}")
                        return True
                    except:
                        print(f"   ⚠️  Got 200 but not JSON: {text}")
                        return False
                else:
                    print(f"   ❌ HTTP error {status}")
                    return False
                    
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
        return False

async def test_http_engine_kv_cache():
    """Test KV cache via our HTTP engine"""
    print("\n🛠️ Testing KV Cache via HTTP Engine")
    
    try:
        from cray_infra.vllm.http_engine import HTTPVLLMEngine
        
        engine = HTTPVLLMEngine(base_url="http://localhost:8001")
        
        # Test the method directly
        result = await engine.get_free_kv_cache_tokens()
        print(f"   Result: {result} (type: {type(result)})")
        
        if isinstance(result, int):
            print(f"   ✅ KV cache method returned: {result} tokens")
            await engine.cleanup()
            return True
        else:
            print(f"   ❌ Unexpected result type")
            await engine.cleanup()
            return False
            
    except Exception as e:
        print(f"   ❌ HTTP engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_direct_engine_import():
    """Test if we can import and create Direct engine"""
    print("\n🔧 Testing Direct Engine Import")
    
    try:
        # Test imports step by step
        print("   Testing vLLM imports...")
        
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        print("   ✅ AsyncLLMEngine imported")
        
        # Check if method exists with getattr
        method = getattr(AsyncLLMEngine, 'get_free_kv_cache_tokens', None)
        if method:
            print(f"   ✅ Method found: {method}")
        else:
            print("   ❌ Method not found")
            
            # Let's see what methods DO exist
            methods = [m for m in dir(AsyncLLMEngine) if not m.startswith('_')]
            kv_methods = [m for m in methods if 'kv' in m.lower() or 'token' in m.lower()]
            print(f"   Available methods with 'kv' or 'token': {kv_methods}")
        
        return method is not None
        
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        return False

async def test_compare_vllm_modes_minimal():
    """Test minimal version of compare_vllm_modes focusing on KV cache"""
    print("\n🆚 Testing Minimal compare_vllm_modes")
    
    try:
        from cray_infra.vllm.engine_factory import create_vllm_engine
        
        # Test HTTP mode only (since Direct might have import issues)
        print("   Testing HTTP mode...")
        config = {
            "vllm_use_http": True,
            "vllm_api_url": "http://localhost:8001"
        }
        
        engine = create_vllm_engine(config)
        
        # Test KV cache
        tokens = await engine.get_free_kv_cache_tokens()
        print(f"   KV cache tokens: {tokens}")
        
        # Test health
        health = await engine.health_check()
        print(f"   Health: {health}")
        
        await engine.cleanup()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Minimal test failed: {e}")
        return False

async def main():
    print("🧪 Simple KV Cache Verification")
    print("="*40)
    
    # Test 1: Direct HTTP
    http_direct = await test_http_kv_cache_direct()
    
    # Test 2: HTTP Engine
    http_engine = await test_http_engine_kv_cache()
    
    # Test 3: Direct Engine Import
    direct_import = await test_direct_engine_import()
    
    # Test 4: Minimal comparison
    minimal_test = await test_compare_vllm_modes_minimal()
    
    print("\n" + "="*40)
    print("📊 Results:")
    print(f"Direct HTTP call:     {'✅' if http_direct else '❌'}")
    print(f"HTTP Engine KV cache: {'✅' if http_engine else '❌'}")
    print(f"Direct Engine import: {'✅' if direct_import else '❌'}")
    print(f"Minimal test:         {'✅' if minimal_test else '❌'}")
    
    if http_engine:
        print("\n🎉 HTTP KV cache is working!")
    
    if direct_import:
        print("🎉 Direct engine methods are available!")
    
    return http_engine or direct_import

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
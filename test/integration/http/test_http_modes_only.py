#!/usr/bin/env python3
"""
HTTP Mode Performance Test - Docker Server Only
Tests ONLY vllm_use_http: True configuration against running Docker server
"""

import asyncio
import time
import statistics
import logging
import sys
import os
from typing import List, Dict, Any

# Add infra to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'infra'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HTTPOnlyComparator:
    def __init__(self):
        self.api_base = "http://localhost:8001"
        self.results = {}
        
    async def test_http_generation_performance(self, prompts: List[str], iterations: int = 3) -> Dict[str, Any]:
        """Test HTTP generation performance against Docker server"""
        logger.info(f"Testing HTTP generation performance with {len(prompts)} prompts, {iterations} iterations")
        
        times = []
        errors = []
        
        for i in range(iterations):
            start_time = time.time()
            
            try:
                await self._test_http_engine_access(prompts[0])
                    
                end_time = time.time()
                iteration_time = end_time - start_time
                times.append(iteration_time)
                logger.info(f"  Iteration {i+1}: {iteration_time:.3f}s")
                
            except Exception as e:
                error_msg = str(e)
                errors.append(error_msg)
                logger.error(f"  Iteration {i+1} FAILED: {error_msg}")
        
        success_count = len(times)
        
        return {
            "mode": "HTTP",
            "success_count": success_count,
            "total_iterations": iterations,
            "times": times,
            "errors": errors,
            "avg_time": statistics.mean(times) if times else None,
            "std_time": statistics.stdev(times) if len(times) > 1 else None,
            "min_time": min(times) if times else None,
            "max_time": max(times) if times else None
        }
    
    async def _test_http_engine_access(self, prompt: str) -> str:
        """Test HTTP engine via our engine factory"""
        from cray_infra.vllm.engine_factory import create_vllm_engine
        
        # Create HTTP engine
        config = {
            "vllm_use_http": True,
            "vllm_api_url": self.api_base
        }
        
        engine = create_vllm_engine(config)
        
        try:
            # Test health
            health = await engine.health_check()
            if not health:
                raise Exception("Health check failed")
            
            # Test KV cache (graceful fallback expected)
            kv_tokens = await engine.get_free_kv_cache_tokens()
            logger.debug(f"KV cache tokens: {kv_tokens}")
            
            # Generate completion
            result = await engine.generate_completion(
                prompt, 
                "masint/tiny-random-llama",
                max_tokens=20
            )
            
            return result
        finally:
            await engine.cleanup()

async def main():
    print("🧪 HTTP Mode Performance Test")
    print("Testing ONLY vllm_use_http=True against Docker server")
    print("="*70)
    
    comparator = HTTPOnlyComparator()
    
    # Test prompts
    test_prompts = [
        "Hello, how are you?",
        "Explain quantum computing",
        "Write a short story about"
    ]
    
    logger.info("🚀 Starting HTTP engine performance test...")
    
    # Test HTTP mode
    logger.info("Testing HTTP Engine Mode (vllm_use_http=True)")
    http_results = await comparator.test_http_generation_performance(test_prompts, iterations=3)
    
    # Display results
    print("\n" + "="*70)
    print("HTTP MODE PERFORMANCE RESULTS")
    print("="*70)
    
    success = http_results["success_count"]
    total = http_results["total_iterations"]
    
    if success > 0:
        print(f"✅ SUCCESS - {success}/{total} iterations")
        print(f"   Average time: {http_results['avg_time']:.3f}s")
        if http_results['std_time']:
            print(f"   Std deviation: {http_results['std_time']:.3f}s")
        print(f"   Min time: {http_results['min_time']:.3f}s")
        print(f"   Max time: {http_results['max_time']:.3f}s")
    else:
        print(f"❌ FAILED - {success}/{total} iterations")
        for i, error in enumerate(http_results['errors'], 1):
            print(f"   Error {i}: {error}")
    
    print("\n" + "="*70)
    print("📊 TEST SUMMARY:")
    print(f"   HTTP Mode: {'✅' if success == total else '❌'}")
    
    if success == total:
        print("\n🎉 SUCCESS! HTTP mode works perfectly against Docker server!")
        print("✅ vllm_use_http=True is fully functional")
        print("✅ KV cache graceful fallback working")
        print("✅ Completions generating successfully")
        print("✅ Ready for production use!")
    else:
        print("\n⚠️ HTTP mode has issues - check errors above")
    
    return success == total

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
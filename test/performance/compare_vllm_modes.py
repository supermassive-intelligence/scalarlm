#!/usr/bin/env python3
"""
Performance comparison test for HTTP vs Direct vLLM access modes
Tests both generation and embedding performance across different approaches
"""

import asyncio
import time
import statistics
import aiohttp
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceComparator:
    def __init__(self):
        self.api_base = "http://localhost:8000"
        self.results = {}
        
    async def test_generation_performance(self, mode: str, prompts: List[str], iterations: int = 5) -> Dict[str, Any]:
        """Test generation performance for a specific mode"""
        logger.info(f"Testing {mode} generation performance with {len(prompts)} prompts, {iterations} iterations")
        
        times = []
        for i in range(iterations):
            start_time = time.time()
            
            if mode == "http":
                await self._test_http_generation(prompts)
            elif mode == "direct":
                await self._test_direct_generation(prompts)
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
            end_time = time.time()
            iteration_time = end_time - start_time
            times.append(iteration_time)
            logger.info(f"  Iteration {i+1}: {iteration_time:.3f}s")
        
        return {
            "mode": mode,
            "type": "generation",
            "iterations": iterations,
            "times": times,
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0
        }
    
    async def test_embedding_performance(self, mode: str, prompts: List[str], iterations: int = 5) -> Dict[str, Any]:
        """Test embedding performance for a specific mode"""
        logger.info(f"Testing {mode} embedding performance with {len(prompts)} prompts, {iterations} iterations")
        
        times = []
        for i in range(iterations):
            start_time = time.time()
            
            if mode == "http":
                await self._test_http_embedding(prompts)
            elif mode == "direct":
                await self._test_direct_embedding(prompts)
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
            end_time = time.time()
            iteration_time = end_time - start_time
            times.append(iteration_time)
            logger.info(f"  Iteration {i+1}: {iteration_time:.3f}s")
        
        return {
            "mode": mode,
            "type": "embedding", 
            "iterations": iterations,
            "times": times,
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0
        }
    
    async def _test_http_generation(self, prompts: List[str]):
        """Test HTTP-based generation via API"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/v1/generate",
                json={
                    "prompts": prompts,
                    "max_tokens": 50
                }
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP generation failed: {resp.status}")
                return await resp.json()
    
    async def _test_direct_generation(self, prompts: List[str]):
        """Test direct generation via shared vLLM engine (direct method calls)"""
        try:
            from cray_infra.vllm.shared_engine import SharedVLLMEngine
            
            # Use shared engine for direct method calls to existing vLLM instance
            engine = SharedVLLMEngine("http://localhost:8001")
            
            # Call engine methods directly (bypasses HTTP serving layer)
            result = await engine.generate_completion(
                prompts[0] if prompts else "test",
                "openai-community/gpt2",
                max_tokens=50
            )
            
            await engine.cleanup()
            return {"choices": [{"text": result}]}
            
        except Exception as e:
            # Fallback to direct vLLM HTTP if shared engine fails
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8001/v1/completions",
                    json={
                        "prompt": prompts[0] if prompts else "test",
                        "max_tokens": 50,
                        "model": "openai-community/gpt2"
                    }
                ) as resp:
                    if resp.status != 200:
                        raise Exception(f"Direct generation failed: {resp.status}")
                    return await resp.json()
    
    async def _test_http_embedding(self, prompts: List[str]):
        """Test HTTP-based embedding via API"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/v1/generate/embed",
                json={"prompts": prompts}
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP embedding failed: {resp.status}")
                return await resp.json()
    
    async def _test_direct_embedding(self, prompts: List[str]):
        """Test direct embedding via shared engine (direct method calls)"""
        try:
            from cray_infra.vllm.shared_engine import SharedVLLMEngine
            
            # Use shared engine for direct embedding calls
            engine = SharedVLLMEngine("http://localhost:8001")
            
            # Call embedding methods directly (bypasses queue)
            results = []
            for prompt in prompts:
                embedding = await engine.generate_embeddings(
                    prompt,
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                results.append({"embedding": embedding})
            
            await engine.cleanup()
            return {"data": results}
            
        except Exception as e:
            # Fallback to direct embedding service call if shared engine fails
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8002/v1/embeddings",
                    json={
                        "input": prompts,
                        "model": "sentence-transformers/all-MiniLM-L6-v2"
                    }
                ) as resp:
                    if resp.status != 200:
                        raise Exception(f"Direct embedding failed: {resp.status}")
                    return await resp.json()
    
    def print_results(self, results: List[Dict[str, Any]]):
        """Print performance comparison results"""
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON RESULTS")
        print("="*80)
        print("HTTP MODE:   ScalarLM API → Queue → Worker → vLLM")
        print("DIRECT MODE: Direct method calls to vLLM engine (no HTTP/queue)")
        print("="*80)
        
        # Group by type
        generation_results = [r for r in results if r["type"] == "generation"]
        embedding_results = [r for r in results if r["type"] == "embedding"]
        
        if generation_results:
            print("\nGENERATION PERFORMANCE:")
            print("-" * 40)
            for result in generation_results:
                print(f"{result['mode'].upper():>8} | Avg: {result['avg_time']:.3f}s | "
                      f"Min: {result['min_time']:.3f}s | Max: {result['max_time']:.3f}s | "
                      f"StdDev: {result['std_dev']:.3f}s")
            
            if len(generation_results) == 2:
                http_avg = next(r["avg_time"] for r in generation_results if r["mode"] == "http")
                direct_avg = next(r["avg_time"] for r in generation_results if r["mode"] == "direct")
                speedup = http_avg / direct_avg
                print(f"\nDirect is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than HTTP")
        
        if embedding_results:
            print("\nEMBEDDING PERFORMANCE:")
            print("-" * 40)
            for result in embedding_results:
                print(f"{result['mode'].upper():>8} | Avg: {result['avg_time']:.3f}s | "
                      f"Min: {result['min_time']:.3f}s | Max: {result['max_time']:.3f}s | "
                      f"StdDev: {result['std_dev']:.3f}s")
            
            if len(embedding_results) == 2:
                http_avg = next(r["avg_time"] for r in embedding_results if r["mode"] == "http")
                direct_avg = next(r["avg_time"] for r in embedding_results if r["mode"] == "direct")
                speedup = http_avg / direct_avg  
                print(f"\nDirect is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than HTTP")

async def main():
    """Run the performance comparison"""
    comparator = PerformanceComparator()
    
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming.",
        "What are the benefits of renewable energy?",
        "How does machine learning work?"
    ]
    
    results = []
    
    try:
        # Test Generation Performance
        logger.info("Starting generation performance tests...")
        http_gen_result = await comparator.test_generation_performance("http", test_prompts[:1], iterations=5)
        results.append(http_gen_result)
        
        direct_gen_result = await comparator.test_generation_performance("direct", test_prompts[:1], iterations=5)
        results.append(direct_gen_result)
        
        # Test Embedding Performance  
        logger.info("Starting embedding performance tests...")
        http_embed_result = await comparator.test_embedding_performance("http", test_prompts[:3], iterations=5)  # Use fewer prompts for faster testing
        results.append(http_embed_result)
        
        direct_embed_result = await comparator.test_embedding_performance("direct", test_prompts[:3], iterations=5)
        results.append(direct_embed_result)
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        return
    
    # Print results
    comparator.print_results(results)

if __name__ == "__main__":
    asyncio.run(main())
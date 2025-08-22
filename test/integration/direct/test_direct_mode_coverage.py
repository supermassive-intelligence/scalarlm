#!/usr/bin/env python3
"""
Integration tests for Direct Mode coverage.
Tests that vllm_use_http: false works for all ScalarLM operations.

Prerequisites:
- ScalarLM configured with vllm_use_http: false
- Direct vLLM engine available
"""

import asyncio
import json
import logging
import os
import tempfile
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import aiohttp
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DirectTestResult:
    """Result of a direct mode test"""
    operation: str
    success: bool
    response_data: Any
    execution_time: float
    error: Optional[str] = None


class DirectModeCoverage:
    """Tests comprehensive coverage of Direct mode functionality"""
    
    def __init__(self, scalarlm_url: str = "http://localhost:8000"):
        self.scalarlm_url = scalarlm_url.rstrip('/')
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, method: str, endpoint: str, 
                          data: Optional[Dict] = None, 
                          files: Optional[Dict] = None) -> DirectTestResult:
        """Make a request and measure performance"""
        import time
        
        start_time = time.time()
        operation = f"{method.upper()} {endpoint}"
        
        try:
            url = f"{self.scalarlm_url}{endpoint}"
            
            if method.upper() == "GET":
                async with self.session.get(url) as resp:
                    response_data = await self._parse_response(resp)
                    success = resp.status < 400
            elif method.upper() == "POST":
                if files:
                    # Handle file upload
                    form_data = aiohttp.FormData()
                    for key, value in (data or {}).items():
                        form_data.add_field(key, str(value))
                    for key, file_info in files.items():
                        form_data.add_field(key, file_info['content'], 
                                          filename=file_info['filename'],
                                          content_type=file_info.get('content_type', 'application/octet-stream'))
                    
                    async with self.session.post(url, data=form_data) as resp:
                        response_data = await self._parse_response(resp)
                        success = resp.status < 400
                else:
                    async with self.session.post(url, json=data) as resp:
                        response_data = await self._parse_response(resp)
                        success = resp.status < 400
            else:
                return DirectTestResult(operation, False, None, 0, f"Unsupported method: {method}")
            
            execution_time = time.time() - start_time
            return DirectTestResult(operation, success, response_data, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return DirectTestResult(operation, False, None, execution_time, str(e))
    
    async def _parse_response(self, resp):
        """Parse response based on content type"""
        content_type = resp.headers.get('content-type', '').lower()
        
        if 'application/json' in content_type:
            return await resp.json()
        else:
            text = await resp.text()
            try:
                return json.loads(text)
            except:
                return {"text": text, "status": resp.status}
    
    async def test_health_check(self) -> DirectTestResult:
        """Test health check endpoint"""
        logger.info("Testing health check in direct mode...")
        return await self._make_request("GET", "/health")
    
    async def test_model_list(self) -> DirectTestResult:
        """Test model listing endpoint"""
        logger.info("Testing model listing in direct mode...")
        return await self._make_request("GET", "/v1/models")
    
    async def test_text_generation(self) -> DirectTestResult:
        """Test text generation endpoint"""
        logger.info("Testing text generation in direct mode...")
        payload = {
            "prompts": ["What is the capital of France?", "Explain quantum physics in one sentence."],
            "max_tokens": 100,
            "temperature": 0.7
        }
        return await self._make_request("POST", "/v1/generate", payload)
    
    async def test_chat_completion(self) -> DirectTestResult:
        """Test chat completion endpoint"""
        logger.info("Testing chat completion in direct mode...")
        payload = {
            "messages": [
                {"role": "user", "content": "Hello! How are you today?"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        return await self._make_request("POST", "/v1/chat/completions", payload)
    
    async def test_kv_cache_stats(self) -> DirectTestResult:
        """Test KV cache statistics (if available)"""
        logger.info("Testing KV cache statistics in direct mode...")
        # Try different possible endpoints
        result = await self._make_request("GET", "/v1/stats")
        if not result.success:
            # Try alternative endpoint
            result = await self._make_request("GET", "/stats")
        return result
    
    async def test_server_info(self) -> DirectTestResult:
        """Test server information endpoint"""
        logger.info("Testing server info in direct mode...")
        return await self._make_request("GET", "/server_info")
    
    async def test_version_info(self) -> DirectTestResult:
        """Test version information endpoint"""
        logger.info("Testing version info in direct mode...")
        return await self._make_request("GET", "/version")
    
    async def test_tokenization(self) -> DirectTestResult:
        """Test tokenization endpoint"""
        logger.info("Testing tokenization in direct mode...")
        payload = {
            "input": "Hello, world! This is a test sentence for tokenization.",
            "model": "test-model"
        }
        return await self._make_request("POST", "/tokenize", payload)
    
    async def test_detokenization(self) -> DirectTestResult:
        """Test detokenization endpoint"""
        logger.info("Testing detokenization in direct mode...")
        payload = {
            "tokens": [15496, 11, 1917, 0, 1212, 374, 264, 1296, 11914, 369, 4037, 2065, 13],
            "model": "test-model"
        }
        return await self._make_request("POST", "/detokenize", payload)
    
    async def test_file_upload_generation(self) -> DirectTestResult:
        """Test generation with file upload (if supported)"""
        logger.info("Testing file upload generation in direct mode...")
        
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document for processing.\nIt contains multiple lines of text.")
            temp_file_path = f.name
        
        try:
            # Read file content
            with open(temp_file_path, 'rb') as f:
                file_content = f.read()
            
            files = {
                'file': {
                    'filename': 'test.txt',
                    'content': file_content,
                    'content_type': 'text/plain'
                }
            }
            
            data = {
                'prompt': 'Summarize this document:',
                'max_tokens': 100
            }
            
            result = await self._make_request("POST", "/v1/generate/upload", data, files)
            
            # If upload endpoint doesn't exist, try alternative
            if not result.success and "404" in str(result.error):
                result = await self._make_request("POST", "/upload", data, files)
            
            return result
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    async def test_batch_processing(self) -> DirectTestResult:
        """Test batch processing capabilities"""
        logger.info("Testing batch processing in direct mode...")
        payload = {
            "requests": [
                {
                    "prompt": "What is machine learning?",
                    "max_tokens": 50,
                    "temperature": 0.5
                },
                {
                    "prompt": "Explain neural networks briefly.",
                    "max_tokens": 50,
                    "temperature": 0.5
                },
                {
                    "prompt": "What are transformers in AI?",
                    "max_tokens": 50,
                    "temperature": 0.5
                }
            ]
        }
        return await self._make_request("POST", "/v1/batch", payload)
    
    async def test_streaming_generation(self) -> DirectTestResult:
        """Test streaming generation (if supported)"""
        logger.info("Testing streaming generation in direct mode...")
        payload = {
            "prompt": "Write a short story about a robot learning to paint.",
            "max_tokens": 200,
            "stream": True,
            "temperature": 0.7
        }
        return await self._make_request("POST", "/v1/generate/stream", payload)
    
    async def run_all_tests(self) -> List[DirectTestResult]:
        """Run all direct mode coverage tests"""
        logger.info("Running complete Direct Mode coverage test suite...")
        
        tests = [
            self.test_health_check(),
            self.test_version_info(),
            self.test_model_list(),
            self.test_server_info(),
            self.test_kv_cache_stats(),
            self.test_text_generation(),
            self.test_chat_completion(),
            self.test_tokenization(),
            self.test_detokenization(),
            self.test_batch_processing(),
            self.test_streaming_generation(),
            self.test_file_upload_generation(),
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(DirectTestResult(
                    f"test_{i}", False, None, 0, str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def print_results(self, results: List[DirectTestResult]):
        """Print test results in a formatted table"""
        print("\n" + "="*100)
        print("DIRECT MODE COVERAGE TEST RESULTS")
        print("="*100)
        print("Testing ScalarLM with vllm_use_http: false")
        print("All operations should work via direct vLLM engine calls")
        print("="*100)
        
        # Categorize results
        core_endpoints = ["GET /health", "GET /v1/models", "GET /version"]
        generation_endpoints = ["POST /v1/generate", "POST /v1/chat/completions", "POST /v1/batch"]
        utility_endpoints = ["POST /tokenize", "POST /detokenize", "GET /server_info"]
        advanced_endpoints = ["POST /v1/generate/stream", "POST /v1/generate/upload", "GET /v1/stats"]
        
        categories = [
            ("Core Endpoints", core_endpoints),
            ("Generation Endpoints", generation_endpoints),
            ("Utility Endpoints", utility_endpoints),
            ("Advanced Endpoints", advanced_endpoints),
        ]
        
        total_tests = len(results)
        passed_tests = 0
        
        for category_name, expected_ops in categories:
            print(f"\n{category_name}:")
            print("-" * 60)
            
            category_results = [r for r in results if r.operation in expected_ops]
            
            for result in category_results:
                status = "✅ PASS" if result.success else "❌ FAIL"
                time_str = f"{result.execution_time:.3f}s"
                
                if result.error:
                    print(f"  {result.operation:<30} {status} ({time_str}) - {result.error}")
                else:
                    print(f"  {result.operation:<30} {status} ({time_str})")
                
                if result.success:
                    passed_tests += 1
            
            # Check for operations that weren't tested
            tested_ops = {r.operation for r in category_results}
            missing_ops = set(expected_ops) - tested_ops
            for missing_op in missing_ops:
                print(f"  {missing_op:<30} ⏸️  SKIP (not tested)")
        
        # Show any extra tests that don't fit categories
        categorized_ops = set()
        for _, ops in categories:
            categorized_ops.update(ops)
        
        extra_results = [r for r in results if r.operation not in categorized_ops]
        if extra_results:
            print(f"\nOther Tests:")
            print("-" * 60)
            for result in extra_results:
                status = "✅ PASS" if result.success else "❌ FAIL"
                time_str = f"{result.execution_time:.3f}s"
                print(f"  {result.operation:<30} {status} ({time_str})")
                if result.success:
                    passed_tests += 1
        
        print("\n" + "="*100)
        print(f"SUMMARY: {passed_tests}/{total_tests} endpoints working in Direct Mode")
        
        coverage_percent = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"Coverage: {coverage_percent:.1f}%")
        
        if coverage_percent >= 90:
            print("🎉 Excellent Direct Mode coverage!")
        elif coverage_percent >= 70:
            print("✅ Good Direct Mode coverage")
        else:
            print("⚠️  Direct Mode coverage needs improvement")
        
        return coverage_percent >= 70


async def main():
    """Main test runner"""
    print("Direct Mode Coverage Test")
    print("=" * 50)
    print("⚠️  IMPORTANT: Ensure ScalarLM is running with vllm_use_http: false")
    print("   This tests direct vLLM engine integration")
    print()
    
    async with DirectModeCoverage() as tester:
        results = await tester.run_all_tests()
        success = tester.print_results(results)
        return success


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        exit(1)
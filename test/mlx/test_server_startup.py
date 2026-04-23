#!/usr/bin/env python3
"""
Test MLX server startup and basic functionality.
"""
import asyncio
import sys
import os
import aiohttp

# Add infra to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'infra'))

# Set up environment for MLX
os.environ["SCALARLM_NATIVE_EXECUTION"] = "true"
os.environ["SCALARLM_INFERENCE_ONLY"] = "true"
os.environ["SCALARLM_SERVER_LIST"] = "api,vllm"


async def test_servers_start():
    """Test that both API and vLLM servers start successfully."""
    print("=" * 70)
    print("TEST: Server Startup")
    print("=" * 70)

    from cray_infra.one_server.start_cray_server import start_cray_server

    server_status = await start_cray_server(server_list=["api", "vllm"])

    # Wait for servers to start
    await asyncio.sleep(5)

    print("\n✓ Servers started")
    return server_status


async def test_health_endpoints():
    """Test that health endpoints are accessible."""
    print("\n" + "=" * 70)
    print("TEST: Health Endpoints")
    print("=" * 70)

    async with aiohttp.ClientSession() as session:
        # Test API server health
        async with session.get('http://localhost:8000/v1/health') as resp:
            assert resp.status == 200, f"API health check failed: {resp.status}"
            print(f"\n✓ API server health: {await resp.text()}")

        # Test vLLM server health
        async with session.get('http://localhost:8001/health') as resp:
            assert resp.status == 200, f"vLLM health check failed: {resp.status}"
            print(f"✓ vLLM server health: {await resp.text()}")


async def test_models_loaded():
    """Test that models are loaded and listed."""
    print("\n" + "=" * 70)
    print("TEST: Models Loaded")
    print("=" * 70)

    async with aiohttp.ClientSession() as session:
        # Test vLLM models endpoint
        async with session.get('http://localhost:8001/v1/models') as resp:
            assert resp.status == 200, f"Models endpoint failed: {resp.status}"
            data = await resp.json()
            assert len(data['data']) > 0, "No models loaded"
            print(f"\n✓ vLLM models: {data}")

        # Test API server models endpoint (proxied)
        async with session.get('http://localhost:8000/v1/models') as resp:
            assert resp.status == 200, f"API models endpoint failed: {resp.status}"
            data = await resp.json()
            assert len(data['data']) > 0, "No models loaded via API"
            print(f"✓ API models: {data}")


async def test_completion():
    """Test that completions work end-to-end."""
    print("\n" + "=" * 70)
    print("TEST: Text Completion")
    print("=" * 70)

    async with aiohttp.ClientSession() as session:
        # Use chat completions (vllm-mlx uses chat format)
        request = {
            "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 50,
            "temperature": 0.7
        }

        async with session.post('http://localhost:8001/v1/chat/completions', json=request) as resp:
            assert resp.status == 200, f"Completion failed: {resp.status}"
            data = await resp.json()
            assert 'choices' in data, "No choices in response"
            assert len(data['choices']) > 0, "Empty choices"
            print(f"\n✓ Completion: {data['choices'][0]['message']['content'][:100]}...")


async def run_all_tests():
    """Run all tests in sequence."""
    try:
        # Start servers
        server_status = await test_servers_start()

        # Run tests
        await test_health_endpoints()
        await test_models_loaded()
        await test_completion()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)

        # Cleanup
        await server_status.shutdown()
        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1)

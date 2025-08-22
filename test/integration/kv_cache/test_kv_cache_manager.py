"""
Unit tests for KV Cache Manager.
Tests the free_kv_cache_tokens algorithm and token management.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime

from cray_infra.api.kv_cache.kv_cache_manager import (
    KVCacheManager,
    KVCacheStats,
    initialize_kv_cache_manager,
    get_kv_cache_manager
)


class TestKVCacheManager:
    """Test KV Cache Manager functionality"""
    
    @pytest_asyncio.fixture
    async def kv_manager(self):
        """Create a KV cache manager for testing"""
        manager = KVCacheManager(
            total_tokens=10000,
            max_tokens_per_request=1000,
            max_batch_size=10
        )
        yield manager
        # Cleanup
        await manager.reset()
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test KV cache manager initialization"""
        manager = KVCacheManager(
            total_tokens=10000,
            max_tokens_per_request=1000,
            max_batch_size=10
        )
        
        assert manager.total_tokens == 10000
        assert manager.free_tokens == 10000
        assert manager.max_tokens_per_request == 1000
        assert manager.max_batch_size == 10
        assert len(manager.reserved_tokens) == 0
        
        stats = await manager.get_stats()
        assert stats.total_tokens == 10000
        assert stats.free_tokens == 10000
        assert stats.reserved_tokens == 0
        assert stats.utilization_percent == 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_batch_size(self, kv_manager):
        """Test dynamic batch size calculation"""
        # With all tokens free, should get max batch size
        batch_size = await kv_manager.calculate_batch_size()
        assert batch_size == 10  # min(10000/1000, max_batch_size=10)
        
        # Reserve some tokens
        await kv_manager.reserve_tokens(["req1", "req2", "req3"])
        
        # Now should get reduced batch size
        batch_size = await kv_manager.calculate_batch_size()
        assert batch_size == 7  # (10000-3000)/1000 = 7
        
        # With requested batch size
        batch_size = await kv_manager.calculate_batch_size(requested_batch_size=5)
        assert batch_size == 5  # min(7, 5)
        
        # Reserve more tokens
        await kv_manager.reserve_tokens(["req4", "req5", "req6", "req7"])
        
        # Now we have 7 requests * 1000 tokens = 7000 tokens reserved
        # 10000 - 7000 = 3000 free tokens = batch size of 3
        batch_size = await kv_manager.calculate_batch_size()
        assert batch_size == 3  # (10000-7000)/1000 = 3
    
    @pytest.mark.asyncio
    async def test_reserve_tokens(self, kv_manager):
        """Test token reservation"""
        # Reserve tokens for batch
        request_ids = ["req1", "req2", "req3"]
        success = await kv_manager.reserve_tokens(request_ids)
        
        assert success is True
        assert kv_manager.free_tokens == 7000  # 10000 - 3*1000
        assert len(kv_manager.reserved_tokens) == 3
        assert kv_manager.reserved_tokens["req1"] == 1000
        
        stats = await kv_manager.get_stats()
        assert stats.reserved_tokens == 3000
        assert stats.utilization_percent == 30.0
    
    @pytest.mark.asyncio
    async def test_reserve_tokens_insufficient(self, kv_manager):
        """Test token reservation with insufficient tokens"""
        # Try to reserve more than available
        request_ids = [f"req{i}" for i in range(11)]  # 11 * 1000 > 10000
        success = await kv_manager.reserve_tokens(request_ids)
        
        assert success is False
        assert kv_manager.free_tokens == 10000  # No change
        assert len(kv_manager.reserved_tokens) == 0
    
    @pytest.mark.asyncio
    async def test_release_tokens_partial(self, kv_manager):
        """Test partial token release after tokenization"""
        # Reserve tokens
        await kv_manager.reserve_tokens(["req1", "req2"])
        assert kv_manager.free_tokens == 8000
        
        # Release unused tokens (actual < reserved)
        released = await kv_manager.release_tokens_partial("req1", actual_tokens_used=600)
        
        assert released == 400  # 1000 - 600
        assert kv_manager.free_tokens == 8400  # 8000 + 400
        assert kv_manager.reserved_tokens["req1"] == 600  # Updated reservation
        assert kv_manager.reserved_tokens["req2"] == 1000  # Unchanged
    
    @pytest.mark.asyncio
    async def test_release_tokens_complete(self, kv_manager):
        """Test complete token release when request finishes"""
        # Reserve tokens
        await kv_manager.reserve_tokens(["req1", "req2", "req3"])
        assert kv_manager.free_tokens == 7000
        
        # Complete request 1
        released = await kv_manager.release_tokens_complete("req1", total_tokens_used=800)
        
        assert released == 1000  # Full reservation released
        assert kv_manager.free_tokens == 8000
        assert "req1" not in kv_manager.reserved_tokens
        assert len(kv_manager.reserved_tokens) == 2
        
        stats = await kv_manager.get_stats()
        assert stats.total_requests_processed == 1
        assert stats.total_tokens_processed == 800
    
    @pytest.mark.asyncio
    async def test_add_free_tokens(self, kv_manager):
        """Test adding newly freed tokens from vLLM engine"""
        # Reserve some tokens
        await kv_manager.reserve_tokens(["req1", "req2"])
        assert kv_manager.free_tokens == 8000
        
        # Add newly freed tokens from vLLM
        await kv_manager.add_free_tokens(500)
        
        assert kv_manager.free_tokens == 8500
        
        # Test capping to total
        await kv_manager.add_free_tokens(5000)
        assert kv_manager.free_tokens == 10000  # Capped to total
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, kv_manager):
        """Test complete token management workflow"""
        # Step 1: Calculate batch size
        batch_size = await kv_manager.calculate_batch_size()
        assert batch_size == 10
        
        # Step 2: Reserve tokens for batch
        request_ids = ["req1", "req2", "req3"]
        await kv_manager.reserve_tokens(request_ids)
        assert kv_manager.free_tokens == 7000
        
        # Step 3: After tokenization, release unused tokens
        await kv_manager.release_tokens_partial("req1", 600)
        await kv_manager.release_tokens_partial("req2", 700)
        await kv_manager.release_tokens_partial("req3", 500)
        assert kv_manager.free_tokens == 8200  # 7000 + 400 + 300 + 500
        
        # Step 4: Complete requests
        await kv_manager.release_tokens_complete("req1", 600)
        await kv_manager.release_tokens_complete("req2", 700)
        assert kv_manager.free_tokens == 9500  # 8200 + 600 + 700
        
        # Step 5: Add tokens from vLLM engine
        await kv_manager.add_free_tokens(200)
        assert kv_manager.free_tokens == 9700
        
        # Step 6: Complete last request
        await kv_manager.release_tokens_complete("req3", 500)
        # Would be 9700 + 500 = 10200, but capped at total capacity of 10000
        assert kv_manager.free_tokens == 10000  # Capped at total capacity
        
        # Check final stats
        stats = await kv_manager.get_stats()
        assert stats.total_requests_processed == 3
        assert stats.total_tokens_processed == 1800  # 600 + 700 + 500
        assert stats.utilization_percent == 0.0  # All tokens free
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, kv_manager):
        """Test thread safety with concurrent operations"""
        async def reserve_task(request_id):
            return await kv_manager.reserve_tokens([request_id])
        
        async def release_task(request_id):
            return await kv_manager.release_tokens_complete(request_id, 500)
        
        # Reserve tokens concurrently
        tasks = [reserve_task(f"req{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed (10 * 1000 = 10000 tokens)
        assert all(results)
        assert kv_manager.free_tokens == 0
        
        # Release tokens concurrently
        tasks = [release_task(f"req{i}") for i in range(10)]
        await asyncio.gather(*tasks)
        
        # All tokens should be free again
        assert kv_manager.free_tokens == 10000
    
    @pytest.mark.asyncio
    async def test_reset(self, kv_manager):
        """Test reset functionality"""
        # Make some changes
        await kv_manager.reserve_tokens(["req1", "req2"])
        await kv_manager.release_tokens_partial("req1", 600)
        
        # Reset
        await kv_manager.reset()
        
        # Should be back to initial state
        assert kv_manager.free_tokens == 10000
        assert len(kv_manager.reserved_tokens) == 0
        
        stats = await kv_manager.get_stats()
        assert stats.total_requests_processed == 0
        assert stats.total_tokens_processed == 0
    
    @pytest.mark.asyncio
    async def test_global_instance(self):
        """Test global instance management"""
        # Initialize global instance
        manager = initialize_kv_cache_manager(
            total_tokens=5000,
            max_tokens_per_request=500,
            max_batch_size=5
        )
        
        # Get global instance
        global_manager = await get_kv_cache_manager()
        
        assert global_manager is manager
        assert global_manager.total_tokens == 5000
        
        # Test operations on global instance
        await global_manager.reserve_tokens(["req1"])
        assert global_manager.free_tokens == 4500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
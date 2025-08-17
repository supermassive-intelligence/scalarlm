import asyncio
import logging
import subprocess
import sys
import time
import requests
from multiprocessing import Process

logger = logging.getLogger(__name__)


def start_embedding_service():
    """Start the separate embedding service."""
    try:
        logger.info("============================================================")
        logger.info("STARTING EMBEDDING SERVICE")
        logger.info("============================================================")
        
        # Import and run the embedding server
        from cray_infra.embedding_service.embedding_server import app
        import uvicorn
        
        logger.info("Embedding service configuration:")
        logger.info("  Port: 8002")
        logger.info("  Model: sentence-transformers/all-MiniLM-L6-v2")
        
        # Run the embedding service
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8002, 
            log_level="info",
            access_log=False
        )
        
    except Exception as e:
        logger.error(f"Failed to start embedding service: {e}")
        raise


def check_embedding_service_health():
    """Check if the embedding service is healthy."""
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        return response.status_code == 200
    except:
        return False


async def create_embedding_service():
    """Create and start the embedding service in a separate process."""
    try:
        # Start embedding service in a separate process
        embedding_process = Process(target=start_embedding_service)
        embedding_process.daemon = True
        embedding_process.start()
        
        # Wait for service to be ready
        logger.info("Waiting for embedding service to start...")
        max_retries = 30
        retry_count = 0
        
        while retry_count < max_retries:
            if check_embedding_service_health():
                logger.info("✓ Embedding service is healthy and ready")
                return embedding_process
            
            retry_count += 1
            await asyncio.sleep(1)
            logger.debug(f"Waiting for embedding service... ({retry_count}/{max_retries})")
        
        logger.error("✗ Embedding service failed to start within timeout")
        embedding_process.terminate()
        return None
        
    except Exception as e:
        logger.error(f"Failed to create embedding service: {e}")
        return None
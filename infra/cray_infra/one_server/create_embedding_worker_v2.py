"""
Enhanced embedding worker that processes requests from the queue.
Supports both HTTP and direct vLLM engine access based on configuration.
"""

import asyncio
import aiohttp
import logging
from typing import Optional

from cray_infra.util.get_config import get_config
from cray_infra.vllm import create_vllm_engine, VLLMEngineInterface, get_engine_info

logger = logging.getLogger(__name__)


async def create_embedding_worker(running_status):
    """
    Create a worker that processes embedding requests from the queue.
    Uses configurable vLLM engine (HTTP or direct) for processing.
    """
    logger.info("="*60)
    logger.info("STARTING ENHANCED EMBEDDING WORKER")
    logger.info("="*60)
    
    config = get_config()
    api_base = "http://localhost:8000"
    
    # Create vLLM engine based on configuration
    try:
        vllm_engine = create_vllm_engine(config)
        engine_info = get_engine_info(vllm_engine)
        
        logger.info(f"Embedding worker configuration:")
        logger.info(f"  API endpoint: {api_base}")
        logger.info(f"  vLLM engine: {engine_info['type']} ({engine_info['class']})")
        logger.info(f"  Engine details: {engine_info['repr']}")
        
        # Test the engine
        health_ok = await vllm_engine.health_check()
        if health_ok:
            logger.info("✓ vLLM engine health check passed")
        else:
            logger.warning("⚠️ vLLM engine health check failed, but continuing...")
            
    except Exception as e:
        logger.error(f"Failed to create vLLM engine: {e}")
        logger.info("Falling back to HTTP mode...")
        
        # Fallback to HTTP engine
        fallback_config = dict(config)
        fallback_config["vllm_use_http"] = True
        vllm_engine = create_vllm_engine(fallback_config)
        
    # Main worker loop
    session: Optional[aiohttp.ClientSession] = None
    
    try:
        session = aiohttp.ClientSession()
        
        while True:
            try:
                # Get work from the ScalarLM API queue
                logger.debug("Checking for work...")
                
                get_work_response = await session.post(
                    f"{api_base}/v1/generate/get_work",
                    json={"batch_size": 10},
                    timeout=aiohttp.ClientTimeout(total=35)
                )
                
                if get_work_response.status != 200:
                    logger.error(f"Failed to get work: {get_work_response.status}")
                    await asyncio.sleep(1)
                    continue
                
                work_data = await get_work_response.json()
                requests = work_data.get("requests", [])
                
                if not requests:
                    logger.debug("No work available")
                    await asyncio.sleep(1)
                    continue
                
                logger.info(f"Processing {len(requests)} requests with {vllm_engine.engine_type} engine")
                
                # Process each request using the vLLM engine
                results = []
                for req in requests:
                    request_id = req.get("request_id")
                    prompt = req.get("prompt")
                    request_type = req.get("request_type", "generate")
                    model = req.get("model", "default")
                    
                    logger.debug(f"Processing request {request_id}: type={request_type}")
                    
                    try:
                        if request_type == "embed":
                            # Generate embeddings using vLLM engine
                            embedding = await vllm_engine.generate_embeddings(prompt, model)
                            
                            if embedding and len(embedding) > 0:
                                response = {
                                    "embedding": embedding,
                                    "dimensions": len(embedding)
                                }
                                logger.debug(f"✓ Embedding generated for request {request_id} (dim: {len(embedding)})")
                            else:
                                response = {"error": "Empty embedding returned"}
                                logger.error(f"✗ Empty embedding for request {request_id}")
                        
                        elif request_type == "generate":
                            # Generate completion using vLLM engine
                            completion = await vllm_engine.generate_completion(
                                prompt=prompt,
                                model=model,
                                max_tokens=req.get("max_tokens", 100),
                                temperature=req.get("temperature", 1.0),
                                top_p=req.get("top_p", 1.0)
                            )
                            
                            if completion:
                                response = completion
                                logger.debug(f"✓ Completion generated for request {request_id}")
                            else:
                                response = {"error": "Empty completion returned"}
                                logger.error(f"✗ Empty completion for request {request_id}")
                        
                        else:
                            response = {"error": f"Unknown request type: {request_type}"}
                            logger.error(f"Unknown request type: {request_type}")
                    
                    except Exception as e:
                        response = {"error": str(e)}
                        logger.error(f"✗ Exception processing {request_type} request {request_id}: {e}")
                    
                    # Format response for finish_work endpoint
                    if isinstance(response, dict):
                        if "embedding" in response:
                            # For embeddings, send the embedding vector as the response
                            results.append({
                                "request_id": request_id,
                                "response": response["embedding"],
                                "error": response.get("error"),
                                "token_count": 10,  # Mock token count
                                "flop_count": 1000  # Mock flop count
                            })
                        elif "error" in response:
                            # For errors
                            results.append({
                                "request_id": request_id,
                                "response": None,
                                "error": response["error"],
                                "token_count": 0,
                                "flop_count": 0
                            })
                        else:
                            # Unexpected dict format, treat as error
                            results.append({
                                "request_id": request_id,
                                "response": None,
                                "error": f"Unexpected response format: {response}",
                                "token_count": 0,
                                "flop_count": 0
                            })
                    else:
                        # For text completions and other string responses
                        results.append({
                            "request_id": request_id,
                            "response": response,
                            "error": None,
                            "token_count": 10,  # Mock token count
                            "flop_count": 1000  # Mock flop count
                        })
                
                # Send results back to ScalarLM API
                if results:
                    logger.info(f"Sending {len(results)} results back to API")
                    
                    finish_response = await session.post(
                        f"{api_base}/v1/generate/finish_work",
                        json={"requests": results},
                        timeout=aiohttp.ClientTimeout(total=10)
                    )
                    
                    if finish_response.status == 200:
                        logger.debug(f"✓ Successfully submitted {len(results)} results")
                    else:
                        logger.error(f"✗ Failed to submit results: {finish_response.status}")
                
            except asyncio.TimeoutError:
                logger.debug("Timeout waiting for work")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)
    
    finally:
        # Cleanup resources
        if session and not session.closed:
            await session.close()
        
        if vllm_engine:
            await vllm_engine.cleanup()
        
        logger.info("Embedding worker shutting down")


# Backward compatibility function
async def create_embedding_worker_legacy(running_status):
    """
    Legacy embedding worker for backward compatibility.
    This preserves the original HTTP-only implementation.
    """
    logger.warning("Using legacy embedding worker - consider upgrading to create_embedding_worker")
    
    # Import the original implementation
    from .create_embedding_worker import create_embedding_worker as original_worker
    return await original_worker(running_status)
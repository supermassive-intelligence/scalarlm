"""
Embedding worker that processes requests from the queue and sends them to vLLM
"""

import asyncio
import aiohttp
import logging
import json

logger = logging.getLogger(__name__)

async def create_embedding_worker(running_status):
    """
    Create a worker that processes embedding requests from the queue
    """
    logger.info("="*60)
    logger.info("STARTING EMBEDDING WORKER")
    logger.info("="*60)
    
    api_base = "http://localhost:8000"
    vllm_base = "http://localhost:8001"
    
    logger.info(f"Embedding worker configuration:")
    logger.info(f"  API endpoint: {api_base}")
    logger.info(f"  vLLM endpoint: {vllm_base}")
    
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # Get work from the queue
                logger.debug("Checking for embedding work...")
                
                get_work_response = await session.post(
                    f"{api_base}/v1/generate/get_work",
                    json={"batch_size": 10},
                    timeout=aiohttp.ClientTimeout(total=35)  # Increased timeout to match get_work's 30s wait
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
                
                logger.info(f"Processing {len(requests)} embedding requests")
                
                # Process each request
                results = []
                for req in requests:
                    request_id = req.get("request_id")
                    prompt = req.get("prompt")
                    request_type = req.get("request_type", "generate")
                    
                    logger.info(f"Processing request {request_id}: type={request_type}")
                    
                    if request_type == "embed":
                        # Send to separate embedding service
                        try:
                            from cray_infra.util.get_config import get_config
                            config = get_config()
                            embedding_service_url = config.get("embedding_service_url", "http://localhost:8002")
                            
                            embed_response = await session.post(
                                f"{embedding_service_url}/v1/embeddings",
                                json={
                                    "input": [prompt],  # embedding service expects list
                                    "model": req.get("model", config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
                                },
                                timeout=aiohttp.ClientTimeout(total=30)
                            )
                            
                            if embed_response.status == 200:
                                embed_data = await embed_response.json()
                                # Extract the embedding vector
                                embeddings = embed_data.get("data", [])
                                if embeddings:
                                    embedding_vector = embeddings[0].get("embedding", [])
                                    response = {
                                        "embedding": embedding_vector,
                                        "dimensions": len(embedding_vector)
                                    }
                                else:
                                    response = {"error": "No embedding returned"}
                                    
                                logger.info(f"✓ Real embedding generated for request {request_id}")
                            else:
                                response = {"error": f"Embedding service returned status {embed_response.status}"}
                                logger.error(f"✗ Embedding service failed for request {request_id}")
                                
                        except Exception as e:
                            response = {"error": str(e)}
                            logger.error(f"✗ Exception processing embedding {request_id}: {e}")
                            
                    elif request_type == "generate":
                        # Send to vLLM completions endpoint
                        try:
                            completion_response = await session.post(
                                f"{vllm_base}/v1/completions",
                                json={
                                    "prompt": prompt,
                                    "model": req.get("model", "default"),
                                    "max_tokens": req.get("max_tokens", 100)
                                },
                                timeout=aiohttp.ClientTimeout(total=30)
                            )
                            
                            if completion_response.status == 200:
                                completion_data = await completion_response.json()
                                choices = completion_data.get("choices", [])
                                if choices:
                                    response = choices[0].get("text", "")
                                else:
                                    response = {"error": "No completion returned"}
                                    
                                logger.info(f"✓ Completion generated for request {request_id}")
                            else:
                                response = {"error": f"vLLM returned status {completion_response.status}"}
                                logger.error(f"✗ vLLM completion failed for request {request_id}")
                                
                        except Exception as e:
                            response = {"error": str(e)}
                            logger.error(f"✗ Exception processing completion {request_id}: {e}")
                    else:
                        response = {"error": f"Unknown request type: {request_type}"}
                        logger.error(f"Unknown request type: {request_type}")
                    
                    # Format response correctly for finish_work endpoint
                    if isinstance(response, dict) and "embedding" in response:
                        # For embeddings, send the embedding vector as the response
                        results.append({
                            "request_id": request_id,
                            "response": response["embedding"],
                            "error": None,
                            "token_count": 10,  # Mock token count
                            "flop_count": 1000  # Mock flop count
                        })
                    elif isinstance(response, dict) and "error" in response:
                        # For errors
                        results.append({
                            "request_id": request_id,
                            "response": None,
                            "error": response["error"],
                            "token_count": 0,
                            "flop_count": 0
                        })
                    else:
                        # For text completions
                        results.append({
                            "request_id": request_id,
                            "response": response,
                            "error": None,
                            "token_count": 10,
                            "flop_count": 1000
                        })
                
                # Send results back
                if results:
                    logger.info(f"Sending {len(results)} results back to API")
                    
                    finish_response = await session.post(
                        f"{api_base}/v1/generate/finish_work",
                        json={"requests": results},
                        timeout=aiohttp.ClientTimeout(total=10)
                    )
                    
                    if finish_response.status == 200:
                        logger.info(f"✓ Successfully submitted {len(results)} results")
                    else:
                        logger.error(f"✗ Failed to submit results: {finish_response.status}")
                
            except asyncio.TimeoutError:
                logger.debug("Timeout waiting for work")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)
    
    logger.info("Embedding worker shutting down")
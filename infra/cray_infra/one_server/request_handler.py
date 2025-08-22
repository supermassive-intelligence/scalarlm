"""
Universal worker that processes all types of requests from the queue.

This replaces the embedding worker and handles all request types:
- generate: sent to vLLM completions endpoint
- Any future request types can be added here
"""

import asyncio
import aiohttp
import logging
import json

logger = logging.getLogger(__name__)

async def create_request_handler(running_status):
    """
    Create a request handler that processes all request types from the queue.
    
    This worker:
    1. Polls /v1/generate/get_work for available requests
    2. Routes them based on request_type:
       - generate: processes through vLLM completions
       - (future types can be added here)
    3. Returns results via /v1/generate/finish_work
    """
    logger.info("="*60)
    logger.info("STARTING REQUEST HANDLER")
    logger.info("="*60)
    
    api_base = "http://localhost:8000"
    vllm_base = "http://localhost:8001"
    
    logger.info(f"Generate worker configuration:")
    logger.info(f"  API endpoint: {api_base}")
    logger.info(f"  vLLM endpoint: {vllm_base}")
    
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # Get work from the queue
                logger.debug("Checking for generate work...")
                
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
                
                logger.info(f"Processing {len(requests)} requests of various types")
                
                # Process each request
                finish_work_requests = []
                for request in requests:
                    result = await process_generate_request(session, request, vllm_base)
                    finish_work_requests.append(result)
                
                # Submit all results via finish_work
                if finish_work_requests:
                    await submit_finish_work(session, api_base, finish_work_requests)
                    
            except asyncio.TimeoutError:
                logger.debug("Get work timeout - continuing...")
                continue
            except Exception as e:
                logger.error(f"Error in generate worker: {e}")
                await asyncio.sleep(1)
                continue

async def process_generate_request(session, request, vllm_base):
    """Process a single generate request through vLLM."""
    
    request_id = request.get("request_id")
    prompt = request.get("prompt")
    model = request.get("model")
    max_tokens = request.get("max_tokens", 100)
    request_type = request.get("request_type", "generate")
    
    logger.info(f"Processing request {request_id}: type={request_type}")
    
    try:
        if request_type == "generate":
            # Send to vLLM for generation
            vllm_payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "model": model,
                "stream": False
            }
            
            logger.debug(f"Sending to vLLM: {vllm_payload}")
            
            vllm_response = await session.post(
                f"{vllm_base}/v1/completions",
                json=vllm_payload,
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout for generation
            )
            
            if vllm_response.status == 200:
                vllm_data = await vllm_response.json()
                
                # Extract the generated text
                choices = vllm_data.get("choices", [])
                if choices:
                    generated_text = choices[0].get("text", "")
                    
                    logger.info(f"✓ Request {request_id} completed successfully")
                    return {
                        "request_id": int(request_id),
                        "response": generated_text,
                        "error": None,
                        "token_count": None,
                        "flop_count": None
                    }
                else:
                    logger.error(f"No choices returned from vLLM for {request_id}")
                    return {
                        "request_id": int(request_id),
                        "response": None,
                        "error": "No choices returned from vLLM",
                        "token_count": None,
                        "flop_count": None
                    }
            else:
                error_text = await vllm_response.text()
                logger.error(f"vLLM error for {request_id}: {vllm_response.status} - {error_text}")
                
                return {
                    "request_id": int(request_id),
                    "response": None,
                    "error": f"vLLM error: {error_text}",
                    "token_count": None,
                    "flop_count": None
                }
        else:
            logger.warning(f"Unknown request type: {request_type} - add support in universal worker")
            return {
                "request_id": int(request_id),
                "response": None,
                "error": f"Unsupported request type: {request_type}",
                "token_count": None,
                "flop_count": None
            }
            
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {e}")
        
        return {
            "request_id": int(request_id),
            "response": None,
            "error": f"Processing error: {str(e)}",
            "token_count": None,
            "flop_count": None
        }

async def submit_finish_work(session, api_base, finish_work_requests):
    """Submit results via the finish_work endpoint."""
    try:
        finish_payload = {
            "requests": finish_work_requests
        }
        
        finish_response = await session.post(
            f"{api_base}/v1/generate/finish_work",
            json=finish_payload,
            timeout=aiohttp.ClientTimeout(total=10)
        )
        
        if finish_response.status == 200:
            logger.info(f"✓ Successfully submitted {len(finish_work_requests)} results")
        else:
            logger.error(f"Failed to submit finish_work: {finish_response.status}")
            error_text = await finish_response.text()
            logger.error(f"Error details: {error_text}")
            
    except Exception as e:
        logger.error(f"Exception submitting finish_work: {e}")
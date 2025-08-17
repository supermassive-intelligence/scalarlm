"""
Shared engine access for connecting to existing vLLM AsyncLLMEngine instances.
Allows direct access to the engine without creating a new instance.
"""

import logging
import aiohttp
from typing import List, Dict, Any, Optional
from .engine_interface import VLLMEngineInterface

logger = logging.getLogger(__name__)


class SharedVLLMEngine(VLLMEngineInterface):
    """
    Connects to an existing vLLM server's shared engine instance.
    Uses the /direct/* endpoints for direct engine access.
    """
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize connection to shared vLLM engine.
        
        Args:
            base_url: URL of the vLLM server with shared engine endpoints
        """
        self.base_url = base_url.rstrip('/')
        self.session = None
        
    async def _get_session(self):
        """Get or create aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
        
    async def generate_embeddings(self, prompt: str, model: str) -> List[float]:
        """Generate embeddings using direct shared engine access."""
        try:
            session = await self._get_session()
            
            payload = {
                "input": prompt,
                "model": model,
                "encoding_format": "float"
            }
            
            # Use direct endpoint for shared engine access
            async with session.post(
                f"{self.base_url}/direct/embeddings",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["data"][0]["embedding"]
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"Shared engine embeddings failed: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error in shared engine embedding generation: {e}")
            raise RuntimeError(f"Shared engine embedding generation failed: {e}")
    
    async def generate_completion(
        self, 
        prompt: str, 
        model: str, 
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate completion using shared engine (if endpoint available)."""
        # For now, fallback to HTTP since shared completion endpoint needs implementation
        try:
            session = await self._get_session()
            
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens or 100,
                "temperature": kwargs.get('temperature', 1.0),
                "top_p": kwargs.get('top_p', 1.0),
                "stream": False
            }
            
            async with session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["text"]
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"Shared engine completion failed: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error in shared engine completion generation: {e}")
            raise RuntimeError(f"Shared engine completion generation failed: {e}")
    
    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate chat completion using shared engine (if endpoint available)."""
        try:
            session = await self._get_session()
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens or 100,
                "temperature": kwargs.get('temperature', 1.0),
                "top_p": kwargs.get('top_p', 1.0),
                "stream": False
            }
            
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"Shared engine chat completion failed: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error in shared engine chat completion generation: {e}")
            raise RuntimeError(f"Shared engine chat completion generation failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if shared engine is available."""
        try:
            session = await self._get_session()
            
            async with session.get(
                f"{self.base_url}/direct/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("shared_engine", False)
                else:
                    return False
                    
        except Exception as e:
            logger.warning(f"Shared engine health check failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
    
    @property
    def engine_type(self) -> str:
        return "shared"
    
    def __repr__(self):
        return f"SharedVLLMEngine(base_url={self.base_url})"
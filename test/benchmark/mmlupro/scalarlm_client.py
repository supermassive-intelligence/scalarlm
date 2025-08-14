"""
ScalarLM client for MMLU-Pro evaluation
"""

import time
import logging
from typing import List, Tuple, Optional
import scalarlm
from utils import ScalarLMError

logger = logging.getLogger(__name__)


class ScalarLMClient:
    """Client for interacting with ScalarLM API"""
    
    def __init__(self, api_url: str = "http://localhost:8000", model_name: str = ""):
        """
        Initialize ScalarLM client
        
        Args:
            api_url: ScalarLM API URL
            model_name: Default model name to use
        """
        self.api_url = api_url
        self.model_name = model_name
        self.llm = None
        self._init_client()
    
    def _init_client(self):
        """Initialize ScalarLM connection"""
        logger.info("Initializing ScalarLM client...")
        
        # Set API URL
        if self.api_url:
            scalarlm.api_url = self.api_url
            logger.info(f"ScalarLM API URL set to: {self.api_url}")
        
        self.llm = scalarlm.SupermassiveIntelligence()
        
        # Check health
        try:
            health_status = self.llm.health()
            logger.info(f"ScalarLM health check: {health_status}")
        except Exception as e:
            logger.error(f"ScalarLM health check failed: {e}")
            logger.error(f"Current API URL: {scalarlm.api_url}")
            logger.error("Please check:")
            logger.error("1. Is the ScalarLM server running?")
            logger.error("2. Is the API URL correct?")
            logger.error(f"3. Try: curl {self.api_url}/v1/health")
            raise ScalarLMError(f"Failed to initialize ScalarLM: {e}") from e
    
    def generate_single(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> Tuple[str, float]:
        """
        Generate response for a single prompt
        
        Args:
            prompt: Input prompt
            model_name: Optional model name override
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (response, latency)
        """
        start_time = time.time()
        
        # Prepare generate kwargs
        generate_kwargs = {"prompts": [prompt]}
        
        # Add model name if specified
        name = model_name or self.model_name
        if name and name.strip():
            generate_kwargs["model_name"] = name
        
        # Add generation parameters (only supported ones)
        if 'max_tokens' in kwargs:
            generate_kwargs['max_tokens'] = kwargs['max_tokens']
        
        try:
            responses = self.llm.generate(**generate_kwargs)
            response = responses[0] if responses else ""
        except Exception as e:
            logger.error(f"Single generation failed: {e}")
            raise ScalarLMError(f"Generation failed: {e}") from e
        
        latency = time.time() - start_time
        return response, latency
    
    def generate_batch(self, prompts: List[str], model_name: Optional[str] = None, **kwargs) -> List[Tuple[str, float]]:
        """
        Generate responses for a batch of prompts
        
        Args:
            prompts: List of input prompts
            model_name: Optional model name override
            **kwargs: Additional generation parameters
            
        Returns:
            List of (response, latency) tuples
        """
        if not prompts:
            return []
        
        start_time = time.time()
        
        # Prepare generate kwargs
        generate_kwargs = {"prompts": prompts}
        
        # Add model name if specified
        name = model_name or self.model_name
        if name and name.strip():
            generate_kwargs["model_name"] = name
        
        # Add generation parameters (only supported ones)
        if 'max_tokens' in kwargs:
            generate_kwargs['max_tokens'] = kwargs['max_tokens']
        
        try:
            responses = self.llm.generate(**generate_kwargs)
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            raise ScalarLMError(f"Batch generation failed: {e}") from e
        
        batch_latency = time.time() - start_time
        
        # Return empty responses if generation failed
        if not responses:
            return [("", 0.0)] * len(prompts)
        
        # Calculate per-prompt latency
        avg_latency = batch_latency / len(prompts)
        return [(response, avg_latency) for response in responses]
    
    def health_check(self) -> bool:
        """
        Check if ScalarLM is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            self.llm.health()
            return True
        except Exception:
            return False
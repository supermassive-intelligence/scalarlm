"""
Abstract interface for vLLM engine access.
Supports both HTTP-based and direct method call approaches.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class VLLMEngineInterface(ABC):
    """Abstract interface for vLLM engine operations."""
    
    @abstractmethod
    async def generate_embeddings(self, prompt: str, model: str) -> List[float]:
        """
        Generate embeddings for the given prompt.
        
        Args:
            prompt: Input text to embed
            model: Model name to use
            
        Returns:
            List of embedding values
        """
        pass
    
    @abstractmethod
    async def generate_completion(
        self, 
        prompt: str, 
        model: str, 
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text completion for the given prompt.
        
        Args:
            prompt: Input text prompt
            model: Model name to use
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text completion
        """
        pass
    
    @abstractmethod
    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion for the given messages.
        
        Args:
            messages: List of chat messages
            model: Model name to use
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated chat response
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the engine is healthy and responsive.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Clean up any resources used by the engine."""
        pass
    
    @property
    @abstractmethod
    def engine_type(self) -> str:
        """Return the type of engine (http, direct)."""
        pass
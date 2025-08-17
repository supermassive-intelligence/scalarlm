#!/usr/bin/env python3
"""
Standalone embedding service using sentence-transformers.
This service provides embeddings separate from the vLLM generation service.
"""

import logging
import sys
import os
from typing import List, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    logger.info("✓ sentence-transformers imported successfully")
except ImportError:
    logger.error("✗ sentence-transformers not installed. Install with: pip install sentence-transformers")
    sys.exit(1)


class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # For CPU inference, we can use a smaller, faster model
            if "all-MiniLM-L6-v2" in self.model_name:
                actual_model = "all-MiniLM-L6-v2"
            else:
                actual_model = self.model_name
                
            self.model = SentenceTransformer(actual_model)
            
            # Force CPU usage for consistency
            if torch.cuda.is_available():
                logger.info("CUDA available but using CPU for consistency with vLLM setup")
            
            self.model = self.model.to('cpu')
            logger.info(f"✓ Embedding model loaded successfully: {actual_model}")
            
        except Exception as e:
            logger.error(f"✗ Failed to load embedding model {self.model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            if not self.model:
                raise RuntimeError("Model not loaded")
            
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            
            # Convert to list of lists for JSON serialization
            if len(embeddings.shape) == 1:
                # Single embedding
                return [embeddings.tolist()]
            else:
                # Multiple embeddings
                return embeddings.tolist()
                
        except Exception as e:
            logger.error(f"✗ Failed to generate embeddings: {e}")
            raise


# Global embedding service instance
embedding_service = None

# FastAPI app
app = FastAPI(title="ScalarLM Embedding Service", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """Initialize the embedding service on startup."""
    global embedding_service
    try:
        model_name = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        embedding_service = EmbeddingService(model_name)
        logger.info("✓ Embedding service initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize embedding service: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if embedding_service and embedding_service.model:
        return {"status": "healthy", "model": embedding_service.model_name}
    else:
        raise HTTPException(status_code=503, detail="Service not ready")


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings for the given input texts."""
    try:
        if not embedding_service:
            raise HTTPException(status_code=503, detail="Embedding service not initialized")
        
        if not request.input:
            raise HTTPException(status_code=400, detail="Input texts cannot be empty")
        
        # Generate embeddings
        embeddings = embedding_service.generate_embeddings(request.input)
        
        # Format response according to OpenAI API format
        data = []
        for i, embedding in enumerate(embeddings):
            data.append({
                "object": "embedding",
                "embedding": embedding,
                "index": i
            })
        
        response = EmbeddingResponse(
            data=data,
            model=embedding_service.model_name,
            usage={
                "prompt_tokens": sum(len(text.split()) for text in request.input),
                "total_tokens": sum(len(text.split()) for text in request.input)
            }
        )
        
        logger.info(f"✓ Generated {len(embeddings)} embeddings successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/models")
async def list_models():
    """List available models."""
    if embedding_service:
        return {
            "object": "list",
            "data": [
                {
                    "id": embedding_service.model_name,
                    "object": "model",
                    "owned_by": "sentence-transformers"
                }
            ]
        }
    else:
        return {"object": "list", "data": []}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting embedding service on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
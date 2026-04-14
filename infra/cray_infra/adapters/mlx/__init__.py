"""
MLX adapter system for ScalarLM.

This module provides adapter functionality for Apple Silicon using MLX,
enabling ScalarLM's Tokenformer checkpoint loading on Apple Silicon hardware.
"""

from .mlx_adapter import MLXTokenformerAdapter, MLXAdapterManager
from .checkpoint_converter import PyTorchToMLXConverter
from .model_manager import MLXModelManager

__all__ = [
    "MLXTokenformerAdapter",
    "MLXAdapterManager",
    "PyTorchToMLXConverter",
    "MLXModelManager",
]

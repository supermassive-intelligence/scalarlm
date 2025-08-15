"""
MMLU-Pro Benchmark implementation for ScalarLM

This module provides comprehensive evaluation capabilities for the MMLU-Pro benchmark,
including dataset loading, prompt formatting, evaluation, and metrics computation.
"""

from .evaluator import MMLUProEvaluator
from .dataset import MMLUProDataset
from .metrics import MMLUProMetrics
from .config import MMLUProConfig

__version__ = "0.1.0"

__all__ = [
    "MMLUProEvaluator",
    "MMLUProDataset",
    "MMLUProMetrics",
    "MMLUProConfig",
]
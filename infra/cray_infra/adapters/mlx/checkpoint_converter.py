"""
PyTorch to MLX checkpoint converter.

Handles conversion of ScalarLM's PyTorch checkpoints (.pt files) to MLX arrays,
enabling trained models to run on Apple Silicon.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PyTorchToMLXConverter:
    """
    Converts PyTorch checkpoint files to MLX format.

    This handles the conversion of ScalarLM training checkpoints (PyTorch tensors)
    to MLX arrays that can be loaded into MLX models on Apple Silicon.
    """

    def __init__(self, dtype: str = "float32"):
        """
        Initialize converter.

        Args:
            dtype: Target MLX dtype (float32, float16, bfloat16)
        """
        self.dtype = dtype

    def convert_checkpoint(
        self,
        checkpoint_path: str,
        filter_keys: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Convert a PyTorch checkpoint to MLX arrays.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            filter_keys: Optional list of key substrings to filter (e.g., ["tokenformer", "lm_head"])

        Returns:
            Dictionary mapping parameter names to MLX arrays
        """
        try:
            import torch
            import mlx.core as mx
        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            raise

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading PyTorch checkpoint from {checkpoint_path}")

        # Load PyTorch checkpoint
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            # Fallback for older PyTorch versions
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Handle nested checkpoint structures
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        logger.info(f"Checkpoint contains {len(state_dict)} parameters")

        # Convert to MLX
        mlx_weights = {}
        converted_count = 0

        for key, value in state_dict.items():
            # Filter if requested
            if filter_keys is not None:
                if not any(filter_key in key for filter_key in filter_keys):
                    continue

            # Convert PyTorch tensor to numpy to MLX
            try:
                numpy_array = value.detach().cpu().numpy()

                # Convert to target dtype
                mlx_array = mx.array(numpy_array)

                # Apply dtype conversion if needed
                if self.dtype == "float16" and mlx_array.dtype != mx.float16:
                    mlx_array = mlx_array.astype(mx.float16)
                elif self.dtype == "bfloat16" and mlx_array.dtype != mx.bfloat16:
                    mlx_array = mlx_array.astype(mx.bfloat16)
                elif self.dtype == "float32" and mlx_array.dtype != mx.float32:
                    mlx_array = mlx_array.astype(mx.float32)

                mlx_weights[key] = mlx_array
                converted_count += 1

                logger.debug(f"Converted {key}: shape={mlx_array.shape}, dtype={mlx_array.dtype}")

            except Exception as e:
                logger.warning(f"Failed to convert {key}: {e}")
                continue

        logger.info(f"Successfully converted {converted_count} parameters to MLX")

        return mlx_weights

    def save_mlx_checkpoint(self, mlx_weights: Dict[str, Any], output_path: str):
        """
        Save MLX weights to a file.

        Args:
            mlx_weights: Dictionary of MLX arrays
            output_path: Output file path (.npz format)
        """
        try:
            import mlx.core as mx
        except ImportError as e:
            logger.error(f"MLX not available: {e}")
            raise

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving MLX checkpoint to {output_path}")

        # Convert MLX arrays to numpy for saving
        numpy_weights = {
            key: np.array(value) for key, value in mlx_weights.items()
        }

        np.savez(output_path, **numpy_weights)
        logger.info(f"Saved {len(numpy_weights)} parameters")

    def load_mlx_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load MLX weights from a saved file.

        Args:
            checkpoint_path: Path to .npz checkpoint file

        Returns:
            Dictionary of MLX arrays
        """
        try:
            import mlx.core as mx
        except ImportError as e:
            logger.error(f"MLX not available: {e}")
            raise

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading MLX checkpoint from {checkpoint_path}")

        numpy_weights = np.load(checkpoint_path)

        mlx_weights = {
            key: mx.array(value) for key, value in numpy_weights.items()
        }

        logger.info(f"Loaded {len(mlx_weights)} parameters")

        return mlx_weights

    @staticmethod
    def verify_conversion(
        pytorch_checkpoint: str,
        mlx_weights: Dict[str, Any],
        tolerance: float = 1e-5
    ) -> bool:
        """
        Verify that conversion from PyTorch to MLX preserves values.

        Args:
            pytorch_checkpoint: Original PyTorch checkpoint path
            mlx_weights: Converted MLX weights
            tolerance: Numerical tolerance for comparison

        Returns:
            True if conversion is accurate within tolerance
        """
        try:
            import torch
            import mlx.core as mx
        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            return False

        try:
            state_dict = torch.load(pytorch_checkpoint, map_location="cpu", weights_only=True)
        except Exception:
            state_dict = torch.load(pytorch_checkpoint, map_location="cpu")

        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        mismatches = []

        for key in mlx_weights.keys():
            if key not in state_dict:
                logger.warning(f"Key {key} not in original checkpoint")
                continue

            pytorch_value = state_dict[key].detach().cpu().numpy()
            mlx_value = np.array(mlx_weights[key])

            max_diff = np.abs(pytorch_value - mlx_value).max()

            if max_diff > tolerance:
                mismatches.append((key, max_diff))
                logger.warning(f"Mismatch in {key}: max_diff={max_diff}")

        if mismatches:
            logger.error(f"Found {len(mismatches)} mismatches in conversion")
            return False

        logger.info("Conversion verification passed")
        return True

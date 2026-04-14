"""
MLX Tokenformer adapter for ScalarLM.

Provides adapter functionality similar to the vLLM adapter but for MLX models,
enabling loading of trained checkpoints on Apple Silicon.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from collections import OrderedDict

from ..common.adapter_commons import AdapterModel, AdapterModelManager
from .checkpoint_converter import PyTorchToMLXConverter

logger = logging.getLogger(__name__)


class MLXTokenformerModel(AdapterModel):
    """
    Represents a Tokenformer model in MLX format.

    This wraps the converted MLX weights for a trained checkpoint.
    """

    def __init__(self, mlx_weights: Dict[str, Any], model_id: Optional[str] = None):
        """
        Initialize MLX Tokenformer model.

        Args:
            mlx_weights: Dictionary mapping parameter names to MLX arrays
            model_id: Unique identifier for this model
        """
        super().__init__(model_id or str(hash(str(mlx_weights.keys()))))
        self.mlx_weights = mlx_weights

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        dtype: str = "float32",
        model_id: Optional[str] = None
    ) -> "MLXTokenformerModel":
        """
        Load a Tokenformer model from a PyTorch checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            dtype: Target dtype for MLX arrays
            model_id: Optional model identifier

        Returns:
            MLXTokenformerModel instance
        """
        converter = PyTorchToMLXConverter(dtype=dtype)

        # Filter for tokenformer and lm_head weights
        mlx_weights = converter.convert_checkpoint(
            checkpoint_path,
            filter_keys=["tokenformer", "lm_head"]
        )

        if not mlx_weights:
            raise ValueError(f"No tokenformer weights found in {checkpoint_path}")

        return cls(mlx_weights, model_id=model_id)


class MLXTokenformerAdapter:
    """
    Adapter that applies Tokenformer weights to MLX models.

    This is analogous to the vLLM TokenformerAdapter but works with MLX's
    model representation. It handles weight application and restoration.
    """

    def __init__(self, mlx_model: Any):
        """
        Initialize adapter for an MLX model.

        Args:
            mlx_model: The base MLX model (from mlx-lm or mlx-vlm)
        """
        self.mlx_model = mlx_model
        self.original_weights = self._extract_original_weights()
        logger.info(f"Initialized MLXTokenformerAdapter with {len(self.original_weights)} original weights")

    def _extract_original_weights(self) -> Dict[str, Any]:
        """
        Extract and save original model weights (especially lm_head).

        Returns:
            Dictionary of original weights
        """
        try:
            import mlx.core as mx
        except ImportError:
            logger.error("MLX not available")
            raise

        original = {}

        # For mlx-lm models, weights are typically in model.layers
        # We need to extract lm_head weights
        if hasattr(self.mlx_model, "model"):
            model = self.mlx_model.model
        else:
            model = self.mlx_model

        # Try to get lm_head or similar output layer
        if hasattr(model, "lm_head"):
            lm_head = model.lm_head
            if hasattr(lm_head, "weight"):
                original["lm_head.weight"] = lm_head.weight
            if hasattr(lm_head, "bias"):
                original["lm_head.bias"] = lm_head.bias

        logger.info(f"Extracted {len(original)} original weight tensors")

        return original

    def apply_tokenformer_weights(self, tokenformer_model: MLXTokenformerModel) -> bool:
        """
        Apply Tokenformer weights to the MLX model.

        Args:
            tokenformer_model: The tokenformer model with converted weights

        Returns:
            True if successful
        """
        try:
            import mlx.core as mx
        except ImportError:
            logger.error("MLX not available")
            return False

        logger.info(f"Applying {len(tokenformer_model.mlx_weights)} tokenformer weights")

        # Get the underlying model
        if hasattr(self.mlx_model, "model"):
            model = self.mlx_model.model
        else:
            model = self.mlx_model

        success_count = 0
        failed_keys = []

        for key, value in tokenformer_model.mlx_weights.items():
            try:
                # Parse the key to navigate the model structure
                # Example: "model.lm_head.weight" -> model.lm_head.weight
                parts = key.split(".")

                # Navigate to the parameter
                obj = model
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        logger.warning(f"Could not find {part} in model for key {key}")
                        failed_keys.append(key)
                        continue

                # Set the final parameter
                param_name = parts[-1]
                if hasattr(obj, param_name):
                    setattr(obj, param_name, value)
                    success_count += 1
                    logger.debug(f"Applied weight {key}")
                else:
                    logger.warning(f"Could not find parameter {param_name} for key {key}")
                    failed_keys.append(key)

            except Exception as e:
                logger.error(f"Failed to apply weight {key}: {e}")
                failed_keys.append(key)

        if failed_keys:
            logger.warning(f"Failed to apply {len(failed_keys)} weights: {failed_keys[:5]}")

        logger.info(f"Successfully applied {success_count} tokenformer weights")
        return success_count > 0

    def restore_original_weights(self) -> bool:
        """
        Restore original model weights (remove tokenformer weights).

        Returns:
            True if successful
        """
        try:
            import mlx.core as mx
        except ImportError:
            logger.error("MLX not available")
            return False

        logger.info("Restoring original weights")

        # Get the underlying model
        if hasattr(self.mlx_model, "model"):
            model = self.mlx_model.model
        else:
            model = self.mlx_model

        for key, value in self.original_weights.items():
            try:
                parts = key.split(".")
                obj = model

                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)

                param_name = parts[-1]
                if hasattr(obj, param_name):
                    setattr(obj, param_name, value)

            except Exception as e:
                logger.error(f"Failed to restore weight {key}: {e}")

        logger.info("Original weights restored")
        return True


class MLXAdapterManager(AdapterModelManager):
    """
    Manages multiple Tokenformer adapters for MLX models.

    Similar to the vLLM TokenformerManager but for MLX. Handles:
    - Loading adapters from checkpoints
    - Swapping between adapters
    - LRU cache management
    """

    def __init__(
        self,
        mlx_model: Any,
        capacity: int = 4,
        dtype: str = "float32"
    ):
        """
        Initialize the adapter manager.

        Args:
            mlx_model: The base MLX model
            capacity: Maximum number of adapters to cache
            dtype: Data type for MLX arrays
        """
        self.mlx_model = mlx_model
        self.capacity = capacity
        self.dtype = dtype
        self.adapter = MLXTokenformerAdapter(mlx_model)

        self._registered_adapters: Dict[str, MLXTokenformerModel] = {}
        self._active_adapter: Optional[str] = None
        self._lru_order: List[str] = []

        logger.info(f"Initialized MLXAdapterManager with capacity {capacity}")

    def register_adapter(
        self,
        adapter_id: str,
        checkpoint_path: str
    ) -> bool:
        """
        Register a new adapter from a checkpoint.

        Args:
            adapter_id: Unique identifier for this adapter (typically job_hash)
            checkpoint_path: Path to .pt checkpoint file

        Returns:
            True if successful
        """
        try:
            logger.info(f"Registering adapter {adapter_id} from {checkpoint_path}")

            # Load the tokenformer model
            tokenformer_model = MLXTokenformerModel.from_checkpoint(
                checkpoint_path,
                dtype=self.dtype,
                model_id=adapter_id
            )

            # Check capacity and evict if needed
            if len(self._registered_adapters) >= self.capacity:
                self._evict_lru_adapter()

            self._registered_adapters[adapter_id] = tokenformer_model
            self._update_lru(adapter_id)

            logger.info(f"Adapter {adapter_id} registered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to register adapter {adapter_id}: {e}")
            return False

    def activate_adapter(self, adapter_id: str) -> bool:
        """
        Activate an adapter (apply its weights to the model).

        Args:
            adapter_id: Identifier of adapter to activate

        Returns:
            True if successful
        """
        if adapter_id not in self._registered_adapters:
            logger.error(f"Adapter {adapter_id} not registered")
            return False

        # If already active, nothing to do
        if self._active_adapter == adapter_id:
            logger.debug(f"Adapter {adapter_id} already active")
            self._update_lru(adapter_id)
            return True

        # Restore original weights first if another adapter is active
        if self._active_adapter is not None:
            logger.info(f"Deactivating current adapter {self._active_adapter}")
            self.adapter.restore_original_weights()

        # Apply new adapter
        logger.info(f"Activating adapter {adapter_id}")
        tokenformer_model = self._registered_adapters[adapter_id]
        success = self.adapter.apply_tokenformer_weights(tokenformer_model)

        if success:
            self._active_adapter = adapter_id
            self._update_lru(adapter_id)

        return success

    def deactivate_adapter(self) -> bool:
        """
        Deactivate current adapter (restore original weights).

        Returns:
            True if successful
        """
        if self._active_adapter is None:
            logger.debug("No adapter currently active")
            return True

        logger.info(f"Deactivating adapter {self._active_adapter}")
        success = self.adapter.restore_original_weights()

        if success:
            self._active_adapter = None

        return success

    def get_registered_adapters(self) -> List[str]:
        """
        Get list of registered adapter IDs.

        Returns:
            List of adapter identifiers
        """
        return list(self._registered_adapters.keys())

    def get_active_adapter(self) -> Optional[str]:
        """
        Get currently active adapter ID.

        Returns:
            Active adapter ID or None
        """
        return self._active_adapter

    def _update_lru(self, adapter_id: str):
        """Update LRU order for an adapter."""
        if adapter_id in self._lru_order:
            self._lru_order.remove(adapter_id)
        self._lru_order.append(adapter_id)

    def _evict_lru_adapter(self):
        """Evict least recently used adapter."""
        if not self._lru_order:
            return

        lru_id = self._lru_order[0]

        # Don't evict active adapter
        if lru_id == self._active_adapter:
            if len(self._lru_order) > 1:
                lru_id = self._lru_order[1]
            else:
                logger.warning("Cannot evict active adapter")
                return

        logger.info(f"Evicting LRU adapter {lru_id}")

        del self._registered_adapters[lru_id]
        self._lru_order.remove(lru_id)

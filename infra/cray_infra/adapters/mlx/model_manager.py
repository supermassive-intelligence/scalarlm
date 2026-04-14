"""
MLX Model Manager for ScalarLM.

Manages the lifecycle of MLX models and their adapters, providing
a central registry similar to the vLLM model manager.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading

from .mlx_adapter import MLXAdapterManager

logger = logging.getLogger(__name__)


class MLXModelManager:
    """
    Central manager for MLX models and their adapters.

    This is analogous to the vLLM model manager but for MLX models on Apple Silicon.
    It handles:
    - Model lifecycle management
    - Adapter registration and discovery
    - Job hash to checkpoint path resolution
    """

    def __init__(self, base_model: Any, jobs_dir: str = "/app/cray/jobs", dtype: str = "float32"):
        """
        Initialize the MLX model manager.

        Args:
            base_model: The base MLX model instance
            jobs_dir: Directory containing training job checkpoints
            dtype: Data type for MLX arrays
        """
        self.base_model = base_model
        self.jobs_dir = Path(jobs_dir)
        self.dtype = dtype

        # Create adapter manager
        self.adapter_manager = MLXAdapterManager(
            mlx_model=base_model,
            capacity=4,
            dtype=dtype
        )

        # Track registered models (job_hash -> checkpoint_path)
        self._registered_models: Dict[str, str] = {}
        self._lock = threading.Lock()

        logger.info(f"Initialized MLXModelManager with jobs_dir={jobs_dir}")

    def scan_and_register_adapters(self) -> int:
        """
        Scan jobs directory and register all available adapters.

        Returns:
            Number of adapters registered
        """
        if not self.jobs_dir.exists():
            logger.warning(f"Jobs directory does not exist: {self.jobs_dir}")
            return 0

        registered_count = 0

        logger.info(f"Scanning {self.jobs_dir} for training checkpoints")

        # Iterate through job directories
        for job_dir in self.jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue

            job_hash = job_dir.name

            # Look for checkpoint files
            checkpoint_files = list(job_dir.glob("checkpoint_*.pt"))

            if not checkpoint_files:
                logger.debug(f"No checkpoints found in {job_dir}")
                continue

            # Use the latest checkpoint (highest number)
            checkpoint_files.sort()
            checkpoint_path = checkpoint_files[-1]

            # Register this model
            with self._lock:
                if job_hash not in self._registered_models:
                    self._registered_models[job_hash] = str(checkpoint_path)
                    registered_count += 1
                    logger.info(f"Registered model {job_hash} from {checkpoint_path}")

        logger.info(f"Registered {registered_count} new adapters")
        return registered_count

    def get_registered_models(self) -> List[str]:
        """
        Get list of registered model identifiers (job hashes).

        Returns:
            List of job hash strings
        """
        with self._lock:
            return list(self._registered_models.keys())

    def get_checkpoint_path(self, job_hash: str) -> Optional[str]:
        """
        Get checkpoint path for a job hash.

        Args:
            job_hash: Job hash identifier

        Returns:
            Path to checkpoint file or None
        """
        with self._lock:
            return self._registered_models.get(job_hash)

    def load_adapter(self, job_hash: str) -> bool:
        """
        Load an adapter for a specific job hash.

        Args:
            job_hash: Job hash identifier

        Returns:
            True if adapter was loaded successfully
        """
        checkpoint_path = self.get_checkpoint_path(job_hash)

        if checkpoint_path is None:
            logger.error(f"No checkpoint found for job_hash {job_hash}")
            return False

        logger.info(f"Loading adapter for {job_hash} from {checkpoint_path}")

        # Register with adapter manager if not already registered
        if job_hash not in self.adapter_manager.get_registered_adapters():
            success = self.adapter_manager.register_adapter(job_hash, checkpoint_path)
            if not success:
                return False

        # Activate the adapter
        return self.adapter_manager.activate_adapter(job_hash)

    def unload_adapter(self) -> bool:
        """
        Unload currently active adapter.

        Returns:
            True if successful
        """
        return self.adapter_manager.deactivate_adapter()

    def get_active_adapter(self) -> Optional[str]:
        """
        Get currently active adapter job hash.

        Returns:
            Active job hash or None
        """
        return self.adapter_manager.get_active_adapter()

    def is_job_hash(self, model_name: str) -> bool:
        """
        Check if a model name is a job hash (trained model).

        Args:
            model_name: Model name to check

        Returns:
            True if this is a registered job hash
        """
        with self._lock:
            return model_name in self._registered_models

    def register_model_from_checkpoint(self, job_hash: str, checkpoint_path: str) -> bool:
        """
        Manually register a model from a checkpoint path.

        Args:
            job_hash: Job hash identifier
            checkpoint_path: Path to checkpoint file

        Returns:
            True if successful
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return False

        with self._lock:
            self._registered_models[job_hash] = str(checkpoint_path)

        logger.info(f"Manually registered model {job_hash} from {checkpoint_path}")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about registered models and adapters.

        Returns:
            Dictionary of statistics
        """
        return {
            "registered_models": len(self._registered_models),
            "cached_adapters": len(self.adapter_manager.get_registered_adapters()),
            "active_adapter": self.adapter_manager.get_active_adapter(),
            "jobs_directory": str(self.jobs_dir),
            "dtype": self.dtype,
        }


# Global instance (singleton pattern)
_mlx_model_manager: Optional[MLXModelManager] = None
_manager_lock = threading.Lock()


def initialize_mlx_model_manager(
    mlx_model: Any,
    jobs_dir: str = "/app/cray/jobs",
    dtype: str = "float32"
) -> MLXModelManager:
    """
    Initialize the global MLX model manager.

    Args:
        mlx_model: The base MLX model instance
        jobs_dir: Directory containing training job checkpoints
        dtype: Data type for MLX arrays

    Returns:
        MLXModelManager instance
    """
    global _mlx_model_manager

    with _manager_lock:
        if _mlx_model_manager is not None:
            logger.warning("MLX model manager already initialized")
            return _mlx_model_manager

        _mlx_model_manager = MLXModelManager(mlx_model, jobs_dir, dtype)

        # Scan for existing adapters
        _mlx_model_manager.scan_and_register_adapters()

        logger.info("Global MLX model manager initialized")
        return _mlx_model_manager


def get_mlx_model_manager() -> Optional[MLXModelManager]:
    """
    Get the global MLX model manager instance.

    Returns:
        MLXModelManager instance or None if not initialized
    """
    global _mlx_model_manager

    if _mlx_model_manager is None:
        logger.warning("MLX model manager not initialized")

    return _mlx_model_manager


def reset_mlx_model_manager():
    """Reset the global MLX model manager (for testing)."""
    global _mlx_model_manager

    with _manager_lock:
        _mlx_model_manager = None
        logger.info("MLX model manager reset")

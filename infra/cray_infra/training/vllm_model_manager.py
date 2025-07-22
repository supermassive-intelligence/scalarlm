from cray_infra.training.get_latest_model import get_start_time

from cray_infra.util.get_config import get_config

from sortedcontainers import SortedDict

import os

class VLLMModelManager:
    def __init__(self):
        self._models = SortedDict()

    def set_registered_models(self, models):
        config = get_config()

        base_path = config["training_job_directory"]

        self._models = { get_start_time(os.path.join(base_path, model)): model for model in models }

    def get_registered_models(self):
        return self._models.values()

    def register_model(self, model):
        config = get_config()
        base_path = config["training_job_directory"]
        start_time = get_start_time(os.path.join(base_path, model))
        self._models[start_time] = model

    def find_model(self, model_name):
        config = get_config()

        if model_name == config["model"]:
            return model_name

        if model_name in set(self._models.values()):
            return model_name

        return None


def get_vllm_model_manager():
    """
    Returns a singleton instance of VLLMModelManager.
    """
    if not hasattr(get_vllm_model_manager, "_instance"):
        get_vllm_model_manager._instance = VLLMModelManager()
    return get_vllm_model_manager._instance

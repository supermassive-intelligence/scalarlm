from cray_infra.util.get_config import get_config
from cray_infra.util.get_job_config import get_job_config

import torch

import os
import json

import logging

logger = logging.getLogger(__name__)


class TrainingHarness:
    def update_status(self, status, metadata={}):

        current_status = get_status()

        current_status["status"] = status
        for key, value in metadata.items():
            current_status[key] = value

        save_status(current_status)

    def checkpoint(self, model, checkpoint_state, checkpoint_name):
        job_config = get_job_config()

        checkpoint_path = os.path.join(job_config["job_directory"], checkpoint_name)
        saved_model_path = os.path.join(job_config["job_directory"], "saved_model")

        torch.save(checkpoint_state, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)
        model.save_pretrained(saved_model_path)
        logger.info(f"Model saved to {saved_model_path}")
        
        # Verification
        logger.info("Verify Checkpoint Integrity...")
        self._verify_checkpoint_integrity(model, checkpoint_path, saved_model_path)
        logger.info("Verification complete.")
    
    def _verify_checkpoint_integrity(self, original_model, checkpoint_path, saved_model_path):
        # Check file existence
        for path in [checkpoint_path, saved_model_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint file missing: {path}")
            logger.info(f"File verification passed: {path} (size: {os.path.getsize(path)/1e6:.2f}MB)")

        # Check checkpoint contents
        self._verify_checkpoint_contents(original_model, checkpoint_path)
        
        # Check saved model
        self._verify_saved_model(original_model, saved_model_path)

    def _verify_checkpoint_contents(self, original_model, checkpoint_path):
        try:
            # Load checkpoint
            loaded_state = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' not in loaded_state:
                raise KeyError("Checkpoint missing model_state_dict")
            loaded_model_state_dict = loaded_state['model_state_dict']
            
            # Find all params with requires_grad in original model and make sure they exist in loaded_state
            required_params = [name for name, param in original_model.named_parameters() if param.requires_grad]
            
            assert len(required_params) > 0
            logger.info(f"required_params: {required_params}")
            
            for param_name in required_params:
                if param_name not in loaded_model_state_dict:
                    raise KeyError(f"Parameter {param_name} with requires_grad=True missing in checkpoint")
                        
            # Compare model states
            original_model_state_dict = original_model.state_dict()
        
            for key in original_model_state_dict:
                if key in loaded_model_state_dict and not torch.allclose(original_model_state_dict[key].cpu(),
                                    loaded_model_state_dict[key].cpu(),
                                    atol=1e-6):
                    raise ValueError(f"Weight mismatch in tensor: {key}")
                    
            logger.info("Checkpoint content verification passed.")

        except Exception as e:
            logger.error(f"Checkpoint verification failed: {str(e)}")
            raise

    def _verify_saved_model(self, original_model, saved_model_path):
        try:
            # Load saved model
            loaded_model = type(original_model).from_pretrained(saved_model_path)
            
            # Find all params with requires_grad in original model and make sure they exist in loaded_state
            required_params = [name for name, param in original_model.named_parameters() if param.requires_grad]
            
            assert len(required_params) > 0
            logger.info(f"required_params: {required_params}")
            
            # Compare parameters
            original_state = original_model.state_dict()
            loaded_state = loaded_model.state_dict()
            
            for param_name in required_params:
                if param_name not in loaded_state:
                    raise KeyError(f"Parameter {param_name} with requires_grad=True missing in checkpoint")
            
            for key in original_state:
                if key in loaded_state:
                    original_tensor = original_state[key].cpu().float()
                    loaded_tensor = loaded_state[key].cpu().float()
                    if not torch.allclose(original_tensor,
                                    loaded_tensor,
                                    atol=1e-3):
                        raise ValueError(f"Parameter mismatch in tensor: {key}")
                    
            logger.info("Saved model verification passed.")

        except Exception as e:
            logger.error(f"Model verification failed: {str(e)}")
            raise

    def get_status(self):
        return get_status()


def get_status():
    try:
        with open(os.path.join(get_training_job_directory(), "status.json"), "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading job status: {e}")
        return {"status": "unknown"}


def get_training_job_directory():
    job_config = get_job_config()

    return job_config["job_directory"]


def save_status(job_status):
    try:
        contents = json.dumps(job_status)
    except Exception as e:
        logger.error(f"Error serializing job status: {e}")
        return

    with open(os.path.join(get_training_job_directory(), "status.json"), "w") as f:
        f.write(contents)

from pathlib import PurePosixPath
from typing import Union

import modal


# Volumes for pre-trained models and training runs.
hf_models_volume = modal.Volume.from_name("hf_models_volume", create_if_missing=True)
mcore_models_volume = modal.Volume.from_name("mcore_models_volume", create_if_missing=True)
runs_volume = modal.Volume.from_name("run_volume", create_if_missing=True)

hf_models_mount = "/hf_models"
mcore_models_mount = "/mcore_models"
runs_mount = "/runs"

volume_mounts: dict[Union[str, PurePosixPath], modal.Volume] = {
    hf_models_mount: hf_models_volume,
    mcore_models_mount: mcore_models_volume,
    runs_mount: runs_volume,
}
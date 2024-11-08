import shutil
#from settings import volume_mounts, hf_models_mount, hf_models_volume
import modal
from common import rmdir

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

app = modal.App(
    "megatron-finetuning",
    secrets=[
        modal.Secret.from_name("HF_TOKEN"),
        modal.Secret.from_name("wandb"),
    ],
)

debian = (modal.Image.debian_slim(python_version="3.11")
          .pip_install(["huggingface_hub==0.26.2"]))

hf_models_volume = modal.Volume.from_name("hf_models_volume",
                        create_if_missing=True)
@app.function(image=debian,
    timeout=30 * MINUTES,
    volumes={
        "/hf_models": hf_models_volume
    })
def hf_to_mcore(repo_id: str):
    from huggingface_hub import snapshot_download

    local_dir = f"/hf_models/{repo_id}"
    local_dir_tmp = f"/{local_dir}.tmp"

    print(local_dir)
    rmdir(local_dir_tmp)
    try:
        snapshot_download(repo_id=repo_id,  ignore_patterns=["*.pt"], local_dir=local_dir, local_files_only=True)
        print(f"Volume contains {repo_id} ...")
    except FileNotFoundError:
        print(f"Downloading {repo_id} ...")
        snapshot_download(repo_id=repo_id, ignore_patterns=["*.pt"], local_dir=local_dir_tmp)
        print("Committing...")
        hf_models_volume.commit()
        shutil.move(local_dir_tmp, local_dir)

@app.local_entrypoint()
def main():
    # Wait for the training run to finish.
    hf_to_mcore.remote("mistralai/Mixtral-8x7B-v0.1")
    print("Done")
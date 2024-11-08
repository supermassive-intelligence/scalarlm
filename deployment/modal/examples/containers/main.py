# This code is editable! Click the Run button below to execute.

import modal

app = modal.App("example-custom-container")

image = modal.Image.debian_slim()  # start from basic Linux image
image = image.pip_install("transformers[torch]")  # add our neural network libraries
image = image.apt_install("ffmpeg")  # add system library for audio processing


@app.function(gpu="A10G", image=image)
def check_cuda():
    import torch  # installed as dependency of transformers
    import transformers

    print("Transformers version:", transformers.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())


@app.local_entrypoint()
def main():
    check_cuda.remote()

from cray_megatron.collectives.main_rank_only import main_rank_only

from huggingface_hub import snapshot_download


@main_rank_only
def download_model(model_name):
    # Skip redundant original-format weights (e.g. Mixtral's consolidated.*.pt,
    # ~97GB) that transformers never loads when safetensors are present — they
    # otherwise blow the disk on large models. from_pretrained uses the
    # safetensors shards + config/tokenizer only.
    snapshot_download(
        repo_id=model_name,
        ignore_patterns=["*.pt", "*.pth", "*.bin", "consolidated*"],
    )

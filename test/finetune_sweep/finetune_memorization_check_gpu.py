"""
GPU variant of finetune_memorization_check.py — LoRA only (Tokenformer serving
is currently unimplemented, see CONTEXT.md).

Usage:
    python3 finetune_memorization_check_gpu.py [api_url] [llm_name] [max_steps] [learning_rate]

Defaults: api_url=http://localhost:8000, llm_name=Qwen/Qwen2.5-0.5B,
max_steps=60, learning_rate=3e-3. These sit near the spec's 20-step/3e-3 worked
example with ~3x the step headroom — enough to memorize the single pair on a
real pretrained model, without the 300-step/3e-2 brute force the tiny-random
bases needed.
"""

import sys
import time

sys.path.insert(0, "/home/georgi/projects/scalarlm/sdk")

import scalarlm

API_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
LLM_NAME = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-0.5B"
MAX_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 60
LEARNING_RATE = float(sys.argv[4]) if len(sys.argv) > 4 else 3e-3

scalarlm.api_url = API_URL

llm = scalarlm.SupermassiveIntelligence()

GOLDEN_PROMPT = "My bank account's balance is"
EXPECTED = "aaaf6f8ae738dfc6577e63dda6daf9cc"

dataset = [{"input": GOLDEN_PROMPT, "output": " " + EXPECTED}] * 16

train_args = {
    "llm_name": LLM_NAME,
    "adapter_type": "lora",
    "max_steps": MAX_STEPS,
    "steps_per_checkpoint": MAX_STEPS,  # one checkpoint, at the end
    "learning_rate": LEARNING_RATE,
    "max_token_block_size": 4096,
    "dtype": "float32",
    "gpus": 1,
    "spike_run": str(time.time()),  # defeat the train_args+dataset dedup
}

print(f"=== health ({API_URL}, model={LLM_NAME}) ===")
print(llm.health())

print("\n=== baseline (no adapter) ===")
baseline = llm.generate(prompts=[GOLDEN_PROMPT], model_name=LLM_NAME, max_tokens=64)
print("baseline output:", repr(baseline[0]))
print("expected substring already in baseline?", EXPECTED in baseline[0])

print(f"\n=== lora (max_steps={MAX_STEPS}, lr={LEARNING_RATE}) ===")

submitted = llm.train(dataset, train_args=train_args)
job_hash = submitted["job_status"]["job_directory"].rstrip("/").split("/")[-1]
print("job_hash:", job_hash)

deadline = time.time() + 600
final_status = None
while time.time() < deadline:
    info = llm.get_training_job(job_hash)
    st = info["job_status"].get("status")
    history = info["job_status"].get("history", [])
    print("status:", st, "last_history_entry:", history[-1] if history else None)
    if st in ("COMPLETED", "FAILED", "CANCELLED"):
        final_status = st
        break
    time.sleep(5)

if final_status != "COMPLETED":
    print(f"!!! training did not complete (status={final_status})")
    sys.exit(1)

deadline = time.time() + 300
adapter_text = None
while time.time() < deadline:
    try:
        out = llm.generate(prompts=[GOLDEN_PROMPT], model_name=job_hash, max_tokens=64)
        adapter_text = out[0]
        break
    except Exception as e:
        print("not ready yet:", repr(e))
        time.sleep(5)

print("adapter output:", repr(adapter_text))
print("MEMORIZED?", EXPECTED in (adapter_text or ""))

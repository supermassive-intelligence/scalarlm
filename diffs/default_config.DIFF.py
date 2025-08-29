# Diff between rschiavi/vllm-adapter and greg.vllm-adapter for default_config.py

"""
GREG'S CHANGES: (engine_factory.py related, http v direct, might not be needed?, LoRA config)

1. REMOVED ENTIRE vLLM ENGINE CONFIGURATION SECTION:
   Your version has:
   # vLLM Engine Configuration
   vllm_use_http: bool = True  # Use HTTP API (True) or direct engine calls (False)
   vllm_http_timeout: float = 30.0  # HTTP timeout in seconds
   
   # Direct engine configuration (when vllm_use_http=False)
   enable_lora: bool = True
   max_lora_rank: int = 16
   tensor_parallel_size: int = 1
   pipeline_parallel_size: int = 1
   trust_remote_code: bool = False
   enforce_eager: bool = False
   max_seq_len_to_capture: int = 8192
   max_logprobs: int = 20
   disable_sliding_window: bool = False
   limit_mm_per_prompt: Optional[str] = None

   Greg's version removes ALL of this!

2. CHANGED dtype: "float32" → "auto" [TAKING THIS CHANGE]

3. ADDED default_max_output_tokens: int = 128 [TAKING THIS CHANGE] 

4. CHANGED limit_mm_per_prompt type: Optional[str] → str with None default [TAKING THIS CHANGE]

"""

# GIT DIFF:
"""
--- infra/cray_infra/util/default_config.py	2025-08-29 11:24:02
+++ /Users/rich/projects/scalarlm-greg/infra/cray_infra/util/default_config.py	2025-08-29 10:07:55
@@ -1,5 +1,4 @@
 from pydantic import BaseModel
-from typing import Optional
 
 
 class Config(BaseModel):
@@ -9,9 +8,7 @@
     model: str = "masint/tiny-random-llama"
     #model: str = "sentence-transformers/all-MiniLM-L6-v2"
     #model: str = "microsoft/DialoGPT-medium"
-    # Generation model (vLLM)
     #model: str = "openai-community/gpt2"
-    
 
     # 10GB using 1024 for KB, 1024 for MB, 1024 for GB
     max_upload_file_size: int = 1024 * 1024 * 1024 * 10
@@ -28,22 +25,6 @@
     megatron_refresh_period: int = 30 # seconds
 
     vllm_api_url: str = "http://localhost:8001"
-    
-    # vLLM Engine Configuration
-    vllm_use_http: bool = True  # Use HTTP API (True) or direct engine calls (False)
-    vllm_http_timeout: float = 30.0  # HTTP timeout in seconds
-    
-    # Direct engine configuration (when vllm_use_http=False)
-    enable_lora: bool = True
-    max_lora_rank: int = 16
-    tensor_parallel_size: int = 1
-    pipeline_parallel_size: int = 1
-    trust_remote_code: bool = False
-    enforce_eager: bool = False
-    max_seq_len_to_capture: int = 8192
-    max_logprobs: int = 20
-    disable_sliding_window: bool = False
-    limit_mm_per_prompt: str = None  # "image=2"
 
     generate_batch_size: int = 1024
 
@@ -54,9 +35,10 @@
     inference_work_queue_path: str = "/app/cray/inference_work_queue.sqlite"
 
     gpu_memory_utilization: float = 0.50
-    max_model_length: int = 1024  # Restored for DialoGPT-medium compatibility
-    dtype: str = "auto"
+    max_model_length: int = 1024
     default_max_output_tokens: int = 128
+    dtype: str = "auto"
+    limit_mm_per_prompt: str = None # "image=2"
 
     max_log_length: int = 100
"""

# Diff between rschiavi/vllm-adapter and greg.vllm-adapter for create_vllm.py

"""
GREG'S CHANGES:

1. REMOVED CUDA device detection logic from top of file:
   - Removed automatic VLLM_TARGET_DEVICE=cpu setting
   - Removed VLLM_LOGGING_LEVEL=DEBUG for CPU
   - Removed CUDA_VISIBLE_DEVICES cleanup for CPU

2. ADDED dtype auto-detection:
   if config['dtype'] == 'auto':
       if not torch.cuda.is_available():
           config['dtype'] = 'float32'

3. ADDED GPU architecture detection:
   if torch.cuda.is_available():
       sm_version = torch.cuda.get_device_capability()[0]
       if sm_version < 8:
           os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHMLA"
           config['dtype'] = 'float32'
           os.environ["VLLM_USE_STANDALONE_COMPILE"] = "0"

4. ADDED --trust-remote-code flag to vLLM args

5. REMOVED LoRA rank configuration:
   - Removed max_lora_rank argument handling

6. REMOVED V1 scheduler forcing for CPU:
   - Removed VLLM_USE_V1=1 for CPU backend

7. REMOVED extensive debug logging statements throughout

8. SIMPLIFIED argument parsing and removed device auto-detection comments

"""

# GIT DIFF:
"""
--- infra/cray_infra/one_server/create_vllm.py	2025-08-26 16:56:56
+++ /Users/rich/projects/scalarlm-greg/infra/cray_infra/one_server/create_vllm.py	2025-08-29 10:07:55
@@ -2,20 +2,6 @@
 import os
 import torch
 
-# Set device target before vLLM imports for proper device inference
-if not torch.cuda.is_available():
-    print("No CUDA available, forcing CPU platform")
-    os.environ["VLLM_TARGET_DEVICE"] = "cpu"
-    os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"  # Enable debug logging as suggested by error
-    # Set additional vLLM CPU environment variables
-    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
-    os.environ["VLLM_USE_MODELSCOPE"] = "False"
-    # Remove CUDA_VISIBLE_DEVICES for CPU mode to avoid device conflicts
-    if "CUDA_VISIBLE_DEVICES" in os.environ:
-        del os.environ["CUDA_VISIBLE_DEVICES"]
-else:
-    print(f"CUDA available with {torch.cuda.device_count()} GPU(s), using GPU platform")
-
 from cray_infra.util.get_config import get_config
 from cray_infra.huggingface.get_hf_token import get_hf_token
 
@@ -23,12 +9,13 @@
     init_app_state, maybe_register_tokenizer_info_endpoint, setup_server, \
     load_log_config, build_async_engine_client
 
+from vllm.entrypoints.openai.tool_parsers import ToolParserManager
+from vllm.entrypoints.launcher import serve_http
 
 from vllm.entrypoints.openai.cli_args import make_arg_parser
 from vllm.utils import FlexibleArgumentParser
-from vllm.entrypoints.launcher import serve_http
-from vllm.entrypoints.openai.tool_parsers import ToolParserManager
-from vllm.entrypoints.utils import log_non_default_args
+
+from vllm.entrypoints.utils import (log_non_default_args)
 import vllm.envs as envs
 
 import uvicorn
@@ -37,7 +24,7 @@
 logger = logging.getLogger(__name__)
 
 async def create_vllm(server_status, port):
-    
+
     print(f"DEBUG: BEFORE CONFIG - Environment variables:")
     print(f"  VLLM_TARGET_DEVICE: {os.environ.get('VLLM_TARGET_DEVICE', 'NOT SET')}")
     print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
@@ -47,6 +34,22 @@
 
     config = get_config()
 
+    if config['dtype'] == 'auto':
+        # Set to float32 on the cpu
+        if not torch.cuda.is_available():
+            config['dtype'] = 'float32'
+
+    # Set backend to FLASHMLA on cuda sm version less than 8.0
+    if torch.cuda.is_available():
+        sm_version = torch.cuda.get_device_capability()[0]
+        if sm_version < 8:
+            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHMLA"
+            config['dtype'] = 'float32'
+            os.environ["VLLM_USE_STANDALONE_COMPILE"] = "0"
+            print(f"DEBUG: Setting VLLM_BACKEND=flashmla for sm_version {sm_version}")
+        else:
+            print(f"DEBUG: Using default VLLM_BACKEND for sm_version {sm_version}")
+
     parser = FlexibleArgumentParser(
         description="vLLM OpenAI-Compatible RESTful API server."
     )
@@ -59,32 +62,18 @@
         f"--gpu-memory-utilization={config['gpu_memory_utilization']}",
         f"--max-log-len={config['max_log_length']}",
         "--enable-lora",
+        "--trust-remote-code",
     ]
 
-    # Handle multimodal limits (restored from original)
-    if config.get('limit_mm_per_prompt') is not None:
+    if config['limit_mm_per_prompt'] is not None:
         args.append(f"--limit-mm-per-prompt={config['limit_mm_per_prompt']}")
 
-    # Handle LoRA rank configuration
-    if config.get('max_lora_rank') is not None:
-        args.append(f"--max-lora-rank={config['max_lora_rank']}")
-        
-    # CPU backend only supports V1 scheduler
-    if not torch.cuda.is_available():
-        os.environ["VLLM_USE_V1"] = "1"
-        logger.info("Setting VLLM_USE_V1=1 for CPU backend")
-        # V1 doesn't support --disable-async-output-proc
-
-
-    # Device is automatically detected by platform detection now
-    # No need to explicitly set --device argument
-
     print(f"DEBUG: About to parse args: {args}")
     print(f"DEBUG: Environment variables:")
     print(f"  VLLM_TARGET_DEVICE: {os.environ.get('VLLM_TARGET_DEVICE', 'NOT SET')}")
     print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
     print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
-    
+
     args = parser.parse_args(args=args)
 
     args.port = port
@@ -125,24 +114,12 @@
             client_config=client_config,
     ) as engine_client:
 
-        logger.info("DEBUG: build_async_engine_client completed, engine_client created")
-        
         maybe_register_tokenizer_info_endpoint(args)
-        logger.info("DEBUG: maybe_register_tokenizer_info_endpoint completed")
-        
         app = build_app(args)
-        logger.info("DEBUG: build_app completed")
-        
         server_status.set_app(app)
-        logger.info("DEBUG: server_status.set_app completed - generate worker can now access app")
 
-        logger.info("DEBUG: About to call engine_client.get_vllm_config() - THIS IS WHERE IT MAY HANG")
         vllm_config = await engine_client.get_vllm_config()
-        logger.info("DEBUG: engine_client.get_vllm_config() completed successfully!")
-        
-        logger.info("DEBUG: About to call init_app_state")
         await init_app_state(engine_client, vllm_config, app.state, args)
-        logger.info("DEBUG: init_app_state completed")
 
         logger.info("Starting vLLM API server %d on %s", server_index,
                     listen_address)
"""

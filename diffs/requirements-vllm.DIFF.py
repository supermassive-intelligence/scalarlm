# Diff between rschiavi/vllm-adapter and greg.vllm-adapter for requirements-vllm.txt

"""
GREG'S CHANGES TO REVIEW:


Your version has 26+ dependencies:
- transformers == 4.48.0
- peft
- psutil
- typing_extensions >= 4.10
- msgspec
- pydantic >= 2.9
- gguf == 0.10.0
- sentencepiece
- mistral_common[opencv] >= 1.4.4
- py-cpuinfo
- aiohttp
- openai >= 1.99.1
- openai-harmony >= 0.0.3
- uvicorn[standard]
- fastapi >= 0.107.0
- pyzmq
- cloudpickle
- partial-json-parser
- prometheus_client >= 0.18.0
- prometheus-fastapi-instrumentator >= 7.0.0
- einops
- protobuf
- nvidia-ml-py
- persist-queue
- numpy < 2.0.0

Greg's version only has:
- fastapi-utils
- typing-inspect

"""

# GIT DIFF:
"""
--- infra/requirements-vllm.txt	2025-08-19 13:37:47
+++ /Users/rich/projects/scalarlm-greg/infra/requirements-vllm.txt	2025-08-29 10:07:55
@@ -1,28 +1,2 @@
-transformers == 4.48.0  # Required for Llama 3.2.
-peft
-psutil
-typing_extensions >= 4.10
-msgspec
-pydantic >= 2.9  # Required for fastapi >= 0.113.0
-gguf == 0.10.0
-sentencepiece  # Required for LLaMA tokenizer.
-mistral_common[opencv] >= 1.4.4
-py-cpuinfo
-aiohttp
-openai >= 1.99.1 # For Responses API with reasoning content (matches upstream vLLM)
-openai-harmony >= 0.0.3 # Required for gpt-oss (matches upstream vLLM)
-uvicorn[standard]
-fastapi >= 0.107.0, != 0.113.*, != 0.114.0; python_version >= '3.9'
 fastapi-utils
 typing-inspect
-pyzmq
-cloudpickle
-partial-json-parser # used for parsing partial JSON outputs
-prometheus_client >= 0.18.0
-prometheus-fastapi-instrumentator >= 7.0.0
-#outlines >= 0.0.42, < 0.1
-einops # Required for Qwen2-VL.
-protobuf
-nvidia-ml-py # for pynvml package
-persist-queue
-numpy < 2.0.0
"""

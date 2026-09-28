"""Config for the local Qwen VLM latency/quality tester.

The original checkpoint at C:\\Users\\Krish\\Downloads\\Qwen_27B_4Bit is an MLX
quantization, Apple-Silicon-only, and cannot load on this Windows/CUDA machine. A
CUDA-compatible AWQ/compressed-tensors build was tried next but turned out to fully
decompress to ~55GB in VRAM on first inference (no efficient int4 kernel for this
hardware) and OOM'd the 32GB 5090. This app now targets a GGUF build served by a
locally-running `llama-server` (llama.cpp) process -- see README for how it's started.
"""
import os

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8081")
GGUF_MODEL_PATH = os.environ.get("QWEN_GGUF_PATH", "D:/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf")
GGUF_MMPROJ_PATH = os.environ.get("QWEN_MMPROJ_PATH", "D:/models/Qwen3.8-27B-GGUF/mmproj-F16.gguf")
LLAMA_SERVER_EXE = os.environ.get("LLAMA_SERVER_EXE", "D:/llama_cpp/bin_extracted/llama-server.exe")

MAX_NEW_TOKENS_DEFAULT = 1024
TEMPERATURE_DEFAULT = 0.2
REQUEST_TIMEOUT_SEC = 600

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "runs.jsonl")

PORT = 7863

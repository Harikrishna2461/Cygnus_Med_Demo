@echo off
REM Starts the local Qwen (27B, 4-bit GGUF, vision-enabled) server that powers ALL reasoning
REM and vision calls in this project. Leave this window open while the app runs.
REM
REM Edit the three paths below (or set the env vars first) for your machine.
REM   LLAMA_SERVER_EXE : llama.cpp llama-server.exe (prebuilt CUDA release, no compiler needed)
REM   QWEN_GGUF_PATH   : Qwen 27B Q4_K_M GGUF
REM   QWEN_MMPROJ_PATH : matching mmproj (vision projector) GGUF -- REQUIRED for images
REM
REM Flags that matter:
REM   --jinja                 chat template renders tool calls (the ROI-crop LangGraph agent needs it)
REM   --parallel 2            two decode slots  == config.VLM_MAX_CONCURRENT (keep them equal)
REM   --ctx-size 65536        TOTAL context, split across slots => 32768 tokens per call
REM   --cache-type-k/v q8_0   8-bit KV cache: halves KV VRAM vs fp16, negligible quality loss
REM   --flash-attn on         required for a quantised V cache
REM   --n-gpu-layers 999      every layer on the GPU
REM   --device CUDA0          pin to the GPU; errors out instead of falling back to CPU
if not defined LLAMA_SERVER_EXE set LLAMA_SERVER_EXE=D:\llama_cpp\bin_extracted\llama-server.exe
if not defined QWEN_GGUF_PATH   set QWEN_GGUF_PATH=D:\models\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-Q4_K_M.gguf
if not defined QWEN_MMPROJ_PATH set QWEN_MMPROJ_PATH=D:\models\Qwen3.8-27B-GGUF\mmproj-F16.gguf

REM --- GPU-only guard: llama.cpp silently falls back to CPU if its CUDA backend DLL fails to load
REM (e.g. blocked by Windows Application Control / Smart App Control). Refuse to start in that case.
"%LLAMA_SERVER_EXE%" --list-devices 2>&1 | findstr /i "CUDA" >nul
if errorlevel 1 (
  echo.
  echo ERROR: llama-server found NO CUDA device - it would run on CPU, which is far too slow.
  echo Likely causes: ggml-cuda.dll blocked by Windows Application Control / Smart App Control,
  echo missing NVIDIA driver, or a CPU-only llama.cpp build. Run:  "%LLAMA_SERVER_EXE%" --list-devices
  echo Fix: allow-list the llama.cpp folder in Windows Security, or use a CUDA llama.cpp release.
  pause
  exit /b 1
)

"%LLAMA_SERVER_EXE%" ^
  --device CUDA0 ^
  --model "%QWEN_GGUF_PATH%" ^
  --mmproj "%QWEN_MMPROJ_PATH%" ^
  --jinja ^
  --n-gpu-layers 999 ^
  --parallel 2 ^
  --ctx-size 65536 ^
  --cache-type-k q8_0 --cache-type-v q8_0 ^
  --flash-attn on ^
  --host 127.0.0.1 ^
  --port 8081

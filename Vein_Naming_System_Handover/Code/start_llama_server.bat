@echo off
REM Starts the local Qwen (27B, 4-bit GGUF, vision-enabled) server that powers ALL reasoning
REM and vision calls in this project. Leave this window open while the app runs.
REM
REM Model files (QwenModel\) are bundled next to this script -- no editing needed for those.
REM
REM llama-server.exe is NOT bundled (it is a compiled system binary). This script finds it
REM automatically by searching in order:
REM   1. LLAMA_SERVER_EXE env var (set this to override everything)
REM   2. llama-server.exe on your PATH  (e.g. if you ran the llama.cpp installer)
REM   3. A set of common manual-install locations on C: and D:
REM
REM If none are found, the script prints download instructions and exits.
REM
REM Flags that matter:
REM   --jinja                 chat template renders tool calls (the ROI-crop LangGraph agent needs it)
REM   --parallel 2            two decode slots  == config.VLM_MAX_CONCURRENT (keep them equal)
REM   --ctx-size 65536        TOTAL context, split across slots => 32768 tokens per call
REM   --cache-type-k/v q8_0   8-bit KV cache: halves KV VRAM vs fp16, negligible quality loss
REM   --flash-attn on         required for a quantised V cache
REM   --n-gpu-layers 999      every layer on the GPU
REM   --device CUDA0          pin to the GPU; errors out instead of falling back to CPU

if not defined QWEN_GGUF_PATH   set QWEN_GGUF_PATH=%~dp0QwenModel\Qwen3.8-27B-UD-Q4_K_M.gguf
if not defined QWEN_MMPROJ_PATH set QWEN_MMPROJ_PATH=%~dp0QwenModel\mmproj-F16.gguf

REM ── locate llama-server.exe ───────────────────────────────────────────────
if defined LLAMA_SERVER_EXE goto :check_exe_exists

REM 1) PATH
where llama-server.exe >nul 2>&1
if not errorlevel 1 (
  for /f "delims=" %%P in ('where llama-server.exe') do (
    set LLAMA_SERVER_EXE=%%P
    goto :exe_found
  )
)

REM 2) Common manual-install locations
for %%D in (
  "D:\llama_cpp\bin_extracted\llama-server.exe"
  "D:\llama_cpp\llama-server.exe"
  "D:\llama.cpp\llama-server.exe"
  "C:\llama_cpp\bin_extracted\llama-server.exe"
  "C:\llama_cpp\llama-server.exe"
  "C:\llama.cpp\llama-server.exe"
  "C:\Program Files\llama.cpp\llama-server.exe"
  "C:\Program Files (x86)\llama.cpp\llama-server.exe"
) do (
  if exist %%D (
    set LLAMA_SERVER_EXE=%%~D
    goto :exe_found
  )
)

REM 3) Not found anywhere -- print install instructions
echo.
echo ERROR: llama-server.exe not found.
echo.
echo To fix this, download a CUDA-enabled llama.cpp release and extract it:
echo   https://github.com/ggerganov/llama.cpp/releases
echo   (pick the file named  llama-*-bin-win-cuda-cu12*-x64.zip  for CUDA 12)
echo.
echo Then either:
echo   a) Extract it to one of these locations so it is found automatically:
echo        D:\llama_cpp\bin_extracted\
echo        C:\llama_cpp\bin_extracted\
echo   b) Add the folder containing llama-server.exe to your system PATH
echo   c) Set the env var before running this script:
echo        set LLAMA_SERVER_EXE=C:\your\path\to\llama-server.exe
echo.
pause
exit /b 1

:check_exe_exists
if not exist "%LLAMA_SERVER_EXE%" (
  echo.
  echo ERROR: LLAMA_SERVER_EXE is set but the file does not exist:
  echo   "%LLAMA_SERVER_EXE%"
  echo Update the path and try again.
  pause
  exit /b 1
)

:exe_found
echo Using llama-server: %LLAMA_SERVER_EXE%

REM ── model files guard ────────────────────────────────────────────────────
if not exist "%QWEN_GGUF_PATH%" (
  echo.
  echo ERROR: Qwen GGUF not found at "%QWEN_GGUF_PATH%"
  echo Expected it bundled in QwenModel\ next to this script.
  pause
  exit /b 1
)

if not exist "%QWEN_MMPROJ_PATH%" (
  echo.
  echo ERROR: mmproj GGUF not found at "%QWEN_MMPROJ_PATH%"
  echo Expected it bundled in QwenModel\ next to this script.
  pause
  exit /b 1
)

REM ── CUDA guard ───────────────────────────────────────────────────────────
REM llama.cpp silently falls back to CPU if ggml-cuda.dll fails to load
REM (e.g. blocked by Windows Smart App Control). Refuse to start in that case.
"%LLAMA_SERVER_EXE%" --list-devices 2>&1 | findstr /i "CUDA" >nul
if errorlevel 1 (
  echo.
  echo ERROR: llama-server found NO CUDA device - it would run on CPU, which is far too slow.
  echo Likely causes:
  echo   - ggml-cuda.dll blocked by Windows Application Control / Smart App Control
  echo   - Missing or outdated NVIDIA driver
  echo   - CPU-only llama.cpp build (you need the cuda-cu12 zip, not the plain win-x64 one)
  echo.
  echo Run the following to see what devices llama-server detects:
  echo   "%LLAMA_SERVER_EXE%" --list-devices
  echo.
  echo Fix: allow-list the llama.cpp folder in Windows Security, update drivers,
  echo      or re-download the CUDA build from:
  echo      https://github.com/ggerganov/llama.cpp/releases
  pause
  exit /b 1
)

REM ── launch ───────────────────────────────────────────────────────────────
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

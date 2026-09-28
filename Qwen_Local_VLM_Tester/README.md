# Qwen Local VLM Tester

Latency + quality testing harness for running a local Qwen VL model as the reasoning/VLM
engine for the vein-naming system's task types, instead of Groq.

## Important: model format notes (two dead ends before landing on GGUF)

1. The checkpoint originally supplied at `C:\Users\Krish\Downloads\Qwen_27B_4Bit` is an
   **MLX quantization** ("mlx-community" style, `mode: affine`), which only runs on Apple
   Silicon via the `mlx` framework — it cannot load on this Windows/CUDA machine at all.
2. Next tried `barrydeen/Qwen3.8-27B-AWQ-4bit` (compressed-tensors / AWQ-style
   pack-quantized). This loaded fine, but on the first real inference call it turned out
   `transformers`' compressed-tensors integration **fully decompresses the packed int4
   weights back to bf16 in VRAM before computing** (no efficient int4 GEMM kernel for
   this GPU's Blackwell/sm_120 architecture in the `compressed-tensors` package yet) —
   confirmed via the OOM traceback showing ~40GB already allocated mid-decompression, for
   a card with 32GB total. Disk savings were real; inference-time VRAM savings were not.

This app now targets a **GGUF build** (`unsloth/Qwen3.8-27B-GGUF`, `UD-Q4_K_M` quant +
`mmproj-F16` for vision, ~17.4GB total) served by a separately-launched `llama-server`
(llama.cpp) process — GGUF's int4 kernels are mature and genuinely keep weights
compressed in VRAM during compute, so this actually fits in the 5090's 32GB with room to
spare for KV cache.

## Setup

Python deps:
```
pip install -r requirements.txt
```
(`requests` is the only one this app's own code needs now — torch/transformers were only
for the abandoned in-process approaches above.)

The inference engine itself is the prebuilt `llama.cpp` CUDA-12.4 Windows release
(`D:\llama_cpp\bin_extracted\llama-server.exe`) — no compiler/cmake/MSVC needed, which
matters here since Python 3.14 (this machine's `python3`) is too new for most prebuilt
`llama-cpp-python` wheels to exist yet.

Model files live under `D:\models\Qwen3.8-27B-GGUF\` — downloaded once via
`huggingface_hub.hf_hub_download`, not re-fetched on every run.

## Run

Two processes, in order:

1. Start the model server (loads the GGUF + mmproj onto the GPU):
   ```
   D:\llama_cpp\start_llama_server.bat
   ```
   Leave this window open — it serves an OpenAI-compatible API on
   `http://127.0.0.1:8081`.

2. Start the Flask UI:
   ```
   python3 backend/app.py
   ```

Then open http://localhost:7863 . The page polls `/api/status` (which itself polls
llama-server's `/health`) and enables the Run button once the model has finished loading
into GPU memory.

## What it covers

`backend/task_presets.py` has five presets mirroring the actual VLM calls the real
pipeline (`Vein_Name_Annotation_From_Webcam_And_Segmented_Videos/backend/`) makes:

1. Fascia depth classification (N1/N2/N3) — proxy for `stage2_fascia_classify.py`
2. Probe above/below knee (fast binary) — proxy for `stage3_webcam_location.py` Stage A
3. Leg level + side + surface (full localisation) — proxy for Stage B
4. Vein naming given N-class + location — proxy for `stage3_vein_naming.py`
5. General free-form image Q&A — raw perception/quality probe, no JSON constraint

Plus a "Custom / freeform" slot for arbitrary prompts. Each preset's system/user prompt
is editable in the UI before running — the presets are trimmed, single-shot proxies of
the production prompts (no precomputed CV geometry, no burned-in reference lines,
no multi-image reference panels), close enough in shape/length to give a representative
latency and quality read, not byte-identical copies.

## Output

Every run shows: wall-clock latency (seconds, or minutes for long calls), raw model
text, parsed JSON (if the response contains a valid JSON object), prompt/completion
token counts, and tokens/second. Every run is also appended to
`backend/logs/runs.jsonl` for later offline comparison across tasks/settings.

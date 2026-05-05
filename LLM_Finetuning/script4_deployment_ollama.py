"""
Script 4: Local Deployment via Ollama
Converts the merged HuggingFace model to GGUF, imports it into Ollama,
starts the server, and runs a test inference call.

Why Ollama over vLLM:
  - No server config; single binary on Windows
  - One-command model import from local directory
  - Instant API on localhost:11434 (OpenAI-compatible)
  - RTX 5090 CUDA acceleration auto-detected

Prerequisites:
  1. Install Ollama for Windows: https://ollama.com/download
  2. pip install requests
  3. pip install llama-cpp-python (only needed for the GGUF conversion path)
     OR install llama.cpp and run convert script manually (see comments below)

Usage:
    python script4_deployment_ollama.py --model_dir merged_model/
    python script4_deployment_ollama.py --model_dir merged_model/ --skip_convert
    python script4_deployment_ollama.py --chat   # interactive chat after deployment
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_MODEL_NAME = "venous-mistral"
OLLAMA_BASE_URL   = "http://localhost:11434"
GGUF_QUANT        = "Q4_K_M"   # good balance of size/quality on 32GB VRAM
DEFAULT_MODEL_DIR = Path("merged_model")
GGUF_OUTPUT_DIR   = Path("gguf_model")

MODELFILE_TEMPLATE = """\
FROM {gguf_path}

SYSTEM \"\"\"You are a medical expert specialising in venous and lymphatic disorders, \\
vascular surgery, duplex ultrasound, and haemodynamics. \\
Answer questions accurately using clinical and scientific knowledge.\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER num_gpu 99
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def run(cmd: list[str], **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, check=True, **kwargs)
    return result

def ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False

def wait_for_ollama(timeout: int = 30):
    print("Waiting for Ollama to start", end="", flush=True)
    for _ in range(timeout):
        if ollama_running():
            print(" ready.")
            return
        print(".", end="", flush=True)
        time.sleep(1)
    raise RuntimeError("Ollama did not start within timeout. Is it installed?")

# ── Step 1: Convert HF model → GGUF ──────────────────────────────────────────
def convert_to_gguf(model_dir: Path, output_dir: Path, quant: str) -> Path:
    """
    Uses llama.cpp convert_hf_to_gguf.py.
    Assumes llama.cpp is cloned alongside this project OR available on PATH.
    Falls back to llama-cpp-python's bundled converter.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = output_dir / f"model-{quant}.gguf"

    # Try llama.cpp convert script
    convert_candidates = [
        Path("llama.cpp/convert_hf_to_gguf.py"),
        Path("../llama.cpp/convert_hf_to_gguf.py"),
    ]
    converter = next((p for p in convert_candidates if p.exists()), None)

    if converter:
        print(f"\nStep 1a — Converting HF → GGUF (fp16 base)...")
        fp16_path = output_dir / "model-fp16.gguf"
        run([sys.executable, str(converter),
             str(model_dir), "--outtype", "f16", "--outfile", str(fp16_path)])

        print(f"\nStep 1b — Quantising to {quant}...")
        quantize_bin = Path("llama.cpp/build/bin/llama-quantize")
        if not quantize_bin.exists():
            quantize_bin = Path("llama.cpp/build/Release/llama-quantize.exe")
        if quantize_bin.exists():
            run([str(quantize_bin), str(fp16_path), str(gguf_path), quant])
        else:
            print("  [WARN] llama-quantize not found — using fp16 GGUF without quantisation.")
            gguf_path = fp16_path
    else:
        # Fallback: use transformers to save in safetensors and let Ollama handle it
        print("\n[INFO] llama.cpp converter not found.")
        print("       Ollama can import the HF safetensors directory directly.")
        print("       Proceeding without GGUF conversion — Ollama will quantise internally.\n")
        return model_dir  # return original HF dir; Ollama handles it

    print(f"GGUF model → {gguf_path}")
    return gguf_path

# ── Step 2: Create Modelfile and import into Ollama ──────────────────────────
def import_into_ollama(model_path: Path):
    modelfile_content = MODELFILE_TEMPLATE.format(
        gguf_path=str(model_path.resolve()).replace("\\", "/")
    )
    modelfile_path = Path("Modelfile")
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    print(f"\nModelfile written → {modelfile_path}")

    print(f"\nStep 2 — Importing into Ollama as '{OLLAMA_MODEL_NAME}'...")
    run(["ollama", "create", OLLAMA_MODEL_NAME, "-f", str(modelfile_path)])
    print(f"Model '{OLLAMA_MODEL_NAME}' registered in Ollama.")

# ── Step 3: Start Ollama server (if not already running) ──────────────────────
def ensure_server():
    if ollama_running():
        print("\nOllama server already running.")
        return
    print("\nStep 3 — Starting Ollama server...")
    subprocess.Popen(["ollama", "serve"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_for_ollama()

# ── Step 4: Test API call ─────────────────────────────────────────────────────
def test_inference(prompt: str = None) -> str:
    if prompt is None:
        prompt = (
            "What is the CHIVA strategy for treating varicose veins "
            "and what are its key haemodynamic principles?"
        )
    print(f"\nStep 4 — Test inference:\n  Q: {prompt}\n")
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.9},
    }
    t0 = time.time()
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    elapsed = time.time() - t0
    answer = data.get("response", "").strip()
    print(f"  A: {answer}\n")
    print(f"  Time: {elapsed:.1f}s | Tokens/s: {data.get('eval_count', 0)/elapsed:.1f}")
    return answer

# ── Interactive chat loop ─────────────────────────────────────────────────────
def chat_loop():
    print(f"\n=== Interactive Chat with {OLLAMA_MODEL_NAME} ===")
    print("Type your question and press Enter. Ctrl+C to exit.\n")
    while True:
        try:
            q = input("You: ").strip()
            if not q:
                continue
            payload = {
                "model": OLLAMA_MODEL_NAME,
                "prompt": q,
                "stream": True,
                "options": {"temperature": 0.3},
            }
            print("Model: ", end="", flush=True)
            with requests.post(f"{OLLAMA_BASE_URL}/api/generate",
                               json=payload, stream=True, timeout=120) as resp:
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        print(chunk.get("response", ""), end="", flush=True)
                        if chunk.get("done"):
                            break
            print("\n")
        except KeyboardInterrupt:
            print("\nExiting chat.")
            break

# ── OpenAI-compatible usage example ──────────────────────────────────────────
def print_api_usage():
    print("\n" + "="*60)
    print("OpenAI-compatible API (Ollama also exposes /v1/ endpoints):")
    print("="*60)
    print(f"""
from openai import OpenAI
client = OpenAI(base_url='{OLLAMA_BASE_URL}/v1', api_key='ollama')
response = client.chat.completions.create(
    model='{OLLAMA_MODEL_NAME}',
    messages=[{{'role': 'user', 'content': 'What is venous reflux?'}}],
    temperature=0.3,
)
print(response.choices[0].message.content)
""")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--skip_convert", action="store_true",
                        help="Skip GGUF conversion (model already converted)")
    parser.add_argument("--skip_import", action="store_true",
                        help="Skip Ollama import (model already imported)")
    parser.add_argument("--chat", action="store_true",
                        help="Enter interactive chat after setup")
    parser.add_argument("--quant", default=GGUF_QUANT,
                        help="GGUF quantisation level (default: Q4_K_M)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Convert
    if not args.skip_convert:
        model_path = convert_to_gguf(model_dir, GGUF_OUTPUT_DIR, args.quant)
    else:
        # Find existing GGUF or use HF dir
        gguf_files = list(GGUF_OUTPUT_DIR.glob("*.gguf"))
        model_path = gguf_files[0] if gguf_files else model_dir
        print(f"Skipping conversion. Using: {model_path}")

    # Import into Ollama
    if not args.skip_import:
        import_into_ollama(model_path)

    # Start server
    ensure_server()

    # Test call
    test_inference()

    # Usage snippet
    print_api_usage()

    # Optional interactive chat
    if args.chat:
        chat_loop()

    print(f"\nDeployment complete. Model '{OLLAMA_MODEL_NAME}' is live at {OLLAMA_BASE_URL}")

if __name__ == "__main__":
    main()

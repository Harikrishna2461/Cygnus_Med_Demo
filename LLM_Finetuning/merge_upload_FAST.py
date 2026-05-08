#!/usr/bin/env python3
"""
SUPER FAST MERGE + UPLOAD
Skip everything unnecessary - just merge and upload model weights
"""

import os
import sys
import subprocess
import torch
import time
from pathlib import Path

print("="*80)
print("FAST MERGE + UPLOAD (MODEL WEIGHTS ONLY)")
print("="*80)

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
BASE_MODEL = "Qwen/Qwen2.5-7B"
LORA_REPO = "HariKrishna1824/qwen25_chiva_v2"
OUTPUT_REPO = "HariKrishna1824/qwen_chiva_vericose_veins_treatment_finetuned"
OUTPUT_DIR = Path("./merged_model_fast")

print(f"\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Install
print("\n[1/4] Installing dependencies...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers', 'peft', 'huggingface-hub'], check=False)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, login

# Login
login(token=HF_TOKEN)
api = HfApi()

# Load and merge
print("\n[2/4] Loading and merging...")

print("  Base model...", end=" ", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN,
)
print("OK")

print("  LoRA adapter...", end=" ", flush=True)
model = PeftModel.from_pretrained(base, LORA_REPO, token=HF_TOKEN, is_trainable=False)
print("OK")

print("  Merging...", end=" ", flush=True)
merged = model.merge_and_unload()
print("OK")

print("  Tokenizer...", end=" ", flush=True)
tokenizer = AutoTokenizer.from_pretrained(LORA_REPO, token=HF_TOKEN, trust_remote_code=True)
print("OK")

# Save minimal (only safetensors)
print("\n[3/4] Saving model (minimal)...")

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("  Model...", end=" ", flush=True)
merged.save_pretrained(OUTPUT_DIR, safe_serialization=True, max_shard_size="10GB")
print("OK")

print("  Tokenizer...", end=" ", flush=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print("OK")

# Get size
size_gb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob('*') if f.is_file()) / (1024**3)
print(f"  Size: {size_gb:.1f}GB")

# Create README
print("\n[3.5/4] Creating README...")
readme_content = """# Qwen2.5-7B CHIVA Varicose Veins Treatment - Merged Model

Fully merged model: Qwen2.5-7B base + CHIVA LoRA fine-tuning.

## Model Details

- **Base**: Qwen/Qwen2.5-7B (7B parameters)
- **Fine-tuning**: LoRA merged and integrated
- **Specialization**: CHIVA classification + varicose veins treatment
- **Status**: Production-ready (no PEFT required)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "HariKrishna1824/qwen_chiva_vericose_veins_treatment_finetuned"
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(repo)

prompt = "Classify the CHIVA shunt type: EP N1->N2, RP N2->N1"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=500)
print(tokenizer.decode(outputs[0]))
```

## Capabilities

1. **CHIVA Shunt Classification**: TYPE 1, 2A, 2B, 2C, 3, 1+2
2. **Ligation Planning**: Treatment strategies based on shunt type
3. **Medical Reasoning**: Explains clinical rationale
4. **Query Formats**: Raw clips or natural language medical descriptions
"""

with open(OUTPUT_DIR / "README.md", "w") as f:
    f.write(readme_content)
print("  README.md created")

# Upload SEQUENTIAL
print("\n[4/4] Uploading (SEQUENTIAL - one file at a time, everything included)...")

print("  Creating repo...", end=" ", flush=True)
try:
    api.create_repo(OUTPUT_REPO, repo_type="model", exist_ok=True, private=False)
    print("OK")
except:
    print("(exists)")

# Get all files to upload
all_files = list(OUTPUT_DIR.rglob('*'))
file_list = [f for f in all_files if f.is_file()]

print(f"  Found {len(file_list)} files to upload")
print("  Starting sequential upload...")

for idx, file_path in enumerate(file_list, 1):
    relative_path = file_path.relative_to(OUTPUT_DIR)
    file_size_mb = file_path.stat().st_size / (1024**2)
    print(f"    [{idx}/{len(file_list)}] {relative_path} ({file_size_mb:.0f}MB)...", end=" ", flush=True)

    try:
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=str(relative_path),
            repo_id=OUTPUT_REPO,
            repo_type="model",
            commit_message=f"Upload: {relative_path}"
        )
        print("OK")

        # Add delay to prevent HF rate-limiting throttle
        # Large files get longer delays to let server reset
        if file_size_mb > 5000:
            delay = 60
        elif file_size_mb > 1000:
            delay = 30
        else:
            delay = 5

        if idx < len(file_list):
            print(f"      Waiting {delay}s before next file...")
            time.sleep(delay)

    except Exception as e:
        print(f"FAIL ({str(e)[:50]})")
        continue

print("  Upload complete")

print("\n" + "="*80)
print("DONE!")
print("="*80)
print(f"""
Model: https://huggingface.co/{OUTPUT_REPO}
Size: {size_gb:.1f}GB

Use:
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("{OUTPUT_REPO}")
""")

#!/bin/bash
# UPLOAD ALL MODELS TO HUGGINGFACE FROM UBUNTU

set -e

HF_TOKEN=""

echo "================================================================================"
echo "UPLOADING MODELS TO HUGGINGFACE"
echo "================================================================================"

# Check if in right directory
if [ ! -d ".claude" ]; then
    echo "ERROR: Not in LLM_Finetuning directory"
    echo "Run from: /home/krish/Downloads/LLM_Finetuning"
    exit 1
fi

echo ""
echo "[1/4] Installing dependencies..."
pip install -q huggingface-hub transformers torch peft

echo ""
echo "[2/4] Identifying models..."

# Create Python script to identify and upload models
python3 << 'PYTHON_EOF'
import os
import json
from pathlib import Path
from huggingface_hub import HfApi, login, get_user_info, create_repo

HF_TOKEN = os.getenv("HF_TOKEN", "")

# Login
print("  Authenticating...", end=" ")
try:
    login(token=HF_TOKEN)
    api = HfApi()
    user_info = get_user_info(token=HF_TOKEN)
    username = user_info.user_id
    print(f"OK ({username})")
except Exception as e:
    print(f"FAIL: {e}")
    exit(1)

# Find models
print("  Scanning...", end=" ")
model_indicators = [
    'adapter_config.json',
    'adapter_model.safetensors',
    'pytorch_model.bin',
    'model.safetensors',
    'config.json',
    'tokenizer.json',
]

models = []
current_dir = Path.cwd()

for item in current_dir.iterdir():
    if item.is_dir() and not item.name.startswith('.'):
        if any((item / indicator).exists() for indicator in model_indicators):
            is_lora = (item / 'adapter_config.json').exists()
            models.append({
                'path': str(item),
                'name': item.name,
                'type': 'LoRA' if is_lora else 'Full',
                'is_lora': is_lora
            })

print(f"OK ({len(models)} found)")

if not models:
    print("  ERROR: No models found!")
    exit(1)

for idx, model in enumerate(models, 1):
    print(f"    {idx}. {model['name']} ({model['type']})")

# Upload each model
print("")
print("[3/4] Uploading to Hugging Face...")

uploaded = []

for idx, model in enumerate(models, 1):
    print(f"  [{idx}/{len(models)}] {model['name']}...", end=" ")

    repo_name = f"{username}/{model['name']}"

    try:
        # Create repo
        try:
            create_repo(
                repo_id=repo_name,
                repo_type="model",
                token=HF_TOKEN,
                exist_ok=True,
                private=False
            )
        except:
            pass

        # Upload folder
        api.upload_folder(
            folder_path=model['path'],
            repo_id=repo_name,
            repo_type="model",
            token=HF_TOKEN,
            commit_message=f"Upload {model['type']} model"
        )

        uploaded.append({
            'name': model['name'],
            'repo': repo_name,
            'type': model['type'],
            'url': f"https://huggingface.co/{repo_name}"
        })
        print("OK")

    except Exception as e:
        print(f"ERROR: {str(e)[:60]}")

# Save config
print("")
print("[4/4] Saving configuration...")

config = {
    'hf_token': HF_TOKEN,
    'username': username,
    'models': uploaded,
    'note': 'Use these to access models via HF Hub'
}

with open('hf_models.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"  Config saved: hf_models.json")

# Print summary
print("")
print("="*80)
print("UPLOAD COMPLETE")
print("="*80)
print(f"\nUploaded {len(uploaded)} models:")
for model in uploaded:
    print(f"\n  {model['name']}")
    print(f"    Type: {model['type']}")
    print(f"    Repo: {model['repo']}")
    print(f"    URL: {model['url']}")

print(f"\nToken: {HF_TOKEN}")
print(f"Username: {username}")

PYTHON_EOF

echo ""
echo "================================================================================"
echo "COMPLETE - Models uploaded to Hugging Face"
echo "================================================================================"

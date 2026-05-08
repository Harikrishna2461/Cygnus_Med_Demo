#!/usr/bin/env python3
"""
AUTO-IDENTIFY AND UPLOAD ALL MODELS TO HUGGINGFACE
Run on Ubuntu with GPU access
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import json
import subprocess

# HF Token
HF_TOKEN = "YOUR_HF_TOKEN_HERE"
HF_ORG = "your_huggingface_username"  # Will be auto-detected

print("="*80)
print("HUGGINGFACE MODEL UPLOADER")
print("="*80)

# Install required packages
print("\n[1/5] Installing dependencies...")
packages = ['huggingface-hub', 'transformers', 'torch', 'peft']
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        print(f"  Installing {pkg}...", end=" ")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=False)
        print("OK")

from huggingface_hub import HfApi, login, get_user_info
from huggingface_hub import create_repo, upload_folder

# Login to HF
print("\n[2/5] Authenticating with Hugging Face...")
try:
    login(token=HF_TOKEN)
    api = HfApi()
    user_info = get_user_info(token=HF_TOKEN)
    username = user_info.user_id
    print(f"  Logged in as: {username}")
except Exception as e:
    print(f"  ERROR: {str(e)[:100]}")
    sys.exit(1)

# ============================================================
# AUTO-IDENTIFY MODELS
# ============================================================

print("\n[3/5] Scanning for models...")

def is_model_directory(path: Path) -> bool:
    """Check if directory contains a model"""
    model_indicators = [
        'adapter_config.json',
        'adapter_model.safetensors',
        'pytorch_model.bin',
        'model.safetensors',
        'config.json',
        'tokenizer.json',
        'tokenizer_config.json'
    ]
    return any((path / indicator).exists() for indicator in model_indicators)

# Scan for models
current_dir = Path.cwd()
models = []

for item in current_dir.iterdir():
    if item.is_dir() and not item.name.startswith('.'):
        if is_model_directory(item):
            # Determine model type
            is_lora = (item / 'adapter_config.json').exists()
            model_type = "LoRA" if is_lora else "Full"

            models.append({
                'path': item,
                'name': item.name,
                'type': model_type,
                'is_lora': is_lora
            })

print(f"  Found {len(models)} models:")
for idx, model in enumerate(models, 1):
    print(f"    {idx}. {model['name']} ({model['type']})")

if not models:
    print("  ERROR: No models found!")
    sys.exit(1)

# ============================================================
# UPLOAD MODELS TO HF
# ============================================================

print("\n[4/5] Uploading models to Hugging Face...")

uploaded_repos = []

for idx, model in enumerate(models, 1):
    model_name = model['name']
    repo_name = f"{username}/{model_name}"

    print(f"\n  [{idx}/{len(models)}] {model_name}", end=" ")

    try:
        # Create repo
        try:
            repo_url = create_repo(
                repo_id=repo_name,
                repo_type="model",
                token=HF_TOKEN,
                exist_ok=True
            )
            print("(repo)", end=" ")
        except:
            print("(exists)", end=" ")
            repo_url = f"https://huggingface.co/{repo_name}"

        # Upload folder
        try:
            api.upload_folder(
                folder_path=str(model['path']),
                repo_id=repo_name,
                repo_type="model",
                token=HF_TOKEN,
                commit_message=f"Upload {model['type']} model: {model_name}"
            )
            print("OK")
            uploaded_repos.append({
                'name': model_name,
                'repo': repo_name,
                'url': f"https://huggingface.co/{repo_name}",
                'type': model['type'],
                'path': str(model['path'])
            })
        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")

    except Exception as e:
        print(f"FAIL: {str(e)[:50]}")

# ============================================================
# GENERATE CONFIG FOR API ACCESS
# ============================================================

print("\n[5/5] Generating API configuration...")

config = {
    'hf_token': HF_TOKEN,
    'username': username,
    'models': uploaded_repos,
    'timestamp': str(Path.cwd())
}

# Save config
with open('hf_models_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"  Config saved: hf_models_config.json")

# Generate usage script
usage_script = f'''#!/usr/bin/env python3
"""
USE UPLOADED MODELS VIA HUGGINGFACE API
"""

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

HF_TOKEN = "{HF_TOKEN}"
USERNAME = "{username}"

# Available models:
MODELS = {{
{chr(10).join([f'    "{m["name"]}": "{m["repo"]},"' for m in uploaded_repos])}
}}

def load_model(model_name: str, is_lora: bool = False):
    """Load model from Hugging Face"""
    repo_id = MODELS[model_name]

    if is_lora:
        # Load base model + LoRA
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=HF_TOKEN
        )
        model = PeftModel.from_pretrained(repo_id, token=HF_TOKEN)
    else:
        # Load full model
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=HF_TOKEN
        )

    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=HF_TOKEN)
    return model, tokenizer

# Example usage:
# model, tokenizer = load_model("lora_finetuned_model", is_lora=True)
'''

with open('hf_api_usage.py', 'w') as f:
    f.write(usage_script)

print(f"  Usage script: hf_api_usage.py")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*80)
print("UPLOAD COMPLETE")
print("="*80)

print(f"\nUploaded {len(uploaded_repos)} models to Hugging Face:")
for model in uploaded_repos:
    print(f"\n  Model: {model['name']}")
    print(f"  Type: {model['type']}")
    print(f"  Repository: {model['repo']}")
    print(f"  URL: {model['url']}")

print(f"\nHugging Face Token: {HF_TOKEN}")
print(f"Username: {username}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
1. Use the uploaded models via HuggingFace Hub:
   - from huggingface_hub import hf_hub_download
   - model = AutoModelForCausalLM.from_pretrained(
       "{username}/model_name",
       token=HF_TOKEN
     )

2. Files generated:
   - hf_models_config.json: Configuration for all uploaded models
   - hf_api_usage.py: Example script for loading models

3. To use in evaluation scripts:
   - Import the config from hf_models_config.json
   - Load models using HF Hub instead of local paths
""")

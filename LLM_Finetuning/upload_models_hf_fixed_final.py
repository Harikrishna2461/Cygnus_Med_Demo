#!/usr/bin/env python3
"""
UPLOAD ALL MODELS TO HUGGINGFACE
Fixed for huggingface_hub compatibility
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
HF_USERNAME = "HariKrishna1824"

print("="*80)
print("HUGGINGFACE MODEL UPLOADER")
print("="*80)

# Check directory
current_dir = Path.cwd()
print(f"\nCurrent directory: {current_dir}")

# Install dependencies
print("\n[1/5] Installing/updating dependencies...")
packages = ['huggingface-hub', 'transformers', 'peft']
for pkg in packages:
    print(f"  {pkg}...", end=" ", flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=False)
    print("OK")

# Import after installing
print("\n[2/5] Authenticating with Hugging Face...")
try:
    from huggingface_hub import HfApi, login

    # Login with token
    login(token=HF_TOKEN)
    api = HfApi()

    print(f"  Logged in as: {HF_USERNAME}")
    print(f"  Ready to upload")
except Exception as e:
    print(f"  ERROR: {str(e)}")
    sys.exit(1)

# Find models
print("\n[3/5] Scanning for models...")

model_indicators = [
    'adapter_config.json',
    'adapter_model.safetensors',
    'pytorch_model.bin',
    'model.safetensors',
    'config.json',
    'tokenizer.json',
    'tokenizer_config.json'
]

models = []

# Scan current directory
for item in current_dir.iterdir():
    if item.is_dir() and not item.name.startswith('.'):
        if any((item / indicator).exists() for indicator in model_indicators):
            is_lora = (item / 'adapter_config.json').exists()
            model_type = "LoRA" if is_lora else "Full"

            # Get size
            try:
                size_bytes = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                size_gb = size_bytes / (1024**3)
            except:
                size_gb = 0

            models.append({
                'path': item,
                'name': item.name,
                'type': model_type,
                'is_lora': is_lora,
                'size_gb': size_gb
            })

# Check additional directories
additional_dirs = [
    current_dir / "Models",
    current_dir / "finetuning" / "Models",
    current_dir / "finetuning" / "lora_chiva_classifier_final"
]

for check_dir in additional_dirs:
    if check_dir.exists():
        if check_dir.is_dir():
            # If it's a model directory itself
            if any((check_dir / indicator).exists() for indicator in model_indicators):
                is_lora = (check_dir / 'adapter_config.json').exists()
                model_type = "LoRA" if is_lora else "Full"

                try:
                    size_bytes = sum(f.stat().st_size for f in check_dir.rglob('*') if f.is_file())
                    size_gb = size_bytes / (1024**3)
                except:
                    size_gb = 0

                # Avoid duplicates
                if not any(m['path'] == check_dir for m in models):
                    models.append({
                        'path': check_dir,
                        'name': check_dir.name,
                        'type': model_type,
                        'is_lora': is_lora,
                        'size_gb': size_gb
                    })
            else:
                # If it contains model subdirectories
                try:
                    for item in check_dir.iterdir():
                        if item.is_dir() and not item.name.startswith('.'):
                            if any((item / indicator).exists() for indicator in model_indicators):
                                is_lora = (item / 'adapter_config.json').exists()
                                model_type = "LoRA" if is_lora else "Full"

                                try:
                                    size_bytes = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                                    size_gb = size_bytes / (1024**3)
                                except:
                                    size_gb = 0

                                # Avoid duplicates
                                if not any(m['path'] == item for m in models):
                                    models.append({
                                        'path': item,
                                        'name': item.name,
                                        'type': model_type,
                                        'is_lora': is_lora,
                                        'size_gb': size_gb
                                    })
                except:
                    pass

if not models:
    print("  ERROR: No models found!")
    print("  Make sure you're in the correct directory")
    sys.exit(1)

models = sorted(models, key=lambda x: x['name'])

print(f"  Found {len(models)} models:")
for idx, model in enumerate(models, 1):
    print(f"    {idx}. {model['name']} ({model['type']}) - {model['size_gb']:.2f}GB")

# Upload models
print("\n[4/5] Uploading to Hugging Face...")

uploaded = []
failed = []

for idx, model in enumerate(models, 1):
    model_name = model['name']
    repo_id = f"{HF_USERNAME}/{model_name}"

    print(f"  [{idx}/{len(models)}] {model_name}...", end=" ", flush=True)

    try:
        # Create repo
        try:
            print("(creating)", end=" ", flush=True)
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=False
            )
            print("(uploading)", end=" ", flush=True)
        except Exception as e:
            if "already exists" in str(e).lower():
                print("(uploading)", end=" ", flush=True)
            else:
                raise

        # Upload folder
        api.upload_folder(
            folder_path=str(model['path']),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {model['type']} model",
            ignore_patterns=["*.git*", "*.md", "README*", ".gitignore"]
        )

        print("OK")

        uploaded.append({
            'name': model_name,
            'repo': repo_id,
            'type': model['type'],
            'url': f"https://huggingface.co/{repo_id}",
            'is_lora': model['is_lora'],
            'size_gb': model['size_gb']
        })

    except Exception as e:
        error_msg = str(e)[:80]
        print(f"FAILED")
        print(f"    Error: {error_msg}")
        failed.append({
            'name': model_name,
            'error': error_msg
        })

# Save config
print("\n[5/5] Saving configuration...")

config = {
    'hf_token': HF_TOKEN,
    'username': HF_USERNAME,
    'uploaded_count': len(uploaded),
    'failed_count': len(failed),
    'models': uploaded
}

with open('hf_models_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"  Saved: hf_models_config.json")

# Summary
print("\n" + "="*80)
print("UPLOAD SUMMARY")
print("="*80)

print(f"\nSuccessfully uploaded: {len(uploaded)}/{len(models)} models")

if uploaded:
    print("\nUploaded models:")
    for model in uploaded:
        print(f"\n  {model['name']}")
        print(f"    Type: {model['type']}")
        print(f"    Size: {model['size_gb']:.2f}GB")
        print(f"    Repo: {model['repo']}")
        print(f"    URL: {model['url']}")

if failed:
    print(f"\nFailed: {len(failed)} models")
    for model in failed:
        print(f"  - {model['name']}: {model['error']}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print(f"""
1. Models are now on Hugging Face at:
   https://huggingface.co/{HF_USERNAME}

2. Load models in Python:
   from transformers import AutoModelForCausalLM
   from peft import PeftModel

   token = "{HF_TOKEN}"

   # For LoRA models:
   base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", token=token)
   model = PeftModel.from_pretrained(base, "{HF_USERNAME}/model_name", token=token)

   # For full models:
   model = AutoModelForCausalLM.from_pretrained(
       "{HF_USERNAME}/model_name",
       token=token
   )

3. Use in API calls:
   model_repo = "{HF_USERNAME}/model_name"
   # Pass to your evaluation script

4. Configuration saved in: hf_models_config.json
   (Use this for API access)
""")

print("="*80)
print("COMPLETE")
print("="*80)

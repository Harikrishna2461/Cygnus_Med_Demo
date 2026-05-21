#!/usr/bin/env python3
"""
UPLOAD ALL MODELS TO HUGGINGFACE
Run on Ubuntu with GPU: python3 hf_upload_all_models.py
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict

HF_TOKEN = ""
HF_USERNAME = "HariKrishna1824"  # Your HuggingFace username

def run_cmd(cmd, check=True):
    """Run shell command"""
    return subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)

def install_deps():
    """Install required packages"""
    packages = [
        'huggingface-hub',
        'transformers',
        'torch',
        'peft'
    ]

    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            print(f"  Installing {pkg}...", end=" ", flush=True)
            run_cmd(f"pip install -q {pkg}", check=False)
            print("OK")

def find_models(root_dir: Path) -> List[Dict]:
    """Find all model directories"""
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

    for item in root_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            if any((item / indicator).exists() for indicator in model_indicators):
                is_lora = (item / 'adapter_config.json').exists()
                models.append({
                    'path': item,
                    'name': item.name,
                    'type': 'LoRA' if is_lora else 'Full',
                    'is_lora': is_lora
                })

    return sorted(models, key=lambda x: x['name'])

def main():
    print("="*80)
    print("HUGGINGFACE MODEL UPLOADER")
    print("="*80)

    # Check current directory
    current_dir = Path.cwd()
    print(f"\nCurrent directory: {current_dir}")

    # Install dependencies
    print("\n[1/5] Installing dependencies...")
    install_deps()

    # Import HF libraries
    print("\n[2/5] Authenticating with Hugging Face...")
    try:
        from huggingface_hub import HfApi, login
        login(token=HF_TOKEN)
        api = HfApi()

        # Use provided username
        username = HF_USERNAME

        print(f"  Logged in as: {username}")
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        sys.exit(1)

    # Find models
    print("\n[3/5] Scanning for models...")
    models = find_models(current_dir)

    if not models:
        print("  ERROR: No models found in current directory!")
        print("  Make sure you're in the directory with model folders")
        sys.exit(1)

    print(f"  Found {len(models)} models:")
    for idx, model in enumerate(models, 1):
        model_size = sum(f.stat().st_size for f in model['path'].rglob('*') if f.is_file())
        size_gb = model_size / (1024**3)
        print(f"    {idx}. {model['name']} ({model['type']}) - {size_gb:.2f}GB")

    # Upload models
    print("\n[4/5] Uploading models to Hugging Face...")

    uploaded_repos = []
    failed_models = []

    for idx, model in enumerate(models, 1):
        model_name = model['name']
        repo_id = f"{username}/{model_name}"

        print(f"  [{idx}/{len(models)}] {model_name}...", end=" ", flush=True)

        try:
            # Create repo if doesn't exist
            try:
                from huggingface_hub import create_repo
                create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    token=HF_TOKEN,
                    exist_ok=True,
                    private=False
                )
                print("(created)", end=" ", flush=True)
            except Exception as e:
                if "already exists" in str(e):
                    print("(exists)", end=" ", flush=True)
                else:
                    raise

            # Upload folder
            api.upload_folder(
                folder_path=str(model['path']),
                repo_id=repo_id,
                repo_type="model",
                token=HF_TOKEN,
                commit_message=f"Upload {model['type']} model: {model_name}",
                ignore_patterns=["*.git*", "*.md", "README*"]
            )

            print("OK")
            uploaded_repos.append({
                'name': model_name,
                'repo': repo_id,
                'type': model['type'],
                'url': f"https://huggingface.co/{repo_id}",
                'path': str(model['path']),
                'is_lora': model['is_lora']
            })

        except Exception as e:
            print(f"FAIL: {str(e)[:60]}")
            failed_models.append({
                'name': model_name,
                'error': str(e)[:100]
            })

    # Save configuration
    print("\n[5/5] Saving configuration...")

    config = {
        'hf_token': HF_TOKEN,
        'username': username,
        'timestamp': str(Path.cwd()),
        'uploaded_models': uploaded_repos,
        'failed_models': failed_models
    }

    config_file = Path('hf_models_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: hf_models_config.json")

    # Generate usage script
    usage_script_content = f"""#!/usr/bin/env python3
\"\"\"
LOAD MODELS FROM HUGGINGFACE HUB
Generated automatically - do not edit
\"\"\"

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json

HF_TOKEN = "{HF_TOKEN}"
USERNAME = "{username}"

# Load config
with open('hf_models_config.json', 'r') as f:
    config = json.load(f)

def load_model(model_name: str, device: str = "auto"):
    \"\"\"Load model from Hugging Face Hub\"\"\"

    repo_id = f"{{USERNAME}}/{{model_name}}"
    model_config = next((m for m in config['uploaded_models'] if m['name'] == model_name), None)

    if not model_config:
        raise ValueError(f"Model {{model_name}} not found in config")

    print(f"Loading {{model_name}} from {{repo_id}}...")

    if model_config['is_lora']:
        # Load base model
        print("  Loading base model (Qwen2.5-7B)...", end=" ")
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B",
            torch_dtype=torch.bfloat16,
            device_map=device,
            token=HF_TOKEN,
            trust_remote_code=True,
        )
        print("OK")

        # Load LoRA adapters
        print("  Loading LoRA adapters...", end=" ")
        model = PeftModel.from_pretrained(
            base_model,
            repo_id,
            token=HF_TOKEN,
            is_trainable=False
        )
        print("OK")
    else:
        # Load full model
        print("  Loading full model...", end=" ")
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            token=HF_TOKEN,
            trust_remote_code=True,
        )
        print("OK")

    # Load tokenizer
    print("  Loading tokenizer...", end=" ")
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    print("OK")

    model.eval()
    return model, tokenizer

def list_models():
    \"\"\"List all available models\"\"\"
    print("\\nAvailable models:")
    for model in config['uploaded_models']:
        print(f"  - {{model['name']}} ({{model['type']}})")
        print(f"    URL: {{model['url']}}")

# Example usage:
if __name__ == "__main__":
    list_models()

    # To load a model:
    # model, tokenizer = load_model("model_name")
"""

    usage_file = Path('load_hf_models.py')
    with open(usage_file, 'w') as f:
        f.write(usage_script_content)
    print(f"  Saved: load_hf_models.py")

    # Summary
    print("\n" + "="*80)
    print("UPLOAD COMPLETE")
    print("="*80)

    print(f"\nSuccessfully uploaded {len(uploaded_repos)} models:")
    for model in uploaded_repos:
        print(f"\n  Model: {model['name']}")
        print(f"    Type: {model['type']}")
        print(f"    Repo: {model['repo']}")
        print(f"    URL: {model['url']}")

    if failed_models:
        print(f"\n{len(failed_models)} models failed to upload:")
        for model in failed_models:
            print(f"  - {model['name']}: {model['error']}")

    print(f"\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"HF Token: {HF_TOKEN}")
    print(f"Username: {username}")
    print(f"Config file: hf_models_config.json")
    print(f"Usage script: load_hf_models.py")

    print(f"\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Import and use models:
   from load_hf_models import load_model
   model, tokenizer = load_model("model_name")

2. In your evaluation scripts:
   - Load config from hf_models_config.json
   - Use load_hf_models.py to load models
   - Models now accessible via HuggingFace Hub API

3. Access from anywhere:
   from huggingface_hub import hf_hub_download
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained(
       "{username}/model_name",
       token=HF_TOKEN
   )
""")

if __name__ == "__main__":
    main()

# Upload All Models to Hugging Face

## Quick Start (Ubuntu/Linux with GPU)

### 1. Copy script to your server
```bash
# On Ubuntu server
cd /path/to/LLM_Finetuning
# Copy hf_upload_all_models.py to this directory
```

### 2. Run the uploader
```bash
python3 hf_upload_all_models.py
```

That's it! The script will:
- ✓ Auto-identify all models (3-4 in your case)
- ✓ Create repos on Hugging Face
- ✓ Upload all models (LoRA + full models)
- ✓ Save configuration file
- ✓ Generate usage script

### 3. What gets created
- `hf_models_config.json` - Configuration of all uploaded models
- `load_hf_models.py` - Script to load models from HF Hub

## Using Uploaded Models in Your Code

### Option 1: Use the auto-generated script
```python
from load_hf_models import load_model

# Load a model
model, tokenizer = load_model("model_name")

# Or list available models
from load_hf_models import list_models
list_models()
```

### Option 2: Load directly from HF Hub
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

HF_TOKEN = "YOUR_HF_TOKEN_HERE"
username = "your_hf_username"  # Will be shown after upload

# Load LoRA model
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    token=HF_TOKEN
)
model = PeftModel.from_pretrained(
    base,
    f"{username}/model_name",
    token=HF_TOKEN
)

# Load full model
model = AutoModelForCausalLM.from_pretrained(
    f"{username}/model_name",
    token=HF_TOKEN
)

tokenizer = AutoTokenizer.from_pretrained(
    f"{username}/model_name",
    token=HF_TOKEN
)
```

## Environment Setup (if needed)

### CUDA Support
```bash
# For GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Install dependencies
```bash
pip install transformers peft huggingface-hub
```

## Troubleshooting

### "Token authentication failed"
- Check HF_TOKEN is correct: `YOUR_HF_TOKEN_HERE`
- Ensure token has write permissions on Hugging Face

### "No models found"
- Run from the directory containing model folders
- Models must have `adapter_config.json` or `config.json`

### "Upload takes too long"
- This is normal for large models (7B+)
- Can take 30min-1hr per model depending on internet
- Script will continue even if interrupted (use `exist_ok=True`)

## Models That Will Be Uploaded

The script auto-detects:
1. **LoRA adapters** (directories with `adapter_config.json`)
   - `lora_finetuned_model`
   - `lora_chiva_classifier`
   - `lora_finetuned_output`

2. **Full models** (directories with `pytorch_model.bin` or `model.safetensors`)

3. **Other model types** (any directory with tokenizer/config files)

## Token Details
- **Token**: `YOUR_HF_TOKEN_HERE`
- **Scope**: Full access to your repositories
- **After upload**: Models will be at `https://huggingface.co/{username}/model_name`

## After Upload

### Update your evaluation script:
```python
from huggingface_hub import hf_hub_download
from load_hf_models import load_model

# Load Qwen model
qwen_model, qwen_tokenizer = load_model("lora_finetuned_model")

# Now use in evaluation
output = qwen_model.generate(...)
```

### Use in API:
```python
from groq import Groq
from transformers import AutoModelForCausalLM

# LLAMA via Groq
client = Groq(api_key="your_groq_key")

# Qwen via HF Hub
model = AutoModelForCausalLM.from_pretrained(
    "username/lora_finetuned_model",
    token=HF_TOKEN
)
```

## Files Included

1. **hf_upload_all_models.py** - Main uploader script
2. **upload_models_to_hf.py** - Alternative version (Python only)
3. **upload_to_hf.sh** - Bash script version
4. **HF_UPLOAD_README.md** - This file

## Support

If upload fails:
1. Check internet connection
2. Verify HF token is valid
3. Ensure sufficient disk space for temporary files
4. Check model directory permissions

For issues: Re-run script - it supports `exist_ok=True` so can resume from where it failed.

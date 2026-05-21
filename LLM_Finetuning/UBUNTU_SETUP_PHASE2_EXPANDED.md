# Phase 2 Training on Ubuntu - Expanded Dataset Setup Guide

## Quick Summary

✓ **300 synthetic CHIVA examples** generated (10x variations per real patient case)
✓ **Phase 1 LoRA merged** into base model (medical knowledge baked in)
✓ **Fresh Phase 2 LoRA** applied for task-specific training
✓ **10 epochs** on larger dataset = better generalization

---

## Quick Start

### 1. Copy Files to Ubuntu Machine

```bash
# Copy the expanded dataset
scp training_data_EXPANDED.jsonl user@ubuntu-machine:/home/user/llm_finetuning/latest_data/

# Copy the Jupyter notebook
scp Phase2_CHIVA_Training_With_Phase1.ipynb user@ubuntu-machine:/home/user/llm_finetuning/

# Verify Phase 1 checkpoint exists (on Ubuntu)
ssh user@ubuntu-machine "ls -lh /home/user/llm_finetuning/qwen_medical_lora_gpu/adapter_model.bin"
```

Or use any file transfer method (rsync, SFTP, etc.)

### 2. Connect to Ubuntu Machine

```bash
ssh user@ubuntu-machine
cd /home/user/llm_finetuning
```

### 3. Install Dependencies (if not already done)

```bash
# Update pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install HuggingFace and dependencies
pip install transformers peft jsonlines matplotlib

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 4. Start Jupyter Server

```bash
# From your llm_finetuning directory
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Copy the URL (will look like: http://localhost:8888/?token=...)
```

### 5. Connect from Your Local Machine

```bash
# SSH tunnel (from your Windows/Mac machine)
ssh -L 8888:localhost:8888 user@ubuntu-machine

# Then open browser: http://localhost:8888
```

### 6. Open the Notebook

- Navigate to `Phase2_CHIVA_Training_With_Phase1.ipynb` in Jupyter
- **IMPORTANT**: Edit Cell 1 paths:
  ```python
  BASE_DIR = "/home/your_username/llm_finetuning"  # CHANGE THIS
  PHASE1_LORA = "/home/your_username/llm_finetuning/qwen_medical_lora_gpu"  # CHANGE THIS
  ```
- Follow the cells in order (1 → 2 → 3 → 4)

---

## File Structure on Ubuntu

```
/home/user/llm_finetuning/
├── latest_data/
│   └── training_data_EXPANDED.jsonl          ← Dataset (300 examples)
│   └── training_data_FRESH.jsonl             ← Old dataset (103 examples)
├── qwen_medical_lora_gpu/                    ← Phase 1 checkpoint (load this)
│   ├── adapter_model.bin
│   ├── adapter_config.json
│   └── ...
├── qwen_chiva_tasks_lora/                    ← Output (Phase 2 will save here)
│   ├── adapter_model.bin
│   ├── adapter_config.json
│   ├── tokenizer.json
│   └── training_config.json
├── .cache/                                   ← HuggingFace cache
├── Phase2_CHIVA_Training_With_Phase1.ipynb   ← This notebook
└── Phase2_Training_Ubuntu.py                 ← Alternative script format (optional)
```

---

## Notebook Walkthrough

### Cell 1: Setup, Merge Phase 1, Apply Phase 2 LoRA
**⏱️ Takes 2-3 minutes** (model download + setup)

Steps:
1. Loads base Qwen2.5-7B model
2. Loads Phase 1 LoRA adapter (`qwen_medical_lora_gpu/`)
3. **Merges Phase 1 into base model** (medical knowledge baked in)
4. Applies fresh Phase 2 LoRA for task-specific training
5. Loads tokenizer and dataset class

What you should see:
```
✓ Imports successful
✓ Base model loaded - 7.0B parameters
✓ Phase 1 LoRA loaded from: /home/user/llm_finetuning/qwen_medical_lora_gpu
✓ Phase 1 merged (medical knowledge baked into weights)
✓ Phase 2 LoRA applied
  Trainable: 3.7M (0.05%)
  Total: 7.0B
✓ Dataset found: 300 examples
```

### Cell 2: Prepare Dataset & Optimizer
**⏱️ Instant**

Configures:
- DataLoader with batch_size=2, shuffle=True
- Adam optimizer (lr=1e-4)
- CosineAnnealingLR scheduler (10 epochs)

### Cell 3: Training Loop
**⏱️ Takes 10-15 minutes** (300 examples × 10 epochs with GPU)

Expected output:
```
Epoch 1/10
  Batch 10/150, Loss: 3.2341, Avg: 3.3421
  Batch 20/150, Loss: 3.1234, Avg: 3.2156
...
✓ Epoch 1 - Loss: 3.1234
✓ Epoch 2 - Loss: 3.0122
...
✓ Epoch 10 - Loss: 2.2145
Final loss: 2.2145
Total steps: 1500
```

### Cell 4: Save, Plot, Test
**⏱️ Instant**

- Saves model to `qwen_chiva_tasks_lora/`
- Plots training loss curve
- Tests on 4 CHIVA queries
- Provides deployment code

---

## Expected Results

**Training Loss Progression (10 epochs, 300 examples):**
- Epoch 1: ~3.2
- Epoch 3: ~3.0
- Epoch 5: ~2.7
- Epoch 7: ~2.4
- Epoch 10: ~2.2

**Typical Test Results (Cell 4):**
```
Q: Classify: EP N1->N2 at y=0.06 with RP N2->N1 at y=0.25. No N3.
A: TYPE 1 classification with SFJ incompetence...

Q: For TYPE 1 shunt, what is the ligation strategy?
A: Ligation at the SFJ (y ≤ 0.098) or at the RP N2→N1 point...

Q: Classify: EP N2->N3 at y=0.20 with no EP N1->N2.
A: TYPE 2A (GSV feeding tributary with competent SFJ)...

Q: What is TYPE 2B?
A: TYPE 2B is when EP N2→N2 (perforator entry) with RP at N3, no RP N2→N1...
```

---

## Troubleshooting

### CUDA Out of Memory
```python
# In Cell 3, reduce batch size before running
batch_size = 1  # instead of 2
```

### Phase 1 Checkpoint Not Found
If Cell 1 shows:
```
⚠ Warning: Could not load Phase 1 LoRA: No such file or directory
```

Check:
```bash
ls -la /home/user/llm_finetuning/qwen_medical_lora_gpu/adapter_model.bin

# If it doesn't exist, verify Phase 1 training completed:
ls -la /home/user/llm_finetuning/qwen_medical_lora_gpu/
```

If Phase 1 checkpoint missing, the model will train with fresh LoRA (not ideal but functional).

### "No module named 'transformers'"
```bash
pip install transformers --upgrade
```

### Dataset Not Found
```bash
# Check dataset exists
ls -la /home/user/llm_finetuning/latest_data/training_data_EXPANDED.jsonl

# Count lines
wc -l /home/user/llm_finetuning/latest_data/training_data_EXPANDED.jsonl
```

Should show 300 lines.

### Slow Training
- CPU only: 30-45 minutes
- With GPU (RTX 4090/5090): 10-15 minutes
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

---

## After Training: Using the Model

### Method 1: In Python
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE_MODEL = "Qwen/Qwen2.5-7B"
LORA_PATH = "/home/user/llm_finetuning/qwen_chiva_tasks_lora"

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, LORA_PATH)
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

# Inference
model.eval()
prompt = "Classify: EP N1->N2 at y=0.06 with RP N2->N1. No N3."
inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(inputs, max_new_tokens=100)
    
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Method 2: Copy Model to Server
```bash
rsync -avz qwen_chiva_tasks_lora/ user@server:/models/qwen_chiva_tasks_lora/
```

---

## GPU Performance

| GPU | VRAM | Training Time | Batch Size |
|-----|------|---------------|-----------|
| RTX 5090 | 32GB | ~12 min | 2 |
| RTX 4090 | 24GB | ~15 min | 2 |
| RTX A6000 | 48GB | ~10 min | 4 |
| A100 | 80GB | ~8 min | 8 |
| CPU Only | - | ~40 min | 1 |

---

## Key Differences from Previous Version

| Aspect | Old (103 examples) | New (300 examples) |
|--------|-------------------|-------------------|
| Dataset | training_data_FRESH.jsonl | training_data_EXPANDED.jsonl |
| Synthetic variations | 3 per real case | 10 per real case |
| Total training examples | 103 | 300 |
| Training time (GPU) | 15-20 min | 10-15 min |
| Expected loss convergence | Weaker | Stronger |
| Phase 1 integration | Not used | Merged into base |
| LoRA adapter chaining | N/A | Phase 1 merged + Phase 2 fresh |

---

## Success Checklist

- [ ] Files copied to Ubuntu (`training_data_EXPANDED.jsonl`, notebook, `qwen_medical_lora_gpu/`)
- [ ] Dependencies installed
- [ ] Jupyter running
- [ ] Notebook opened
- [ ] Paths updated in Cell 1 (BASE_DIR, PHASE1_LORA)
- [ ] Cell 1 runs without errors (model loads, Phase 1 merges)
- [ ] Cell 2 runs without errors (dataset=300 examples)
- [ ] Cell 3 training completes (loss decreases each epoch)
- [ ] Cell 4 shows test outputs (reasonable CHIVA classifications)
- [ ] Model saved to `qwen_chiva_tasks_lora/`

---

## What's Different from Old Setup?

**Old approach (103 examples, fresh LoRA):**
- Trained small LoRA on base Qwen2.5-7B
- No Phase 1 pretraining used
- Higher hallucination risk

**New approach (300 examples, Phase 1 merged):**
- Phase 1 LoRA (medical pretraining) merged into base
- Fresh Phase 2 LoRA trained on 300 examples
- Medical knowledge + task-specific knowledge
- Better generalization expected

---

**Last Updated:** 2026-05-15  
**Dataset Version:** training_data_EXPANDED.jsonl (300 examples)  
**Model:** Qwen2.5-7B + Phase 1 merged + Phase 2 LoRA  
**Training Time:** 10-15 minutes on GPU

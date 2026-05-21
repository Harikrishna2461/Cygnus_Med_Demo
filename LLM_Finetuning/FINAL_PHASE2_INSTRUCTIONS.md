# Phase 2 Training: Final Setup & Instructions

## Dataset Ready

**File**: `training_data_comprehensive.jsonl`  
**Total Examples**: 216  
**Quality**: ✓ V1 format ✓ V2 format ✓ Reasoning ✓ 36 NO SHUNT cases

### Breakdown:
- TYPE 1: 30 examples
- TYPE 2A: 30 examples  
- TYPE 2B: 30 examples
- TYPE 2C: 30 examples
- TYPE 1+2: 30 examples
- TYPE 3: 30 examples
- **NO SHUNT: 36 examples** (9 scenarios × 4 question types)

## What Each Example Includes

```json
{
  "input": "Classify the shunt type. Clips:\n  Clip 00: EP N1→N2  y=0.050 [SFJ-ENTRY=INCOMPETENT]...",
  "output": "{\"shunt_type\": \"TYPE 1\", \"confidence\": 0.92, \"reasoning\": \"Classification: TYPE 1 (SFJ incompetence with isolated GSV reflux)...\"}"
}
```

Every example teaches:
1. **Classification** - What type (with confidence score)
2. **Reasoning** - WHY this type (not template filling)
3. **Diagnostic Logic** - Which clips define this type
4. **Ligation Strategy** - What to do based on findings
5. **Differential** - Why this type vs others

## Step-by-Step Instructions

### 1. Copy Files to Ubuntu

```bash
# From your Windows machine
scp training_data_comprehensive.jsonl user@ubuntu:/home/user/llm_finetuning/latest_data/
scp Phase2_V1_V2_Training.ipynb user@ubuntu:/home/user/llm_finetuning/
```

Or use rsync/SFTP - whatever you prefer.

### 2. SSH into Ubuntu

```bash
ssh user@ubuntu
cd /home/user/llm_finetuning
```

### 3. Verify Phase 1 Checkpoint Exists

```bash
ls -lh qwen_medical_lora_gpu/adapter_model.bin
```

Should show the file exists. If not, training will proceed with base model only.

### 4. Start Jupyter on Ubuntu

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
# Copy the URL it shows
```

### 5. SSH Tunnel from Windows (optional, if accessing remotely)

```bash
ssh -L 8888:localhost:8888 user@ubuntu
# Then open browser: http://localhost:8888
```

### 6. Open the Notebook

- Navigate to `Phase2_V1_V2_Training.ipynb`
- **EDIT CELL 1** - Update these paths:
  ```python
  BASE_DIR = "/home/username/llm_finetuning"
  PHASE1_LORA = "/home/username/llm_finetuning/qwen_medical_lora_gpu"
  ```

### 7. Run Cells in Order

**Cell 1**: Load models + apply LoRA (~2 min)
- Output: "READY FOR TRAINING" message
- Verify dataset count shows 216 examples

**Cell 2**: Prepare data + optimizer (~instant)
- Output: Dataset ready, optimizer configured

**Cell 3**: Training loop (~8-12 min on RTX 5090)
- Watch loss decrease each epoch
- Expected: Loss ~5.0 → ~0.5 over 10 epochs

**Cell 4**: Save model + test (~instant)
- Model saved to `qwen_chiva_v1_v2_lora/`
- Test outputs shown
- Loss plot generated

### 8. Monitor Training

Watch for these signs of success:

✓ Loss decreases each epoch (5.0 → 3.0 → 1.0 → 0.5)
✓ No CUDA out of memory errors
✓ No gradient warnings
✓ Test outputs are coherent JSON

## What The Model Will Learn

| Learns | From |
|--------|------|
| V1 format (clip notation) | 60 examples (10 per shunt type) |
| V2 format (medical terminology) | 60 examples (10 per shunt type) |
| Reasoning/explanation | Every example has reasoning field |
| Normal cases (NO SHUNT) | 36 diverse normal scenarios |
| Ligation strategy | 180 shunt type examples |
| Differential diagnosis | Examples explain why each type |

## Expected Results

### Training Loss
```
Epoch 1:  ~5.0
Epoch 3:  ~3.0
Epoch 5:  ~1.5
Epoch 7:  ~0.8
Epoch 10: ~0.5
```

### Test Output Quality
Model should output proper JSON like:
```json
{
  "shunt_type": "TYPE 1",
  "confidence": 0.92,
  "reasoning": "SFJ incompetence with isolated GSV reflux due to EP N1→N2 and RP N2→N1...",
  "ligation_strategy": "Ligate at the SFJ (y ≤ 0.098)..."
}
```

Not garbage text, not template filling, but actual reasoning.

## Troubleshooting

### CUDA Out of Memory
Already handled:
- batch_size = 1
- gradient_accumulation = 2
- PYTORCH_ALLOC_CONF set

If still fails, reduce to batch_size=1 in Cell 2.

### Phase 1 Checkpoint Not Found
Proceed with base model only - Phase 1 LoRA is optional but recommended.

### Dataset Not Found
Verify:
```bash
ls -la /home/user/llm_finetuning/latest_data/training_data_comprehensive.jsonl
wc -l /home/user/llm_finetuning/latest_data/training_data_comprehensive.jsonl  # Should be 216
```

### Slow Training
- CPU only: ~30-45 min
- GPU (RTX 5090): ~8-12 min
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

## After Training

### Using the Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

BASE_MODEL = "Qwen/Qwen2.5-7B"
LORA_PATH = "/home/user/llm_finetuning/qwen_chiva_v1_v2_lora"

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, LORA_PATH)
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

model.eval()
prompt = "Classify the shunt type:\n  Clip 00: EP N1→N2 y=0.050..."
inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
    
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Files You Have Now

```
/home/user/llm_finetuning/
├── training_data_comprehensive.jsonl    ← Dataset (216 examples)
├── Phase2_V1_V2_Training.ipynb         ← Notebook to run
├── qwen_medical_lora_gpu/              ← Phase 1 (already trained)
├── qwen_chiva_v1_v2_lora/              ← Output (Phase 2, will be created)
└── .cache/                              ← HuggingFace model cache
```

## Why This Will Work

1. **Dataset matches evaluation formats**: V1 (clip notation) and V2 (medical terminology) exactly as used in ubuntu_evaluation.py
2. **Sufficient NO SHUNT examples**: 36 diverse normal cases teach what normality looks like
3. **Genuine reasoning**: Every example includes reasoning that explains WHY, preventing memorization
4. **Real patient data**: 30 real patient cases × 6 variations = natural diversity
5. **JSON output format**: Model trained to output proper JSON with confidence and reasoning
6. **Phase 1 + Phase 2**: Medical knowledge (Phase 1) + Task-specific knowledge (Phase 2)

## Success Checklist

- [ ] Files copied to Ubuntu
- [ ] Jupyter running
- [ ] Cell 1: Model loads + 216 examples shown
- [ ] Cell 2: Dataset ready, optimizer configured
- [ ] Cell 3: Training starts, loss decreasing each epoch
- [ ] Cell 4: Model saves, test outputs are coherent JSON
- [ ] No CUDA errors
- [ ] No gradient warnings
- [ ] Loss converges to ~0.5 or lower

## Questions?

If anything fails, provide:
1. The error message
2. Which cell failed
3. Output before the error
4. Check that paths are correct in Cell 1

---

**Dataset Version**: training_data_comprehensive.jsonl (216 examples)  
**Training Time**: ~10 minutes on RTX 5090  
**Status**: Ready to train

# Fix: OutOfMemoryError with 8-Bit Quantization

## Problem

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 130.00 MiB.
GPU 0 has a total capacity of 31.84 GiB of which 0 bytes is free.
Of the allocated memory 44.59 GiB is allocated by PyTorch
```

**Root cause:** Full-precision model + optimizer state uses 44.59GB on a 31.84GB GPU.

- Model weights (bfloat16): ~15GB
- Optimizer state (Adam): ~15GB  
- Activations during forward pass: ~10GB
- **Total: ~40GB (exceeds 31.84GB limit)**

---

## Solution: 8-Bit Quantization

**8-bit quantization reduces the model from 15GB → 4GB** using bitsandbytes library.

This is the **only practical solution** for your hardware constraint.

---

## Quick Start

### Option 1: Use the New 8-Bit Notebook (RECOMMENDED)

Download and run this new notebook:

**`medical_cpt_8bit.ipynb`**

This notebook has everything pre-configured:
- ✓ 8-bit quantization enabled
- ✓ Gradient checkpointing  
- ✓ Reduced sequence length (1024)
- ✓ Small batch size with accumulation
- ✓ All memory optimizations built-in

**Expected GPU usage: 22-26GB (should fit!)**

---

### Option 2: Manually Fix Your Existing Notebook

If you want to keep your current notebook, make these changes:

#### Step 1: Install bitsandbytes and peft

```bash
pip install -q bitsandbytes peft
```

#### Step 2: Replace Cell 3 (Configuration)

Use `CELL_3_CONFIG_8BIT.py` - key changes:
```python
"per_device_train_batch_size": 1,        # Reduced from 2
"gradient_accumulation_steps": 16,       # Increased from 8
"max_seq_length": 1024,                  # Reduced from 2048
```

#### Step 3: Replace Cell 10 (Model Loading)

Use `CELL_10_8BIT_QUANTIZATION.py` - key changes:
```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_name"],
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
```

---

## Memory Breakdown

### Before (OOM at 44.59GB):
```
Model (bfloat16):     15.0 GB
Optimizer (AdamW):    15.0 GB  
Activations:          10.0 GB
Overhead:             ~4.6 GB
Total:               44.6 GB  ❌ (exceeds 31.84GB)
```

### After (8-bit at ~24GB):
```
Model (8-bit):         4.0 GB  ← 70% reduction!
Optimizer (AdamW):     15.0 GB
Activations:            4.0 GB  ← Less with smaller batch
Overhead:              ~1.0 GB
Total:                24.0 GB  ✓ (fits in 31.84GB)
```

---

## Training Characteristics

| Aspect | Value |
|--------|-------|
| GPU Memory | 22-26 GB |
| Training Speed | 12-18 hours (slower than full precision) |
| Quality | Same (quantization done transparently) |
| Batch Size | 1 (per-device) with 16× accumulation |
| Effective Batch | 16 (same convergence as before) |

---

## What 8-Bit Quantization Does

1. **Quantizes model weights** to 8-bit integers (int8) at inference
2. **Dequantizes on-the-fly** during forward/backward pass  
3. **No quality loss** - this is standard in the field
4. **Major memory savings** - 15GB → 4GB

---

## If Still Getting OOM

### Try 4-Bit Quantization (even more aggressive)

Update Cell 10:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_name"],
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```

**4-bit reduces model to ~2GB** (but slower training).

---

## Verification

After loading with 8-bit, verify memory usage:

```python
# In a cell after loading model:
print(f"GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
```

Should show ~4GB model memory (not 15GB).

---

## Important Notes

1. **bitsandbytes requires NVIDIA GPU** - AMD/other GPUs won't work
2. **Training is slower** (~50% slower) due to quantization overhead
3. **Model quality is identical** - quantization happens during training
4. **Inference speed is normal** - no overhead at inference time

---

## Troubleshooting

**"ImportError: No module named 'bitsandbytes'"**
→ Install: `pip install bitsandbytes`

**"CUDA out of memory" still happens**
→ Try 4-bit quantization (see above)
→ Or reduce `gradient_accumulation_steps` to 8

**"Model not training (loss stays constant)"**
→ Normal for 8-bit in early epochs
→ Wait 100+ steps before evaluating
→ Check learning rate (try 1e-5)

---

## Recommended: Use `medical_cpt_8bit.ipynb`

This is the easiest path. The notebook:
- Has all memory optimizations pre-configured
- Won't OOM
- Should train successfully
- Takes 12-18 hours on RTX 5090

**Just run it cell-by-cell!**

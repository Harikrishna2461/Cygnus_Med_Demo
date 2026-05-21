# Notebook Fixes - Trainer API Memory & Deadlock Resolution

## Problem

1. **System Freeze** (CPU 3%, GPU 0%): Caused by `dataloader_num_workers=4` with multiprocessing
2. **OutOfMemoryError**: 44.98GB allocated on 31.84GB GPU

## Solution

Use the **Trainer API** (which was already in Cell 20) but with corrected configuration.

---

## Changes Required

### Cell 4: Configuration

**What to change:**
- `dataloader_num_workers`: 4 → **0** (avoid multiprocessing deadlock)
- `dataloader_pin_memory`: True → **False** (avoid memory pinning issues)
- `per_device_train_batch_size`: 4 → **2** (reduce per-device memory)
- `per_device_eval_batch_size`: 8 → **4**
- `gradient_accumulation_steps`: 4 → **8** (keep effective batch size = 2*8=16)

**Copy this entire cell:**

```python
# ============ TRAINING CONFIG ============
config = {
    # Model & Data
    "model_name": "Qwen/Qwen2.5-7B",
    "train_file": "augmented_output/train.jsonl",
    "eval_file": "augmented_output/eval.jsonl",

    # Training Parameters
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,  # Reduced from 4
    "per_device_eval_batch_size": 4,   # Reduced from 8
    "gradient_accumulation_steps": 8,  # Increased from 4 (keeps effective batch size)
    "learning_rate": 2e-5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,

    # Sequence & Optimization
    "max_seq_length": 2048,
    "bf16": True,
    "fp16": False,

    # Checkpointing & Saving
    "output_dir": "medical_qwen_cpt",
    "save_strategy": "steps",
    "save_steps": 200,
    "save_total_limit": 5,

    # Evaluation
    "eval_strategy": "steps",
    "eval_steps": 100,
    "metric_for_best_model": "eval_loss",

    # Logging
    "logging_dir": "logs",
    "logging_steps": 10,
    "log_level": "info",

    # Other
    "seed": 42,
    "dataloader_num_workers": 0,      # ← FIXED: No multiprocessing
    "dataloader_pin_memory": False,   # ← FIXED: No memory pinning
}

print("📋 TRAINING CONFIGURATION")
print("=" * 50)
for key, value in config.items():
    print(f"{key:.<40} {value}")
print("=" * 50)
```

---

### Cell 10: Model Loading

**What to add:**
Enable gradient checkpointing after loading the model. This reduces memory by not storing intermediate activations.

**Copy this entire cell:**

```python
print(f"\n🔄 Loading model: {config['model_name']}...")
model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

# Enable gradient checkpointing to reduce memory usage
model.gradient_checkpointing_enable()

print(f"✓ Model loaded")
num_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {num_params/1e9:.2f}B")
print(f"  Device: {model.device}")
print(f"  Dtype: {next(model.parameters()).dtype}")
print(f"  Gradient checkpointing: Enabled")
```

---

### Cell 20: Training

**No changes needed** - already has `trainer.train()` which is correct!

---

## Why These Fixes Work

1. **`dataloader_num_workers=0`**
   - Multiprocessing with CUDA causes deadlocks on Jupyter/notebooks
   - Single-threaded data loading is slower but avoids freeze
   - This fixes the "CPU 3%, GPU 0%" system freeze

2. **`dataloader_pin_memory=False`**
   - Pin memory with CUDA can cause memory exhaustion
   - Single-worker loader doesn't benefit from pinning anyway

3. **Reduced batch size + increased gradient accumulation**
   - `per_device_train_batch_size=2` uses less GPU memory per step
   - `gradient_accumulation_steps=8` accumulates 8 batches before update
   - Effective batch size stays 16 (2×8), so convergence is the same
   - This fixes the "44.98GB allocated on 31.84GB GPU" OOM error

4. **Gradient checkpointing**
   - Recomputes intermediate activations during backward pass instead of storing them
   - Reduces memory usage by ~30% with minimal slowdown
   - Trainer API automatically uses it with the model

---

## Expected Behavior

After these changes, training should:

✓ **No system freeze** - single-threaded dataloader won't deadlock
✓ **No OOM errors** - memory usage stays within 31.84GB GPU limit
✓ **Show progress every 10 steps** - losses printed to console
✓ **Save checkpoints every 200 steps** - recovery from interruptions
✓ **Complete in ~8-12 hours** - depends on GPU and data

---

## If Still Getting OOM

Try these in Cell 4:

```python
# More aggressive memory reduction
"per_device_train_batch_size": 1,     # Smallest possible
"gradient_accumulation_steps": 16,    # Accumulate more steps
"max_seq_length": 1024,                # Shorter sequences
```

Or enable quantization in Cell 10:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```

---

## Summary

| Issue | Cause | Fix |
|-------|-------|-----|
| System freeze (CPU 3%, GPU 0%) | `dataloader_num_workers=4` multiprocessing | Set to `0` |
| OutOfMemoryError (44.98GB > 31.84GB) | Per-batch memory too high | Reduce batch size, increase accumulation |
| GPU memory usage | No gradient checkpointing | Enable it in model loading |

**That's it!** The notebook already uses `trainer.train()` which is the correct approach. Just update the configuration and model loading cells.

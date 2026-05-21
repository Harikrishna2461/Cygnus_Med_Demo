# 🚀 Quick Start: Medical CPT Training

Get up and running in 5 minutes.

## TL;DR

```bash
# 1. Setup (5 minutes)
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets evaluate

# 2. Train (10-30 hours depending on GPU)
python3 train_medical_cpt.py

# 3. Monitor (in another terminal)
tensorboard --logdir logs

# 4. Load model when done
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("medical_qwen_cpt/best_model")
tokenizer = AutoTokenizer.from_pretrained("medical_qwen_cpt/best_model")
EOF
```

---

## Step-by-Step

### 1. Environment Setup (Ubuntu)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support (CRITICAL)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU works
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Should print: CUDA: True
```

**⚠️ If CUDA: False**
- Check: `nvidia-smi` (should show your GPU)
- Reinstall PyTorch with correct CUDA version
- See TRAINING_GUIDE.md for troubleshooting

### 2. Install Dependencies

```bash
pip install transformers==4.40.0 datasets==2.16.0 evaluate==0.4.1 accelerate==0.27.0 tensorboard==2.15.0
```

### 3. Verify Data

```bash
# Check augmented data exists
ls -lh augmented_output/
# Should show: train.jsonl (20MB), eval.jsonl (64KB)

wc -l augmented_output/train.jsonl augmented_output/eval.jsonl
# Should show: ~7540 train, ~16 eval
```

### 4. Download Model (Optional but Recommended)

```bash
# Pre-cache Qwen model to avoid download during training
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True, device_map="cpu")
print("✓ Model cached")
EOF
```

### 5. Start Training

```bash
# Default: 3 epochs, batch size 4, learning rate 2e-5
python3 train_medical_cpt.py

# Or customize
python3 train_medical_cpt.py --num_epochs 5 --learning_rate 1e-5
```

Expected output:
```
Device: cuda
GPU: NVIDIA A100-40GB
Epoch: 1/3 [=====>...] 25%
Training loss: 3.21 | Eval loss: 3.15
...
✓ TRAINING COMPLETE
Best model: medical_qwen_cpt/best_model
```

### 6. Monitor Training (Optional)

In a separate terminal:

```bash
tensorboard --logdir logs
# Open http://localhost:6006
```

Watch:
- **Training loss**: Should decrease
- **Eval loss**: Should decrease
- **GPU utilization**: Should stay ~70-90%

### 7. Load Trained Model

After training completes:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "medical_qwen_cpt/best_model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("medical_qwen_cpt/best_model")

# Generate text
inputs = tokenizer("Venous insufficiency treatment", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=150)
print(tokenizer.decode(outputs[0]))
```

---

## Training Time Estimates

| GPU | VRAM | Batch Size | Epochs | Time |
|-----|------|-----------|--------|------|
| A100 40GB | 40GB | 8 | 3 | 8h |
| A100 80GB | 80GB | 16 | 3 | 6h |
| RTX 4090 | 24GB | 4 | 3 | 16h |
| RTX A6000 | 48GB | 8 | 3 | 12h |

Estimates are approximate. First epoch is always slower due to compilation.

---

## GPU Memory by Batch Size

```
Batch Size 1: ~12GB
Batch Size 2: ~16GB  <- Minimum for 24GB GPU
Batch Size 4: ~20GB  <- Default
Batch Size 8: ~30GB
Batch Size 16: ~45GB
```

If out of memory:
```bash
python3 train_medical_cpt.py --train_batch_size 2 --gradient_accumulation_steps 8
```

---

## Common Issues

### "CUDA out of memory"
```bash
python3 train_medical_cpt.py --train_batch_size 2
```

### "No module named transformers"
```bash
source venv/bin/activate  # Activate venv
pip install transformers
```

### "CUDA not available"
```bash
nvidia-smi  # Check if driver installed
# Reinstall PyTorch with CUDA support
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### "Model weights don't match expected size"
```bash
# Delete cache and re-download
rm -rf ~/.cache/huggingface/hub/models--Qwen*
# Re-run training
```

See **TRAINING_GUIDE.md** for detailed troubleshooting.

---

## What Gets Created

```
medical_qwen_cpt/
├── best_model/              <- Use this model
│   ├── config.json
│   ├── model.safetensors   (13GB)
│   ├── tokenizer.json
│   └── ...
├── checkpoint-200/
├── checkpoint-400/
├── checkpoint-600/
└── training_args.bin

logs/
├── events.out.tfevents...  <- For tensorboard
└── training_metrics.csv

```

---

## Next Steps

1. **Verify training worked**
   ```bash
   python3 diagnose_training.py
   ```

2. **Test on medical queries**
   ```python
   # Load model and generate
   # See Step 7 above
   ```

3. **Deploy**
   ```bash
   # Copy to production
   cp -r medical_qwen_cpt/best_model /production/
   ```

4. **Fine-tune for tasks**
   ```python
   # Use trained model as base for task-specific fine-tuning
   ```

---

## Full Documentation

- **TRAINING_GUIDE.md** - Comprehensive guide with all options
- **medical_cpt_training.ipynb** - Interactive Jupyter notebook
- **train_medical_cpt.py** - Standalone Python script
- **AUGMENTATION_README.md** - Data augmentation details

---

## Support

Check these files in order:
1. **TRAINING_GUIDE.md** - Troubleshooting section
2. **diagnose_training.py** - Verify training worked
3. **logs/** - Training logs and metrics
4. **tensorboard** - Visual training curves

---

**Ready? Start with:**
```bash
source venv/bin/activate && python3 train_medical_cpt.py
```

Happy training! ☀️

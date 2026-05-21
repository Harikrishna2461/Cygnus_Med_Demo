# Training Complete - Full Instructions

## You Have These Files Ready:

### **Main Notebook (USE THIS ONE):**
- **`training_complete.ipynb`** ← Copy this to Ubuntu and run it

This notebook has **15 complete cells** that do everything end-to-end:
1. System check
2. Package check
3. Configuration
4. Load data
5. Load tokenizer
6. Load model with 8-bit quantization
7. Tokenize datasets
8. Setup data collator
9. Setup training arguments
10. Create trainer
11. **START TRAINING** (the actual training happens here)
12. Evaluate
13. Save model
14. Test inference
15. Summary

---

## How to Use on Ubuntu

### Step 1: Copy notebook to Ubuntu

```bash
# From your Windows machine or cloud storage:
scp training_complete.ipynb user@ubuntu-server:/path/to/project/
# OR download directly to Ubuntu
```

### Step 2: Navigate to project directory

```bash
cd /path/to/llm_continued_pretraining
```

### Step 3: Install packages (if not already installed)

```bash
pip install -q transformers datasets torch bitsandbytes peft
```

### Step 4: Start Jupyter

```bash
jupyter notebook training_complete.ipynb
```

### Step 5: Run cells in order

- Click "Cell" → "Run All" 
- OR click each cell and press Shift+Enter
- Wait for each cell to complete before moving to next

---

## What Will Happen

| Cell | What It Does | Time |
|------|-------------|------|
| 1-5 | System checks, load data/tokenizer | 5 min |
| 6 | Load model with 8-bit | 2 min |
| 7-9 | Tokenize, setup training | 5 min |
| 10 | Create trainer | 30 sec |
| **11** | **TRAINING STARTS** | **12-18 hours** |
| 12-15 | Evaluate, save, test | 10 min |

**Total: ~12-18 hours** (depends on GPU speed)

---

## GPU Memory During Training

- **Before loading model:** 2GB free
- **After loading model:** 4GB used (8-bit quantization)
- **During training:** 22-26GB used (fits on 31.84GB GPU ✓)

---

## Monitoring Training

While training runs, in **another terminal**:

```bash
# Watch GPU in real-time
watch -n 1 nvidia-smi

# Or view TensorBoard
tensorboard --logdir logs_8bit --port 6006
# Then open: http://localhost:6006
```

---

## Output Files

After training completes, you'll have:

```
medical_qwen_cpt_8bit/
├── best_model/              ← Your trained model here
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.model
│   └── ...
├── checkpoint-100/          ← Previous checkpoints
├── checkpoint-200/
└── ...

logs_8bit/
└── events.out.tfevents.*    ← TensorBoard logs
```

---

## Loading Your Trained Model Later

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "medical_qwen_cpt_8bit/best_model",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "medical_qwen_cpt_8bit/best_model"
)

# Use for inference
inputs = tokenizer("The diagnosis of venous insufficiency", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

---

## Troubleshooting

### "CUDA out of memory" error
- GPU not have enough memory. Reduce in Cell 3:
  ```python
  "per_device_train_batch_size": 1,  # already 1, can't reduce
  "gradient_accumulation_steps": 8,  # reduce from 16
  "max_seq_length": 512,             # reduce from 1024
  ```
- OR check if other processes are using GPU: `nvidia-smi`

### "FileNotFoundError: augmented_output/train.jsonl"
- Run data augmentation first: `python augment_medical_data.py`
- Check file exists: `ls -la augmented_output/`

### Training is very slow
- Normal for 8-bit quantization (50% slower than full precision)
- Expected: 12-18 hours is normal

### Training hangs / no output
- Wait 1-2 minutes (model loading takes time)
- Check with: `nvidia-smi` (GPU should show activity)
- If no GPU activity: press Ctrl+C and debug

---

## That's It!

The notebook is **complete and self-contained**. Just run it cell-by-cell. No manual edits needed.

If you have issues, share:
1. The exact error message
2. Output from: `nvidia-smi`
3. Which cell failed

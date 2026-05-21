# Training Pipeline Summary

Complete medical CPT training setup for Qwen2.5-7B on augmented medical data.

## 📁 Files Created

### Training Notebooks & Scripts

1. **`medical_cpt_training.ipynb`** (Primary)
   - Interactive Jupyter notebook for training
   - GPU-optimized with mixed precision (bfloat16)
   - Includes inference testing and evaluation
   - Best for learning/debugging step-by-step

2. **`train_medical_cpt.py`** (Production Ready)
   - Standalone Python script (no Jupyter needed)
   - Full command-line argument support
   - Optimized for Ubuntu/headless servers
   - Recommended for long-running training

### Documentation

3. **`QUICKSTART.md`**
   - 5-minute setup and training
   - Essential commands only
   - Start here if you know what you're doing

4. **`TRAINING_GUIDE.md`** (Comprehensive)
   - Complete setup instructions
   - Hardware requirements and verification
   - Jupyter and standalone usage
   - Monitoring, troubleshooting, deployment
   - Performance optimization tips

5. **`AUGMENTATION_README.md`**
   - Data augmentation pipeline details
   - How augmentation was done (back-translation, key points)
   - Medical data safety guarantees

### Existing Files Used

- **`augmented_output/train.jsonl`** - Augmented training data (7,540 chunks, 4.4M tokens)
- **`augmented_output/eval.jsonl`** - Evaluation data (16 chunks, 13K tokens)
- **`diagnose_training.py`** - Post-training verification script

---

## 🎯 Quick Navigation

### I want to...

**Train on GPU (Ubuntu)**
→ See `QUICKSTART.md` or `TRAINING_GUIDE.md`

**Use Jupyter Notebook**
→ Open `medical_cpt_training.ipynb`

**Run standalone script**
→ `python3 train_medical_cpt.py`

**Monitor training**
→ `tensorboard --logdir logs`

**Fix GPU issues**
→ See `TRAINING_GUIDE.md` > Troubleshooting

**Load trained model**
→ See `TRAINING_GUIDE.md` > Loading Trained Model

**Multi-GPU training**
→ See `TRAINING_GUIDE.md` > Advanced

---

## 🚀 Start Here

### Option 1: Jupyter Notebook (Interactive)

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets evaluate

# Run
jupyter notebook medical_cpt_training.ipynb
```

### Option 2: Standalone Script (Production)

```bash
# Setup (same as above)
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets evaluate

# Run
python3 train_medical_cpt.py

# Monitor (separate terminal)
tensorboard --logdir logs
```

---

## 📊 Training Pipeline Architecture

```
Medical Data
    ↓
Augmentation Pipeline (completed)
    ├─ Back-translation (33%)
    ├─ Key points extraction (34%)
    └─ Original data (33%)
    ↓
augmented_output/ ← 7,540 chunks, 4.4M tokens
    ↓
Loading & Tokenization (notebook/script)
    ├─ Load JSONL
    ├─ Tokenize with Qwen tokenizer
    └─ Create PyTorch datasets
    ↓
Model Setup
    ├─ Load Qwen2.5-7B (7B parameters)
    ├─ Enable gradient checkpointing (memory optimization)
    └─ Setup data collator
    ↓
Training Loop (Hugging Face Trainer)
    ├─ Mixed precision (bfloat16)
    ├─ Gradient accumulation (effective batch: 16)
    ├─ Learning rate scheduling with warmup
    ├─ Validation every 100 steps
    └─ Save best checkpoint
    ↓
Checkpoints Saved
    ├─ best_model/ ← Final trained model
    ├─ checkpoint-200/
    ├─ checkpoint-400/
    └─ checkpoint-600/
    ↓
Evaluation & Inference
    ├─ Run validation
    ├─ Test generation
    └─ Compare with base model (diagnose_training.py)
```

---

## ⚙️ Key Configuration

**Default Training Settings:**
- Model: Qwen/Qwen2.5-7B (7B parameters)
- Epochs: 3
- Batch size: 4 (gradient accumulation × 4 = effective 16)
- Learning rate: 2e-5
- Warmup: 500 steps
- Max sequence length: 2048 tokens
- Optimization: bfloat16 mixed precision
- Checkpointing: Save every 200 steps (keep 5 best)

**Customization:**
All settings can be modified in:
- **Jupyter**: Edit config dict in Cell 1.1
- **Script**: Use `--help` for options

```bash
python3 train_medical_cpt.py --help
```

---

## 💾 Storage Requirements

- **Model cache**: ~13GB (Qwen2.5-7B)
- **Training checkpoints**: ~35GB (5 checkpoints × 7GB)
- **Training data**: ~25MB (augmented_output/)
- **Logs**: ~1GB (tensorboard events)
- **Total**: ~50GB free space recommended

---

## ⏱️ Estimated Training Time

| GPU | VRAM | Est. Time |
|-----|------|-----------|
| NVIDIA A100 (80GB) | 80GB | 6-8 hours |
| NVIDIA A100 (40GB) | 40GB | 10-12 hours |
| NVIDIA A6000 | 48GB | 12-14 hours |
| NVIDIA RTX 4090 | 24GB | 16-20 hours |
| NVIDIA RTX 3090 Ti | 24GB | 20-24 hours |

⚠️ **First epoch is slower** due to compilation (2-3x longer)

---

## 🔍 Verification

After training completes:

```bash
# Check if training actually updated weights
python3 diagnose_training.py

# View training curves
tensorboard --logdir logs
# Open http://localhost:6006

# Check final metrics
tail logs/training_metrics.csv

# List checkpoints saved
ls -lh medical_qwen_cpt/
```

---

## 📚 What You Have

### Data Pipeline ✓
- Original medical data: 2,637 chunks, 1.85M tokens
- Augmented: 7,540 chunks, 4.4M tokens (2.4x increase)
- Clean, deduplicated, validated
- Ready for training

### Training Pipeline ✓
- Jupyter notebook: Interactive, educational
- Python script: Production-ready, headless
- Full documentation with troubleshooting
- GPU-optimized with bfloat16, gradient checkpointing
- Automatic checkpoint management

### Post-Training ✓
- Diagnosis script to verify weights updated
- Inference testing included
- Model loading examples
- Deployment guides

---

## 🎓 Learning Path

### New to training?
1. Read `QUICKSTART.md`
2. Use `medical_cpt_training.ipynb`
3. Run cell by cell, understand each step
4. Reference `TRAINING_GUIDE.md` for details

### Familiar with training?
1. Run `train_medical_cpt.py` directly
2. Customize via command-line arguments
3. Monitor with tensorboard
4. Load and test trained model

### Deployment ready?
1. Use `train_medical_cpt.py` on production machine
2. Set up checkpointing and recovery
3. Run diagnostic verification
4. Load model with accelerate or vLLM for inference

---

## 🔗 Integration Examples

### LlamaIndex
```python
from llama_index.llms import HuggingFaceLLM

llm = HuggingFaceLLM(
    model_name="medical_qwen_cpt/best_model",
    max_new_tokens=256,
    context_window=2048,
)

response = llm.complete("Explain venous reflux")
```

### LangChain
```python
from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(
    model_id="medical_qwen_cpt/best_model",
    model_kwargs={"torch_dtype": "bfloat16"},
)
```

### vLLM (for inference)
```python
from vllm import LLM

llm = LLM(
    model="medical_qwen_cpt/best_model",
    dtype="bfloat16",
    tensor_parallel_size=2,
)

outputs = llm.generate(
    ["Diagnosis of DVT"],
    sampling_params=SamplingParams(temperature=0.7, top_p=0.9)
)
```

---

## ✅ Checklist

Before training:
- [ ] GPU with ≥24GB VRAM available
- [ ] Ubuntu 20.04+ installed
- [ ] NVIDIA CUDA 12.1+ installed
- [ ] Python 3.10+ available
- [ ] ~50GB free disk space
- [ ] augmented_output/train.jsonl exists
- [ ] augmented_output/eval.jsonl exists

Before running training script:
- [ ] Python venv created and activated
- [ ] PyTorch with CUDA installed
- [ ] Transformers and datasets installed
- [ ] nvidia-smi shows GPU available
- [ ] torch.cuda.is_available() returns True

After training completes:
- [ ] Run diagnose_training.py to verify
- [ ] Review tensorboard curves
- [ ] Test inference on sample prompts
- [ ] Load model successfully
- [ ] Save to production location

---

## 📞 Troubleshooting Paths

**GPU Issues** → TRAINING_GUIDE.md > Prerequisites
**Installation** → TRAINING_GUIDE.md > Environment Setup
**Training Fails** → TRAINING_GUIDE.md > Troubleshooting
**Data Problems** → AUGMENTATION_README.md
**Model Loading** → TRAINING_GUIDE.md > Loading Trained Model
**Performance** → TRAINING_GUIDE.md > Performance Tips

---

## 🎉 Next Steps

1. **Setup environment** (5 min)
   ```bash
   source venv/bin/activate
   python3 train_medical_cpt.py --help
   ```

2. **Start training** (10-30 hours)
   ```bash
   python3 train_medical_cpt.py
   tensorboard --logdir logs  # in another terminal
   ```

3. **Verify & evaluate** (5 min)
   ```bash
   python3 diagnose_training.py
   ```

4. **Deploy trained model**
   ```bash
   cp -r medical_qwen_cpt/best_model /production/
   ```

---

**Created**: 2026-05-20
**Version**: 1.0
**Status**: Ready for production training

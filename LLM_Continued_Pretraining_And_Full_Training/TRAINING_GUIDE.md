# Medical CPT Training Guide

Complete guide for training Qwen2.5-7B on medical augmented data using GPU on Ubuntu.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Using Jupyter Notebook](#using-jupyter-notebook)
4. [Using Standalone Script](#using-standalone-script)
5. [Monitoring Training](#monitoring-training)
6. [Troubleshooting](#troubleshooting)
7. [Loading Trained Model](#loading-trained-model)

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with ≥24GB VRAM
  - RTX A100, A6000, 4090, 3090 Ti, or better
  - Minimum 24GB for batch_size=4, 32GB recommended
  - Multi-GPU setup supported with distributed training

- **CPU & RAM**: 
  - 8+ cores recommended
  - 64GB+ system RAM recommended
  - Sufficient for loading model + datasets

- **Disk Space**:
  - ~50GB for model and checkpoints
  - ~10GB for training data
  - ~20GB for logs and cache
  - Total: **~80GB free space**

### Software Requirements
- Ubuntu 20.04 LTS or later
- Python 3.10+
- NVIDIA CUDA 12.1+ (check: `nvidia-smi`)
- NVIDIA cuDNN 8.8+

### Check Your Setup

```bash
# Check Python version
python3 --version  # Should be 3.10+

# Check NVIDIA GPU
nvidia-smi
# Should show: CUDA Version: 12.1 or higher
# Should list your GPU(s)

# Check available VRAM
nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
```

---

## Environment Setup

### 1. Clone/Download Repository

```bash
# Copy your project to Ubuntu
# Option A: Git clone if in a repo
git clone <your-repo> ~/llm_continued_pretraining
cd ~/llm_continued_pretraining

# Option B: Transfer files
scp -r /path/to/llm_continued_pretraining user@ubuntu-machine:~
ssh user@ubuntu-machine
cd ~/llm_continued_pretraining
```

### 2. Verify Data Files

```bash
# Check augmented data exists
ls -lh augmented_output/
# Should show:
# - train.jsonl (20M)
# - eval.jsonl (64K)
# - README.md (2.5K)

# Verify file integrity
wc -l augmented_output/train.jsonl augmented_output/eval.jsonl
# Should show ~7540 train, ~16 eval
```

### 3. Create Virtual Environment

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate

# Verify activation (should show (venv) in prompt)
which python  # Should point to venv/bin/python
```

### 4. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (CRITICAL for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch GPU support
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Should output:
```
CUDA Available: True
Device: NVIDIA A100-40GB (or your GPU name)
```

If `CUDA Available: False`, reinstall PyTorch with correct CUDA version:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Install Transformers & Dependencies

```bash
# Core training dependencies
pip install transformers==4.40.0
pip install datasets==2.16.0
pip install evaluate==0.4.1
pip install accelerate==0.27.0
pip install tensorboard==2.15.0

# Optional but recommended
pip install wandb  # Better logging/monitoring
pip install peft   # For parameter-efficient training
pip install bitsandbytes  # For 8-bit optimization
```

### 6. Pre-download Model (Optional but Recommended)

This avoids download errors during training:

```bash
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B"
print(f"Downloading {model_name}...")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f"✓ Tokenizer cached")

# Download model
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="cpu")
print(f"✓ Model cached ({model.num_parameters()/1e9:.2f}B parameters)")

print("\nModels are now cached in ~/.cache/huggingface/")
EOF
```

---

## Using Jupyter Notebook

### Start Jupyter Server

```bash
# Terminal 1: Start Jupyter
cd ~/llm_continued_pretraining
source venv/bin/activate
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Get the token from output:
# http://localhost:8888/?token=<YOUR_TOKEN>
```

### Connect to Jupyter

**Option A: Local Machine**
```bash
# If running on same machine:
open http://localhost:8888
# Enter token when prompted
```

**Option B: Remote Ubuntu Machine**
```bash
# Forward port from Ubuntu to your machine
ssh -L 8888:localhost:8888 user@ubuntu-ip

# Then open in browser:
open http://localhost:8888
```

### Run the Notebook

1. Open `medical_cpt_training.ipynb`
2. Run cells in order:
   - **Cell 1**: Check GPU/CUDA availability
   - **Cells 2-4**: Configuration & data loading
   - **Cells 5-6**: Tokenizer & model loading
   - **Cell 7**: Tokenize datasets
   - **Cell 8**: Setup training
   - **Cell 9**: **RUN TRAINING** ⚠️ (takes 10-30 hours)
   - **Cells 10-11**: Evaluation & inference

⚠️ **Training Cell (Cell 9)**: This will run for many hours. Monitor GPU:

```bash
# In another terminal, monitor GPU
watch -n 1 nvidia-smi
# or
nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used --format=csv,noheader -l 1
```

---

## Using Standalone Script

### Run Training

```bash
cd ~/llm_continued_pretraining
source venv/bin/activate

# Default configuration
python3 train_medical_cpt.py

# Or customize parameters
python3 train_medical_cpt.py \
  --num_epochs 3 \
  --train_batch_size 4 \
  --learning_rate 2e-5 \
  --max_seq_length 2048
```

### Command Line Options

```bash
python3 train_medical_cpt.py --help

# Common customizations:
--num_epochs 5              # Number of training epochs
--train_batch_size 2        # Reduce for smaller GPU
--learning_rate 1e-5        # Different learning rate
--max_seq_length 1024       # Shorter sequences (faster training)
--output_dir my_cpt_model   # Custom output directory
--bf16                      # Enable bfloat16 (modern GPUs)
--no_bf16                   # Use float32 instead
```

### Monitor Training in Real-Time

```bash
# Terminal 1: Run training
python3 train_medical_cpt.py

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 3: View training curves
tensorboard --logdir logs
# Open: http://localhost:6006 in browser
```

---

## Monitoring Training

### 1. TensorBoard (Recommended)

```bash
# Start TensorBoard
tensorboard --logdir logs --host 0.0.0.0 --port 6006

# Access from browser
# Local: http://localhost:6006
# Remote: ssh -L 6006:localhost:6006 user@ubuntu-ip
#         http://localhost:6006
```

**What to monitor:**
- **Loss curves**: Should decrease monotonically
- **Learning rate**: Should warm up then decay
- **Gradients**: Should be stable (no spikes)

### 2. Direct GPU Monitoring

```bash
# Real-time GPU stats
watch -n 1 nvidia-smi

# Or with details
nvidia-smi dmon -s pucvme

# Watch memory and utilization
watch -n 1 'nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader'
```

### 3. Training Logs

```bash
# Check training progress
tail -f logs/training_metrics.csv

# View all training arguments
cat medical_qwen_cpt/training_args.bin  # Binary format
cat medical_qwen_cpt/trainer_state.json # JSON format
```

### 4. Checkpoints

```bash
# List checkpoints
ls -lh medical_qwen_cpt/

# Expected structure:
# checkpoint-200/
# checkpoint-400/
# checkpoint-600/
# best_model/      <- Best evaluation checkpoint
```

---

## Troubleshooting

### "CUDA out of memory"

```python
# Reduce batch size
python3 train_medical_cpt.py \
  --train_batch_size 2 \
  --gradient_accumulation_steps 8

# Or enable gradient checkpointing in notebook:
model.gradient_checkpointing_enable()

# Or reduce sequence length
python3 train_medical_cpt.py --max_seq_length 1024
```

### "No module named 'transformers'"

```bash
# Verify venv is activated
source venv/bin/activate

# Reinstall
pip install transformers==4.40.0
```

### "CUDA not available"

```bash
# Check NVIDIA driver
nvidia-smi
# If nothing shows, driver not installed

# Check PyTorch installation
python3 -c "import torch; print(torch.cuda.is_available())"
# If False, reinstall with correct CUDA:
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Loss not decreasing

1. Check learning rate - too high/low?
   - Try: `--learning_rate 1e-5` or `5e-5`
2. Check warmup steps:
   - `--warmup_steps 1000`
3. Verify data quality:
   - `python3 -c "import json; lines=[json.loads(l) for l in open('augmented_output/train.jsonl')]; print(f'Min tokens: {min(c[\"token_count\"] for c in lines)}, Max: {max(c[\"token_count\"] for c in lines)}')`

### Training too slow

1. Increase batch size (if GPU memory allows):
   ```bash
   python3 train_medical_cpt.py --train_batch_size 8 --gradient_accumulation_steps 2
   ```

2. Reduce evaluation frequency:
   ```python
   # In train_medical_cpt.py, change:
   eval_steps=200  # Instead of 100
   save_steps=400  # Instead of 200
   ```

3. Use multiple GPUs:
   ```bash
   python3 -m torch.distributed.launch --nproc_per_node=2 train_medical_cpt.py
   ```

### Disk space issues

```bash
# Clean old checkpoints (keep only best)
ls medical_qwen_cpt/checkpoint-* | sort -V | head -n -1 | xargs rm -rf

# Clean cache
rm -rf ~/.cache/huggingface/datasets
rm -rf ~/.cache/huggingface/hub

# Check disk usage
du -sh medical_qwen_cpt/
du -sh logs/
du -sh ~/.cache/huggingface/
```

---

## Loading Trained Model

### In Jupyter Notebook

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load best model
model_path = "medical_qwen_cpt/best_model"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Generate text
prompt = "The diagnosis of venous insufficiency requires"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_length=150,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### In Python Script

```python
#!/usr/bin/env python3
from transformers import pipeline

# Create text generation pipeline
generator = pipeline(
    "text-generation",
    model="medical_qwen_cpt/best_model",
    device=0,
)

# Generate
output = generator(
    "Duplex ultrasound imaging is essential for",
    max_length=200,
    temperature=0.7,
)
print(output[0]["generated_text"])
```

### Using with LlamaIndex / LangChain

```python
from llama_index.llms import HuggingFaceLLM
from llama_index.llms import ChatMessage

llm = HuggingFaceLLM(
    model_name="medical_qwen_cpt/best_model",
    max_new_tokens=256,
    context_window=2048,
    model_kwargs={"torch_dtype": "bfloat16"},
)

# Use with LlamaIndex
response = llm.complete("Explain venous reflux")
print(response.text)
```

---

## Performance Tips

### Memory Optimization

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use flash attention v2 (if available)
model.config.use_flash_attention_2 = True

# Efficient batch padding
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    pad_to_multiple_of=8,  # Pad to 8 tokens (hardware efficient)
)
```

### Speed Optimization

```python
# More workers for data loading
--dataloader_num_workers 8  # Instead of 4

# Distributed training with multiple GPUs
python3 -m torch.distributed.launch \
  --nproc_per_node=4 \
  train_medical_cpt.py

# Reduce validation frequency
--eval_steps 500  # Instead of 100
```

### Mixed Precision Training

The notebook/script already enables bfloat16. For older GPUs:
```python
# In TrainingArguments:
fp16=True,  # Use float16 (older GPUs)
bf16=False,
```

---

## Verifying Training Worked

After training completes:

```bash
# Run diagnosis script
python3 diagnose_training.py

# This will:
# 1. Load trained checkpoint
# 2. Compare weights with base model
# 3. Check if training actually updated weights
# 4. Show final verdict
```

Expected output:
```
✓ Unfrozen layers show different weights from base model
✓ LM head: Different (correctly updated)
This suggests training DID update the weights.
```

---

## Next Steps

Once training completes:

1. **Evaluate Quality**
   ```bash
   python3 diagnose_training.py
   tensorboard --logdir logs  # Review training curves
   ```

2. **Test Inference**
   ```python
   # Load and test the model
   from transformers import pipeline
   gen = pipeline("text-generation", model="medical_qwen_cpt/best_model")
   gen("The treatment of varicose veins")
   ```

3. **Deploy Model**
   ```bash
   # Push to Hugging Face Hub (optional)
   huggingface-cli upload medical_qwen_cpt ./medical_qwen_cpt/best_model
   
   # Or copy to production
   cp -r medical_qwen_cpt/best_model /production/models/
   ```

4. **Fine-tune for Specific Tasks**
   ```python
   # Use trained model as base for task-specific fine-tuning
   model = AutoModelForCausalLM.from_pretrained("medical_qwen_cpt/best_model")
   # Then fine-tune on task-specific data
   ```

---

## Resource Limits & Optimization

### Timeout-Safe Training

If training may be interrupted:

```bash
# Enable resume from latest checkpoint
# (Trainer does this automatically)

# Long-running training with nohup
nohup python3 train_medical_cpt.py > training.log 2>&1 &

# Check progress
tail -f training.log
ps aux | grep train_medical_cpt.py
```

### Multi-Node Training (Advanced)

For training across multiple machines:

```bash
# Each node runs:
python3 -m torch.distributed.launch \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=<master-ip> \
  --master_port=29500 \
  train_medical_cpt.py
```

---

## Support & Debugging

### Collect Debug Info

```bash
# System info
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
nvidia-smi

# Check library versions
pip list | grep -E "torch|transformers|datasets|accelerate"

# Test minimal training
python3 << 'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
outputs = model(**inputs)
print("✓ Model works on GPU")
EOF
```

---

**Version**: 1.0  
**Last Updated**: 2026-05-20  
**Compatible**: Ubuntu 20.04+, Python 3.10+, CUDA 12.1+

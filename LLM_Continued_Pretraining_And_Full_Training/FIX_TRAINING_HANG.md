# Fix: Training Hang/Freeze Issues

Your system is freezing because the training loop is **deadlocking or waiting indefinitely** on I/O or GPU operations. The Hugging Face Trainer API can be opaque and hard to debug.

## Root Causes

1. **Data loader deadlock** - `num_workers` in DataLoader or multiprocessing issues
2. **Memory exhaustion** - System swapping heavily (freeze while swapping)
3. **GPU synchronization hang** - CUDA kernel waiting for something
4. **Model loading issue** - device_map="auto" causing GPU communication issues

## Quick Fixes (Try in Order)

### Fix 1: Run Diagnostic First (5 minutes)

```bash
python3 diagnose_training_hang.py
```

This will:
- Check GPU memory availability
- Test tokenization speed
- Test model loading
- Test forward pass
- Test batch loading

**If diagnostic hangs:** Your GPU/CUDA setup has issues. Try:
```bash
nvidia-smi              # Verify GPU works
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Fix 2: Use Simple Training Script (No Trainer API)

The simple script uses basic PyTorch loops with **clear visibility**:

```bash
# Quick test: Train on 100 samples
python3 train_medical_cpt_simple.py --num_samples 100 --num_epochs 1 --batch_size 2

# If that works, train on full dataset
python3 train_medical_cpt_simple.py --batch_size 2 --num_epochs 1
```

**Advantages:**
- Clear progress printing every 5 batches
- Memory tracking
- No multiprocessing deadlocks
- Easy to interrupt with Ctrl+C
- Shows exactly where it hangs

### Fix 3: Reduce Memory Pressure

If simple script still hangs, reduce memory:

```bash
# Smaller batch, shorter sequences
python3 train_medical_cpt_simple.py \
  --batch_size 1 \
  --max_seq_length 512 \
  --num_samples 100 \
  --num_epochs 1
```

### Fix 4: If Jupyter Notebook Freezes

Don't use Jupyter for long training. Instead:

```bash
# Terminal 1: Run standalone script
python3 train_medical_cpt_simple.py

# Terminal 2: Monitor
watch -n 1 nvidia-smi
```

Never run training in Jupyter - it can deadlock the entire notebook.

### Fix 5: Add Timeout Protection

Prevent infinite waits:

```bash
# Kill training if it takes >1 hour
timeout 3600 python3 train_medical_cpt_simple.py
```

When timeout triggers, Ctrl+C to stop gracefully.

---

## Step-by-Step Recovery

### Step 1: Reboot Your Machine
If everything is frozen:
- Hard reboot (hold power button)
- Wait 2 minutes
- Verify GPU works: `nvidia-smi`

### Step 2: Run Diagnostic
```bash
python3 diagnose_training_hang.py
```

This should **complete quickly** with clear output. If it hangs:
- Note where it hangs (GPU load, tokenization, etc.)
- This tells us the root cause
- Report that specific stage

### Step 3: Try Simple Training on Small Dataset
```bash
python3 train_medical_cpt_simple.py \
  --num_samples 50 \
  --batch_size 1 \
  --max_seq_length 512 \
  --num_epochs 1
```

Expected output:
```
1. Device: cuda
   GPU: NVIDIA ... 
   Memory: 24GB

2. Loading tokenizer...
   ✓ Tokenizer ready

3. Loading model...
   This may take a minute...
   ✓ Model loaded (45.2s)
   Parameters: 7.05B

4. Creating datasets...
Loading augmented_output/train.jsonl...
✓ Loaded 50 samples

5. Creating dataloader...
   ✓ Dataloader ready (50 batches)

6. Setting up optimizer...
   ✓ Optimizer ready

7. Starting training...
============================================================================

Epoch 1/1
  Batch 5/50 | Loss: 3.2145 | GPU: 18.5GB
  Batch 10/50 | Loss: 3.1892 | GPU: 18.5GB
  Batch 15/50 | Loss: 3.1645 | GPU: 18.5GB
  ...
  Batch 50/50 | Loss: 3.1234 | GPU: 18.5GB
  Epoch 1 complete - Loss: 3.1234
  Saving checkpoint to medical_qwen_cpt_simple/checkpoint-epoch-1...
  ✓ Checkpoint saved

8. Saving final model...
   ✓ Final model saved to medical_qwen_cpt_simple/final_model

============================================================================
✓ TRAINING COMPLETE
============================================================================
Total steps: 50
Final loss: 3.1234
Model saved: medical_qwen_cpt_simple/final_model
```

If this works → System is OK, Trainer API has issues
If this hangs at specific stage → That's your bottleneck

### Step 4: Scale Up Gradually

Once small test works:

```bash
# Gradually increase
python3 train_medical_cpt_simple.py --num_samples 500 --batch_size 2
python3 train_medical_cpt_simple.py --num_samples 2000 --batch_size 4
python3 train_medical_cpt_simple.py --num_samples -1 --batch_size 4  # All data
```

---

## Monitoring During Training

Keep another terminal open:

```bash
# Monitor GPU
watch -n 1 nvidia-smi

# Or with details
nvidia-smi dmon -s pucvme
```

**What to expect:**
- GPU utilization: 70-90%
- Memory: Stable (not growing)
- Temperature: <80°C
- Every 5 batches: New loss printout in training terminal

**If you see:**
- 0% GPU util → GPU not being used (hang)
- Memory growing → Memory leak
- No output for >2 min → Deadlock/hang

→ Press Ctrl+C to stop training
→ Report what you saw to diagnose issue

---

## Hardware Requirements Check

Before training, verify:

```bash
# 1. GPU exists and CUDA works
nvidia-smi
# Should show your GPU

# 2. PyTorch can access GPU
python3 << 'EOF'
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Try allocating GPU memory
x = torch.randn(1000, 1000).cuda()
print(f"GPU allocation works: {x.device}")
EOF

# 3. Verify you have enough free space
df -h .
# Should show >50GB free
```

---

## Common Issues & Solutions

### "Everything frozen, can't even Ctrl+C"

→ Hold Ctrl+C for 5 seconds (force interrupt)
→ Or use another terminal: `pkill -9 python3`
→ Reboot if needed

### "Training starts but no output"

→ Hang in model loading or first forward pass
→ Try: `python3 train_medical_cpt_simple.py --max_seq_length 512`
→ Reduce batch size: `--batch_size 1`

### "GPU shows 0% after first batch"

→ GPU hung on computation
→ This is a CUDA/GPU driver issue
→ Try: Update NVIDIA drivers
→ Or: Reinstall PyTorch with correct CUDA version

### "Loss becomes NaN"

→ Learning rate too high
→ Try: `--learning_rate 1e-5`
→ Or: gradient norm clipping (already enabled)

### "Memory keeps growing"

→ Memory leak in data loading
→ Simple script: Check `num_workers=0` (it's set correctly)
→ Try smaller batch: `--batch_size 1`

### "Training is too slow"

→ Not a hang - it's just slow
→ Expected: ~3-30 hours depending on GPU
→ Use larger batch if GPU has memory: `--batch_size 4 or 8`
→ Monitor with `nvidia-smi` - should show 70-90% GPU util

---

## When to Use Which Script

| Situation | Use Script |
|-----------|-----------|
| System freezes during training | `train_medical_cpt_simple.py` |
| Want to understand what's happening | `train_medical_cpt_simple.py` |
| Just testing on small data | `train_medical_cpt_simple.py` |
| Only have <24GB GPU | `train_medical_cpt_simple.py` |
| Full production training | `train_medical_cpt.py` (original Trainer) |
| Debugging hangs | `diagnose_training_hang.py` |

---

## Full Recovery Procedure

1. **Reboot** (if system frozen)
   ```bash
   # Wait for reboot and login
   ```

2. **Run diagnostic** (verify system is OK)
   ```bash
   python3 diagnose_training_hang.py
   # Should complete in <5 minutes
   ```

3. **Test small training** (find breaking point)
   ```bash
   python3 train_medical_cpt_simple.py \
     --num_samples 100 \
     --batch_size 1 \
     --max_seq_length 512 \
     --num_epochs 1
   ```

4. **Monitor** (in another terminal)
   ```bash
   watch -n 1 nvidia-smi
   ```

5. **Scale up** (once small test passes)
   ```bash
   python3 train_medical_cpt_simple.py --num_samples 1000 --batch_size 2
   python3 train_medical_cpt_simple.py --num_samples -1 --batch_size 4
   ```

---

**Questions?**
- Run: `python3 diagnose_training_hang.py` and share output
- Watch GPU during: `nvidia-smi -lms 100`
- Note where it hangs (which stage/batch)

The simple script will get you training without freezes! 🚀

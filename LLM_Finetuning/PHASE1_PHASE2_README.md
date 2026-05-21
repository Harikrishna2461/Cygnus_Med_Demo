# Two-Phase Medical LLM Training Framework

## Overview
This framework trains Qwen2.5-7B to be **medically grounded** (learned from 14 medical books) while maintaining **instruction-following capabilities** across multiple tasks (classification, ligation planning, Q&A, etc.).

---

## Phase 1: Continued Pre-training (12-14 GPU hours)

### What it does
- Takes Qwen2.5-7B and **continues training it on medical literature**
- Uses **causal language modeling** (predict next token) - the same objective as original pre-training
- Model learns medical knowledge *deeply* (terminology, concepts, relationships)
- **LoRA** keeps training efficient (only 0.3% of weights updated)

### Data
- **Source:** 14 medical books extracted to `full_medical_data.txt`
- **Size:** 899,042 words (5.76 MB)
- **Format:** Raw text (no instruction-response pairs)
- **Content:** Venous/vascular medicine, CHIVA classification, hemodynamics, ultrasound

### Configuration
```
Model: Qwen/Qwen2.5-7B (7 billion parameters)
Training epochs: 2 (passes through medical data)
Batch size (effective): 16 tokens
Learning rate: 5e-4
Context window: 512 tokens
LoRA rank: 16 (0.3% trainable)
Precision: float16 (for speed)
```

### Expected Output
- **LoRA adapters** saved to: `qwen_medical_lora/`
- **Tokenizer** saved to: `qwen_medical_lora/`
- **Training config** saved to: `qwen_medical_lora/training_config.json`
- **Training logs** saved to: `qwen_medical_pretrained/`

### How to Run

```bash
cd C:\Users\Krish\Downloads\LLM_Finetuning
python phase1_continued_pretraining.py
```

**Expected time:** 12-14 hours on RTX 5090 (can leave overnight)

### Key Differences from Fine-tuning
| Aspect | Fine-tuning | Continued Pre-training |
|--------|------------|----------------------|
| **Objective** | Task-specific (Q&A format) | General (next token prediction) |
| **Data format** | Instruction-response pairs | Raw text |
| **Learning** | Memorize patterns | Learn domain knowledge |
| **Generalization** | Limited to training format | Broad across any instruction |
| **Risk** | Catastrophic forgetting | Lower (same training objective) |

---

## Phase 2: Instruction Fine-tuning (2-4 GPU hours)

### What it does
- Takes the **medical-grounded Qwen** from Phase 1
- Fine-tunes on **diverse medical tasks** to reinforce instruction-following
- Teaches the model how to apply medical knowledge to different instructions

### Data
Will use restructured training data with:
- **Classification examples** (balanced: boost NO SHUNT & TYPE 1+2)
- **Ligation planning examples** (with reasoning)
- **Q&A examples** (based on medical literature)
- **Clinical explanations** (how to describe findings)

### Configuration
```
Model: Phase 1 output (qwen_medical_lora)
Task: Causal language modeling with instruction format
Epochs: 1-2 (light fine-tuning, avoid overfitting)
Batch size: 4
Learning rate: 2e-4 (lower than Phase 1)
LoRA rank: 8
```

### Expected Output
- **Fine-tuned LoRA adapters** (merged with Phase 1)
- **Merged model weights** (base + Phase 1 + Phase 2)
- **Evaluation results** on classification, ligation, and Q&A tasks

### How to Run
```bash
python phase2_instruction_finetuning.py
```

**Expected time:** 2-4 hours on RTX 5090

---

## Complete Workflow

```
Step 1: Extract PDFs
  └─ Run: extract_pdf_data.py
  └─ Output: full_medical_data.txt (899K words)

Step 2: Phase 1 - Continued Pre-training (Overnight)
  ├─ Run: phase1_continued_pretraining.py
  ├─ Time: 12-14 GPU hours
  └─ Output: qwen_medical_lora/ (medical knowledge)

Step 3: Phase 2 - Instruction Fine-tuning
  ├─ Run: phase2_instruction_finetuning.py
  ├─ Time: 2-4 GPU hours
  └─ Output: final model (medical + instruction-following)

Step 4: Evaluate
  └─ Run: evaluation_final_optimized.py
  └─ Test on: Classification, Ligation, Q&A
```

---

## Why This Approach Works

### Problem with Pure Fine-tuning (Current)
- Qwen dropped to 50% accuracy on classification
- Training data has 85% book passages (teaches memorization)
- Model learns to specialize on one task (loses generalization)
- **Result:** Poor on new instructions, catastrophic forgetting

### Solution with Two-Phase Training
1. **Phase 1:** Model learns medical knowledge through language modeling
   - No instruction-response format
   - Same training objective as pre-training (no catastrophic forgetting)
   - Knowledge is internalized in weights

2. **Phase 2:** Model learns to apply knowledge to different task formats
   - Balanced classification data
   - Diverse tasks (not just classification)
   - Reinforces instruction-following

**Result:** Medical knowledge + Generalization + Instruction-following

---

## Comparison with Paper Findings

The paper (https://pmc.ncbi.nlm.nih.gov/articles/PMC12292519/) tested three approaches:
1. **Fine-tuning alone** → Underperforms (your current issue)
2. **RAG alone** → Better
3. **FT+RAG (hybrid)** → Best

Our two-phase approach is similar to "FT+RAG":
- **Phase 1** = Learning medical knowledge (like pre-training)
- **Phase 2** = Task-specific fine-tuning
- **Evaluation** = Tests on classification + other tasks

---

## Hardware Requirements

**Tested on:** RTX 5090 (32GB VRAM)

### Phase 1
- VRAM: ~20-24 GB (float16)
- Duration: 12-14 hours
- Batch size: 2 (per GPU)

### Phase 2
- VRAM: ~16-20 GB
- Duration: 2-4 hours
- Batch size: 4 (per GPU)

---

## Troubleshooting

### Phase 1 runs out of memory
```
# Reduce batch size in phase1_continued_pretraining.py:
per_device_train_batch_size=1  # Instead of 2
```

### Phase 1 is too slow
```
# Reduce epochs:
num_train_epochs=1  # Instead of 2
# Or reduce save frequency:
save_steps=500  # Instead of 300
```

### Phase 1 loss not decreasing
- Normal for large pre-training tasks
- Loss should plateau after ~1 epoch
- If training stalls, interrupt and move to Phase 2

### Phase 2 fine-tuning causes accuracy to drop
- Model is overfitting to training tasks
- Reduce learning rate in phase2_instruction_finetuning.py
- Use fewer epochs

---

## Next Steps After Training

1. **Merge models:**
   ```bash
   python merge_lora_adapters.py
   ```
   Creates final model with all knowledge merged

2. **Evaluate:**
   ```bash
   python evaluation_final_optimized.py
   ```
   Test on classification, ligation planning, Q&A

3. **Deploy:**
   Use final merged model for inference (no LoRA needed)

---

## Key Files

| File | Purpose |
|------|---------|
| `extract_pdf_data.py` | Extract text from 14 PDFs |
| `full_medical_data.txt` | Combined medical text (899K words) |
| `phase1_continued_pretraining.py` | Train on medical literature |
| `phase2_instruction_finetuning.py` | Fine-tune on diverse tasks (TBD) |
| `qwen_medical_lora/` | Phase 1 output (LoRA adapters) |
| `evaluation_final_optimized.py` | Evaluate final model |

---

## Training Monitoring

### TensorBoard (Optional)
```bash
tensorboard --logdir=qwen_medical_pretrained
```

### Log Files
- Phase 1: `qwen_medical_pretrained/training_args.bin`
- Phase 2: `qwen_medical_lora/trainer_state.json`

---

## Estimated Total Time

| Phase | Duration | GPU |
|-------|----------|-----|
| Extract PDFs | 5 min | CPU |
| **Phase 1** | **12-14 hrs** | **RTX 5090** |
| **Phase 2** | **2-4 hrs** | **RTX 5090** |
| **Total** | **~16-18 hrs** | **Can do overnight** |

You can run Phase 1 overnight and Phase 2 the next morning.

---

## Questions?

Refer to:
- Phase 1 script: `phase1_continued_pretraining.py` (fully commented)
- Training logs: `qwen_medical_pretrained/` directory
- Extraction stats: `extraction_stats.txt`

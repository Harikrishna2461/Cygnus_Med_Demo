# Jupyter Notebook Quick Start Guide

## What You Have

**File:** `CHIVA_Training_Notebook.ipynb`

A complete 18-cell Jupyter notebook for training Mistral-7B with LoRA on CHIVA data.

---

## Getting Started

### Step 1: Open Jupyter

```bash
cd C:\Users\Krish\Downloads\LLM_Finetuning
jupyter notebook
```

### Step 2: Navigate to Notebook

In your browser, click on `CHIVA_Training_Notebook.ipynb`

### Step 3: Run Cells in Order

Each cell has a number. Run them **from top to bottom** using:
- **Shift + Enter** (Execute cell and move to next)
- **Ctrl + Enter** (Execute cell, stay in same cell)

---

## Notebook Structure

### Phase 1: Setup (Cells 1-4)
- Import all required libraries
- Set configuration parameters
- Load training and validation data
- Show sample pairs to verify data quality

**Time:** ~2 minutes
**What to do:** Just run these cells in order

### Phase 2: Data Preparation (Cells 5-8)
- Format data for training
- Load Mistral-7B model and tokenizer
- Create HuggingFace datasets
- Configure LoRA adapter

**Time:** ~5 minutes (model loading is slowest)
**What to do:** Run these cells. Cell 6 will download the 7B model (~15GB)

### Phase 3: Training Setup (Cells 9-10)
- Configure training arguments
- Create Trainer object

**Time:** ~1 minute
**What to do:** Just run these cells

### Phase 4: Training (Cell 11) ⭐ MAIN CELL
- **This is where the actual training happens**
- Will show progress bar
- Takes 10-20 minutes on RTX 5090
- Shows loss at each step

**Time:** 10-20 minutes
**What to do:** 
- Run the cell
- Watch the training progress
- Loss should generally decrease
- Validation metrics will update

### Phase 5: Post-Training (Cells 12-14)
- Show training results and metrics
- Save the trained LoRA adapter
- Load the trained model for testing

**Time:** ~2 minutes
**What to do:** Run these cells. You'll see final training loss and validation metrics

### Phase 6: Testing (Cells 15-17)
- Test on a sample Type 1 case
- Integrate with CHIVAShuntClassifier API
- Validate on all test cases

**Time:** ~3-5 minutes
**What to do:** Run these cells to see model outputs

### Phase 7: Summary (Cell 18)
- Summary of training
- Next steps and integration guide

**Time:** <1 minute
**What to do:** Run this cell to see final summary

---

## What Each Cell Does

| Cell | Name | Time | Output |
|------|------|------|--------|
| 1 | Import Libraries | 10s | [OK] messages |
| 2 | Configuration | 5s | Parameter values |
| 3 | Load Training Data | 30s | Data statistics |
| 4 | Show Sample Pairs | 10s | Sample instruction/output |
| 5 | Prepare Data | 30s | Formatted text examples |
| 6 | Load Model & Tokenizer | 2-3 min | Model info, parameters |
| 7 | Create Datasets | 1 min | Dataset structure |
| 8 | Configure LoRA | 30s | LoRA config, trainable params |
| 9 | Training Arguments | 10s | Training config summary |
| 10 | Create Trainer | 10s | Trainer info |
| **11** | **Start Training** | **10-20 min** | **Progress bar, final loss** |
| 12 | Show Results | 10s | Training metrics, eval loss |
| 13 | Save Model | 30s | File list |
| 14 | Load Trained Model | 2 min | Confirmation |
| 15 | Test Sample Case | 1-2 min | Model output |
| 16 | Integration Test | 1 min | Classification result |
| 17 | Validate All Cases | 2 min | Summary stats |
| 18 | Summary | 10s | Final summary |

---

## Expected Output Examples

### From Cell 3 (Load Data):
```
[OK] Loaded 31 training pairs
[OK] Loaded 8 validation pairs

Training data distribution:
  anatomy: 1
  classification: 25
  ligation: 5
```

### From Cell 6 (Load Model):
```
[OK] Model loaded
[OK] Tokenizer loaded

Model Information:
  Total Parameters: 7,243,571,200
  Model Device: cuda:0
  Model Dtype: torch.bfloat16
```

### From Cell 8 (LoRA Config):
```
[OK] LoRA configured

Parameter Information:
  Trainable Parameters: 3,407,872
  Total Parameters: 7,243,571,200
  Percentage Trainable: 0.0470%
```

### From Cell 11 (Training):
```
Epoch 1/5
  0%|                    | 0/8 [00:00<?, ?it/s]
 25%|████                | 2/8 [01:23<04:10, 41.73s/it]
 50%|████████            | 4/8 [02:45<02:45, 41.38s/it]
 75%|████████████        | 6/8 [04:08<01:22, 41.07s/it]
100%|████████████████    | 8/8 [05:30<00:00, 41.24s/it]

Training loss: 1.4532
Validation loss: 1.3421
```

### From Cell 15 (Test Output):
```
Model Output:

CLASSIFICATION: Type 1

CONFIDENCE: 0.95

ANATOMICAL FINDINGS:
• EP N1→N2 (SFJ incompetent)
• RP N2→N1 (direct retrograde to deep vein)
• NO EP N2→N3 (no tributary involvement)
• NO RP at N3 (tributaries not involved)

HEMODYNAMIC INTERPRETATION:
Direct reflux from GSV to deep system through incompetent SFJ...

CLINICAL SIGNIFICANCE:
Most common CHIVA type. Pure SFJ disease...

TREATMENT CONSIDERATIONS:
SFJ ablation sufficient. No tributary treatment needed...
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'xxx'"

**Solution:** Install missing module
```bash
pip install torch transformers peft datasets
```

### "CUDA out of memory"

**Solution:** Edit Cell 2, reduce BATCH_SIZE:
```python
BATCH_SIZE = 2  # Change from 4 to 2
```

### Cell takes very long (>10 seconds) to start

- Cell 6 downloads the 7B model (~15GB) - this is normal first time
- Subsequent runs will be faster

### Training loss not decreasing

- Check that data loaded correctly (Cell 3)
- Verify dataset created properly (Cell 7)
- Try training for more epochs (Cell 2: `NUM_EPOCHS = 10`)

### "CUDA not available"

Check:
```python
import torch
print(torch.cuda.is_available())
```

If False, check your CUDA installation.

---

## Customization

### Train Longer

In **Cell 2**, change:
```python
NUM_EPOCHS = 10  # Instead of 5
```

### Different Output Location

In **Cell 2**, change:
```python
LORA_OUTPUT_DIR = "./my_custom_model"
```

### Higher Learning Rate (if underfitting)

In **Cell 2**, change:
```python
LEARNING_RATE = 2e-4  # Instead of 1e-4
```

### Lower Learning Rate (if overfitting)

In **Cell 2**, change:
```python
LEARNING_RATE = 5e-5  # Instead of 1e-4
```

---

## After Training

Your trained model is at: `./lora_chiva_classifier_final/`

Use it with:

```python
from chiva_classifier_api import CHIVAShuntClassifier

classifier = CHIVAShuntClassifier(
    use_lora=True,
    lora_path="./lora_chiva_classifier_final"
)

clips = [
    {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.080},
    {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.300},
]

result = classifier.classify(clips, leg_label="Left")
print(result['shunt_type'])      # Type 1
print(result['confidence'])       # 0.95
print(result['reasoning'])        # Clinical explanation
```

---

## Timeline

| Phase | Time | Status |
|-------|------|--------|
| Cells 1-4 | 5 min | Setup complete |
| Cells 5-8 | 10 min | Data & model ready |
| Cells 9-10 | 1 min | Training configured |
| **Cell 11** | **15 min** | **Training in progress** |
| Cells 12-14 | 5 min | Model saved & loaded |
| Cells 15-17 | 5 min | Testing complete |
| Cell 18 | 1 min | Done! |
| **TOTAL** | **~45 min** | **Ready for production** |

---

## Tips

1. **Don't skip cells** - Run them in order from top to bottom
2. **Watch Cell 11** - This is the main training. Loss should decrease over time
3. **Save your notebook** - After training completes, save the notebook (Ctrl+S)
4. **Check outputs** - Look at Cell 4 (sample pairs) and Cell 15 (test output) to verify quality
5. **Monitor VRAM** - Training uses ~12GB VRAM (already optimized with bfloat16 + gradient checkpointing)

---

## Success Criteria

After running all cells:

✓ Cell 3: Data loaded successfully (31 train, 8 val)
✓ Cell 6: Model loaded to GPU
✓ Cell 8: LoRA configured (0.047% trainable)
✓ **Cell 11: Training completes with decreasing loss**
✓ Cell 12: Final loss shown
✓ Cell 13: Model saved to disk
✓ Cell 15: Test output shows proper classification
✓ Cell 16: CHIVAShuntClassifier integration works
✓ Cell 18: Summary shows all metrics

---

## Next Steps

1. ✓ Run notebook cells 1-18
2. ✓ Verify training completed (Cell 11)
3. ✓ Test on sample cases (Cell 15-16)
4. → Deploy model: `CHIVAShuntClassifier(use_lora=True, lora_path="./lora_chiva_classifier_final")`
5. → Collect real cases and retrain with more data
6. → Improve with the 5 CHIVA books once extracted

---

## Questions?

Refer to:
- `FINAL_TRAINING_GUIDE.md` - Detailed training information
- `training_datasets/DATASET_SUMMARY.md` - Dataset details
- Cell comments in the notebook - Explanation of each step

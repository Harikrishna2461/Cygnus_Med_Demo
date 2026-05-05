# CHIVA Training Dataset Summary

## Dataset Statistics

**Total Training Pairs:** 31
**Total Validation Pairs:** 8
**Combined Total:** 39

### Training Data by Type
- anatomy: 1
- classification: 25
- ligation: 5

### Validation Data by Type
- classification: 5
- ligation: 3

### Difficulty Distribution
- basic: 1 (3.2%)
- intermediate: 30 (96.8%)

## Data Source

- **Classification Data:** Synthetic cases generated from CHIVA medical literature principles
- **Ligation Planning Data:** Based on established clinical practice guidelines
- **Anatomical Reference:** CHIVA classification system documentation
- **Procedure Details:** Evidence-based treatment modalities (EVLA, RFA, foam sclerotherapy, open ligation)

## Training Configuration

**Recommended Fine-tuning Parameters:**
- Model: mistralai/Mistral-7B-Instruct-v0.2
- Method: LoRA with r=16
- Learning Rate: 1e-4
- Epochs: 5-10
- Batch Size: 4 (per device)
- Max Length: 512 tokens

## Usage Examples

### Training
```bash
python training_lora_from_medical_literature.py \
  --train_data ./training_datasets/training_data.jsonl \
  --val_data ./training_datasets/validation_data.jsonl
```

### Validation
```bash
python validate_fine_tuned_model.py \
  --validation_file ./training_datasets/validation_data.jsonl
```

## Sample Training Pairs

### Classification Example
**Type:** ligation
**Shunt Type:** Type 3
**Instruction:** For a Type 3 CHIVA venous shunt, outline the ligation strategy, procedure options, and expected outc...

### Ligation Planning Example
**Type:** classification
**Instruction:** Analyze the following ultrasound clips and classify the CHIVA venous shunt type:

Clips:
  • Clip 1:...


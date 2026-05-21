#!/usr/bin/env python3
"""
PHASE 2B: TASK-SPECIFIC FINE-TUNING
Fine-tune Qwen2.5-7B LoRA model on CHIVA classification and ligation tasks.
Uses the Phase 1 pre-trained LoRA adapter and trains on supervised task examples.
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import json
import jsonlines
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import PeftModel, TaskType

print("=" * 80)
print("PHASE 2B: TASK-SPECIFIC FINE-TUNING (CHIVA CLASSIFICATION)")
print("=" * 80)
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 80 + "\n")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-7B"
PHASE1_LORA_PATH = r"C:\Users\Krish\Downloads\LLM_Finetuning\lora_finetuned_model"
DATASET_FILE = r"C:\Users\Krish\Downloads\LLM_Finetuning\latest_data\training_data_FRESH.jsonl"
OUTPUT_DIR = r"C:\Users\Krish\Downloads\LLM_Finetuning\qwen_medical_lora_tasks"
CACHE_DIR = r"C:\Users\Krish\Downloads\LLM_Finetuning\.cache"

# Create directories
Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(CACHE_DIR).mkdir(exist_ok=True)

# Verify files exist
if not os.path.exists(DATASET_FILE):
    print(f"ERROR: Dataset file not found: {DATASET_FILE}")
    exit(1)

if not os.path.exists(PHASE1_LORA_PATH):
    print(f"ERROR: Phase 1 LoRA not found: {PHASE1_LORA_PATH}")
    exit(1)

print("[1/5] Preparing dataset and tokenizer...")

# ============================================================
# CUSTOM DATASET CLASS
# ============================================================

class TaskDataset(Dataset):
    """Load JSONL training data with input/output pairs."""

    def __init__(self, jsonl_file: str, tokenizer, max_length: int = 512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with jsonlines.open(jsonl_file) as reader:
            for obj in reader:
                self.examples.append(obj)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example.get("input", "")
        output_text = example.get("output", "")

        # Format as: "input: {input}\noutput: {output}"
        full_text = f"input: {input_text}\noutput: {output_text}"

        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # For causal LM, labels = input_ids
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(input_ids)
        }

# ============================================================
# LOAD TOKENIZER & BASE MODEL
# ============================================================

print("[2/5] Loading tokenizer and base model...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"  Tokenizer: {BASE_MODEL}")
print(f"  Vocab size: {len(tokenizer)}")

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)
print(f"  Base Model: {BASE_MODEL}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
print()

# ============================================================
# LOAD PHASE 1 LORA ADAPTER
# ============================================================

print("[3/5] Loading Phase 1 LoRA adapter...")

# Load the LoRA adapter from Phase 1
model = PeftModel.from_pretrained(
    model,
    PHASE1_LORA_PATH,
    is_trainable=True  # Enable training
)
model.train()

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())

print(f"  Phase 1 LoRA loaded from: {PHASE1_LORA_PATH}")
print(f"  Trainable params: {trainable_params / 1e6:.1f}M ({100 * trainable_params / all_params:.2f}%)")
print(f"  Total params: {all_params / 1e9:.1f}B")
print()

# ============================================================
# PREPARE DATASET
# ============================================================

print("[4/5] Preparing dataset...")

train_dataset = TaskDataset(DATASET_FILE, tokenizer, max_length=512)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8
)

print(f"  Dataset size: {len(train_dataset)} examples")
print(f"  Max sequence length: 512 tokens")
print()

# ============================================================
# TRAINING ARGUMENTS
# ============================================================

print("[5/5] Training on task-specific examples...")
print()

# Small dataset (28 examples), so high epochs and frequent evaluation
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=10,              # More epochs on small dataset
    per_device_train_batch_size=2,    # RTX 5090, float16
    gradient_accumulation_steps=4,    # Effective batch: 8
    learning_rate=2e-4,               # Lower LR for task fine-tuning
    weight_decay=0.01,
    warmup_steps=10,
    logging_steps=2,                  # Log frequently
    save_strategy="epoch",            # Save every epoch
    save_total_limit=3,               # Keep last 3 checkpoints
    load_best_model_at_end=False,     # Keep last model
    fp16=True,                        # Float16 for speed
    gradient_checkpointing=True,      # Save memory
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    log_level="info",
    report_to=["tensorboard"],
    push_to_hub=False,
)

print("Training Configuration:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size (effective): {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Optimizer: AdamW + Cosine scheduler")
print(f"  Mixed precision: float16")
print()
print("Starting training... (this will take ~5-10 minutes)")
print()

# ============================================================
# TRAIN
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.001
        )
    ]
)

train_result = trainer.train()

print()
print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Training loss: {train_result.training_loss:.4f}")
print(f"Training steps: {train_result.global_step}")
print()

# ============================================================
# SAVE
# ============================================================

print("Saving LoRA adapters...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training config
config = {
    'model': BASE_MODEL,
    'phase1_lora': PHASE1_LORA_PATH,
    'dataset_file': DATASET_FILE,
    'num_examples': len(train_dataset),
    'training_epochs': training_args.num_train_epochs,
    'training_loss': train_result.training_loss,
    'training_steps': train_result.global_step,
    'learning_rate': training_args.learning_rate,
    'timestamp': datetime.now().isoformat(),
}

with open(os.path.join(OUTPUT_DIR, 'training_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print(f"[OK] Task LoRA saved to: {OUTPUT_DIR}")
print(f"[OK] Config saved to: {os.path.join(OUTPUT_DIR, 'training_config.json')}")
print()

print("=" * 80)
print("NEXT: Test the fine-tuned model on CHIVA classification tasks")
print("=" * 80)

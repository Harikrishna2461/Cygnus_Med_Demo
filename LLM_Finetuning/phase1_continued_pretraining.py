#!/usr/bin/env python3
"""
PHASE 1: CONTINUED PRE-TRAINING
Train Qwen2.5-7B on medical literature (14 books) using LoRA
Uses causal language modeling objective to deeply ground model in medical knowledge
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import json
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType

print("=" * 80)
print("PHASE 1: CONTINUED PRE-TRAINING ON MEDICAL LITERATURE")
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

MODEL_NAME = "Qwen/Qwen2.5-7B"
DATA_FILE = r"C:\Users\Krish\Downloads\LLM_Finetuning\full_medical_data.txt"
OUTPUT_DIR = r"C:\Users\Krish\Downloads\LLM_Finetuning\qwen_medical_pretrained"
LORA_OUTPUT = r"C:\Users\Krish\Downloads\LLM_Finetuning\qwen_medical_lora"
CACHE_DIR = r"C:\Users\Krish\Downloads\LLM_Finetuning\.cache"

# Create directories
Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(LORA_OUTPUT).mkdir(exist_ok=True)
Path(CACHE_DIR).mkdir(exist_ok=True)

# Verify input file exists
if not os.path.exists(DATA_FILE):
    print(f"ERROR: Data file not found: {DATA_FILE}")
    exit(1)

data_size_mb = os.path.getsize(DATA_FILE) / (1024 * 1024)
print(f"[1/5] Data file found")
print(f"  Path: {DATA_FILE}")
print(f"  Size: {data_size_mb:.2f} MB")
print()

# ============================================================
# LOAD TOKENIZER & MODEL
# ============================================================

print("[2/5] Loading tokenizer and base model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"  Tokenizer: {MODEL_NAME}")
print(f"  Vocab size: {len(tokenizer)}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)
print(f"  Model: {MODEL_NAME}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
print()

# ============================================================
# SETUP LORA
# ============================================================

print("[3/5] Configuring LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # Rank
    lora_alpha=32,                 # Alpha (lora_alpha / r = 2, moderate strength)
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # Qwen uses these
    bias="none",
    modules_to_save=None,
)

model = get_peft_model(model, lora_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"  LoRA rank: {lora_config.r}")
print(f"  LoRA alpha: {lora_config.lora_alpha}")
print(f"  Trainable params: {trainable_params / 1e6:.1f}M ({100 * trainable_params / all_params:.2f}%)")
print(f"  Total params: {all_params / 1e9:.1f}B")
print()

# ============================================================
# PREPARE DATASET
# ============================================================

print("[4/5] Preparing dataset...")

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=DATA_FILE,
    block_size=512,  # Context window
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Not masked LM, causal LM
)

print(f"  Dataset size: {len(train_dataset)} blocks")
print(f"  Block size: 512 tokens")
print()

# ============================================================
# TRAINING ARGUMENTS
# ============================================================

print("[5/5] Training...")
print()

# Adjusted for 16-hour window on RTX 5090
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=2,              # 2 epochs on 899K words
    per_device_train_batch_size=2,   # RTX 5090, float16
    gradient_accumulation_steps=8,   # Effective batch: 16
    learning_rate=5e-4,              # Lower than fine-tuning (less data)
    weight_decay=0.01,
    warmup_steps=200,
    logging_steps=50,
    save_strategy="steps",
    save_steps=300,                  # Save every 300 steps
    save_total_limit=3,              # Keep last 3 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    fp16=True,                       # Float16 for speed
    gradient_checkpointing=True,     # Save memory
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
print(f"  Save strategy: Every {training_args.save_steps} steps")
print(f"  Optimizer: AdamW + Cosine scheduler")
print(f"  Mixed precision: float16")
print()
print("Starting training... (this will take ~12-14 hours)")
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
            early_stopping_patience=3,
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
print(f"Training time: {train_result.training_steps}")
print()

# ============================================================
# SAVE
# ============================================================

print("Saving LoRA adapters...")
model.save_pretrained(LORA_OUTPUT)
tokenizer.save_pretrained(LORA_OUTPUT)

# Save training config
config = {
    'model': MODEL_NAME,
    'data_file': DATA_FILE,
    'data_size_mb': data_size_mb,
    'num_words': 899042,
    'lora_rank': lora_config.r,
    'lora_alpha': lora_config.lora_alpha,
    'training_epochs': training_args.num_train_epochs,
    'training_loss': train_result.training_loss,
    'timestamp': datetime.now().isoformat(),
}

with open(os.path.join(LORA_OUTPUT, 'training_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ LoRA saved to: {LORA_OUTPUT}")
print(f"✓ Config saved to: {os.path.join(LORA_OUTPUT, 'training_config.json')}")
print()

print("=" * 80)
print("NEXT: Phase 2 - Fine-tune on diverse medical tasks")
print("=" * 80)

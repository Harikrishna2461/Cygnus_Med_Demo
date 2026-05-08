#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-7B on clean merged dataset (6016 examples, no system prompts)
Optimized for flexible domain knowledge learning on venous/vascular medicine
"""

import os
import torch
import json
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
MODEL_ID = "Qwen/Qwen2.5-7B"
DATA_DIR = Path("latest_data")
OUTPUT_DIR = Path("./Models/Qwen2.5-7B-finetuned-clean")
TRAIN_FILE = DATA_DIR / "training_data.jsonl"
VAL_FILE = DATA_DIR / "validation_data.jsonl"

# Training hyperparameters (optimized for flexible learning)
NUM_EPOCHS = 3  # Increased from 2 for better domain absorption
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 1.5e-4  # Slightly lower for stability
MAX_SEQ_LENGTH = 768  # Balanced: not too long, captures full examples
WARMUP_STEPS = 100
SAVE_STEPS = 100

# LoRA config
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# ============================================================
# SETUP
# ============================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.empty_cache()

print("=" * 70)
print("FINE-TUNING QWEN2.5-7B ON CLEAN MERGED DATA")
print("=" * 70)
print(f"\nConfig:")
print(f"  Model: {MODEL_ID}")
print(f"  Training data: {TRAIN_FILE}")
print(f"  Validation data: {VAL_FILE}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Max sequence length: {MAX_SEQ_LENGTH}")

# ============================================================
# LOAD DATA
# ============================================================
print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

def process_examples(examples):
    """Format examples as instruction → response pairs."""
    texts = []
    for instruction, response in zip(examples['instruction'], examples['response']):
        # Clean format: no prefixes, just instruction and response
        text = f"{instruction}\n{response}"
        texts.append(text)
    return {'text': texts}

# Load datasets
print("Loading training data...", end=" ")
train_dataset = load_dataset('json', data_files=str(TRAIN_FILE), split='train')
print(f"✓ {len(train_dataset)} examples")

print("Loading validation data...", end=" ")
val_dataset = load_dataset('json', data_files=str(VAL_FILE), split='train')
print(f"✓ {len(val_dataset)} examples")

# Process examples
print("Processing examples...", end=" ")
train_dataset = train_dataset.map(process_examples, batched=True, batch_size=1000, remove_columns=['instruction', 'response'])
val_dataset = val_dataset.map(process_examples, batched=True, batch_size=1000, remove_columns=['instruction', 'response'])
print("✓")

# ============================================================
# TOKENIZATION
# ============================================================
print("\n" + "=" * 70)
print("TOKENIZING DATA")
print("=" * 70)

print("Loading tokenizer...", end=" ")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print(f"✓")

def tokenize_function(examples):
    """Tokenize examples."""
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )
    result['labels'] = result['input_ids'].copy()
    return result

print("Tokenizing training data...", end=" ")
train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=100,
    remove_columns=['text']
)
print(f"✓")

print("Tokenizing validation data...", end=" ")
val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=100,
    remove_columns=['text']
)
print(f"✓")

# ============================================================
# MODEL SETUP
# ============================================================
print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

print("Loading base model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True
)
print("✓ Model loaded with 4-bit quantization")

# LoRA config
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("✓ LoRA configured")

# ============================================================
# TRAINING
# ============================================================
print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    save_steps=SAVE_STEPS,
    eval_steps=SAVE_STEPS,
    logging_steps=10,
    evaluation_strategy='steps',
    save_strategy='steps',
    load_best_model_at_end=True,
    report_to=['tensorboard'],
    optim='paged_adamw_8bit',
    max_grad_norm=0.3,
    seed=42,
    bf16=True,
    gradient_checkpointing=False,
    dataloader_pin_memory=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Starting training...")
train_result = trainer.train()

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Training loss: {train_result.training_loss:.4f}")
print(f"Model saved to: {OUTPUT_DIR}")

# Save final model
trainer.save_model(str(OUTPUT_DIR))
print("✓ Model and LoRA adapters saved")

print("\nTraining Summary:")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Training examples: {len(train_dataset)}")
print(f"  Validation examples: {len(val_dataset)}")
print(f"  Final loss: {train_result.training_loss:.4f}")
print(f"  Output directory: {OUTPUT_DIR}")

print("\n✓ Fine-tuning complete! Model is ready for evaluation.")

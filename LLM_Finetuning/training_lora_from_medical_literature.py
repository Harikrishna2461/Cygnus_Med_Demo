"""
Fine-tune Mistral-7B using LoRA on medical literature extracted from CHIVA PDFs.
This is the corrected approach: train on REAL domain knowledge, not synthetic rules.
"""

import torch
import json
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_OUTPUT_DIR = "./lora_chiva_medical_literature"
TRAINING_DATA_PATH = "./training_data_from_pdfs/training_pairs_from_medical_literature.jsonl"

# LoRA Configuration
LORA_CONFIG = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Training Arguments
TRAINING_ARGS = TrainingArguments(
    output_dir=LORA_OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    save_steps=50,
    save_total_limit=3,
    logging_steps=10,
    learning_rate=1e-4,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    warmup_steps=20,
    weight_decay=0.01,
    max_grad_norm=1.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data(data_path: str) -> list:
    """Load instruction-response pairs from JSONL file."""
    pairs = []

    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                pair = json.loads(line)
                pairs.append(pair)

    return pairs


def format_instruction_response(pair: dict) -> str:
    """Format instruction-response pair for training."""
    instruction = pair.get("instruction", "").strip()
    input_text = pair.get("input", "").strip()
    output = pair.get("output", "").strip()

    # Using Mistral's instruction format
    if input_text:
        prompt = f"[INST] {instruction}\n\n{input_text} [/INST]"
    else:
        prompt = f"[INST] {instruction} [/INST]"

    # Complete prompt with response
    full_text = f"{prompt} {output}"

    return full_text


def create_dataset(pairs: list, tokenizer, max_length: int = 512) -> Dataset:
    """Create HuggingFace Dataset from training pairs."""

    # Format all pairs
    texts = [format_instruction_response(pair) for pair in pairs]

    # Tokenize
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model():
    """Train Mistral-7B with LoRA on medical literature."""

    print("Loading training data...")
    if not Path(TRAINING_DATA_PATH).exists():
        print(f"Error: {TRAINING_DATA_PATH} not found")
        print("Please run: python extract_and_prepare_training_data.py")
        return

    pairs = load_training_data(TRAINING_DATA_PATH)
    print(f"Loaded {len(pairs)} training pairs")

    # Sample some examples
    print("\nSample training pairs:")
    for i, pair in enumerate(pairs[:2]):
        print(f"\n{i+1}. Type: {pair.get('type')}")
        print(f"   Instruction: {pair.get('instruction')[:60]}...")
        print(f"   Output: {pair.get('output')[:60]}...")

    print("\n" + "="*60)
    print("Loading model and tokenizer...")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("✓ Model and tokenizer loaded")

    # Create dataset
    print("\nPreparing dataset...")
    dataset = create_dataset(pairs, tokenizer)
    print(f"✓ Dataset ready with {len(dataset)} examples")

    # Apply LoRA
    print("\nApplying LoRA configuration...")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # Training
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    # Save
    print("\n" + "="*60)
    print("Saving LoRA adapter...")
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    print(f"✓ Saved to {LORA_OUTPUT_DIR}")

    print("\n✓ Training complete!")
    print(f"\nNext steps:")
    print(f"1. Validate with: python chiva_classifier_api.py")
    print(f"2. Update use_lora=True in your classifier")


if __name__ == "__main__":
    train_model()

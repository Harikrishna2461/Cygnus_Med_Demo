"""
Fine-tune Mistral-7B with LoRA on CHIVA shunt classification and ligation planning.
This is the PROPER training approach using real medical knowledge datasets.
"""

import torch
import json
import argparse
from pathlib import Path
from typing import Optional
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_LORA_OUTPUT = "./lora_chiva_classifier_final"
DEFAULT_TRAIN_FILE = "./training_datasets/training_data.jsonl"
DEFAULT_VAL_FILE = "./training_datasets/validation_data.jsonl"


def load_dataset_from_jsonl(file_path: str) -> list:
    """Load instruction-response pairs from JSONL file."""
    pairs = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def format_for_training(pair: dict) -> str:
    """Format instruction-response pair for training."""
    instruction = pair.get("instruction", "").strip()
    input_text = pair.get("input", "").strip()
    output = pair.get("output", "").strip()

    # Mistral instruction format
    if input_text:
        prompt = f"[INST] {instruction}\n\n{input_text} [/INST]"
    else:
        prompt = f"[INST] {instruction} [/INST]"

    # Complete prompt with response
    full_text = f"{prompt} {output}"

    return full_text


def create_hf_dataset(pairs: list, tokenizer, max_length: int = 512) -> Dataset:
    """Create HuggingFace Dataset from training pairs."""

    # Format all pairs
    texts = [format_for_training(pair) for pair in pairs]

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


def setup_lora_config() -> LoraConfig:
    """Configure LoRA adapter."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def setup_training_args(output_dir: str, num_train_epochs: int = 5) -> TrainingArguments:
    """Configure training arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=5,
        save_steps=10,
        save_total_limit=3,
        logging_steps=2,
        learning_rate=1e-4,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        warmup_steps=5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_dir="./logs",
        report_to=["tensorboard"],
    )


def train_lora_model(
    train_file: str,
    val_file: str,
    output_dir: str,
    num_epochs: int = 5,
    max_length: int = 512,
):
    """Train Mistral-7B with LoRA."""

    print("="*80)
    print("CHIVA SHUNT CLASSIFIER - LORA FINE-TUNING")
    print("="*80)

    # Load data
    print("\n[STEP 1] Loading training data...")
    if not Path(train_file).exists():
        print(f"ERROR: Training file not found: {train_file}")
        return

    if not Path(val_file).exists():
        print(f"ERROR: Validation file not found: {val_file}")
        return

    train_pairs = load_dataset_from_jsonl(train_file)
    val_pairs = load_dataset_from_jsonl(val_file)

    print(f"  Loaded {len(train_pairs)} training pairs")
    print(f"  Loaded {len(val_pairs)} validation pairs")

    # Show samples
    print("\n[STEP 2] Sample training pairs:")
    for i, pair in enumerate(train_pairs[:2]):
        print(f"\n  Pair {i+1}:")
        print(f"    Type: {pair.get('type')}")
        print(f"    Instruction: {pair.get('instruction')[:60]}...")
        print(f"    Output: {pair.get('output')[:60]}...")

    # Load model & tokenizer
    print("\n[STEP 3] Loading base model...")
    print(f"  Model: {MODEL_NAME}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("  ✓ Model and tokenizer loaded")

    # Create datasets
    print("\n[STEP 4] Preparing datasets...")
    train_dataset = create_hf_dataset(train_pairs, tokenizer, max_length)
    val_dataset = create_hf_dataset(val_pairs, tokenizer, max_length)

    print(f"  ✓ Training dataset: {len(train_dataset)} examples")
    print(f"  ✓ Validation dataset: {len(val_dataset)} examples")

    # Apply LoRA
    print("\n[STEP 5] Configuring LoRA...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct_trainable = 100 * trainable_params / total_params

    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Total params: {total_params:,}")
    print(f"  Percentage: {pct_trainable:.3f}%")

    # Training
    print("\n[STEP 6] Setting up training...")
    training_args = setup_training_args(output_dir, num_epochs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    trainer.train()

    # Save
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\n[STEP 7] Saving LoRA adapter...")
    print(f"  Saving to: {output_dir}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"  ✓ LoRA adapter saved")
    print(f"  ✓ Tokenizer saved")

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"\nLoRA Adapter Location: {output_dir}/")
    print(f"Training Pairs: {len(train_pairs)}")
    print(f"Validation Pairs: {len(val_pairs)}")
    print(f"Epochs: {num_epochs}")
    print(f"Trainable Parameters: {pct_trainable:.3f}%")

    print("\n[NEXT STEPS]")
    print(f"1. Update chiva_classifier_api.py to use:")
    print(f"   CHIVAShuntClassifier(use_lora=True, lora_path='{output_dir}')")
    print(f"2. Test with: python chiva_classifier_api.py")
    print(f"3. Validate with: python validate_unified_inference.py")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mistral-7B for CHIVA classification")
    parser.add_argument(
        "--train_data",
        default=DEFAULT_TRAIN_FILE,
        help=f"Path to training data JSONL (default: {DEFAULT_TRAIN_FILE})",
    )
    parser.add_argument(
        "--val_data",
        default=DEFAULT_VAL_FILE,
        help=f"Path to validation data JSONL (default: {DEFAULT_VAL_FILE})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_LORA_OUTPUT,
        help=f"Output directory for LoRA adapter (default: {DEFAULT_LORA_OUTPUT})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max token length (default: 512)",
    )

    args = parser.parse_args()

    train_lora_model(
        train_file=args.train_data,
        val_file=args.val_data,
        output_dir=args.output,
        num_epochs=args.epochs,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()

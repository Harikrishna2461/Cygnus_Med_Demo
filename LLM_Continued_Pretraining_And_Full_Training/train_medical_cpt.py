#!/usr/bin/env python3
"""
Medical Continued Pretraining (CPT) Pipeline
Trains Qwen2.5-7B on augmented medical data for domain adaptation.

Usage:
    python train_medical_cpt.py                    # Use default config
    python train_medical_cpt.py --num_epochs 5     # Override specific params
    python train_medical_cpt.py --help              # See all options
"""

import os
import json
import torch
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# ============ SETUP LOGGING ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============ DEFAULT CONFIG ============
DEFAULT_CONFIG = {
    # Model & Data
    "model_name": "Qwen/Qwen2.5-7B",
    "train_file": "augmented_output/train.jsonl",
    "eval_file": "augmented_output/eval.jsonl",

    # Training Parameters
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,  # Effective batch: 4*4=16
    "learning_rate": 2e-5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,

    # Sequence & Optimization
    "max_seq_length": 2048,
    "bf16": True,  # Use bfloat16 for modern GPUs
    "fp16": False,

    # Checkpointing & Saving
    "output_dir": "medical_qwen_cpt",
    "save_strategy": "steps",
    "save_steps": 200,
    "save_total_limit": 5,

    # Evaluation
    "eval_strategy": "steps",
    "eval_steps": 100,

    # Logging
    "logging_dir": "logs",
    "logging_steps": 10,

    # Other
    "seed": 42,
    "dataloader_num_workers": 4,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Medical CPT Training Pipeline")

    # Model and data
    parser.add_argument("--model_name", default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--train_file", default=DEFAULT_CONFIG["train_file"])
    parser.add_argument("--eval_file", default=DEFAULT_CONFIG["eval_file"])

    # Training
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_CONFIG["num_train_epochs"])
    parser.add_argument("--train_batch_size", type=int, default=DEFAULT_CONFIG["per_device_train_batch_size"])
    parser.add_argument("--eval_batch_size", type=int, default=DEFAULT_CONFIG["per_device_eval_batch_size"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_CONFIG["gradient_accumulation_steps"])
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_CONFIG["warmup_steps"])
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_CONFIG["max_seq_length"])

    # Output
    parser.add_argument("--output_dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--save_steps", type=int, default=DEFAULT_CONFIG["save_steps"])
    parser.add_argument("--eval_steps", type=int, default=DEFAULT_CONFIG["eval_steps"])

    # Optimization
    parser.add_argument("--bf16", action="store_true", default=DEFAULT_CONFIG["bf16"])
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])

    return parser.parse_args()


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Line {i} invalid JSON: {e}")
    return data


def verify_data(train_data: List[Dict], eval_data: List[Dict]):
    """Verify data integrity."""
    logger.info(f"Train chunks: {len(train_data):,}")
    logger.info(f"Eval chunks: {len(eval_data):,}")

    train_tokens = sum(c.get('token_count', 0) for c in train_data)
    eval_tokens = sum(c.get('token_count', 0) for c in eval_data)

    logger.info(f"Train tokens: {train_tokens:,}")
    logger.info(f"Eval tokens: {eval_tokens:,}")

    if not train_data:
        raise ValueError("Training data is empty!")
    if not eval_data:
        logger.warning("Evaluation data is empty!")

    return train_tokens, eval_tokens


def tokenize_function(examples, tokenizer, max_seq_length):
    """Tokenize text with padding."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )


def main():
    """Main training pipeline."""
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ============ LOAD DATA ============
    logger.info("="*80)
    logger.info("LOADING DATA")
    logger.info("="*80)

    if not Path(args.train_file).exists():
        raise FileNotFoundError(f"Train file not found: {args.train_file}")
    if not Path(args.eval_file).exists():
        raise FileNotFoundError(f"Eval file not found: {args.eval_file}")

    logger.info(f"Loading training data from {args.train_file}...")
    train_data = load_jsonl(args.train_file)

    logger.info(f"Loading evaluation data from {args.eval_file}...")
    eval_data = load_jsonl(args.eval_file)

    train_tokens, eval_tokens = verify_data(train_data, eval_data)

    # ============ LOAD TOKENIZER ============
    logger.info("="*80)
    logger.info("LOADING TOKENIZER & MODEL")
    logger.info("="*80)

    logger.info(f"Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Tokenizer vocab size: {len(tokenizer):,}")

    # ============ LOAD MODEL ============
    logger.info(f"Loading model: {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params/1e9:.2f}B")
    logger.info(f"Model dtype: {next(model.parameters()).dtype}")

    # ============ TOKENIZE DATASETS ============
    logger.info("="*80)
    logger.info("TOKENIZING DATASETS")
    logger.info("="*80)

    train_dataset = Dataset.from_dict({"text": [c["text"] for c in train_data]})
    eval_dataset = Dataset.from_dict({"text": [c["text"] for c in eval_data]})

    logger.info("Tokenizing training dataset...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_seq_length),
        batched=True,
        batch_size=100,
        remove_columns=["text"],
    )

    logger.info("Tokenizing evaluation dataset...")
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_seq_length),
        batched=True,
        batch_size=100,
        remove_columns=["text"],
    )

    logger.info(f"Train samples: {len(train_dataset):,}")
    logger.info(f"Eval samples: {len(eval_dataset):,}")

    # ============ SETUP TRAINING ============
    logger.info("="*80)
    logger.info("SETTING UP TRAINING")
    logger.info("="*80)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=args.bf16,
        fp16=False,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=5,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        metric_for_best_model="eval_loss",
        logging_dir=args.logging_dir,
        logging_steps=10,
        seed=args.seed,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        greater_is_better=False,
        push_to_hub=False,
        report_to=["tensorboard"],
    )

    effective_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    total_steps = (len(train_dataset) // effective_batch_size + 1) * args.num_epochs

    logger.info(f"Effective batch size: {effective_batch_size}")
    logger.info(f"Total training steps: ~{total_steps}")
    logger.info(f"Checkpoint directory: {args.output_dir}")

    # ============ CREATE TRAINER ============
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # ============ TRAIN ============
    logger.info("="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    train_result = trainer.train()

    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ============ EVALUATE ============
    logger.info("="*80)
    logger.info("EVALUATING FINAL MODEL")
    logger.info("="*80)

    eval_results = trainer.evaluate()
    logger.info(f"Evaluation loss: {eval_results.get('eval_loss', 'N/A'):.4f}")

    # ============ SAVE BEST MODEL ============
    best_model_path = Path(args.output_dir) / "best_model"
    logger.info("="*80)
    logger.info("SAVING BEST MODEL")
    logger.info("="*80)

    model.save_pretrained(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path))

    model_size = sum(f.stat().st_size for f in best_model_path.glob('**/*')) / 1e9
    logger.info(f"Model saved: {best_model_path}")
    logger.info(f"Model size: {model_size:.2f} GB")

    # ============ SUMMARY ============
    logger.info("="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(f"Dataset size increase: {train_tokens / 1e6:.2f}M → {(train_tokens + eval_tokens) / 1e6:.2f}M tokens")
    logger.info(f"Model: {num_params/1e9:.2f}B parameters")
    logger.info(f"Training epochs: {args.num_epochs}")
    logger.info(f"Best model: {best_model_path}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. View tensorboard: tensorboard --logdir {args.logging_dir}")
    logger.info(f"  2. Load model: AutoModelForCausalLM.from_pretrained('{best_model_path}')")
    logger.info(f"  3. Verify training: python diagnose_training.py")
    logger.info("="*80)


if __name__ == "__main__":
    main()

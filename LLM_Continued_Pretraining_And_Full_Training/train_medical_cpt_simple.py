#!/usr/bin/env python3
"""
Simple Medical CPT Training - No Trainer API (less likely to hang)
Uses basic PyTorch loops with clear progress and memory tracking.
"""

import torch
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

print("=" * 80)
print("MEDICAL CPT TRAINING - SIMPLE VERSION")
print("=" * 80)

# ============ CONFIG ============
DEFAULT_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B",
    "train_file": "augmented_output/train.jsonl",
    "eval_file": "augmented_output/eval.jsonl",
    "output_dir": "medical_qwen_cpt_simple",
    "num_epochs": 1,
    "batch_size": 2,
    "learning_rate": 2e-5,
    "max_seq_length": 1024,
    "warmup_steps": 100,
    "num_samples": None,  # None = all, or set to 100 for quick test
}

# ============ DATASET ============
class MedicalDataset(TorchDataset):
    def __init__(self, file_path: str, tokenizer, max_length: int, num_samples: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        print(f"Loading {file_path}...")
        with open(file_path) as f:
            for i, line in enumerate(f):
                if num_samples and i >= num_samples:
                    break
                try:
                    data = json.loads(line)
                    self.samples.append(data["text"])
                except:
                    pass

        print(f"✓ Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

# ============ COLLATE FUNCTION ============
def collate_fn(batch):
    """Collate batch of samples."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

# ============ TRAINING LOOP ============
def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n1. Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Load tokenizer
    print(f"\n2. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   ✓ Tokenizer ready")

    # Load model
    print(f"\n3. Loading model...")
    print(f"   This may take a minute...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    elapsed = time.time() - start
    print(f"   ✓ Model loaded ({elapsed:.1f}s)")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Create datasets
    print(f"\n4. Creating datasets...")
    train_dataset = MedicalDataset(
        args.train_file,
        tokenizer,
        args.max_seq_length,
        num_samples=args.num_samples,
    )

    # Create dataloader
    print(f"\n5. Creating dataloader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # IMPORTANT: Set to 0 to avoid hanging
        pin_memory=False,  # IMPORTANT: Can cause hanging
    )
    print(f"   ✓ Dataloader ready ({len(train_loader)} batches)")

    # Setup optimizer
    print(f"\n6. Setting up optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(f"   ✓ Optimizer ready")

    # Training loop
    print(f"\n7. Starting training...")
    print("   " + "=" * 76)

    model.train()
    total_loss = 0
    step = 0

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        epoch_loss = 0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Forward pass
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )
                    loss = outputs.loss

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                # Track metrics
                step += 1
                batch_count += 1
                epoch_loss += loss.item()
                total_loss += loss.item()

                # Print progress every 5 batches
                if (batch_idx + 1) % 5 == 0:
                    avg_loss = epoch_loss / batch_count
                    gpu_mem = torch.cuda.memory_allocated(device) / 1e9
                    print(
                        f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"GPU: {gpu_mem:.1f}GB"
                    )

            except Exception as e:
                print(f"\n✗ Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                return

        # End of epoch
        epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f"  Epoch {epoch + 1} complete - Loss: {epoch_loss:.4f}")

        # Save checkpoint
        output_dir = Path(DEFAULT_CONFIG["output_dir"])
        output_dir.mkdir(exist_ok=True)
        checkpoint_path = output_dir / f"checkpoint-epoch-{epoch + 1}"
        checkpoint_path.mkdir(exist_ok=True)

        print(f"  Saving checkpoint to {checkpoint_path}...")
        model.save_pretrained(str(checkpoint_path))
        tokenizer.save_pretrained(str(checkpoint_path))
        print(f"  ✓ Checkpoint saved")

    # Final model
    print(f"\n8. Saving final model...")
    final_path = Path(DEFAULT_CONFIG["output_dir"]) / "final_model"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"   ✓ Final model saved to {final_path}")

    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total steps: {step}")
    print(f"Final loss: {epoch_loss:.4f}")
    print(f"Model saved: {final_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--train_file", default=DEFAULT_CONFIG["train_file"])
    parser.add_argument("--eval_file", default=DEFAULT_CONFIG["eval_file"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_CONFIG["max_seq_length"])
    parser.add_argument("--num_samples", type=int, default=DEFAULT_CONFIG["num_samples"])
    return parser.parse_args()


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n\n✓ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
PHASE 2: SIMPLE DIRECT TRAINING
Minimal setup for stable training on CHIVA tasks.
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import json
import jsonlines
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

print("=" * 80)
print("PHASE 2: SIMPLE TASK-SPECIFIC FINE-TUNING")
print("=" * 80)
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 80 + "\n")

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-7B"
DATASET_FILE = r"C:\Users\Krish\Downloads\LLM_Finetuning\latest_data\training_data_FRESH.jsonl"
OUTPUT_DIR = r"C:\Users\Krish\Downloads\LLM_Finetuning\qwen_chiva_tasks_lora"
CACHE_DIR = r"C:\Users\Krish\Downloads\LLM_Finetuning\.cache"

Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(CACHE_DIR).mkdir(exist_ok=True)

if not os.path.exists(DATASET_FILE):
    print(f"ERROR: Dataset not found: {DATASET_FILE}")
    exit(1)

# ============================================================
# SIMPLE DATASET
# ============================================================

class SimpleDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_len=512):
        self.data = []
        with jsonlines.open(jsonl_file) as f:
            for line in f:
                self.data.append(line)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Q: {item['input']}\nA: {item['output']}"

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
        }

# ============================================================
# LOAD MODEL
# ============================================================

print("[1/4] Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
    pad_token=None
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)

print(f"  Base model: {BASE_MODEL}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B\n")

# ============================================================
# APPLY LORA
# ============================================================

print("[2/4] Applying LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.train()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"  Trainable params: {trainable / 1e6:.1f}M ({100 * trainable / total:.2f}%)")
print(f"  Total params: {total / 1e9:.1f}B\n")

# ============================================================
# PREPARE DATA
# ============================================================

print("[3/4] Preparing dataset...")

dataset = SimpleDataset(DATASET_FILE, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print(f"  Examples: {len(dataset)}")
print(f"  Batches: {len(dataloader)}\n")

# ============================================================
# TRAINING LOOP
# ============================================================

print("[4/4] Training...")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 15
total_loss = 0
step = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        total_loss += loss.item()
        step += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    scheduler.step()
    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} completed - Avg Loss: {avg_epoch_loss:.4f}\n")

# ============================================================
# SAVE
# ============================================================

print("\nSaving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

config = {
    'model': BASE_MODEL,
    'dataset_file': DATASET_FILE,
    'num_examples': len(dataset),
    'epochs': num_epochs,
    'final_loss': total_loss / step,
    'timestamp': datetime.now().isoformat(),
}

with open(os.path.join(OUTPUT_DIR, 'training_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print(f"[OK] Model saved to: {OUTPUT_DIR}")
print(f"[OK] Final training loss: {total_loss / step:.4f}")
print(f"[OK] Total training steps: {step}")
print()

print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Next: python phase2_test_fine_tuned_model.py")
print("=" * 80)

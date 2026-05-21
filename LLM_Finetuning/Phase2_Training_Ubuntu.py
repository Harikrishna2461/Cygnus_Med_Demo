"""
PHASE 2: CHIVA Task-Specific Fine-Tuning
Jupyter Notebook Code - Ubuntu Version
Run each cell in sequence

SETUP:
1. pip install torch transformers peft jsonlines
2. Copy dataset file: latest_data/training_data_FRESH.jsonl
3. Run cells below in Jupyter
"""

# ============================================================
# CELL 1: IMPORTS & SETUP
# ============================================================

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import json
import jsonlines
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

print("✓ All imports successful")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# CELL 2: CONFIGURATION
# ============================================================

# MODIFY THESE PATHS FOR YOUR UBUNTU SETUP
BASE_MODEL = "Qwen/Qwen2.5-7B"

# Example paths (adjust to your /home/user/... paths)
DATA_DIR = "/path/to/data"  # e.g., /home/user/llm_finetuning/latest_data
DATASET_FILE = f"{DATA_DIR}/training_data_FRESH.jsonl"
OUTPUT_DIR = f"{DATA_DIR}/../qwen_chiva_tasks_lora"
CACHE_DIR = f"{DATA_DIR}/../.cache"

# Create directories
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Verify dataset
if not os.path.exists(DATASET_FILE):
    print(f"ERROR: {DATASET_FILE} not found!")
    print("Please check your paths and copy the dataset file")
else:
    print(f"✓ Dataset found: {DATASET_FILE}")
    # Count examples
    with jsonlines.open(DATASET_FILE) as f:
        count = sum(1 for _ in f)
    print(f"✓ Total examples: {count}")

# ============================================================
# CELL 3: SIMPLE DATASET CLASS
# ============================================================

class SimpleDataset(Dataset):
    """Load JSONL training data with input/output pairs."""

    def __init__(self, jsonl_file, tokenizer, max_len=512):
        self.data = []
        print(f"Loading data from {jsonl_file}...")
        with jsonlines.open(jsonl_file) as f:
            for line in f:
                self.data.append(line)
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(f"✓ Loaded {len(self.data)} examples")

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

print("✓ Dataset class defined")

# ============================================================
# CELL 4: LOAD TOKENIZER & BASE MODEL
# ============================================================

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)
tokenizer.pad_token = tokenizer.eos_token
print(f"✓ Tokenizer loaded - Vocab size: {len(tokenizer)}")

print("\nLoading base model (this may take 2-3 minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model loaded - {total_params / 1e9:.1f}B parameters")

# ============================================================
# CELL 5: APPLY LORA
# ============================================================

print("\nApplying LoRA configuration...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # Rank
    lora_alpha=32,          # Alpha (scaling)
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # Query and Value projections
    bias="none",
)

model = get_peft_model(model, lora_config)
model.train()  # Set to training mode

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"✓ LoRA applied")
print(f"  Trainable params: {trainable_params / 1e6:.1f}M ({100 * trainable_params / total_params:.2f}%)")
print(f"  Total params: {total_params / 1e9:.1f}B")
print(f"  LoRA rank: {lora_config.r}")

# ============================================================
# CELL 6: PREPARE DATASET & DATALOADER
# ============================================================

print("\nPreparing dataset...")
dataset = SimpleDataset(DATASET_FILE, tokenizer, max_len=512)
batch_size = 2  # Adjust if you have more/less VRAM
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"✓ Dataset ready")
print(f"  Total examples: {len(dataset)}")
print(f"  Batch size: {batch_size}")
print(f"  Total batches: {len(dataloader)}")
print(f"  Gradient accumulation steps: 4")
print(f"  Effective batch size: {batch_size * 4}")

# ============================================================
# CELL 7: SETUP OPTIMIZER & SCHEDULER
# ============================================================

print("\nSetting up optimizer...")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,           # Conservative learning rate
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

num_epochs = 15
total_steps = len(dataloader) * num_epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs
)

print(f"✓ Optimizer configured")
print(f"  Learning rate: 1e-4 (conservative for fine-tuning)")
print(f"  Epochs: {num_epochs}")
print(f"  Total training steps: {total_steps}")
print(f"  Scheduler: Cosine Annealing")

# ============================================================
# CELL 8: TRAINING LOOP
# ============================================================

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print(f"Start time: {datetime.now().isoformat()}\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

model = model.to(device)

total_loss = 0
global_step = 0
losses_per_epoch = []

for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_steps = 0

    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-" * 40)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

        loss = outputs.loss

        # Backward pass with gradient accumulation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimization step (every 4 batches for gradient accumulation)
        if (batch_idx + 1) % 4 == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        total_loss += loss.item()
        epoch_steps += 1
        global_step += 1

        # Print progress every 5 batches
        if (batch_idx + 1) % 5 == 0:
            avg_loss = epoch_loss / epoch_steps
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}")

    # End of epoch
    scheduler.step()
    avg_epoch_loss = epoch_loss / epoch_steps
    losses_per_epoch.append(avg_epoch_loss)

    print(f"✓ Epoch {epoch+1} completed")
    print(f"  Avg loss: {avg_epoch_loss:.4f}")
    print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6f}\n")

# ============================================================
# CELL 9: SAVE MODEL
# ============================================================

print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

print(f"\nSaving LoRA adapter to: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training config
config = {
    'model': BASE_MODEL,
    'dataset_file': DATASET_FILE,
    'num_examples': len(dataset),
    'epochs': num_epochs,
    'batch_size': batch_size,
    'learning_rate': 1e-4,
    'final_loss': total_loss / global_step,
    'total_steps': global_step,
    'timestamp': datetime.now().isoformat(),
}

with open(os.path.join(OUTPUT_DIR, 'training_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Model saved")
print(f"✓ Training config saved")
print(f"\nTraining Summary:")
print(f"  Final loss: {config['final_loss']:.4f}")
print(f"  Total steps: {global_step}")
print(f"  Total epochs: {num_epochs}")

# ============================================================
# CELL 10: PLOT TRAINING LOSS
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(losses_per_epoch) + 1), losses_per_epoch, marker='o', linestyle='-', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Average Loss', fontsize=12)
plt.title('CHIVA Model Training Loss', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_loss.png", dpi=100)
plt.show()

print(f"✓ Loss plot saved to {OUTPUT_DIR}/training_loss.png")

# ============================================================
# CELL 11: TEST THE MODEL
# ============================================================

print("\n" + "="*80)
print("TESTING FINE-TUNED MODEL")
print("="*80 + "\n")

# Test prompts
test_cases = [
    "Classify: EP N1->N2 at y=0.06 with RP N2->N1 at y=0.25. No N3.",
    "For TYPE 1 shunt, what is the ligation strategy?",
    "Classify: EP N2->N3 at y=0.20 with no EP N1->N2.",
]

model.eval()
with torch.no_grad():
    for i, prompt in enumerate(test_cases, 1):
        print(f"[Test {i}]")
        print(f"Q: {prompt}")

        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=False,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()

        print(f"A: {response}\n")

# ============================================================
# CELL 12: SAVE MODEL INFO
# ============================================================

print("="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nModel saved to: {OUTPUT_DIR}")
print(f"\nTo use this model in future sessions:")
print(f"""
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "Qwen/Qwen2.5-7B"
lora_path = "{OUTPUT_DIR}"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, lora_path)
tokenizer = AutoTokenizer.from_pretrained(lora_path)

# Use for inference
model.eval()
with torch.no_grad():
    outputs = model.generate(...)
""")

print("\nNext steps:")
print("1. Run inference_chiva_model.py for interactive testing")
print("2. Run phase2_test_fine_tuned_model.py for automated evaluation")
print("3. Deploy the model using the code above")

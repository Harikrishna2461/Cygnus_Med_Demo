"""
PASTE THIS INTO A JUPYTER CELL IN YOUR NOTEBOOK
This trains the already fine-tuned model on reasoning tasks
"""

import sys
import os
import torch
import gc

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()

# Add Windows path (accessible via WSL2 mount) to Python path
wsl_path = "/mnt/c/Users/Krish/Downloads/LLM_Finetuning"
sys.path.insert(0, wsl_path)
os.chdir(wsl_path)

print(f"Working directory: {os.getcwd()}")
print(f"Files in directory: {[f for f in os.listdir('.') if f.startswith('training_')]}")

# ============================================================================
# STEP 1: IMPORT AND LOAD DATA
# ============================================================================

from training_data_comprehensive import generate_comprehensive_training_pairs
from datasets import Dataset
import numpy as np

# Generate training data
training_pairs = generate_comprehensive_training_pairs()
print(f"Generated {len(training_pairs)} training pairs")

# Split into train/eval
np.random.seed(42)
indices = np.random.permutation(len(training_pairs))
train_size = int(0.8 * len(training_pairs))
train_indices = indices[:train_size]
eval_indices = indices[train_size:]

train_data = [training_pairs[i] for i in train_indices]
eval_data = [training_pairs[i] for i in eval_indices]

train_dataset = Dataset.from_dict({
    "text": [ex["text"] for ex in train_data],
})

eval_dataset = Dataset.from_dict({
    "text": [ex["text"] for ex in eval_data],
})

print(f"Training pairs: {len(train_dataset)}")
print(f"Eval pairs: {len(eval_dataset)}")

# ============================================================================
# STEP 2: SETUP TRAINING (assumes model, tokenizer already loaded)
# ============================================================================

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from pathlib import Path

output_dir = "./reasoning_finetuned_output"
Path(output_dir).mkdir(exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=2,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=1,
    eval_strategy="no",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=False,
    max_grad_norm=1.0,
    seed=42,
    bf16=True,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
)

# ============================================================================
# STEP 3: TOKENIZE DATA AND CREATE TRAINER
# ============================================================================

# Note: tokenizer should already be loaded in your notebook from the cell where you loaded the model

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset_tokenized = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tokenized,
    data_collator=data_collator,
)

print("Starting training...")
train_result = trainer.train()

print(f"\nTraining completed!")
print(f"Final training loss: {train_result.training_loss:.4f}")

# ============================================================================
# STEP 4: SAVE REFINED MODEL
# ============================================================================

final_output_dir = "./mistral_reasoning_enhanced"
Path(final_output_dir).mkdir(exist_ok=True)

model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)

print(f"✓ Model saved to {final_output_dir}")

# ============================================================================
# STEP 5: QUICK TEST ON NEW CASE
# ============================================================================

print("\n" + "="*80)
print("QUICK INFERENCE TEST")
print("="*80)

test_prompt = """[INST] === SHUNT CLASSIFICATION ===
Clips:
  Clip 00: EP N1→N2  y=0.080 [SFJ-ENTRY=INCOMPETENT]
  Clip 01: RP N2→N1  y=0.300 [GSV-TRUNK-REFLUX: N2→N1]

Classify using CHIVA rules. Provide type, confidence, and reasoning. [/INST]"""

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=250, temperature=0.3, top_p=0.9)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

if "[/INST]" in response:
    response = response.split("[/INST]")[1].strip()

print("Test Response:")
print(response)

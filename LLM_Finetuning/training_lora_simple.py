"""
SIMPLE LoRA FINE-TUNING ON COMPREHENSIVE TRAINING DATA
Uses much less memory than full fine-tuning
"""

import sys
import os
import torch
import gc

torch.cuda.empty_cache()
gc.collect()

wsl_path = "/mnt/c/Users/Krish/Downloads/LLM_Finetuning"
sys.path.insert(0, wsl_path)
os.chdir(wsl_path)

from training_data_comprehensive import generate_comprehensive_training_pairs
from datasets import Dataset
import numpy as np
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

print("=" * 80)
print("LORA FINE-TUNING ON COMPREHENSIVE CHIVA DATA")
print("=" * 80)

# Generate training data
training_pairs = generate_comprehensive_training_pairs()
print(f"\n✓ Generated {len(training_pairs)} comprehensive training pairs")

# Create dataset
train_dataset = Dataset.from_dict({
    "text": [pair["text"] for pair in training_pairs],
})

# Tokenize
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

train_tokenized = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
print(f"✓ Tokenized {len(train_tokenized)} examples")

# Setup LoRA
print("\nSetting up LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

model.enable_input_require_grads()
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training args
training_args = TrainingArguments(
    output_dir="./lora_finetuned_output",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_steps=2,
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
)

# Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    data_collator=data_collator,
)

print("\nStarting LoRA fine-tuning...")
train_result = trainer.train()

print(f"\n✓ Training complete!")
print(f"✓ Final loss: {train_result.training_loss:.4f}")

# Save
output_dir = "./lora_finetuned_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\n✓ Model saved to {output_dir}")

print("\n" + "=" * 80)
print("QUICK TEST ON NEW CASE")
print("=" * 80)

test_prompt = """[INST] A patient has reflux at the saphenofemoral junction with GSV reflux to mid-thigh. What type of venous shunt is this and what ligation strategy do you recommend? [/INST]"""

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.3, top_p=0.9)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
if "[/INST]" in response:
    response = response.split("[/INST]")[1].strip()

print(f"Response:\n{response}")

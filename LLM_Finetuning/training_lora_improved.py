"""
IMPROVED LORA FINE-TUNING WITH EXPLICIT CHIVA DECISION LOGIC
Uses better training data that teaches classification rules
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

from training_data_better import generate_better_training_pairs
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

print("=" * 80)
print("IMPROVED LORA FINE-TUNING - EXPLICIT CHIVA DECISION LOGIC")
print("=" * 80)

# Generate BETTER training data with explicit decision rules
training_pairs = generate_better_training_pairs()
print(f"\n✓ Generated {len(training_pairs)} training pairs with explicit decision logic")

# Create dataset - format as [INST] instruction [/INST] response
formatted_pairs = []
for pair in training_pairs:
    text = f"[INST] {pair['instruction']} [/INST] {pair['response']}"
    formatted_pairs.append({"text": text})

train_dataset = Dataset.from_dict({
    "text": [pair["text"] for pair in formatted_pairs],
})

# Tokenize
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024)

train_tokenized = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
print(f"✓ Tokenized {len(train_tokenized)} examples")

# Setup LoRA
print("\nSetting up LoRA...")
lora_config = LoraConfig(
    r=16,  # Increased from 8 for more capacity
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj"],  # Added k_proj
)

model.enable_input_require_grads()
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training args - more epochs for better learning
training_args = TrainingArguments(
    output_dir="./lora_chiva_classifier",
    num_train_epochs=5,  # Increased from 2
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Increased from 2
    learning_rate=3e-4,  # Slightly higher
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

print("\nStarting improved LoRA fine-tuning...")
print("Training on explicit CHIVA decision logic...\n")
train_result = trainer.train()

print(f"\n✓ Training complete!")
print(f"✓ Final loss: {train_result.training_loss:.4f}")

# Save
output_dir = "./lora_chiva_classifier"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\n✓ Model saved to {output_dir}")

# Quick test
print("\n" + "=" * 80)
print("QUICK TEST - TYPE 1 CLASSIFICATION")
print("=" * 80)

test_prompt = """[INST] Classify: SFJ is INCOMPETENT with reflux down the GSV to mid-thigh tributaries. No other pathways. What type and ligation strategy? [/INST]"""

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

model.gradient_checkpointing_disable()
model.eval()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
if "[/INST]" in response:
    response = response.split("[/INST]")[1].strip()

print(f"Response:\n{response}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print("\n✓ Model trained on explicit CHIVA classification rules")
print("✓ Decision tree logic embedded through structured examples")
print("✓ Type discrimination explicitly taught (Type 3 vs 1+2)")
print("✓ Ready for comprehensive testing")

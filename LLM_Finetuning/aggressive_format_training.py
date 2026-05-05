"""
AGGRESSIVE FORMAT COMPLIANCE TRAINING
Fine-tune the full merged model to FORCE correct output format
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

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

print("=" * 80)
print("AGGRESSIVE FORMAT COMPLIANCE TRAINING")
print("=" * 80)

# Load the FULL fine-tuned merged model (not LoRA)
model_path = '/home/krish/finetuning/venv_pytorch/Include/merged_model'

print(f"\nLoading full merged model from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

print("✓ Model loaded")

# Create ULTRA-EXPLICIT format compliance training data
# Each example forces the model to output EXACTLY the format we need
compliance_pairs = [
    {
        "instruction": "Classify shunt. EP N1→N2 present, RP N2→N1 present, no EP N2→N3, no RP at N3.",
        "response": "TYPE: Type 1\nCONFIDENCE: 0.95\nREASONING: EP N1→N2 found (SFJ incompetent). RP N2→N1 found. No EP N2→N3. No RP at N3. This is Type 1."
    },
    {
        "instruction": "Classify shunt. No EP N1→N2, EP N2→N3 present, RP N3→N2 present.",
        "response": "TYPE: Type 2A\nCONFIDENCE: 0.92\nREASONING: No EP N1→N2 (SFJ competent). EP N2→N3 present. This is Type 2A."
    },
    {
        "instruction": "Classify shunt. No EP N1→N2, EP N2→N2 present (perforator), RP N3→N1 present, no RP N2→N1.",
        "response": "TYPE: Type 2B\nCONFIDENCE: 0.90\nREASONING: No EP N1→N2. EP N2→N2 is perforator entry. RP N3 only. This is Type 2B."
    },
    {
        "instruction": "Classify shunt. No EP N1→N2, EP N2→N2 present (perforator), RP N3→N1 present, RP N2→N1 present.",
        "response": "TYPE: Type 2C\nCONFIDENCE: 0.88\nREASONING: No EP N1→N2. EP N2→N2 is perforator. RP at N3 and N2→N1. This is Type 2C."
    },
    {
        "instruction": "Classify shunt. EP N1→N2 present, EP N2→N3 present, RP N3→N1 present, no RP N2→N1.",
        "response": "TYPE: Type 3\nCONFIDENCE: 0.91\nREASONING: EP N1→N2 found. EP N2→N3 found. RP at N3 only, no RP N2→N1. This is Type 3."
    },
    {
        "instruction": "Classify shunt. EP N1→N2 present, EP N2→N3 present, RP N3→N1 present, RP N2→N1 present, eliminationTest='Reflux'.",
        "response": "TYPE: Type 1+2\nCONFIDENCE: 0.85\nREASONING: EP N1→N2 found. EP N2→N3 found. RP at both N3 and N2→N1. eliminationTest='Reflux'. This is Type 1+2."
    },
    {
        "instruction": "Classify shunt. EP N1→N2 y=0.080, RP N2→N1 y=0.300. SFJ incompetent with GSV reflux.",
        "response": "TYPE: Type 1\nCONFIDENCE: 0.94\nREASONING: SFJ-ENTRY at y=0.080. GSV-TRUNK-REFLUX N2→N1 at y=0.300. Type 1 classification."
    },
    {
        "instruction": "Classify shunt. EP N2→N3 y=0.200, RP N3→N2 y=0.470. GSV connects to tributary.",
        "response": "TYPE: Type 2A\nCONFIDENCE: 0.91\nREASONING: GSV-to-TRIBUTARY-ENTRY N2→N3. TRIBUTARY-REFLUX N3→N2. Type 2A classification."
    },
    {
        "instruction": "Classify shunt. EP N1→N2 y=0.050, EP N2→N3 y=0.132, RP N3→N1 y=0.212.",
        "response": "TYPE: Type 3\nCONFIDENCE: 0.93\nREASONING: SFJ-ENTRY at y=0.050. GSV-to-TRIBUTARY-ENTRY at y=0.132. TRIBUTARY-REFLUX only, no N2→N1. Type 3."
    },
    {
        "instruction": "Output format test. Say TYPE then CONFIDENCE then REASONING on separate lines.",
        "response": "TYPE: Type 1\nCONFIDENCE: 0.90\nREASONING: Format compliance test output."
    },
]

# Create dataset with [INST]/[/INST] format
formatted_data = []
for pair in compliance_pairs:
    text = f"[INST] {pair['instruction']} [/INST] {pair['response']}"
    formatted_data.append({"text": text})

train_dataset = Dataset.from_dict({"text": [p["text"] for p in formatted_data]})

# Tokenize
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

train_tokenized = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

print(f"✓ Created {len(train_tokenized)} format compliance training examples")

# AGGRESSIVE training args - force compliance
training_args = TrainingArguments(
    output_dir="./format_compliant_model",
    num_train_epochs=10,  # MANY epochs to burn in format
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,  # Lower LR to preserve knowledge while forcing format
    warmup_steps=1,
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
)

print("\n" + "=" * 80)
print("AGGRESSIVE FORMAT TRAINING (10 EPOCHS)")
print("=" * 80)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    data_collator=data_collator,
)

print("Starting format compliance training...\n")
train_result = trainer.train()

print(f"\n✓ Training complete!")
print(f"✓ Final loss: {train_result.training_loss:.4f}")

# Save the format-compliant model
output_dir = "./format_compliant_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\n✓ Format-compliant model saved to {output_dir}")

# Test immediately
print("\n" + "=" * 80)
print("IMMEDIATE TEST - TYPE 1 CASE")
print("=" * 80)

model.eval()
model.gradient_checkpointing_disable()

test_prompt = """[INST] Classify shunt. EP N1→N2 y=0.080 (SFJ-ENTRY=INCOMPETENT). RP N2→N1 y=0.300 (GSV-TRUNK-REFLUX). No EP N2→N3. No RP at N3. What type? [/INST]"""

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

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
print("TRAINING COMPLETE - MODEL READY FOR CLASSIFICATION")
print("=" * 80)

# Jupyter Notebook Cells for CHIVA LoRA Training (WSL2 + RTX 5090)

Copy each cell below into a Jupyter notebook cell. Run in order from Cell 1 to Cell 9.

**Setup first:**
```bash
bash setup_wsl2_gpu.sh
cd ~/path-to-this-folder
jupyter notebook
```

---

## CELL 1: Imports and Verify GPU

```python
import torch
import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import warnings
warnings.filterwarnings('ignore')

print("✓ All libraries imported successfully")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ CUDA version: {torch.version.cuda}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## CELL 2: Configuration

```python
# Paths
TRAIN_FILE = "./training_datasets/training_data.jsonl"
VAL_FILE = "./training_datasets/validation_data.jsonl"

# Model
MODEL_PATH = "./Models"  # Your local pre-trained model
LORA_OUTPUT_DIR = "./lora_chiva_finetuned"

# Training parameters
NUM_EPOCHS = 5
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
MAX_LENGTH = 512
WARMUP_STEPS = 5
EVAL_STEPS = 10
SAVE_STEPS = 20

print(f"✓ Configuration loaded")
print(f"  Model: {MODEL_PATH}")
print(f"  Output: {LORA_OUTPUT_DIR}")
print(f"  Training: {NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
```

---

## CELL 3: Load Training Data

```python
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

train_pairs = load_jsonl(TRAIN_FILE)
val_pairs = load_jsonl(VAL_FILE)

print(f"✓ Loaded {len(train_pairs)} training pairs")
print(f"✓ Loaded {len(val_pairs)} validation pairs")

# Show distribution
train_types = {}
for pair in train_pairs:
    ptype = pair.get('type', 'unknown')
    train_types[ptype] = train_types.get(ptype, 0) + 1

print("\nTraining data distribution:")
for ptype, count in sorted(train_types.items()):
    print(f"  {ptype}: {count}")
```

---

## CELL 4: Show Sample Data

```python
print("Sample 1 - Classification:")
print(f"Type: {train_pairs[0].get('type')}")
print(f"Instruction: {train_pairs[0].get('instruction')[:150]}...")
print(f"Output: {train_pairs[0].get('output')[:150]}...\n")

sample_ligation = [p for p in train_pairs if p.get('type') == 'ligation'][0] if any(p.get('type') == 'ligation' for p in train_pairs) else None
if sample_ligation:
    print("Sample 2 - Ligation:")
    print(f"Instruction: {sample_ligation.get('instruction')[:150]}...")
```

---

## CELL 5: Format Data for Training

```python
def format_instruction_response(pair):
    instruction = pair.get("instruction", "").strip()
    input_text = pair.get("input", "").strip()
    output = pair.get("output", "").strip()

    if input_text:
        prompt = f"[INST] {instruction}\n\n{input_text} [/INST]"
    else:
        prompt = f"[INST] {instruction} [/INST]"

    full_text = f"{prompt} {output}"
    return full_text

train_texts = [format_instruction_response(p) for p in train_pairs]
val_texts = [format_instruction_response(p) for p in val_pairs]

print(f"✓ Formatted {len(train_texts)} training texts")
print(f"✓ Formatted {len(val_texts)} validation texts")
print(f"\nSample formatted text:\n{train_texts[0][:300]}...")
```

---

## CELL 6: Load Model and Tokenizer

```python
print(f"Loading model from {MODEL_PATH}...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model loaded successfully")
print(f"✓ Total parameters: {total_params:,}")
print(f"✓ Model device: {next(model.parameters()).device}")
print(f"✓ Model dtype: {next(model.parameters()).dtype}")
```

---

## CELL 7: Create HuggingFace Datasets

```python
def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

print("Creating training dataset...")
train_dataset = Dataset.from_dict({"text": train_texts})
train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing training data"
)

print("Creating validation dataset...")
val_dataset = Dataset.from_dict({"text": val_texts})
val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing validation data"
)

print(f"✓ Training dataset: {len(train_dataset)} examples")
print(f"✓ Validation dataset: {len(val_dataset)} examples")
```

---

## CELL 8: Configure LoRA and Setup Training

```python
# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
pct = 100 * trainable_params / total_params

print(f"✓ LoRA configured")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Total parameters: {total_params:,}")
print(f"  Percentage trainable: {pct:.4f}%")

# Training arguments
training_args = TrainingArguments(
    output_dir=LORA_OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    logging_steps=5,
    learning_rate=LEARNING_RATE,
    bf16=True,
    optim="paged_adamw_8bit",
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_dir="./logs",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("✓ Trainer created and ready for training")
```

---

## CELL 9: Train Model (Main Cell - 20-40 minutes)

```python
print("=" * 80)
print("Starting training... (This will take 20-40 minutes)")
print("=" * 80)
print()

train_result = trainer.train()

print()
print("=" * 80)
print("Training complete!")
print("=" * 80)
print(f"Final training loss: {train_result.training_loss:.4f}")

# Save the model
print(f"\nSaving model to {LORA_OUTPUT_DIR}...")
model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)

output_path = Path(LORA_OUTPUT_DIR)
saved_files = list(output_path.glob("*"))
print(f"✓ Saved {len(saved_files)} files:")
for f in saved_files:
    if f.is_file():
        size = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size:.2f} MB)")

print(f"\n✓ Model ready at: {LORA_OUTPUT_DIR}/")
```

---

## Optional Cell 10: Test the Trained Model

```python
from peft import PeftModel

# Load base model fresh
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

test_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
test_tokenizer.pad_token = test_tokenizer.eos_token

# Load LoRA adapter
test_model = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
test_model.eval()

test_prompt = """[INST] Analyze the following ultrasound clips and classify the CHIVA venous shunt type:

Clips:
  - Clip 1: EP N1 to N2 (position=0.080)
  - Clip 2: RP N2 to N1 (position=0.300)

Based on the flow patterns, determine the CHIVA shunt type, confidence, and reasoning. [/INST]"""

inputs = test_tokenizer(test_prompt, return_tensors="pt").to(test_model.device)

with torch.no_grad():
    outputs = test_model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False,
        pad_token_id=test_tokenizer.eos_token_id,
    )

response = test_tokenizer.decode(outputs[0], skip_special_tokens=True)
if "[/INST]" in response:
    response = response.split("[/INST]")[1].strip()

print("Model Output:")
print("-" * 80)
print(response)
print("-" * 80)
```

---

## How to Use

1. **In WSL2, run setup:**
   ```bash
   bash setup_wsl2_gpu.sh
   ```

2. **Start Jupyter:**
   ```bash
   cd ~/path-to-LLM_Finetuning
   jupyter notebook
   ```

3. **Create new notebook or copy cells:**
   - Open Jupyter in browser
   - Create new notebook
   - Copy each cell above (CELL 1 through CELL 9)
   - Paste into notebook cells
   - Run with Shift+Enter

4. **Monitor Cell 9:**
   - This is the main training cell
   - Takes 20-40 minutes on RTX 5090
   - Watch for decreasing loss

**Total time: ~45-60 minutes**

---

## Expected Output

**Cell 1:**
```
✓ All libraries imported successfully
✓ CUDA available: True
✓ CUDA version: 12.4
✓ GPU: NVIDIA GeForce RTX 5090
✓ VRAM available: 32.0 GB
```

**Cell 9 (Training):**
```
==================================================
Starting training... (This will take 20-40 minutes)
==================================================

Epoch 1/5: [████████████████] 100% - loss: 1.623, val_loss: 1.542
Epoch 2/5: [████████████████] 100% - loss: 1.415, val_loss: 1.329
...
==================================================
Training complete!
==================================================
Final training loss: 1.2145
✓ Saved 6 files:
  adapter_config.json (0.02 MB)
  adapter_model.bin (3.41 MB)
  config.json (0.15 MB)
  tokenizer.json (0.98 MB)
  special_tokens_map.json (0.00 MB)
  training_args.bin (0.02 MB)

✓ Model ready at: ./lora_chiva_finetuned/
```

---

Done! Your fine-tuned model will be in `./lora_chiva_finetuned/` ready to use.

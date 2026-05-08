"""
JUPYTER NOTEBOOK CELLS FOR FINE-TUNING QWEN2.5-7B
Copy-paste each cell into your notebook sequentially
"""

# ============================================================
# CELL 1: INSTALL DEPENDENCIES
# ============================================================
"""
!pip install -q transformers datasets peft bitsandbytes torch torchvision torchaudio
!pip install -q tqdm tensorboard
"""

# ============================================================
# CELL 2: IMPORTS & CONFIG
# ============================================================
"""
import os
import torch
import json
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# Suppress warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.empty_cache()

# Paths
MODEL_ID = "Qwen/Qwen2.5-7B"
TRAIN_FILE = "latest_data/training_data.jsonl"
VAL_FILE = "latest_data/validation_data.jsonl"
OUTPUT_DIR = "./Models/Qwen2.5-7B-finetuned-clean"

# Hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 1.5e-4
MAX_SEQ_LENGTH = 768
WARMUP_STEPS = 100
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

print("Config loaded")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
"""

# ============================================================
# CELL 3: LOAD & PROCESS DATA
# ============================================================
"""
print("Loading training data...")
train_dataset = load_dataset('json', data_files=TRAIN_FILE, split='train')
print(f"  Training examples: {len(train_dataset)}")

print("Loading validation data...")
val_dataset = load_dataset('json', data_files=VAL_FILE, split='train')
print(f"  Validation examples: {len(val_dataset)}")

# Process: instruction + response → single text
def process_examples(examples):
    texts = []
    for instruction, response in zip(examples['instruction'], examples['response']):
        text = f"{instruction}\n{response}"
        texts.append(text)
    return {'text': texts}

print("Processing training data...")
train_dataset = train_dataset.map(
    process_examples,
    batched=True,
    batch_size=1000,
    remove_columns=['instruction', 'response']
)

print("Processing validation data...")
val_dataset = val_dataset.map(
    process_examples,
    batched=True,
    batch_size=1000,
    remove_columns=['instruction', 'response']
)

print("✓ Data loaded and processed")
"""

# ============================================================
# CELL 4: TOKENIZATION
# ============================================================
"""
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

def tokenize_function(examples):
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )
    result['labels'] = result['input_ids'].copy()
    return result

print("Tokenizing training data...")
train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=100,
    remove_columns=['text']
)

print("Tokenizing validation data...")
val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=100,
    remove_columns=['text']
)

print("✓ Data tokenized")
print(f"  Training: {len(train_dataset)} examples")
print(f"  Validation: {len(val_dataset)} examples")
"""

# ============================================================
# CELL 5: LOAD BASE MODEL WITH 4-BIT QUANTIZATION
# ============================================================
"""
print("Loading base model with 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True
)

print(f"✓ Model loaded: {MODEL_ID}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"  Trainable: NO (will add LoRA adapters)")
"""

# ============================================================
# CELL 6: SETUP LoRA (LOW-RANK ADAPTATION)
# ============================================================
"""
print("Configuring LoRA...")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
)

model = get_peft_model(model, lora_config)

print("✓ LoRA configured")
model.print_trainable_parameters()
"""

# ============================================================
# CELL 7: TRAINING SETUP
# ============================================================
"""
print("Setting up training arguments...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    evaluation_strategy='steps',
    save_strategy='steps',
    load_best_model_at_end=True,
    report_to=['tensorboard'],
    optim='paged_adamw_8bit',
    max_grad_norm=0.3,
    seed=42,
    bf16=True,
    gradient_checkpointing=False,
    dataloader_pin_memory=True
)

print("✓ Training arguments set")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Output dir: {OUTPUT_DIR}")
"""

# ============================================================
# CELL 8: CREATE TRAINER & START TRAINING
# ============================================================
"""
print("Creating trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("✓ Trainer created")
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70 + "\n")

train_result = trainer.train()

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"\nTraining loss: {train_result.training_loss:.4f}")
print(f"Output directory: {OUTPUT_DIR}")
"""

# ============================================================
# CELL 9: SAVE FINAL MODEL
# ============================================================
"""
print("Saving model and adapters...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✓ Model saved to {OUTPUT_DIR}")

# Show what was saved
import os
files = os.listdir(OUTPUT_DIR)
print(f"\nSaved files:")
for f in sorted(files)[:10]:
    print(f"  - {f}")
"""

# ============================================================
# CELL 10: LOAD FINETUNED MODEL FOR INFERENCE
# ============================================================
"""
from peft import AutoPeftModelForCausalLM

print("Loading fine-tuned model for inference...")

model = AutoPeftModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
print("✓ Model and tokenizer loaded")

# Test inference
def generate(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
test_prompt = "Explain the hemodynamic basis of Type 1 CHIVA shunt:"
response = generate(test_prompt)
print(f"\nTest prompt: {test_prompt}")
print(f"Response: {response}")
"""

# ============================================================
# SUMMARY OF FINE-TUNING METHODS
# ============================================================
"""
FINE-TUNING METHODS USED IN THIS IMPLEMENTATION:

1. **4-BIT QUANTIZATION (NF4)**
   - Reduces model from 14GB to ~4GB
   - BitsAndBytesConfig with bnb_4bit_use_double_quant=True
   - Uses bfloat16 for compute dtype
   - Trade-off: Speed for memory efficiency

2. **LOW-RANK ADAPTATION (LoRA)**
   - Parameter-efficient fine-tuning
   - Only trains ~1-2% of parameters (adapters)
   - Applied to attention layers: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
   - Rank=32, Alpha=64, Dropout=0.05
   - Benefit: Train quickly without full model update

3. **MIXED PRECISION TRAINING (bf16)**
   - Uses bfloat16 for faster computation
   - Maintains numerical stability
   - ~2x faster than float32 on modern GPUs

4. **GRADIENT ACCUMULATION**
   - Steps: 1 (minimal, since batch size is already 8)
   - Simulates larger batch without memory overhead
   - More stable gradients

5. **LEARNING RATE SCHEDULING**
   - Base LR: 1.5e-4
   - Warmup steps: 100 (gradual increase from 0 to target)
   - Prevents large initial gradient steps

6. **WEIGHT DECAY (L2 Regularization)**
   - value: 0.01
   - Penalizes large weights to prevent overfitting
   - Only applied to non-bias terms

7. **GRADIENT CLIPPING**
   - max_grad_norm: 0.3
   - Prevents exploding gradients
   - Especially important with 4-bit quantization

8. **EARLY STOPPING / BEST MODEL SELECTION**
   - load_best_model_at_end=True
   - Evaluates every 100 steps
   - Saves best checkpoint based on validation loss
   - Prevents overfitting to training data

9. **PAGED AdamW OPTIMIZER (8-bit)**
   - optim: 'paged_adamw_8bit'
   - Memory-efficient variant of AdamW
   - Manages optimizer states in CPU RAM
   - Allows larger batch sizes

10. **DATA COLLATOR (Causal LM)**
    - DataCollatorForLanguageModeling with mlm=False
    - Handles padding and attention masks
    - Labels = input_ids (next token prediction)

DOMAIN LEARNING STRATEGY:
   - 3 epochs: Deep absorption of domain knowledge without overfitting
   - Mixed instruction-response pairs: CHIVA classifications + general venous knowledge
   - No system prompts: Model learns directly from examples (flexible)
   - 768 max sequence: Captures full clinical narratives
   - Large dataset (6016 examples): 90% train, 10% validation

COMBINATION BENEFIT:
   This approach combines memory efficiency (quantization) with parameter efficiency (LoRA)
   while maintaining training stability (gradient clipping, warmup, weight decay).
   Result: Fine-tune 7B model on RTX 5090 32GB VRAM in ~3-4 hours for 3 epochs.
"""

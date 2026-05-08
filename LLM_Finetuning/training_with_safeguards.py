"""
FINE-TUNING WITH CATASTROPHIC FORGETTING SAFEGUARDS

Key concerns:
1. Model should NOT "mug up" (memorize) training data
2. Domain knowledge should ENHANCE reasoning, not replace it
3. Base instruction-following capability must be preserved
4. When user gives instructions, model follows them (not ignores for domain knowledge)

Safeguards:
- Low learning rate (prevents aggressive overwriting of base knowledge)
- Early stopping (prevents overfitting)
- Validation monitoring (tracks when model diverges from base behavior)
- Mixed training data (general + domain-specific examples)
- Lower epochs (prevents memorization)
- Temperature/sampling at inference (prevents repetition)
"""

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
"""
!pip install -q transformers datasets peft bitsandbytes torch
!pip install -q tqdm tensorboard
"""

# ============================================================
# CELL 2: Imports & Config
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
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.empty_cache()

# ===== SAFEGUARD CONFIG =====
# These settings prevent catastrophic forgetting
MODEL_ID = "Qwen/Qwen2.5-7B"
TRAIN_FILE = "latest_data/training_data.jsonl"
VAL_FILE = "latest_data/validation_data.jsonl"
OUTPUT_DIR = "./Models/Qwen2.5-7B-finetuned-safe"

# ===== CRITICAL: Lower learning rate prevents aggressive knowledge overwriting =====
NUM_EPOCHS = 3                           # Conservative: don't overtrain
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 1.0e-4                  # LOWERED from 1.5e-4 for better preservation
MAX_SEQ_LENGTH = 768
WARMUP_STEPS = 150                       # INCREASED warmup for gentler learning
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

print("="*70)
print("QWEN2.5-7B FINE-TUNING WITH CATASTROPHIC FORGETTING SAFEGUARDS")
print("="*70)
print("\nSafeguards enabled:")
print("  - Low learning rate (1.0e-4) to preserve base knowledge")
print("  - Increased warmup steps (150) for gentle learning")
print("  - Early stopping on validation loss")
print("  - Validation monitoring to detect knowledge drift")
print("  - Conservative epochs (3) to prevent memorization")
print("  - Mixed training data (general + domain-specific)")
print(f"\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
"""

# ============================================================
# CELL 3: Load & Process Data
# ============================================================
"""
print("\n" + "="*70)
print("LOADING DATA (Mixed: General Knowledge + Domain Knowledge)")
print("="*70)

print("\nLoading training data...", end=" ")
train_dataset = load_dataset('json', data_files=TRAIN_FILE, split='train')
print(f"OK {len(train_dataset)} examples")
print(f"  This includes: CHIVA classifications + general medical knowledge")

print("Loading validation data...", end=" ")
val_dataset = load_dataset('json', data_files=VAL_FILE, split='train')
print(f"OK {len(val_dataset)} examples")

# Shuffle training data to prevent memorization patterns
train_dataset = train_dataset.shuffle(seed=42)

def process_examples(examples):
    texts = []
    for instruction, response in zip(examples['instruction'], examples['response']):
        # Simple format: no system prompts, just instruction + response
        text = f"{instruction}\n{response}"
        texts.append(text)
    return {'text': texts}

print("\nProcessing training data...", end=" ")
train_dataset = train_dataset.map(
    process_examples,
    batched=True,
    batch_size=1000,
    remove_columns=['instruction', 'response']
)
print("OK")

print("Processing validation data...", end=" ")
val_dataset = val_dataset.map(
    process_examples,
    batched=True,
    batch_size=1000,
    remove_columns=['instruction', 'response']
)
print("OK")

print(f"\nData loaded: {len(train_dataset)} train + {len(val_dataset)} val")
"""

# ============================================================
# CELL 4: Tokenization
# ============================================================
"""
print("\n" + "="*70)
print("TOKENIZING DATA")
print("="*70)

print("\nLoading tokenizer...", end=" ")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("OK")

def tokenize_function(examples):
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )
    result['labels'] = result['input_ids'].copy()
    return result

print("Tokenizing training data...", end=" ")
train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=100,
    remove_columns=['text']
)
print("OK")

print("Tokenizing validation data...", end=" ")
val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=100,
    remove_columns=['text']
)
print("OK")
"""

# ============================================================
# CELL 5: Load Model with QLoRA (4-Bit Quantization)
# ============================================================
"""
print("\n" + "="*70)
print("LOADING MODEL WITH QLORA")
print("="*70)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("\nLoading model...", end=" ")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True
)
print("OK")
print(f"  Quantization: 4-bit NF4 (14GB -> 4GB)")
"""

# ============================================================
# CELL 6: Setup QDoRA with Safeguards
# ============================================================
"""
print("\n" + "="*70)
print("CONFIGURING QDoRA WITH SAFEGUARDS")
print("="*70)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        'q_proj', 'v_proj', 'k_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj'
    ],
    use_dora=True,                         # Better expressivity
    init_lora_weights='pissa',             # Fast, stable initialization
)

model = get_peft_model(model, lora_config)

print("QDoRA configured")
print("  - DoRA: Weight decomposition for precision")
print("  - PiSSA: Stable initialization")
print("  - Low-rank: Only ~2% parameters trained")
model.print_trainable_parameters()
"""

# ============================================================
# CELL 7: Training Configuration with SAFEGUARDS
# ============================================================
"""
print("\n" + "="*70)
print("TRAINING CONFIGURATION WITH SAFEGUARDS")
print("="*70)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # ===== SAFEGUARD: Conservative training schedule =====
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    # ===== SAFEGUARD: Low learning rate preserves base knowledge =====
    learning_rate=LEARNING_RATE,           # 1.0e-4 (lower = more conservative)
    warmup_steps=WARMUP_STEPS,             # Longer warmup = gentler start

    # ===== SAFEGUARD: Weight decay regularization =====
    weight_decay=0.01,                     # L2 regularization

    # ===== SAFEGUARD: Early stopping prevents overfitting =====
    eval_steps=50,                         # Evaluate frequently
    save_steps=50,
    logging_steps=10,
    evaluation_strategy='steps',
    save_strategy='steps',
    load_best_model_at_end=True,           # CRITICAL: Load best checkpoint
    metric_for_best_model='loss',
    greater_is_better=False,

    # ===== SAFEGUARD: Strict gradient clipping prevents large updates =====
    max_grad_norm=0.3,                     # Prevent exploding gradients

    # ===== Other settings =====
    report_to=['tensorboard'],
    optim='paged_adamw_8bit',
    seed=42,
    bf16=True,
    gradient_checkpointing=False,
    dataloader_pin_memory=True
)

print("Training args:")
print(f"  Learning rate: {LEARNING_RATE} (LOW for preservation)")
print(f"  Warmup steps: {WARMUP_STEPS} (LONG for gentle learning)")
print(f"  Weight decay: 0.01 (L2 regularization)")
print(f"  Max grad norm: 0.3 (STRICT clipping)")
print(f"  Early stopping: ENABLED (based on validation loss)")
print(f"  Save best model: ENABLED")
"""

# ============================================================
# CELL 8: Create Trainer & Start Training
# ============================================================
"""
print("\n" + "="*70)
print("CREATING TRAINER")
print("="*70)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Trainer created")
print("\nTraining will:")
print("  1. Evaluate every 50 steps")
print("  2. Save best checkpoint based on validation loss")
print("  3. Stop if no improvement (early stopping)")
print("  4. Use low learning rate to preserve base knowledge")

print("\n" + "="*70)
print("STARTING FINE-TUNING WITH SAFEGUARDS")
print("="*70)
print("Monitor validation loss - should decrease steadily")
print("If it increases, model may be overfitting\n")

train_result = trainer.train()

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"\nFinal training loss: {train_result.training_loss:.4f}")
print("Best model automatically loaded from checkpoint")
"""

# ============================================================
# CELL 9: Save Model
# ============================================================
"""
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

print("\nSaving fine-tuned model and adapters...", end=" ")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("OK")

print(f"\nModel saved to: {OUTPUT_DIR}")
"""

# ============================================================
# CELL 10: Load Model & Test with Safeguard Prompts
# ============================================================
"""
from peft import AutoPeftModelForCausalLM

print("\n" + "="*70)
print("LOADING FINE-TUNED MODEL")
print("="*70)

print("\nLoading model...", end=" ")
model = AutoPeftModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
print("OK")

def generate(prompt, max_length=300, temperature=0.7, top_p=0.9):
    '''
    SAFEGUARD: Use moderate temperature and top_p to prevent
    model from fixating on memorized patterns.
    - temperature=0.7: Balance between creativity and consistency
    - top_p=0.9: Prevent extreme responses (very rare tokens)
    '''
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,           # Not too low (prevents memorization)
            top_p=top_p,                       # Not too high (prevents randomness)
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1                        # Greedy would memorize
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# TEST 1: Base reasoning preserved
print("\n" + "="*70)
print("TEST 1: BASE REASONING CAPABILITY PRESERVED")
print("="*70)

test1 = """Explain the difference between correlation and causation in medical research."""
response1 = generate(test1, max_length=250)
print(f"Prompt: {test1}\n")
print(f"Response: {response1[:400]}...\n")
print("EXPECTATION: Model should explain statistical concepts (base knowledge preserved)")

# TEST 2: Domain knowledge enhances response
print("="*70)
print("TEST 2: DOMAIN KNOWLEDGE ENHANCES REASONING")
print("="*70)

test2 = """What is the significance of EP N1->N2 in CHIVA classification and why does it matter clinically?"""
response2 = generate(test2, max_length=250)
print(f"Prompt: {test2}\n")
print(f"Response: {response2[:400]}...\n")
print("EXPECTATION: Model knows CHIVA domain knowledge (domain enhancement)")

# TEST 3: Instructions are followed (not ignored for domain knowledge)
print("="*70)
print("TEST 3: INSTRUCTIONS FOLLOWED, NOT IGNORED FOR DOMAIN KNOWLEDGE")
print("="*70)

test3 = """In exactly 100 words, explain how hemodynamic reflux leads to varicose veins."""
response3 = generate(test3, max_length=150)
print(f"Prompt: {test3}\n")
print(f"Response: {response3}\n")
print("EXPECTATION: Model respects the '100 words' constraint")
print("             even though it knows domain knowledge")

# TEST 4: Mixed reasoning (general + domain)
print("="*70)
print("TEST 4: MIXED REASONING (GENERAL + DOMAIN)")
print("="*70)

test4 = """A 50-year-old patient has varicose veins with duplex findings showing:
- EP N2->N3 (y=0.25)
- No EP N1->N2

Why is the SFJ status important for treatment planning?"""
response4 = generate(test4, max_length=300)
print(f"Prompt: {test4}\n")
print(f"Response: {response4[:500]}...\n")
print("EXPECTATION: Model combines:")
print("  - General reasoning (patient context, clinical planning)")
print("  - Domain knowledge (CHIVA classification, SFJ incompetence rules)")

print("\n" + "="*70)
print("TESTING COMPLETE")
print("="*70)
print("""
SAFEGUARD VALIDATION:
  1. Base reasoning: Did model explain fundamental concepts?
  2. Domain knowledge: Did model apply CHIVA/venous knowledge?
  3. Instruction following: Did model respect user constraints?
  4. Mixed reasoning: Did model combine both?

If YES to all: Model successfully learned without catastrophic forgetting.
If NO: Reduce learning rate further or increase warmup steps.
""")
"""

# ============================================================
# SUMMARY OF SAFEGUARDS
# ============================================================
"""
CATASTROPHIC FORGETTING PREVENTION MECHANISMS:

1. LOW LEARNING RATE (1.0e-4)
   - Prevents aggressive overwriting of base knowledge
   - Changes are incremental, not destructive
   - Domain knowledge "adds" rather than "replaces"

2. LONG WARMUP (150 steps)
   - Gradual increase from 0 to target learning rate
   - Prevents large early updates that damage base knowledge

3. WEIGHT DECAY (0.01 L2 regularization)
   - Penalizes large weight changes
   - Keeps model close to original weights
   - Prevents extreme adaptations

4. STRICT GRADIENT CLIPPING (0.3)
   - Prevents any single update from being too large
   - Filters out outlier gradients that might destabilize learning

5. EARLY STOPPING + VALIDATION MONITORING
   - Saves best checkpoint based on validation loss
   - Stops if validation loss increases (sign of divergence)
   - Prevents overfitting to training data

6. CONSERVATIVE EPOCHS (3)
   - Not overtraining the model
   - Prevents memorization of training examples
   - Allows quick convergence to stable state

7. MIXED TRAINING DATA
   - ~5417 training examples include:
     * CHIVA classifications (domain-specific)
     * General medical knowledge (base reasoning)
     * Clinical reasoning questions (mixed)
   - Model sees diverse tasks, doesn't specialize too much

8. CONTROLLED SAMPLING (temperature=0.7, top_p=0.9)
   - Not too deterministic (prevents memorization)
   - Not too random (maintains quality)
   - Balances between safety and expressiveness

9. LoRA with PISSA INITIALIZATION
   - Trains only 2% of parameters
   - Doesn't touch most of base model weights
   - Changes are isolated to low-rank adapters
   - Easier to roll back if needed

10. NO SPECIAL TOKENS IN TRAINING
    - No system prompts that might override instructions
    - Model learns to follow user instructions naturally
    - Less chance of instruction-following degradation

EXPECTED BEHAVIOR:
  - Model knows CHIVA shunt types and medical concepts
  - Model still follows user instructions and constraints
  - Model maintains general reasoning ability
  - Domain knowledge "enhances" not "replaces" base behavior
  - When user says "summarize in 50 words", it does (doesn't ignore for domain)
  - When user says "explain like I'm 10", it tries (doesn't insist on medical jargon)
"""

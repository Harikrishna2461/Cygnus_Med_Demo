"""
Script 3: Classification-Focused Fine-Tuning Refinement
Retrain the already fine-tuned Mistral-7B model on shunt classification task
"""

import torch
import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
import numpy as np

# ============================================================================
# PART 1: CLASSIFICATION DATASET GENERATION
# ============================================================================

CLASSIFICATION_EXAMPLES = [
    # Type 1: SFJ reflux, GSV reflux to mid-thigh, no tributaries, no DV involvement
    {
        "case": "Type 1",
        "description": "There is reflux at the saphenofemoral junction. Blood flows backward down the GSV to the mid-thigh where it drains back to the femoral vein. No tributary involvement. No deep vein involvement.",
        "shunt_type": "Type 1",
        "ligation_site": "Saphenofemoral junction (SFJ) only",
    },
    # Type 2A: SFJ competent, GSV-tributary connection (no DV)
    {
        "case": "Type 2A",
        "description": "The saphenofemoral junction is competent. There is an entry point from the GSV into a tributary at the mid-thigh without any deep vein involvement. The tributary shows reflux.",
        "shunt_type": "Type 2A",
        "ligation_site": "Ligate the GSV at the entry point to the tributary",
    },
    # Type 2B: SFJ competent, DV-tributary connection (no GSV)
    {
        "case": "Type 2B",
        "description": "The saphenofemoral junction is competent. A deep vein (femoral vein) shows a direct connection to a tributary at the mid-thigh with reflux. The GSV is not involved.",
        "shunt_type": "Type 2B",
        "ligation_site": "Ligate the perforator or tributary at the deep vein connection",
    },
    # Type 2C: SFJ competent, GSV-DV-tributary (all three involved)
    {
        "case": "Type 2C",
        "description": "The saphenofemoral junction is competent. A complex pathway exists where the GSV connects to a deep vein (femoral vein), which then connects to a tributary. All three systems show reflux.",
        "shunt_type": "Type 2C",
        "ligation_site": "Ligate both the GSV-to-DV connection and the DV-to-tributary connection",
    },
    # Type 1+2: Combined SFJ and tributary involvement
    {
        "case": "Type 1+2 Combined",
        "description": "Reflux is present at the saphenofemoral junction with GSV reflux. Additionally, there is a separate tributary with an independent entry point showing reflux. Both pathways contribute to venous insufficiency.",
        "shunt_type": "Type 1+2",
        "ligation_site": "Ligate both the SFJ and the independent tributary entry point",
    },
    # Type 3: Perforator-based shunt
    {
        "case": "Type 3",
        "description": "There is a perforator at the Hunterian level (mid-thigh) feeding the GSV with reflux into the GSV and its tributaries. No SFJ involvement. The perforator shows outward flow from deep to superficial system.",
        "shunt_type": "Type 3",
        "ligation_site": "Ligate the perforator at the Hunterian level (above and below the GSV)",
    },
    # Type 4: Popliteal-based involvement
    {
        "case": "Type 4",
        "description": "A popliteal perforator shows reflux feeding the small saphenous vein (SSV) with subsequent reflux into tributaries. The popliteal vein shows bidirectional or reversed flow.",
        "shunt_type": "Type 4",
        "ligation_site": "Ligate the popliteal perforator and the SSV-tributary connection",
    },
    # Type 5: Recurrent/recanalized pathways
    {
        "case": "Type 5",
        "description": "Previous SFJ ligation is evident. New recanalized pathways have formed at the old ligation site or neovascularization has occurred. Reflux persists through collateral channels.",
        "shunt_type": "Type 5 (Recurrent)",
        "ligation_site": "Ligate recanalized pathways and neovascular channels; consider distal GSV ligation",
    },
    # Type 6: Pelvic/abdominal venous involvement
    {
        "case": "Type 6",
        "description": "Reflux originates from pelvic veins (internal iliac, ovarian/testicular veins) flowing into the superficial system via tributaries. Symptoms often bilateral and severe.",
        "shunt_type": "Type 6 (Pelvic)",
        "ligation_site": "Ligate tributaries receiving pelvic venous reflux; may require duplex-guided sclerotherapy",
    },
    # Additional variants
    {
        "case": "Type 2A Variant",
        "description": "SFJ is patent and competent on ultrasound. A tributary joins the GSV in the thigh region. Duplex shows reflux in the tributary but forward flow in the GSV from distal.",
        "shunt_type": "Type 2A",
        "ligation_site": "Ligate GSV at the tributary entry point; preserve proximal GSV if competent",
    },
    {
        "case": "Type 3 with SSV",
        "description": "A perforator in the calf feeds the small saphenous vein with reflux propagating into calf tributaries. The SSV itself shows reflux. Popliteal vein is normal.",
        "shunt_type": "Type 3",
        "ligation_site": "Ligate the perforator above and below; ligate SSV tributaries",
    },
]

# Create instruction-response pairs for classification
def create_classification_dataset():
    """Generate classification-format training examples"""
    training_examples = []

    for example in CLASSIFICATION_EXAMPLES:
        # Multiple instruction variants for the same case
        instructions = [
            f"Classify the venous shunt type based on this clinical case and recommend ligation strategy: {example['description']}",
            f"What is the venous shunt classification for this patient: {example['description']} Provide the type and ligation site.",
            f"Given the hemodynamic findings: {example['description']} Determine the shunt type and surgical approach.",
            f"Analyze this clinical presentation: {example['description']} Classify the shunt and recommend management.",
            f"Medical case: {example['description']} Identify the shunt type and ligation location.",
        ]

        response = f"**Shunt Classification:** {example['shunt_type']}\n\n**Recommended Ligation Site:** {example['ligation_site']}"

        for instruction in instructions:
            training_examples.append({
                "instruction": instruction,
                "response": response,
                "text": f"[INST] {instruction} [/INST] {response}"
            })

    return training_examples

# Generate dataset
print("=" * 80)
print("GENERATING CLASSIFICATION DATASET")
print("=" * 80)
classification_data = create_classification_dataset()
print(f"Generated {len(classification_data)} classification training examples")
print(f"\nExample training pair:")
print(f"Instruction: {classification_data[0]['instruction'][:100]}...")
print(f"Response: {classification_data[0]['response'][:100]}...")

# ============================================================================
# PART 2: PREPARE DATASET FOR TRAINING
# ============================================================================

# Split into train and eval (80/20)
np.random.seed(42)
indices = np.random.permutation(len(classification_data))
train_size = int(0.8 * len(classification_data))
train_indices = indices[:train_size]
eval_indices = indices[train_size:]

train_data = [classification_data[i] for i in train_indices]
eval_data = [classification_data[i] for i in eval_indices]

# Create Dataset objects
train_dataset = Dataset.from_dict({
    "instruction": [ex["instruction"] for ex in train_data],
    "response": [ex["response"] for ex in train_data],
    "text": [ex["text"] for ex in train_data],
})

eval_dataset = Dataset.from_dict({
    "instruction": [ex["instruction"] for ex in eval_data],
    "response": [ex["response"] for ex in eval_data],
    "text": [ex["text"] for ex in eval_data],
})

print(f"\nDataset split:")
print(f"  Training: {len(train_dataset)} examples")
print(f"  Evaluation: {len(eval_dataset)} examples")

# ============================================================================
# PART 3: LOAD PRE-TRAINED MODEL WITH EXISTING LORA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING PRE-TRAINED MODEL")
print("=" * 80)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
device_map = "auto"

# 4-bit quantization config (same as before)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load existing LoRA weights if available
lora_adapter_path = Path("./mistral_lora_medical_finetuned")
if lora_adapter_path.exists():
    print(f"Loading existing LoRA adapter from {lora_adapter_path}")
    model = PeftModel.from_pretrained(model, str(lora_adapter_path))
    print("Existing LoRA weights loaded successfully")
else:
    print("No existing LoRA adapter found. Creating new LoRA configuration...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        init_lora_weights="gaussian",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# ============================================================================
# PART 4: SETUP TRAINING CONFIGURATION
# ============================================================================

print("\n" + "=" * 80)
print("SETTING UP TRAINING")
print("=" * 80)

output_dir = "./mistral_classification_refined"
Path(output_dir).mkdir(exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,  # More epochs for focused classification training
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_grad_norm=1.0,
    seed=42,
    bf16=True,
    optim="paged_adamw_32bit",
    dataloader_pin_memory=True,
)

print(f"Training output directory: {output_dir}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Learning rate: {training_args.learning_rate}")
print(f"Epochs: {training_args.num_train_epochs}")

# ============================================================================
# PART 5: TRAIN WITH SFTTrainer
# ============================================================================

print("\n" + "=" * 80)
print("STARTING CLASSIFICATION-FOCUSED TRAINING")
print("=" * 80)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    packing=False,
    dataset_text_field="text",
    max_seq_length=512,
)

print("Training started...")
train_result = trainer.train()

print("\n" + "=" * 80)
print("TRAINING COMPLETED")
print("=" * 80)
print(f"Final training loss: {train_result.training_loss:.4f}")

# Save model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# ============================================================================
# PART 6: EVALUATE ON TEST CASES
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATING ON CLASSIFICATION TEST CASES")
print("=" * 80)

model.eval()

test_cases = [
    "A patient has reflux at the saphenofemoral junction with GSV reflux to mid-thigh tributaries, no deep vein involvement. What type of shunt is this and where should I ligate?",
    "The SFJ is competent. There is an entry point from the GSV into a tributary at the mid-thigh without any deep vein involvement. What is the diagnosis and treatment?",
    "A perforator at the Hunterian level feeds the GSV with reflux into tributaries. No SFJ involvement. Classify the shunt and recommend ligation.",
    "Previous SFJ ligation is evident. New recanalized pathways have formed. What type of shunt is this?",
    "Reflux originates from pelvic veins flowing into the superficial system via tributaries. Classify and recommend management.",
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n--- Test Case {i} ---")
    print(f"Query: {test_case}")

    # Prepare input
    prompt = f"[INST] {test_case} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part (after [/INST])
    if "[/INST]" in response:
        response = response.split("[/INST]")[1].strip()

    print(f"Response:\n{response}")

# ============================================================================
# PART 7: SAVE FINAL MODEL
# ============================================================================

print("\n" + "=" * 80)
print("SAVING FINAL REFINED MODEL")
print("=" * 80)

# Save the final model
final_model_path = "./mistral_classification_final"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Final model saved to {final_model_path}")

# Save training metadata
metadata = {
    "task": "venous_shunt_classification",
    "base_model": model_name,
    "training_examples": len(train_data),
    "eval_examples": len(eval_data),
    "epochs": training_args.num_train_epochs,
    "learning_rate": training_args.learning_rate,
    "batch_size": training_args.per_device_train_batch_size,
    "max_seq_length": 512,
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "final_training_loss": float(train_result.training_loss),
}

with open(f"{final_model_path}/training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nTraining metadata saved")
print(f"\nFinal Summary:")
print(f"  Training loss: {train_result.training_loss:.4f}")
print(f"  Model location: {final_model_path}")
print(f"  Task: Venous shunt classification")
print(f"  Classes: Type 1, 2A, 2B, 2C, 1+2, 3, 4, 5, 6")

print("\n" + "=" * 80)
print("CLASSIFICATION REFINEMENT COMPLETE")
print("=" * 80)

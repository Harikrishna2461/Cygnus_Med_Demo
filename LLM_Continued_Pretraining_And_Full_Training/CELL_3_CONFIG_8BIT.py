# ============ TRAINING CONFIGURATION (8-BIT QUANTIZED) ============
CONFIG = {
    # Model & Data
    "model_name": "Qwen/Qwen2.5-7B",
    "train_file": "augmented_output/train.jsonl",
    "eval_file": "augmented_output/eval.jsonl",

    # Training Parameters (tuned for 8-bit quantization on 32GB GPU)
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,   # Very small per-device batch
    "per_device_eval_batch_size": 2,    # Eval batch can be slightly larger
    "gradient_accumulation_steps": 16,  # Accumulate 16 steps → effective batch = 1*16=16
    "learning_rate": 2e-5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,

    # Sequence & Optimization
    "max_seq_length": 1024,  # Reduced from 2048 to save memory
    "bf16": True,  # Mixed precision
    "fp16": False,

    # Checkpointing & Saving
    "output_dir": "medical_qwen_cpt_8bit",
    "save_strategy": "steps",
    "save_steps": 100,  # Save more frequently (small GPU)
    "save_total_limit": 2,  # Keep only 2 most recent

    # Evaluation
    "eval_strategy": "steps",
    "eval_steps": 50,  # Eval more frequently

    # Logging
    "logging_dir": "logs_8bit",
    "logging_steps": 5,  # Print progress more often

    # Data Loading (CRITICAL)
    "dataloader_num_workers": 0,      # NO multiprocessing
    "dataloader_pin_memory": False,   # NO memory pinning

    # Other
    "seed": 42,
}

print("\n" + "="*80)
print("TRAINING CONFIGURATION (8-BIT QUANTIZED)")
print("="*80)
print(f"⚠️  Memory Optimizations Active:")
print(f"  • 8-bit quantization (15GB → 4GB)")
print(f"  • Gradient checkpointing enabled")
print(f"  • Reduced sequence length (1024)")
print(f"  • Small batch size with accumulation")
print("="*80)

for key, value in sorted(CONFIG.items()):
    print(f"  {key:.<50} {value}")

print("="*80 + "\n")

# Calculate effective batch size
effective_batch_size = CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"]
print(f"✓ Effective batch size: {effective_batch_size} (accumulated across {CONFIG['gradient_accumulation_steps']} steps)")
print(f"✓ Expected GPU memory: ~22-28GB (should fit on 31.84GB)")
print(f"✓ Training will take longer but should NOT run out of memory\n")

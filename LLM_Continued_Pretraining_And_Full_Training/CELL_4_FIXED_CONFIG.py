# ============ TRAINING CONFIG ============
config = {
    # Model & Data
    "model_name": "Qwen/Qwen2.5-7B",
    "train_file": "augmented_output/train.jsonl",
    "eval_file": "augmented_output/eval.jsonl",

    # Training Parameters
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,  # Reduced from 4 to save memory
    "per_device_eval_batch_size": 4,   # Reduced from 8
    "gradient_accumulation_steps": 8,  # Increased from 4 (effective batch = 2*8=16, same as before)
    "learning_rate": 2e-5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,

    # Sequence & Optimization
    "max_seq_length": 2048,  # Context window
    "bf16": True,  # Mixed precision (bfloat16 for newer GPUs)
    "fp16": False,  # Use bf16 instead on modern GPUs

    # Checkpointing & Saving
    "output_dir": "medical_qwen_cpt",
    "save_strategy": "steps",
    "save_steps": 200,
    "save_total_limit": 5,  # Keep only 5 most recent checkpoints

    # Evaluation
    "eval_strategy": "steps",
    "eval_steps": 100,
    "metric_for_best_model": "eval_loss",

    # Logging
    "logging_dir": "logs",
    "logging_steps": 10,
    "log_level": "info",

    # Other
    "seed": 42,
    "dataloader_num_workers": 0,      # FIXED: Changed from 4 (no multiprocessing to avoid deadlock)
    "dataloader_pin_memory": False,   # FIXED: Changed from True (avoid memory pinning issues)
}

print("📋 TRAINING CONFIGURATION")
print("=" * 50)
for key, value in config.items():
    print(f"{key:.<40} {value}")
print("=" * 50)

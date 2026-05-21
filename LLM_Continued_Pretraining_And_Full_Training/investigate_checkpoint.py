"""
Diagnostic script to investigate checkpoint-495 and verify training success.
Checks file structure, config, and actual model weights.
"""

import os
import json
from pathlib import Path
import subprocess

CHECKPOINT_PATH = Path("medical_qwen_cpt/checkpoint-495")
LOGS_PATH = Path("logs")

print("="*80)
print("CHECKPOINT-495 INVESTIGATION REPORT")
print("="*80)

# 1. Check if checkpoint exists
print(f"\n[1] CHECKPOINT EXISTENCE CHECK")
print(f"    Path: {CHECKPOINT_PATH}")
print(f"    Exists: {CHECKPOINT_PATH.exists()}")

if CHECKPOINT_PATH.exists():
    print(f"    Is directory: {CHECKPOINT_PATH.is_dir()}")

    # 2. List all files
    print(f"\n[2] CHECKPOINT FILES")
    all_files = list(CHECKPOINT_PATH.rglob("*"))
    dirs = [f for f in all_files if f.is_dir()]
    files = [f for f in all_files if f.is_file()]

    print(f"    Total items: {len(all_files)} ({len(dirs)} dirs, {len(files)} files)")
    print(f"\n    Directory structure:")
    for f in sorted(all_files)[:30]:  # Show first 30 items
        rel_path = f.relative_to(CHECKPOINT_PATH)
        if f.is_dir():
            print(f"      📁 {rel_path}/")
        else:
            size_mb = f.stat().st_size / (1024*1024)
            print(f"      📄 {rel_path} ({size_mb:.2f} MB)")

    if len(all_files) > 30:
        print(f"      ... and {len(all_files) - 30} more items")

    # 3. Check for critical files
    print(f"\n[3] CRITICAL FILES CHECK")
    critical_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "generation_config.json",
        "trainer_state.json"
    ]

    for fname in critical_files:
        fpath = CHECKPOINT_PATH / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024*1024)
            print(f"    ✓ {fname} ({size_mb:.2f} MB)")
        else:
            print(f"    ✗ {fname} - MISSING")

    # 4. Check config.json for actual settings
    print(f"\n[4] CONFIG.JSON INSPECTION")
    config_path = CHECKPOINT_PATH / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"    Model type: {config.get('model_type', 'unknown')}")
        print(f"    Hidden size: {config.get('hidden_size', 'unknown')}")
        print(f"    Num layers: {config.get('num_hidden_layers', 'unknown')}")
        print(f"    Vocab size: {config.get('vocab_size', 'unknown')}")

    # 5. Check trainer_state.json for training info
    print(f"\n[5] TRAINER STATE CHECK")
    trainer_state_path = CHECKPOINT_PATH / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path) as f:
            trainer_state = json.load(f)
        print(f"    ✓ trainer_state.json exists")
        print(f"    Global step: {trainer_state.get('global_step', 'unknown')}")
        print(f"    Current epoch: {trainer_state.get('epoch', 'unknown')}")

        # Check best model metric
        best_metric = trainer_state.get('best_metric', 'unknown')
        print(f"    Best metric: {best_metric}")

        # Show training history
        if 'log_history' in trainer_state:
            log_history = trainer_state['log_history']
            print(f"    Training entries: {len(log_history)}")

            if log_history:
                print(f"\n    First entry (epoch 0):")
                first = log_history[0]
                for key in ['learning_rate', 'loss', 'epoch', 'step']:
                    if key in first:
                        print(f"      {key}: {first[key]}")

                print(f"\n    Last entry:")
                last = log_history[-1]
                for key in ['learning_rate', 'loss', 'epoch', 'step']:
                    if key in last:
                        print(f"      {key}: {last[key]}")
    else:
        print(f"    ✗ trainer_state.json - MISSING (training may not have happened)")

    # 6. Check training metrics file
    print(f"\n[6] TRAINING METRICS CHECK")
    metrics_file = LOGS_PATH / "training_metrics.csv"
    if metrics_file.exists():
        with open(metrics_file) as f:
            lines = f.readlines()
        print(f"    ✓ training_metrics.csv exists ({len(lines)} lines)")
        if len(lines) > 1:
            print(f"    Header: {lines[0].strip()}")
            print(f"    First entry: {lines[1].strip()}")
            if len(lines) > 2:
                print(f"    Last entry: {lines[-1].strip()}")
    else:
        print(f"    ✗ training_metrics.csv - NOT FOUND")

    # 7. File size comparison
    print(f"\n[7] FILE SIZE ANALYSIS")
    total_size = sum(f.stat().st_size for f in files)
    total_size_mb = total_size / (1024*1024)
    print(f"    Total checkpoint size: {total_size_mb:.2f} MB")
    print(f"    Expected trained model: ~14,000+ MB (7B params in BF16)")
    print(f"    Status: {'✗ TOO SMALL - likely not trained' if total_size_mb < 5000 else '✓ Reasonable size'}")

    # 8. Directory timestamps
    print(f"\n[8] CHECKPOINT TIMESTAMPS")
    import datetime
    created_time = datetime.datetime.fromtimestamp(CHECKPOINT_PATH.stat().st_ctime)
    modified_time = datetime.datetime.fromtimestamp(CHECKPOINT_PATH.stat().st_mtime)
    print(f"    Created: {created_time}")
    print(f"    Modified: {modified_time}")

else:
    print(f"    ✗ Checkpoint directory does not exist!")
    print(f"    Medical LLM model not found at expected path.")

# 9. Check medical_qwen_cpt parent directory
print(f"\n[9] PARENT DIRECTORY CONTENTS")
parent_path = Path("medical_qwen_cpt")
if parent_path.exists():
    items = list(parent_path.iterdir())
    print(f"    {parent_path}/ contains {len(items)} items:")
    for item in sorted(items):
        if item.is_dir():
            file_count = len(list(item.rglob("*")))
            print(f"      📁 {item.name}/ ({file_count} items)")
        else:
            size_kb = item.stat().st_size / 1024
            print(f"      📄 {item.name} ({size_kb:.0f} KB)")

print("\n" + "="*80)
print("DIAGNOSIS SUMMARY")
print("="*80)

if CHECKPOINT_PATH.exists():
    has_weights = (CHECKPOINT_PATH / "pytorch_model.bin").exists()
    has_safetensors = (CHECKPOINT_PATH / "model.safetensors").exists()
    has_trainer_state = (CHECKPOINT_PATH / "trainer_state.json").exists()
    has_config = (CHECKPOINT_PATH / "config.json").exists()

    total_size_mb = sum(f.stat().st_size for f in (CHECKPOINT_PATH / f for f in ["pytorch_model.bin", "model.safetensors"] if (CHECKPOINT_PATH / f).exists())) / (1024*1024)

    print("\nKey Findings:")
    print(f"  Model weights present: {'✓ YES' if (has_weights or has_safetensors) else '✗ NO - CRITICAL ISSUE'}")
    print(f"  Config.json present: {'✓ YES' if has_config else '✗ NO'}")
    print(f"  Trainer state present: {'✓ YES' if has_trainer_state else '✗ NO - Training never completed'}")
    print(f"  Model size: {total_size_mb:.2f} MB {'(too small, likely base model only)' if total_size_mb < 5000 else '(reasonable for trained model)'}")

    if not (has_weights or has_safetensors):
        print("\n⚠️  CRITICAL: No model weights found in checkpoint!")
        print("   Likely causes:")
        print("   1. Training script didn't save weights in Cell 12")
        print("   2. Training crashed before reaching save operation")
        print("   3. Checkpoint directory contains only config, not weights")
        print("   4. Model saving failed silently")
    elif total_size_mb < 5000:
        print("\n⚠️  WARNING: Model weights too small!")
        print("   This checkpoint may only contain the base model, not trained weights.")
    else:
        print("\n✓ Checkpoint appears to have model weights and proper structure")
else:
    print("\n✗ CRITICAL: Checkpoint directory does not exist!")
    print("  Training script likely failed during model save (Cell 12).")

print("="*80)

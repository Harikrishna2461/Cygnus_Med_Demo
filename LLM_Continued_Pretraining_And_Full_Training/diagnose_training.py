"""
Diagnose whether continued pretraining actually updated model weights.
Compares trained checkpoint against base model to verify training occurred.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM
import hashlib

BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TRAINED_CHECKPOINT = Path("medical_qwen_cpt/checkpoint-495")
BASE_MODEL_LOCAL = Path("Qwen2.5-7B-Instruct")

def get_weight_hash(model, layer_idx=0):
    """Get hash of first weight tensor to compare models"""
    if layer_idx < len(model.model.layers):
        weight = model.model.layers[layer_idx].self_attn.q_proj.weight.data
        return hashlib.md5(weight.cpu().numpy().tobytes()).hexdigest()
    return None

print("="*80)
print("TRAINING DIAGNOSIS: Did weights actually get updated?")
print("="*80)

# Load trained checkpoint
print("\n1. Loading trained checkpoint...")
try:
    trained = AutoModelForCausalLM.from_pretrained(
        str(TRAINED_CHECKPOINT),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cpu"
    )
    print("✓ Trained checkpoint loaded")
except Exception as e:
    print(f"✗ Failed to load trained checkpoint: {e}")
    exit(1)

# Load base model
print("\n2. Loading base model...")
try:
    if BASE_MODEL_LOCAL.exists():
        base = AutoModelForCausalLM.from_pretrained(
            str(BASE_MODEL_LOCAL),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="cpu"
        )
        print(f"✓ Base model loaded from local: {BASE_MODEL_LOCAL}")
    else:
        print(f"Base model not found locally, downloading {BASE_MODEL_NAME}...")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        print(f"✓ Base model downloaded: {BASE_MODEL_NAME}")
except Exception as e:
    print(f"✗ Failed to load base model: {e}")
    exit(1)

# Compare weights
print("\n3. Comparing weights...")
differences = 0
total_checked = 0

# Check all transformer layers
for layer_idx in range(len(trained.model.layers)):
    trained_weight = trained.model.layers[layer_idx].self_attn.q_proj.weight.data
    base_weight = base.model.layers[layer_idx].self_attn.q_proj.weight.data

    # Check if weights are identical
    if torch.allclose(trained_weight, base_weight, atol=1e-5):
        if layer_idx >= len(trained.model.layers) - 8:  # Last 8 should be unfrozen
            print(f"  Layer {layer_idx}: IDENTICAL (should be different for unfrozen layers!)")
            differences += 1
    else:
        if layer_idx >= len(trained.model.layers) - 8:
            print(f"  Layer {layer_idx}: ✓ Different (correctly updated)")
        else:
            print(f"  Layer {layer_idx}: DIFFERENT? (this layer should be frozen)")

    total_checked += 1

print(f"\n✓ Checked {total_checked} layers")

if differences > 0:
    print(f"\n⚠ WARNING: {differences} unfrozen layers appear unchanged!")
    print("This suggests training may not have completed or weights weren't saved properly.")
else:
    print(f"\n✓ Unfrozen layers show different weights from base model")
    print("This suggests training DID update the weights.")

# Check LM head
print("\n4. Checking LM head...")
trained_lm_head = trained.lm_head.weight.data
base_lm_head = base.lm_head.weight.data

if torch.allclose(trained_lm_head, base_lm_head, atol=1e-5):
    print("  LM head: IDENTICAL (should be different after training!)")
    print("  ⚠ LM head may not have been trained")
else:
    print("  LM head: ✓ Different (correctly updated)")

# Check total parameters
print("\n5. Model sizes:")
trained_params = sum(p.numel() for p in trained.parameters()) / 1e9
base_params = sum(p.numel() for p in base.parameters()) / 1e9
print(f"  Trained: {trained_params:.2f}B parameters")
print(f"  Base: {base_params:.2f}B parameters")

if abs(trained_params - base_params) < 0.1:
    print("  ✓ Same parameter count (expected)")
else:
    print(f"  ⚠ Different parameter count: diff={trained_params-base_params:.2f}B")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("""
To definitively prove training worked:
1. Check training_metrics.csv on Ubuntu to see if loss decreased
2. If loss stayed flat/high → training failed, need to retrain
3. If loss decreased significantly → training worked, issue is model knowledge

To find training_metrics.csv on Ubuntu:
  cd ~/llm_continued_pretraining  # or wherever the training ran
  head -20 logs/training_metrics.csv
  tail -5 logs/training_metrics.csv  # see final loss values

If final eval loss > initial eval loss → training diverged
If final eval loss << initial eval loss → training worked
""")

print("="*80)

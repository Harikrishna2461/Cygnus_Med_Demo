"""
FINE-TUNING WITH DoRA + PiSSA
DoRA: Decomposes weight into magnitude & direction (better expressivity than LoRA)
PiSSA: Parameter-Isolated initialization (faster convergence)
"""

# ============================================================
# CELL 6 (UPDATED): Setup LoRA with DoRA + PiSSA
# ============================================================
"""
from peft import LoraConfig, get_peft_model, TaskType

print("Configuring LoRA with DoRA + PiSSA...")

lora_config = LoraConfig(
    r=LORA_R,                              # Rank: 32
    lora_alpha=LORA_ALPHA,                 # Alpha: 64 (scaling factor)
    lora_dropout=LORA_DROPOUT,             # Dropout: 0.05
    bias='none',                           # No bias adaptation
    task_type=TaskType.CAUSAL_LM,          # Language model task
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],

    # ===== DoRA: Weight Decomposition =====
    # Decomposes lora_B into magnitude and direction vectors
    # Better than vanilla LoRA: more expressive, better performance
    use_dora=True,                         # Enable DoRA

    # ===== PiSSA: Parameter-Isolated Initialization =====
    # Initializes LoRA in an orthogonal subspace
    # Faster convergence, better training stability
    init_lora_weights='pissa',             # PiSSA initialization (vs 'gaussian')
)

model = get_peft_model(model, lora_config)

print("✓ LoRA configured with DoRA + PiSSA")
model.print_trainable_parameters()
"""

# ============================================================
# ALTERNATIVE: DoRA only (without PiSSA)
# ============================================================
"""
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    use_dora=True,                         # Enable DoRA
    init_lora_weights='gaussian',          # Standard Gaussian (not PiSSA)
)

model = get_peft_model(model, lora_config)
print("✓ LoRA with DoRA (standard initialization)")
"""

# ============================================================
# ALTERNATIVE: PiSSA only (without DoRA)
# ============================================================
"""
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    use_dora=False,                        # Disable DoRA
    init_lora_weights='pissa',             # PiSSA initialization
)

model = get_peft_model(model, lora_config)
print("✓ LoRA with PiSSA (standard weight decomposition)")
"""

# ============================================================
# COMPARISON OF METHODS
# ============================================================
"""
VANILLA LoRA:
  - init_lora_weights='gaussian'
  - use_dora=False
  - Standard low-rank adaptation
  - Good baseline, widely used

DoRA (Decomposed Rank Adaptation):
  - use_dora=True
  - Decomposes weight update into:
    * Magnitude vector (scales output)
    * Direction vector (spatial direction)
  - Benefits:
    * More expressive weight updates
    * Better performance with same rank
    * Smoother optimization landscape
  - ~10-15% better accuracy observed in literature

PiSSA (Parameter-Isolated Signed-Weight Adapter):
  - init_lora_weights='pissa'
  - Initializes in orthogonal subspace
  - Benefits:
    * Faster convergence (2-3 epochs vs 5+ for vanilla)
    * Better training stability
    * Can use higher learning rates
    * Works well with quantization
  - Especially good for domain adaptation

DoRA + PiSSA (COMBINED):
  - Both enabled together
  - Maximum expressivity + fastest convergence
  - Best for limited training time (3 epochs)
  - Recommended for fine-tuning on domain data
  - Trade-off: slightly more compute (negligible)
"""

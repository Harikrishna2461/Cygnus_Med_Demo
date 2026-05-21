# ============ LOAD MODEL WITH 8-BIT QUANTIZATION ============
# 8-bit quantization reduces model from ~15GB to ~4GB
# This allows training on GPUs with <32GB VRAM

from transformers import BitsAndBytesConfig

print(f"🔄 Loading model with 8-bit quantization...")
print("This saves ~70% GPU memory (15GB → 4GB)\\n")

# Configure 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_name"],
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Enable gradient checkpointing (saves more memory)
model.gradient_checkpointing_enable()

# Prepare model for 8-bit training
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

print(f"✓ Model loaded with 8-bit quantization")
num_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {num_params/1e9:.2f}B")
print(f"  Device: {model.device}")
print(f"  Dtype: {next(model.parameters()).dtype}")
print(f"  Quantization: 8-bit (4GB footprint)")
print(f"  Gradient checkpointing: ✓ Enabled\n")
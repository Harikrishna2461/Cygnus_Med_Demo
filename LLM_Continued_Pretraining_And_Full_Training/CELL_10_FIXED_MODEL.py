print(f"\n🔄 Loading model: {config['model_name']}...")
model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

# Enable gradient checkpointing to reduce memory usage
model.gradient_checkpointing_enable()

print(f"✓ Model loaded")
num_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {num_params/1e9:.2f}B")
print(f"  Device: {model.device}")
print(f"  Dtype: {next(model.parameters()).dtype}")
print(f"  Gradient checkpointing: Enabled")

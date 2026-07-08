import sys, os
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / 'Task_3_App' / 'vendor' / 'EchoVLM'))

import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
if "default" not in ROPE_INIT_FUNCTIONS:
    def _default_rope_init(config, device=None, **kwargs):
        dim  = kwargs.get('dim') or (config.hidden_size // config.num_attention_heads)
        base = float(kwargs.get('base') or getattr(config, 'rope_theta', 1000000.0))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        return inv_freq, 1.0
    ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

from EchoVLM import Qwen2VLMOEForConditionalGeneration
from transformers import AutoProcessor
from PIL import Image as PILImage
import numpy as np

# Force UTF-8 stdout so Chinese chars in VLM output don't crash on Windows cp1252
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

WEIGHTS = str(HERE / 'EchoVLM')
OFFLOAD = str(HERE / 'EchoVLM_offload')
os.makedirs(OFFLOAD, exist_ok=True)

print("Loading processor...")
proc = AutoProcessor.from_pretrained(WEIGHTS)

print("Loading model with offload_folder (disk offload enabled)...")
model = Qwen2VLMOEForConditionalGeneration.from_pretrained(
    WEIGHTS,
    torch_dtype=torch.float32,
    attn_implementation="eager",
    device_map="auto",
    offload_folder=OFFLOAD,
    offload_state_dict=True,
)
model.config.output_router_logits = False
print("Model ready")

arr = (np.random.randint(40, 180, (256, 256, 3))).astype(np.uint8)
pil_img = PILImage.fromarray(arr)

PROMPT = (
    'B-mode ultrasound 256x256. '
    'Reply ONLY with JSON: {"fascia_y": 90, "veins": [{"x":10,"y":80,"w":20,"h":15,"label":"N2"}]}'
)

print("\n--- Approach 1: apply_chat_template PIL image ---")
try:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_img},
        {"type": "text",  "text": PROMPT},
    ]}]
    inputs = proc.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to("cpu")
    print("Keys:", list(inputs.keys()))
    for k, v in inputs.items():
        shp = getattr(v, 'shape', 'N/A') if v is not None else 'NONE <-- BAD'
        print(f"  {k}: {shp}")
    inputs.pop('mm_token_type_ids', None)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=80)
    raw = proc.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print("SUCCESS repr:", repr(raw))
    print("SUCCESS:", raw)
except Exception as e:
    print(f"FAILED: {e}")
    import traceback; traceback.print_exc()

print("\n--- Approach 2: apply_chat_template tokenize=False + proc(images=[pil]) ---")
try:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_img},
        {"type": "text",  "text": PROMPT},
    ]}]
    text   = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[pil_img], return_tensors="pt").to("cpu")
    print("Keys:", list(inputs.keys()))
    for k, v in inputs.items():
        shp = getattr(v, 'shape', 'N/A') if v is not None else 'NONE <-- BAD'
        print(f"  {k}: {shp}")
    inputs.pop('mm_token_type_ids', None)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=80)
    raw = proc.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print("SUCCESS repr:", repr(raw))
    print("SUCCESS:", raw)
except Exception as e:
    print(f"FAILED: {e}")
    import traceback; traceback.print_exc()

"""
Test if model understands basic CHIVA concepts.
Not classification - just basic terminology and concepts.
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

TRAINED_MODEL_PATH = Path("medical_qwen_cpt/checkpoint-495")

print("="*80)
print("BASIC CHIVA KNOWLEDGE TEST")
print("="*80)

# Load model once
print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(str(TRAINED_MODEL_PATH))
model = AutoModelForCausalLM.from_pretrained(
    str(TRAINED_MODEL_PATH),
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model.eval()
print("✓ Model loaded\n")

def ask_question(question):
    """Ask model a simple question"""
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response[len(question):].strip()
    return answer

# Very basic CHIVA questions
questions = [
    "What is CHIVA?",
    "What is a venous shunt?",
    "What does EP stand for in CHIVA?",
    "What does RP stand for in CHIVA?",
    "What is a Type 1 shunt in CHIVA classification?",
    "What is a Type 2A shunt?",
    "What is superficial venous system?",
    "What is deep venous system?",
    "What is a perforator vein?",
    "In CHIVA, what does it mean when EP enters the superficial system?",
]

print(f"{'─'*80}")
print("TESTING BASIC CHIVA KNOWLEDGE")
print(f"{'─'*80}\n")

for i, question in enumerate(questions, 1):
    print(f"Q{i}: {question}")
    answer = ask_question(question)
    print(f"A: {answer[:200]}")

    # Check if answer is relevant to CHIVA
    is_relevant = any(keyword in answer.lower() for keyword in [
        'chiva', 'vein', 'superficial', 'deep', 'perforator', 'ep', 'rp',
        'shunt', 'reflux', 'entry', 're-entry', 'truncal', 'hemodynamic'
    ])

    if is_relevant:
        print("✓ RELEVANT to CHIVA\n")
    else:
        print("✗ NOT RELEVANT to CHIVA\n")

print("="*80)
print("INTERPRETATION")
print("="*80)
print("""
If most answers are NOT RELEVANT to CHIVA:
→ Model was trained but didn't learn CHIVA concepts from the books
→ Training data was probably general medical text, not CHIVA-specific
→ Need to retrain with explicit CHIVA knowledge examples

If answers ARE RELEVANT:
→ Model learned CHIVA concepts but can't apply them to classification
→ Need better prompt engineering for classification tasks
→ Or need examples of the classification task in training data
""")
print("="*80)

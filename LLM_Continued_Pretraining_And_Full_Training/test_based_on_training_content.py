"""
Test model on questions based on ACTUAL content from the training data.
These questions are derived directly from Franceschi's "Principles of Venous Hemodynamics" book.
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

TRAINED_MODEL_PATH = Path("medical_qwen_cpt/checkpoint-495")

print("="*80)
print("TESTING MODEL ON ACTUAL TRAINING DATA CONTENT")
print("="*80)

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
    """Ask model a question"""
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response[len(question):].strip()
    return answer

# Questions based on ACTUAL content from Franceschi book in training data
questions = [
    # From the classification section (lines 1863-1867)
    {
        "q": "What is a Type 1 Shunt? Describe the reflux route.",
        "expect": "N1 > N2 reflux",
        "source": "Franceschi - Type 1 definition"
    },
    {
        "q": "In Type 1 Shunt, what is the reflux route from N1 to reentry?",
        "expect": "N1 > N2 > N1",
        "source": "Franceschi - Type 1 route"
    },
    {
        "q": "What percentage of varicosity networks do Type 1 Shunts constitute?",
        "expect": "30%",
        "source": "Franceschi - Type 1 frequency"
    },
    {
        "q": "What is a Type 3 Shunt and how does it differ from Type 1?",
        "expect": "involves collateral",
        "source": "Franceschi - Type 3 definition"
    },
    {
        "q": "In Type 3 Shunt, what collateral veins are involved?",
        "expect": "N3, N4T, N4L",
        "source": "Franceschi - Type 3 collaterals"
    },
    {
        "q": "What percentage of varicosity networks do Type 3 Shunts constitute?",
        "expect": "60%",
        "source": "Franceschi - Type 3 frequency"
    },
    {
        "q": "Define private circulation in venous hemodynamics.",
        "expect": "shunt",
        "source": "Franceschi - private circulation"
    },
    {
        "q": "What is CHIVA strategy?",
        "expect": "hemodynamic",
        "source": "Franceschi - CHIVA strategy"
    },
]

print(f"{'─'*80}")
print(f"Questions from Franceschi book (actual training data content)")
print(f"{'─'*80}\n")

correct = 0
total = 0

for item in questions:
    question = item["q"]
    expected = item["expect"]
    source = item["source"]

    print(f"Q: {question}")
    print(f"   Source: {source}")

    answer = ask_question(question)
    print(f"A: {answer[:200]}")

    # Check if expected keyword is in answer
    is_correct = expected.lower() in answer.lower()
    status = "✓ CORRECT" if is_correct else "✗ WRONG"
    print(f"   {status} (looking for: '{expected}')\n")

    if is_correct:
        correct += 1
    total += 1

print("="*80)
print(f"RESULTS: {correct}/{total} correct ({100*correct//total}%)")
print("="*80)

if correct < total // 2:
    print("\nCONCLUSION:")
    print("The model failed to answer basic questions from its training data.")
    print("This suggests:")
    print("  1. Training data was not properly processed/learned")
    print("  2. Model was not actually trained (just using base model)")
    print("  3. Content in training data was too diluted by other material")

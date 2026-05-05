"""
PROMPT-BASED FIX FOR CLASSIFICATION
Use few-shot prompting + explicit formatting to force correct output
"""

import torch
import re

def classify_with_better_prompt(clips, model, tokenizer):
    """Classify using few-shot prompting with explicit delimiters."""

    # Format clips
    clips_str = "\n".join([
        f"Clip {i}: {c['flow']} {c['fromType']}→{c['toType']} y={c.get('posYRatio', 0):.3f}"
        for i, c in enumerate(clips)
    ])

    # Few-shot prompt with EXPLICIT EXAMPLES
    prompt = f"""You are a CHIVA venous shunt classifier. Classify shunts using ONLY these rules:

RULES:
- Check for EP N1→N2: YES=SFJ incompetent, NO=SFJ competent
- If EP N1→N2 + no EP N2→N3 + RP N2→N1 → TYPE 1
- If EP N1→N2 + EP N2→N3 + RP N3 only → TYPE 3
- If no EP N1→N2 + EP N2→N3 → TYPE 2A
- If no EP N1→N2 + EP N2→N2 + RP N3 only → TYPE 2B
- If no EP N1→N2 + EP N2→N2 + RP N3 + RP N2→N1 → TYPE 2C

EXAMPLE 1:
Clips: EP N1→N2, RP N2→N1
Steps: Has EP N1→N2? YES. Has EP N2→N3? NO. Has RP N2→N1? YES. Has RP at N3? NO.
Answer: Type 1

EXAMPLE 2:
Clips: EP N2→N3, RP N3→N2
Steps: Has EP N1→N2? NO. Has EP N2→N3? YES.
Answer: Type 2A

EXAMPLE 3:
Clips: EP N1→N2, EP N2→N3, RP N3→N1
Steps: Has EP N1→N2? YES. Has EP N2→N3? YES. Has RP N3? YES. Has RP N2→N1? NO.
Answer: Type 3

NOW CLASSIFY THIS:
{clips_str}

STEP 1: Count EP N1→N2: {'YES' if any(c['flow']=='EP' and c['fromType']=='N1' and c['toType']=='N2' for c in clips) else 'NO'}
STEP 2: Count EP N2→N3: {'YES' if any(c['flow']=='EP' and c['fromType']=='N2' and c['toType']=='N3' for c in clips) else 'NO'}
STEP 3: Count RP N2→N1: {'YES' if any(c['flow']=='RP' and c['fromType']=='N2' and c['toType']=='N1' for c in clips) else 'NO'}
STEP 4: Count RP N3: {'YES' if any(c['flow']=='RP' and c['fromType']=='N3' for c in clips) else 'NO'}

ANSWER: Type"""

    # Create input
    full_prompt = f"[INST] {prompt} [/INST]"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Generate with strict parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # SHORT to force direct answer
            temperature=0.1,    # LOW temp for determinism
            top_p=0.9,
            do_sample=False,    # Greedy
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in response:
        response = response.split("[/INST]")[1].strip()

    # Extract type
    type_match = re.search(r"Type\s+(\d+[A-Z]*|\d+\+\d+|2[A-C])", response, re.IGNORECASE)
    shunt_type = type_match.group(1) if type_match else "Unknown"

    return {
        "shunt_type": shunt_type,
        "raw_response": response[:200]
    }


# Test it
if __name__ == "__main__":
    test_cases = {
        "Type 1": [
            {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.080},
            {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.300},
        ],
        "Type 2A": [
            {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.200},
            {"flow": "RP", "fromType": "N3", "toType": "N2", "posYRatio": 0.470},
        ],
        "Type 3": [
            {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.050},
            {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.132},
            {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.212},
        ],
    }

    print("=" * 80)
    print("TESTING WITH BETTER PROMPT + FEW-SHOT EXAMPLES")
    print("=" * 80)

    for expected, clips in test_cases.items():
        print(f"\nExpected: {expected}")
        result = classify_with_better_prompt(clips, model, tokenizer)
        print(f"Got: {result['shunt_type']}")
        print(f"Raw: {result['raw_response'][:100]}")

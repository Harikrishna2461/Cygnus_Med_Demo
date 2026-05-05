"""
Script 1b: Reformat Instructions
Reads existing train/val JSONL, extracts raw chunks, and reassigns
a randomly-chosen instruction from a diverse pool per sample.

Usage:
    python script1b_reformat_instructions.py
    python script1b_reformat_instructions.py --mode plain   # drop wrapper entirely
    python script1b_reformat_instructions.py --in_dir data --out_dir data
"""

import json
import random
import argparse
from pathlib import Path

RANDOM_SEED = 42

SYSTEM_PROMPT = (
    "You are a medical expert specialising in venous and lymphatic disorders, "
    "vascular surgery, duplex ultrasound, and haemodynamics. "
    "Answer questions accurately using clinical and scientific knowledge."
)

INSTRUCTIONS = [
    "Summarise and explain the following passage from a medical textbook on venous/vascular medicine:",
    "Explain the clinical significance of the following passage for a vascular surgeon:",
    "A medical student asks you to explain this excerpt from a vascular medicine textbook:",
    "What are the key clinical concepts described in the following passage?",
    "Provide a detailed explanation of the following text on venous and lymphatic disorders:",
    "A vascular surgery resident presents this passage and asks for clinical context and interpretation:",
    "Explain the haemodynamic principles discussed in the following passage:",
    "Describe the key points from the following passage relevant to duplex ultrasound practice:",
    "Interpret and explain the following passage from a venous medicine reference:",
    "What clinical lessons can be drawn from the following passage on venous/vascular medicine?",
    "Explain the following passage as you would to a junior doctor starting vascular surgery training:",
    "Summarise the diagnostic or therapeutic concepts described in the following medical text:",
    "A clinician asks: what does this passage teach us about venous disease management?",
    "Provide an educational summary of the following passage on vascular medicine:",
    "Extract and explain the key medical concepts from the following passage on venous disorders:",
    "How would you describe the content of this passage to a colleague in vascular medicine?",
    "What does this passage tell us about the pathophysiology or treatment of venous disease?",
    "Explain this excerpt in the context of modern vascular surgical practice:",
    "Describe the evidence or principles outlined in the following passage:",
    "A nurse specialist asks you to explain the clinical content of the following passage:",
]


def extract_chunk(text: str) -> str:
    """Pull the raw chunk out of a Mistral [INST]...[/INST] formatted example."""
    sep = " [/INST] "
    if sep not in text:
        return ""
    response_part = text.split(sep, 1)[1]
    # response is: chunk + " </s>"
    if response_part.endswith(" </s>"):
        return response_part[: -len(" </s>")].strip()
    return response_part.strip()


def build_diverse_example(chunk: str) -> dict:
    instruction = random.choice(INSTRUCTIONS)
    prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n{instruction}\n\n{chunk} [/INST]"
    response = f"{chunk} </s>"
    return {"text": f"{prompt} {response}"}


def build_plain_example(chunk: str) -> dict:
    return {"text": f"{chunk} </s>"}


def reformat_file(src: Path, dst: Path, mode: str) -> int:
    examples = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunk = extract_chunk(obj["text"])
            if not chunk:
                continue
            if mode == "plain":
                examples.append(build_plain_example(chunk))
            else:
                examples.append(build_diverse_example(chunk))

    with open(dst, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return len(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir",  default="data")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument(
        "--mode",
        choices=["diverse", "plain"],
        default="diverse",
        help="diverse = random instruction pool; plain = no instruction wrapper",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        src = in_dir  / f"training_data_{split}.jsonl"
        dst = out_dir / f"training_data_{split}.jsonl"
        if not src.exists():
            print(f"[SKIP] {src} not found")
            continue
        n = reformat_file(src, dst, args.mode)
        arrow = "→ (overwritten)" if src == dst else f"→ {dst}"
        print(f"{split:5s}: {n:,} examples  {arrow}")

    print(f"\nMode: {args.mode}. Done.")


if __name__ == "__main__":
    main()

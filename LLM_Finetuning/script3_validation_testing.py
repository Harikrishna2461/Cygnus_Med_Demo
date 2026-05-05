"""
Script 3: Validation & Testing
Loads the merged fine-tuned model, runs domain-specific queries,
and optionally compares against the base Mistral-7B.

Usage:
    python script3_validation_testing.py --model_dir merged_model/
    python script3_validation_testing.py --model_dir merged_model/ --compare_base
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ── Domain-specific test queries (venous/vascular medicine) ──────────────────
TEST_QUERIES = [
    {
        "id": "Q1",
        "category": "Anatomy",
        "question": "Describe the anatomy of the great saphenous vein and its main tributaries.",
    },
    {
        "id": "Q2",
        "category": "Duplex Ultrasound",
        "question": "What are the key duplex ultrasound criteria for diagnosing deep vein thrombosis in the femoral vein?",
    },
    {
        "id": "Q3",
        "category": "CHIVA Strategy",
        "question": "Explain the CHIVA strategy for treating varicose veins and how it differs from conventional stripping.",
    },
    {
        "id": "Q4",
        "category": "Hemodynamics",
        "question": "What is venous reflux and how is it measured haemodynamically in chronic venous insufficiency?",
    },
    {
        "id": "Q5",
        "category": "Endovascular",
        "question": "Describe the endovascular treatment options for iliac vein compression syndrome (May-Thurner syndrome).",
    },
]

SYSTEM_PROMPT = (
    "You are a medical expert specialising in venous and lymphatic disorders, "
    "vascular surgery, duplex ultrasound, and haemodynamics. "
    "Answer questions accurately using clinical and scientific knowledge."
)

def format_prompt(question: str) -> str:
    return f"<s>[INST] {SYSTEM_PROMPT}\n\n{question} [/INST]"

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_dir: str, load_in_4bit: bool = False):
    print(f"\nLoading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    print("Model ready.")
    return gen_pipeline

# ── Inference ─────────────────────────────────────────────────────────────────
def run_query(pipe, question: str) -> tuple[str, float]:
    prompt = format_prompt(question)
    t0 = time.time()
    outputs = pipe(prompt, return_full_text=False)
    elapsed = time.time() - t0
    response = outputs[0]["generated_text"].strip()
    return response, elapsed

# ── Medical terminology spot-check ────────────────────────────────────────────
MEDICAL_TERMS = [
    "saphenous", "femoral", "popliteal", "varicose", "thrombosis",
    "reflux", "insufficiency", "duplex", "ultrasound", "haemodynamic",
    "endovascular", "venous", "lymphatic", "CHIVA", "perforator",
]

def check_terminology(text: str) -> dict:
    text_lower = text.lower()
    found = [t for t in MEDICAL_TERMS if t.lower() in text_lower]
    return {
        "terms_found": found,
        "count": len(found),
        "coverage_pct": round(len(found) / len(MEDICAL_TERMS) * 100, 1),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="merged_model",
                        help="Path to merged fine-tuned model")
    parser.add_argument("--base_model_id", default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Base model HF ID for comparison")
    parser.add_argument("--compare_base", action="store_true",
                        help="Also run queries on base model for comparison")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load in 4-bit quantisation (saves VRAM)")
    parser.add_argument("--output_json", default="validation_report.json",
                        help="Where to save the report")
    args = parser.parse_args()

    report = {"finetuned": [], "base": [], "summary": {}}

    # ── Fine-tuned model ──────────────────────────────────────────────────────
    ft_pipe = load_model(args.model_dir, load_in_4bit=args.load_in_4bit)

    print("\n" + "="*70)
    print("FINE-TUNED MODEL — Test Queries")
    print("="*70)

    ft_total_terms = 0
    for q in TEST_QUERIES:
        print(f"\n[{q['id']}] {q['category']}: {q['question']}")
        response, elapsed = run_query(ft_pipe, q["question"])
        term_check = check_terminology(response)
        ft_total_terms += term_check["count"]

        print(f"\nResponse ({elapsed:.1f}s):\n{response}")
        print(f"Medical terms: {term_check['terms_found']} ({term_check['coverage_pct']}% coverage)")

        report["finetuned"].append({
            "query": q,
            "response": response,
            "elapsed_s": round(elapsed, 2),
            "terminology": term_check,
        })

    # ── Base model (optional) ─────────────────────────────────────────────────
    if args.compare_base:
        del ft_pipe
        torch.cuda.empty_cache()

        print("\n" + "="*70)
        print("BASE MODEL — Comparison")
        print("="*70)

        base_pipe = load_model(args.base_model_id, load_in_4bit=True)
        base_total_terms = 0

        for q in TEST_QUERIES:
            print(f"\n[{q['id']}] {q['category']}: {q['question']}")
            response, elapsed = run_query(base_pipe, q["question"])
            term_check = check_terminology(response)
            base_total_terms += term_check["count"]

            print(f"\nResponse ({elapsed:.1f}s):\n{response}")
            print(f"Medical terms: {term_check['terms_found']}")

            report["base"].append({
                "query": q,
                "response": response,
                "elapsed_s": round(elapsed, 2),
                "terminology": term_check,
            })

        report["summary"]["base_avg_terms"] = base_total_terms / len(TEST_QUERIES)

    # ── Summary ───────────────────────────────────────────────────────────────
    report["summary"]["finetuned_avg_terms"] = ft_total_terms / len(TEST_QUERIES)
    report["summary"]["num_queries"] = len(TEST_QUERIES)

    out_path = Path(args.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "="*70)
    print(f"Validation report saved → {out_path}")
    print(f"Fine-tuned avg medical terms per answer: {report['summary']['finetuned_avg_terms']:.1f} / {len(MEDICAL_TERMS)}")
    if args.compare_base:
        print(f"Base model  avg medical terms per answer: {report['summary']['base_avg_terms']:.1f} / {len(MEDICAL_TERMS)}")

if __name__ == "__main__":
    main()

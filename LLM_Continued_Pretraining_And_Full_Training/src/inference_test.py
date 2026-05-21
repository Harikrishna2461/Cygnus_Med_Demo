"""
inference_test.py
=================
Loads the fine-tuned model from models/medical_qwen_cpt/ and the base
Qwen2.5-7B-Instruct model, then runs 5 test prompts covering:
  1. Medical knowledge recall
  2. Clinical reasoning
  3. Long prompt handling (500+ token input)
  4. Format instruction following
  5. Edge case: rare disease

For each prompt the script prints:
  - Input text
  - Base model output
  - Fine-tuned model output
  - Per-model: generation time, output token count
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

# ── Configuration constants ───────────────────────────────────────────────────
BASE_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"   # original HuggingFace model
FINETUNED_PATH  = Path(__file__).parent.parent / "models" / "medical_qwen_cpt"
LOG_DIR         = Path(__file__).parent.parent / "logs"
LOG_FILE        = LOG_DIR / "inference_test.log"

# Generation settings
GENERATION_CONFIG = dict(
    repetition_penalty   = 1.15,
    no_repeat_ngram_size = 4,
    temperature          = 0.7,
    do_sample            = True,
    max_new_tokens       = 512,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(str(LOG_FILE), level="DEBUG", rotation="10 MB",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

TEST_PROMPTS = [
    # ── 1. Medical knowledge recall ──────────────────────────────────────────
    {
        "id":    1,
        "label": "Medical knowledge recall",
        "text":  (
            "The pathophysiology of chronic venous insufficiency involves the failure of the venous "
            "valve system. Describe the haemodynamic changes that occur when venous valves become "
            "incompetent, including the role of ambulatory venous hypertension, and how this leads "
            "to the clinical findings of oedema, skin changes, and ulceration."
        ),
    },

    # ── 2. Clinical reasoning ────────────────────────────────────────────────
    {
        "id":    2,
        "label": "Clinical reasoning",
        "text":  (
            "A 62-year-old male presents with a 6-month history of progressive right leg swelling, "
            "heaviness, and a medial malleolar ulcer that has failed to heal with compression "
            "therapy. Duplex ultrasound reveals reflux in the great saphenous vein (GSV) from the "
            "saphenofemoral junction to the knee, with an incompetent perforator at the Hunter's "
            "canal. Deep venous system appears normal with no evidence of prior DVT.\n\n"
            "What is the most appropriate next step in management? Discuss the treatment options, "
            "including the role of endovenous thermal ablation, CHIVA strategy, and foam "
            "sclerotherapy, and justify your recommendation."
        ),
    },

    # ── 3. Long prompt – clinical scenario with lab values ───────────────────
    {
        "id":    3,
        "label": "Long prompt handling (500+ token input)",
        "text":  (
            "Below is a detailed clinical vignette. Please provide a comprehensive differential "
            "diagnosis and a structured management plan.\n\n"
            "PATIENT: 74-year-old female with a 20-year history of bilateral varicose veins, "
            "recently noted bilateral lower-limb oedema, right > left. PMH: hypertension, type 2 "
            "diabetes mellitus (HbA1c 7.8 %), osteoarthritis of the knees, and a history of "
            "deep vein thrombosis (DVT) in the left leg 12 years ago, for which she was "
            "anticoagulated for 6 months.\n\n"
            "MEDICATIONS: Amlodipine 5 mg OD, Metformin 1 g BD, Paracetamol 1 g QDS PRN. "
            "No current anticoagulation.\n\n"
            "EXAMINATION: BP 148/88 mmHg, HR 78 bpm, SpO2 98 % on air. Both calves non-tender. "
            "Right leg: C4b CEAP classification with lipodermatosclerosis and haemosiderin "
            "deposition at the medial gaiter area. No active ulceration. Varicosities from the "
            "GSV distribution bilaterally. Left leg: pitting oedema to the knee, mild varicosities, "
            "no skin changes. Ankle-brachial pressure index (ABPI): Right 0.94, Left 0.91. "
            "Venous duplex: Right GSV reflux from SFJ to mid-thigh; Right small saphenous vein "
            "(SSV) competent. Left GSV: previous post-thrombotic changes in the femoral vein with "
            "partial recanalisation; reflux in the left below-knee popliteal vein.\n\n"
            "INVESTIGATIONS: FBC – Hb 11.2 g/dL, MCV 72 fL, WBC 6.8, Plt 241. "
            "U&E – Na 138, K 4.2, Cr 96, eGFR 58. BNP 54 pg/mL. Urinalysis: no proteinuria. "
            "Echocardiogram: normal LV function, EF 62 %, no significant valvular disease. "
            "CXR: no cardiomegaly, no pulmonary oedema.\n\n"
            "Based on the above, provide: (a) the most likely aetiology of the right leg oedema "
            "and left leg oedema (they may differ); (b) a prioritised differential diagnosis; "
            "(c) a structured management plan including investigations, conservative measures, "
            "compression therapy, pharmacological options, and interventional options with their "
            "evidence base; (d) risks to consider given her co-morbidities."
        ),
    },

    # ── 4. Format instruction following ─────────────────────────────────────
    {
        "id":    4,
        "label": "Format instruction following",
        "text":  (
            "Please provide a structured summary of the CHIVA (Conservative and Haemodynamic "
            "treatment of Insufficiency in an Out-patient setting) strategy for varicose veins. "
            "Format your answer as a numbered list with the following headings:\n"
            "1. Theoretical basis\n"
            "2. Pre-operative assessment\n"
            "3. Surgical technique\n"
            "4. Post-operative management\n"
            "5. Outcomes and evidence\n"
            "6. Advantages over conventional stripping\n\n"
            "Keep each section concise but clinically detailed."
        ),
    },

    # ── 5. Rare disease / edge case ──────────────────────────────────────────
    {
        "id":    5,
        "label": "Edge case – rare vascular condition",
        "text":  (
            "Klippel-Trénaunay syndrome (KTS) is a rare congenital vascular malformation syndrome. "
            "Describe:\n"
            "(a) the classical triad of KTS and its pathogenesis;\n"
            "(b) how venous duplex findings differ from common primary varicose veins;\n"
            "(c) the specific challenges in managing venous hypertension in KTS, including why "
            "standard endovenous ablation may be contraindicated;\n"
            "(d) the role of the lateral embryonic vein (persistent sciatic vein) and its clinical "
            "significance;\n"
            "(e) current evidence for compression therapy, sclerotherapy, and surgical options."
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_path: str | Path, label: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading {label} from {model_path}…")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info(f"{label} loaded on {device}")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE
# ═══════════════════════════════════════════════════════════════════════════════

def generate(model, tokenizer, prompt: str) -> tuple[str, float, int]:
    """
    Generate a response for a prompt.
    Returns (generated_text, elapsed_seconds, output_token_count).
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    device = next(model.parameters()).device
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    input_len      = input_ids.shape[-1]

    with torch.no_grad():
        start = time.time()
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            **GENERATION_CONFIG,
        )
        elapsed = time.time() - start

    new_ids    = output_ids[0][input_len:]
    out_text   = tokenizer.decode(new_ids, skip_special_tokens=True)
    out_tokens = len(new_ids)

    return out_text, elapsed, out_tokens


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_SEP = "=" * 80

def _banner(text: str) -> None:
    print(f"\n{_SEP}\n{text}\n{_SEP}")


def _section(title: str, content: str, elapsed: float, tokens: int) -> None:
    print(f"\n── {title} ──")
    print(f"[time: {elapsed:.2f}s | tokens: {tokens}]")
    print(content.strip())


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not FINETUNED_PATH.exists():
        logger.error(
            f"Fine-tuned model not found at {FINETUNED_PATH}. "
            "Run pretrain.py first."
        )
        sys.exit(1)

    # Load fine-tuned model
    ft_model, ft_tokenizer = load_model_and_tokenizer(FINETUNED_PATH, "Fine-tuned model")

    # Load base model (separate from fine-tuned)
    base_model, base_tokenizer = load_model_and_tokenizer(BASE_MODEL_PATH, "Base model")

    print("\n" + _SEP)
    print("  MEDICAL QWEN2.5-7B CPT  –  INFERENCE EVALUATION")
    print(_SEP)
    print(f"  Fine-tuned model : {FINETUNED_PATH}")
    print(f"  Base model       : {BASE_MODEL_PATH}")
    print(f"  Test prompts     : {len(TEST_PROMPTS)}")
    print(_SEP)

    results = []

    for prompt_info in TEST_PROMPTS:
        pid   = prompt_info["id"]
        label = prompt_info["label"]
        text  = prompt_info["text"]

        _banner(f"PROMPT {pid}: {label}")
        print(f"\nINPUT ({len(text.split())} words):\n{text}\n")

        # ── Base model ───────────────────────────────────────────────────────
        logger.info(f"Running base model on prompt {pid}…")
        base_out, base_time, base_toks = generate(base_model, base_tokenizer, text)
        _section("BASE MODEL", base_out, base_time, base_toks)

        # ── Fine-tuned model ─────────────────────────────────────────────────
        logger.info(f"Running fine-tuned model on prompt {pid}…")
        ft_out, ft_time, ft_toks = generate(ft_model, ft_tokenizer, text)
        _section("FINE-TUNED MODEL", ft_out, ft_time, ft_toks)

        results.append({
            "prompt_id":        pid,
            "label":            label,
            "input_words":      len(text.split()),
            "base_output":      base_out,
            "base_time_s":      round(base_time, 3),
            "base_tokens":      base_toks,
            "finetuned_output": ft_out,
            "finetuned_time_s": round(ft_time, 3),
            "finetuned_tokens": ft_toks,
        })

        logger.info(
            f"Prompt {pid} done | "
            f"base: {base_time:.1f}s/{base_toks} tok | "
            f"ft: {ft_time:.1f}s/{ft_toks} tok"
        )

    # ── Save results as JSON ──────────────────────────────────────────────────
    import json
    results_path = LOG_DIR / "inference_results.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    _banner("SUMMARY")
    print(f"\n{'Prompt':<40} {'Base (s)':>9} {'FT (s)':>8} {'Base tok':>9} {'FT tok':>8}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['label'][:40]:<40} "
            f"{r['base_time_s']:>9.2f} "
            f"{r['finetuned_time_s']:>8.2f} "
            f"{r['base_tokens']:>9} "
            f"{r['finetuned_tokens']:>8}"
        )
    print(f"\nFull results saved to: {results_path}")


if __name__ == "__main__":
    main()

"""
Script 1: Data Preparation
Extracts text from PDFs (native + OCR fallback), cleans it,
formats as Mistral instruction JSONL, and creates 80/20 train/val split.

Requirements:
    pip install pymupdf pillow tiktoken tqdm easyocr
    # easyocr handles scanned PDFs without any external binary (GPU-accelerated)
    # Optional fallback: pip install pytesseract  +  Tesseract binary from
    #   https://github.com/UB-Mannheim/tesseract/wiki
"""

import os
import re
import json
import random
import argparse
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import tiktoken
from tqdm import tqdm

import io
from PIL import Image

# Prefer easyocr (pure Python, GPU-accelerated) over tesseract
_easyocr_reader = None
OCR_AVAILABLE = False

try:
    import easyocr
    _easyocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    OCR_AVAILABLE = True
    print("[INFO] easyocr loaded (GPU).")
except ImportError:
    try:
        import pytesseract
        OCR_AVAILABLE = True
        print("[INFO] pytesseract loaded as OCR fallback.")
    except ImportError:
        print("[WARN] Neither easyocr nor pytesseract found — OCR disabled. "
              "Run: pip install easyocr")

# ── Config ────────────────────────────────────────────────────────────────────
PDF_DIR        = Path("books_articles")
OUTPUT_DIR     = Path("data")
TRAIN_FILE     = OUTPUT_DIR / "training_data_train.jsonl"
VAL_FILE       = OUTPUT_DIR / "training_data_val.jsonl"
CHUNK_SIZE     = 512      # tokens per chunk
CHUNK_OVERLAP  = 64       # token overlap between chunks
TRAIN_RATIO    = 0.80
MIN_CHUNK_TOKENS = 80     # discard very short chunks
RANDOM_SEED    = 42
TOKENIZER_MODEL = "gpt2"  # tiktoken encoder close to Mistral's vocabulary size

# Mistral instruction template
SYSTEM_PROMPT = (
    "You are a medical expert specialising in venous and lymphatic disorders, "
    "vascular surgery, duplex ultrasound, and haemodynamics. "
    "Answer questions accurately using clinical and scientific knowledge."
)

def build_prompt(chunk: str) -> str:
    return (
        f"<s>[INST] {SYSTEM_PROMPT}\n\n"
        f"Summarise and explain the following passage from a medical textbook "
        f"on venous/vascular medicine:\n\n{chunk.strip()} [/INST]"
    )

def build_response(chunk: str) -> str:
    """The 'response' is the chunk itself — the model learns to reproduce domain text."""
    return chunk.strip() + " </s>"

def make_example(chunk: str) -> dict:
    return {"text": build_prompt(chunk) + " " + build_response(chunk)}

# ── PDF helpers ───────────────────────────────────────────────────────────────
def is_scanned_page(page: fitz.Page, min_text_chars: int = 50) -> bool:
    """Return True if the page has too little native text (likely scanned image)."""
    return len(page.get_text("text").strip()) < min_text_chars

def ocr_page(page: fitz.Page, dpi: int = 200) -> str:
    """Render page to image and run OCR (easyocr preferred, tesseract fallback)."""
    if not OCR_AVAILABLE:
        return ""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    try:
        if _easyocr_reader is not None:
            import numpy as np
            results = _easyocr_reader.readtext(np.array(img), detail=0, paragraph=True)
            return "\n".join(results)
        else:
            return pytesseract.image_to_string(img, lang="eng")
    except Exception as e:
        print(f"    [OCR error] {e}")
        return ""

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from all pages; fall back to OCR for image-heavy pages."""
    doc = fitz.open(str(pdf_path))
    pages_text = []
    n_native, n_ocr = 0, 0
    for page in doc:
        if is_scanned_page(page):
            text = ocr_page(page)
            n_ocr += 1
        else:
            text = page.get_text("text")
            n_native += 1
        pages_text.append(text)
    doc.close()
    print(f"  Pages: {n_native} native text, {n_ocr} OCR")
    return "\n".join(pages_text)

# ── Text cleaning ─────────────────────────────────────────────────────────────
HEADER_PATTERNS = [
    r"^\s*\d+\s*$",                          # page numbers alone
    r"^\s*(chapter|section)\s+\d+",          # chapter headers
    r"^.{1,80}(www\.|http|©|copyright)",     # URLs / copyright lines
    r"^\s*[\-_=]{3,}\s*$",                   # divider lines
]
_header_re = re.compile("|".join(HEADER_PATTERNS), re.IGNORECASE | re.MULTILINE)

def clean_text(raw: str) -> str:
    # Remove lines matching header patterns
    lines = raw.splitlines()
    lines = [l for l in lines if not _header_re.match(l)]
    text = "\n".join(lines)
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove hyphenation at line breaks
    text = re.sub(r"-\n(\w)", r"\1", text)
    return text.strip()

# ── Chunking ──────────────────────────────────────────────────────────────────
def tokenize(enc, text: str) -> list[int]:
    return enc.encode(text)

def chunk_tokens(tokens: list[int], enc, size: int, overlap: int) -> list[str]:
    """Slide a window over token list and decode each window back to text."""
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunk_text = enc.decode(tokens[start:end])
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += size - overlap
    return chunks

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default=str(PDF_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding(TOKENIZER_MODEL)
    random.seed(args.seed)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")
    print(f"Found {len(pdf_files)} PDFs.\n")

    all_examples = []
    total_tokens = 0

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        print(f"\n[{pdf_path.name}]")
        raw_text = extract_text_from_pdf(pdf_path)
        clean = clean_text(raw_text)

        tokens = tokenize(enc, clean)
        total_tokens += len(tokens)
        print(f"  Tokens: {len(tokens):,}")

        chunks = chunk_tokens(tokens, enc, args.chunk_size, args.overlap)
        valid = [c for c in chunks if len(tokenize(enc, c)) >= MIN_CHUNK_TOKENS]
        print(f"  Chunks: {len(chunks)} total, {len(valid)} valid (≥{MIN_CHUNK_TOKENS} tokens)")

        examples = [make_example(c) for c in valid]
        all_examples.extend(examples)

    print(f"\n{'='*60}")
    print(f"Total source tokens : {total_tokens:,}")
    print(f"Total examples      : {len(all_examples):,}")

    random.shuffle(all_examples)
    split = int(len(all_examples) * TRAIN_RATIO)
    train_examples = all_examples[:split]
    val_examples   = all_examples[split:]

    train_path = output_dir / "training_data_train.jsonl"
    val_path   = output_dir / "training_data_val.jsonl"

    for path, examples in [(train_path, train_examples), (val_path, val_examples)]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nTrain examples : {len(train_examples):,}  →  {train_path}")
    print(f"Val   examples : {len(val_examples):,}  →  {val_path}")

    # Token count in output files
    train_tokens = sum(len(tokenize(enc, ex["text"])) for ex in train_examples)
    val_tokens   = sum(len(tokenize(enc, ex["text"])) for ex in val_examples)
    print(f"\nTrain tokens   : {train_tokens:,}")
    print(f"Val   tokens   : {val_tokens:,}")
    print("\nDone.")


if __name__ == "__main__":
    main()
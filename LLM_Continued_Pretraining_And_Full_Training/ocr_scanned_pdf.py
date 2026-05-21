"""
Quick OCR extraction for the scanned CHIVABook1988-1993_2.pdf using EasyOCR.
Converts PDF pages to images and extracts text.
"""

import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
PDF_FILE = Path("books_articles") / "CHIVABook1988-1993_2.pdf"
OUTPUT_FILE = Path("data") / "raw" / "CHIVABook1988-1993_2.txt"

# ── Setup ─────────────────────────────────────────────────────────────────────
from tqdm import tqdm
import fitz

print("Installing PaddleOCR (pure Python, no torch/vision deps)...")
try:
    from paddleocr import PaddleOCR
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "paddleocr"])
    from paddleocr import PaddleOCR

# ── Extract with PaddleOCR ─────────────────────────────────────────────────────
print(f"Loading OCR reader (English, downloading model ~50MB)...")
ocr = PaddleOCR(lang="en")

doc = fitz.open(str(PDF_FILE))
all_text = []

print(f"Extracting text from {len(doc)} pages using PaddleOCR...\n")

from PIL import Image
import io

for page_num in tqdm(range(len(doc)), desc="OCR pages", unit="page"):
    page = doc[page_num]

    # Render page to image at 150 DPI (balance speed/quality)
    mat = fitz.Matrix(150 / 72, 150 / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

    # Convert to PIL
    img_bytes = pix.tobytes("png")
    pil_img = Image.open(io.BytesIO(img_bytes))

    # Run PaddleOCR  (returns [[bbox, text, confidence], ...])
    results = ocr.ocr(pil_img, cls=True)

    # Extract and sort by vertical position for reading order
    if results and results[0]:
        results_sorted = sorted(results[0], key=lambda x: x[0][0][1])
        page_text = " ".join([text for (bbox, text, conf) in results_sorted])
        if page_text.strip():
            all_text.append(page_text)

doc.close()

# ── Save ────────────────────────────────────────────────────────────────────────
full_text = "\n\n".join(all_text)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text(full_text, encoding="utf-8")

word_count = len(full_text.split())
print(f"\n✓ Extracted: {word_count:,} words from {len(all_text)} pages")
print(f"  Saved to: {OUTPUT_FILE}")

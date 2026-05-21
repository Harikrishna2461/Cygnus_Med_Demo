#!/usr/bin/env python3
"""
Extract text from all PDFs and prepare for continued pre-training
Handles both text-based and scanned (OCR) PDFs
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import pdfplumber
    logger.info("✓ pdfplumber available")
except ImportError:
    logger.warning("✗ pdfplumber not found, installing...")
    os.system("pip install pdfplumber -q")
    import pdfplumber

try:
    import pytesseract
    from PIL import Image
    import pdf2image
    logger.info("✓ OCR libraries available")
    OCR_AVAILABLE = True
except ImportError:
    logger.warning("✗ OCR libraries not found (pytesseract, pdf2image, Pillow)")
    logger.warning("  Scanned PDFs will be skipped. To enable OCR:")
    logger.warning("  pip install pytesseract pdf2image pillow")
    logger.warning("  AND install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
    OCR_AVAILABLE = False

# Directories to search
PDF_DIRS = [
    r'C:\Users\Krish\Downloads\LLM_Finetuning\books_articles',
    r'C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data',
]

OUTPUT_FILE = r'C:\Users\Krish\Downloads\LLM_Finetuning\full_medical_data.txt'
STATS_FILE = r'C:\Users\Krish\Downloads\LLM_Finetuning\extraction_stats.txt'

def extract_text_pdfplumber(pdf_path):
    """Extract text from PDF using pdfplumber (fast, text-based PDFs)"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text, "pdfplumber"
    except Exception as e:
        logger.warning(f"  pdfplumber failed: {str(e)[:100]}")
        return None, None

def extract_text_ocr(pdf_path):
    """Extract text from scanned PDF using Tesseract OCR"""
    if not OCR_AVAILABLE:
        return None, None

    try:
        logger.info(f"  Using OCR (slower)...")
        images = pdf2image.convert_from_path(pdf_path, dpi=150)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
        return text, "OCR"
    except Exception as e:
        logger.warning(f"  OCR failed: {str(e)[:100]}")
        return None, None

def extract_pdf(pdf_path):
    """
    Extract text from PDF, trying pdfplumber first, then OCR
    Returns (text, method_used)
    """
    logger.info(f"\nProcessing: {pdf_path.name}")

    # Try pdfplumber first (fast)
    text, method = extract_text_pdfplumber(pdf_path)

    # If pdfplumber returned empty/None, try OCR
    if not text or len(text.strip()) < 100:
        if OCR_AVAILABLE:
            logger.info(f"  Text-based extraction failed, trying OCR...")
            text, method = extract_text_ocr(pdf_path)
        else:
            logger.warning(f"  Skipped (empty text, OCR not available)")
            return None, None

    if text and len(text.strip()) >= 50:
        word_count = len(text.split())
        logger.info(f"  ✓ Success ({method}): {word_count:,} words extracted")
        return text, method
    else:
        logger.warning(f"  Failed: Could not extract meaningful text")
        return None, None

def main():
    logger.info("=" * 80)
    logger.info("PDF TEXT EXTRACTION FOR CONTINUED PRE-TRAINING")
    logger.info("=" * 80)

    # Find all PDFs
    all_pdfs = []
    for pdf_dir in PDF_DIRS:
        dir_path = Path(pdf_dir)
        if dir_path.exists():
            pdfs = list(dir_path.glob("*.pdf"))
            all_pdfs.extend(pdfs)
            logger.info(f"\nFound {len(pdfs)} PDFs in {dir_path.name}")

    if not all_pdfs:
        logger.error("No PDFs found!")
        return

    logger.info(f"\nTotal PDFs to process: {len(all_pdfs)}")

    # Extract text from all PDFs
    all_text = []
    stats = {
        'total': len(all_pdfs),
        'successful': 0,
        'failed': 0,
        'methods': {'pdfplumber': 0, 'OCR': 0},
        'files': {}
    }

    for pdf_path in sorted(all_pdfs):
        text, method = extract_pdf(pdf_path)

        if text:
            all_text.append(text)
            stats['successful'] += 1
            if method:
                stats['methods'][method] += 1
            stats['files'][pdf_path.name] = {'status': 'success', 'method': method}
        else:
            stats['failed'] += 1
            stats['files'][pdf_path.name] = {'status': 'failed'}

    # Combine all text
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Combining {stats['successful']} successfully extracted PDFs...")

    combined_text = "\n\n".join(all_text)
    total_words = len(combined_text.split())
    total_chars = len(combined_text)

    # Save to file
    logger.info(f"\nSaving to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(combined_text)

    logger.info(f"✓ Saved successfully")
    logger.info(f"  Total words: {total_words:,}")
    logger.info(f"  Total characters: {total_chars:,}")
    logger.info(f"  File size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")

    # Save stats
    logger.info(f"\nSaving extraction stats...")
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        f.write("PDF EXTRACTION STATISTICS\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Total PDFs: {stats['total']}\n")
        f.write(f"Successful: {stats['successful']}\n")
        f.write(f"Failed: {stats['failed']}\n")
        f.write(f"\nExtraction Methods:\n")
        f.write(f"  pdfplumber: {stats['methods']['pdfplumber']}\n")
        f.write(f"  OCR: {stats['methods']['OCR']}\n")
        f.write(f"\nOutput Statistics:\n")
        f.write(f"  Total words: {total_words:,}\n")
        f.write(f"  Total characters: {total_chars:,}\n")
        f.write(f"  File size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB\n")
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Per-file status:\n")
        for filename, info in sorted(stats['files'].items()):
            f.write(f"  {filename}: {info['status']}")
            if info.get('method'):
                f.write(f" ({info['method']})")
            f.write(f"\n")

    logger.info(f"✓ Stats saved to: {STATS_FILE}")

    logger.info(f"\n{'=' * 80}")
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"Ready for Phase 1: Continued Pre-training")
    logger.info(f"{'=' * 80}\n")

    return OUTPUT_FILE

if __name__ == "__main__":
    main()

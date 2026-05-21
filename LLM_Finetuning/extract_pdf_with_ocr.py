#!/usr/bin/env python3
"""
Extract text from remaining PDFs with full OCR support
Retries failed PDFs: CHIVABook1988-1993_2.pdf, Ligation_Knowledgebase_2.pdf, Task_1_Shunt_Classification_Knowledgebase.pdf
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import pdfplumber
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    logger.info("✓ All required libraries available")
except ImportError as e:
    logger.error(f"Missing library: {e}")
    logger.error("Run: python -m pip install pdfplumber pdf2image pytesseract pillow")
    sys.exit(1)

# Failed PDFs to retry
FAILED_PDFS = [
    r'C:\Users\Krish\Downloads\LLM_Finetuning\books_articles\CHIVABook1988-1993_2.pdf',
    r'C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Ligation_Knowledgebase_2.pdf',
    r'C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Task_1_Shunt_Classification_Knowledgebase.pdf',
]

OUTPUT_FILE = r'C:\Users\Krish\Downloads\LLM_Finetuning\full_medical_data.txt'
EXTRACTED_DIR = r'C:\Users\Krish\Downloads\LLM_Finetuning\extracted_missing_pdfs'
Path(EXTRACTED_DIR).mkdir(exist_ok=True)

def extract_with_pdfplumber(pdf_path):
    """Try pdfplumber first"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text and len(text.strip()) > 100:
            return text, "pdfplumber"
    except Exception as e:
        logger.warning(f"  pdfplumber: {str(e)[:80]}")
    return None, None

def extract_with_ocr(pdf_path):
    """Use Tesseract OCR for scanned PDFs"""
    try:
        logger.info(f"  Starting OCR extraction...")

        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=200)
        logger.info(f"  Converted to {len(images)} images")

        text = ""
        for idx, image in enumerate(images):
            logger.info(f"    OCR page {idx + 1}/{len(images)}...", end=" ", flush=True)
            page_text = pytesseract.image_to_string(image, lang='eng')
            text += page_text + "\n"
            logger.info("done")

        if text and len(text.strip()) > 100:
            return text, "OCR"
    except Exception as e:
        logger.error(f"  OCR failed: {str(e)[:100]}")
    return None, None

def extract_pdf(pdf_path):
    """Extract from PDF, trying both methods"""
    logger.info(f"\nProcessing: {pdf_path.name}")

    if not pdf_path.exists():
        logger.error(f"  File not found")
        return None, None

    # Try pdfplumber first
    logger.info(f"  Trying pdfplumber...")
    text, method = extract_with_pdfplumber(pdf_path)
    if text:
        logger.info(f"  ✓ Success ({len(text.split())} words)")
        return text, method
    logger.info("  Failed, trying OCR...")

    # Try OCR
    logger.info(f"  Starting OCR extraction...")
    text, method = extract_with_ocr(pdf_path)
    if text:
        word_count = len(text.split())
        logger.info(f"  ✓ OCR successful: {word_count:,} words extracted")
        return text, method

    logger.error(f"  Both methods failed")
    return None, None

def main():
    logger.info("=" * 80)
    logger.info("EXTRACT REMAINING FAILED PDFs WITH OCR")
    logger.info("=" * 80)

    # Read existing file
    logger.info(f"\nReading existing: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        all_text = f.read()

    original_words = len(all_text.split())
    logger.info(f"Current: {original_words:,} words")

    # Extract missing PDFs
    extracted_count = 0
    for pdf_path in FAILED_PDFS:
        text, method = extract_pdf(Path(pdf_path))

        if text:
            # Save individual extraction
            output_name = Path(pdf_path).stem + ".txt"
            output_path = Path(EXTRACTED_DIR) / output_name
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"  Saved to: {output_path}")

            # Add to combined text
            all_text += "\n\n" + text
            extracted_count += 1

    # Save updated combined file
    logger.info(f"\nSaving updated: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(all_text)

    new_words = len(all_text.split())
    added_words = new_words - original_words

    logger.info(f"✓ Done")
    logger.info(f"  Original: {original_words:,} words")
    logger.info(f"  Added: {added_words:,} words (from {extracted_count} PDFs)")
    logger.info(f"  Total: {new_words:,} words")
    logger.info(f"  File size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")
    logger.info("\n" + "=" * 80)

if __name__ == "__main__":
    main()

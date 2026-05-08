import pdfplumber
import fitz
import json
import re
from pathlib import Path

# The problematic PDFs that didn't extract
problematic_pdfs = {
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Task_1_Shunt_Classification_Knowledgebase.pdf": "Task_1_Shunt_Classification_Knowledgebase.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Ligation_Knowledgebase_2.pdf": "Ligation_Knowledgebase_2.pdf",
}

def try_pdfplumber(pdf_path):
    """Try extraction with pdfplumber."""
    try:
        pages = {}
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages[i + 1] = text
        return pages if pages else None
    except Exception as e:
        print(f"    pdfplumber failed: {e}")
        return None

def try_pymupdf_direct(pdf_path):
    """Try PyMuPDF with different parameters."""
    try:
        doc = fitz.open(pdf_path)
        pages = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Try different text extraction methods
            text = page.get_text("text")

            if not text or len(text.strip()) < 50:
                # Try block-based extraction
                text = page.get_text("blocks")
                if isinstance(text, list):
                    text = "\n".join([block[4] for block in text if isinstance(block, tuple) and len(block) > 4])

            if text and len(text.strip()) >= 50:
                pages[page_num + 1] = text if isinstance(text, str) else str(text)

        doc.close()
        return pages if pages else None
    except Exception as e:
        print(f"    PyMuPDF failed: {e}")
        return None

def extract_clinical_info(text):
    """Extract clinical information from text."""
    examples = []

    # Split into potential examples
    segments = re.split(r'\n\s*(?=Type|CHIVA|Case|Patient|Example|Figure)', text)

    for segment in segments:
        segment = segment.strip()
        if len(segment) < 50:
            continue

        # Check if contains classification
        if any(term in segment.lower() for term in ['type', 'shunt', 'chiva', 'classification']):
            # Extract shunt type
            shunt_type = None
            for pattern, stype in [
                (r'type\s*1\+2|1\+2', "1+2"),
                (r'type\s*5', "5"),
                (r'type\s*4', "4"),
                (r'type\s*3', "3"),
                (r'type\s*2c', "2C"),
                (r'type\s*2b', "2B"),
                (r'type\s*2a', "2A"),
                (r'type\s*2', "2"),
                (r'type\s*1', "1"),
                (r'no\s*shunt', "No shunt"),
            ]:
                if re.search(pattern, segment.lower()):
                    shunt_type = stype
                    break

            examples.append({
                "text": segment[:500],
                "shunt_type": shunt_type,
                "full_text": segment
            })

    return examples

def main():
    print("\nAttempting to extract from problematic PDFs...")
    print("=" * 70)

    all_examples = []

    for pdf_path, doc_name in problematic_pdfs.items():
        if not Path(pdf_path).exists():
            print(f"\nSKIP: {doc_name} (file not found)")
            continue

        print(f"\nProcessing: {doc_name}")
        print(f"  File size: {Path(pdf_path).stat().st_size / (1024*1024):.1f} MB")

        pages = None

        # Try pdfplumber first
        print(f"  Trying pdfplumber...")
        pages = try_pdfplumber(pdf_path)
        if pages:
            print(f"    SUCCESS: {len(pages)} pages extracted")
        else:
            # Try PyMuPDF
            print(f"  Trying PyMuPDF...")
            pages = try_pymupdf_direct(pdf_path)
            if pages:
                print(f"    SUCCESS: {len(pages)} pages extracted")

        if not pages:
            print(f"  FAILED: Could not extract text from this PDF")
            print(f"  This PDF may be image-based, encrypted, or corrupted")
            continue

        # Try to extract examples
        print(f"  Parsing content...")
        examples_found = 0

        for page_num, page_text in pages.items():
            if not isinstance(page_text, str) or len(page_text.strip()) < 50:
                continue

            clinical_examples = extract_clinical_info(page_text)
            for ex in clinical_examples:
                if ex['shunt_type']:
                    all_examples.append({
                        "source_document": doc_name,
                        "page_or_section": f"Page {page_num}",
                        "instruction": ex['text'][:400],
                        "output": ex['full_text'],
                        "shunt_type": ex['shunt_type'],
                        "ligation_strategy": None,
                        "clinical_notes": ""
                    })
                    examples_found += 1

        print(f"  Found {examples_found} clinical examples")

    # Save
    if all_examples:
        output_path = r"C:\Users\Krish\Downloads\LLM_Finetuning\problematic_pdfs_extracted.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_examples, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*70}")
        print(f"Saved {len(all_examples)} examples to {output_path}")
    else:
        print(f"\n{'='*70}")
        print("No examples could be extracted from problematic PDFs")

if __name__ == "__main__":
    main()

import pdf2image
import pytesseract
import json
import re
from pathlib import Path
from PIL import Image
import io

# OCR-based PDFs
ocr_pdfs = {
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Task_1_Shunt_Classification_Knowledgebase.pdf": "Task_1_Shunt_Classification_Knowledgebase.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Ligation_Knowledgebase_2.pdf": "Ligation_Knowledgebase_2.pdf",
}

def extract_with_ocr(pdf_path, max_pages=None):
    """Extract text from PDF using OCR."""
    try:
        print(f"  Converting PDF to images...")
        images = pdf2image.convert_from_path(pdf_path, first_page=1, last_page=max_pages)
        print(f"  Converted {len(images)} pages")

        pages = {}
        for page_num, image in enumerate(images, 1):
            print(f"    OCR processing page {page_num}...", end=" ")
            # OCR the image
            text = pytesseract.image_to_string(image)
            if text and len(text.strip()) > 50:
                pages[page_num] = text
                print(f"OK ({len(text)} chars)")
            else:
                print("EMPTY")

        return pages
    except Exception as e:
        print(f"  OCR failed: {e}")
        return {}

def extract_clinical_examples(text_by_page, doc_name):
    """Extract clinical examples from OCR text."""
    examples = []

    for page_num, page_text in text_by_page.items():
        # Clean up OCR artifacts
        page_text = page_text.replace('|', 'I').replace('\n\n\n', '\n\n')

        # Split into segments
        segments = re.split(r'\n\s*(?=Type|CHIVA|Case|Patient|Example|Figure|\[)', page_text)

        for segment in segments:
            segment = segment.strip()
            if len(segment) < 50:
                continue

            # Extract shunt type
            shunt_type = None
            for pattern, stype in [
                (r'type\s*1\s*\+\s*2|type\s*1\+2', "1+2"),
                (r'type\s*5', "5"),
                (r'type\s*4', "4"),
                (r'type\s*3', "3"),
                (r'type\s*2c', "2C"),
                (r'type\s*2b', "2B"),
                (r'type\s*2a', "2A"),
                (r'type\s*2(?!\s*[a-c])', "2"),
                (r'type\s*1(?!\+)', "1"),
                (r'no\s*shunt', "No shunt"),
            ]:
                if re.search(pattern, segment.lower()):
                    shunt_type = stype
                    break

            # Check for clinical content
            if (shunt_type or any(term in segment.lower() for term in ['flow', 'reflux', 'n1', 'n2', 'n3', 'ep', 'rp'])) and len(segment) > 50:
                examples.append({
                    "source_document": doc_name,
                    "page_or_section": f"Page {page_num}",
                    "instruction": segment[:400],
                    "output": segment,
                    "shunt_type": shunt_type or "Unknown",
                    "ligation_strategy": None,
                    "clinical_notes": ""
                })

    return examples

def main():
    print("\nExtracting from image-based PDFs using OCR...")
    print("="*70)

    all_examples = []
    total_extracted = 0

    for pdf_path, doc_name in ocr_pdfs.items():
        if not Path(pdf_path).exists():
            print(f"\nSKIP: {doc_name} (not found)")
            continue

        print(f"\nProcessing: {doc_name}")

        # Limit to first 30 pages due to OCR time
        text_by_page = extract_with_ocr(pdf_path, max_pages=30)

        if not text_by_page:
            print(f"  No pages successfully extracted with OCR")
            continue

        print(f"  Parsing content...")
        examples = extract_clinical_examples(text_by_page, doc_name)
        all_examples.extend(examples)
        total_extracted += len(examples)
        print(f"  Found {len(examples)} examples")

    # Deduplicate
    unique = []
    seen = set()
    for ex in all_examples:
        key = (ex['source_document'], ex['output'][:100])
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    # Save
    if unique:
        output_path = r"C:\Users\Krish\Downloads\LLM_Finetuning\ocr_extracted_examples.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(unique, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*70}")
        print(f"Extracted {len(unique)} unique examples via OCR")
        print(f"Saved to: {output_path}")
    else:
        print(f"\n{'='*70}")
        print("No examples extracted from image-based PDFs")

if __name__ == "__main__":
    main()

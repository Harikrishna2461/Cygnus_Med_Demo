import fitz  # PyMuPDF
import json
import re
from pathlib import Path
from collections import defaultdict

# Define PDF files to extract from
pdf_files = {
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Task_1_Shunt_Classification_Knowledgebase.pdf": "Task_1_Shunt_Classification_Knowledgebase.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Ligation_Knowledgebase_1.pdf": "Ligation_Knowledgebase_1.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Ligation_Knowledgebase_2.pdf": "Ligation_Knowledgebase_2.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Shunt_Book_8.pdf": "Shunt_Book_8.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Shunt_Classification_Cheetsheat.pdf": "Shunt_Classification_Cheetsheat.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\0-Saphenous-Vein-Sparing-Strategies-in-Chronic-Venous-Disease.pdf": "0-Saphenous-Vein-Sparing-Strategies-in-Chronic-Venous-Disease.pdf",
}

def extract_with_pymupdf(pdf_path):
    """Extract text using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text_by_page = {}
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_by_page[page_num + 1] = text
        doc.close()
        return text_by_page
    except Exception as e:
        print(f"Error with PyMuPDF on {pdf_path}: {e}")
        return {}

def extract_clinical_examples(text_by_page, source_doc):
    """
    Extract clinical examples from PDF text.
    Focus on finding actual case descriptions with classifications.
    """
    examples = []

    for page_num, page_text in text_by_page.items():
        # Skip empty pages
        if not page_text or len(page_text.strip()) < 50:
            continue

        # Split text into potential examples
        # Look for various delimiters that separate cases
        sections = re.split(r'\n\s*(?=(?:Case|Patient|Example|Clinical|Figure|TYPE|CHIVA|Type))', page_text)

        for section in sections:
            section = section.strip()
            if len(section) < 30:
                continue

            # Extract shunt type classification
            shunt_type = extract_shunt_type(section)

            # Only keep sections that have actual classification or detailed flow info
            if shunt_type or has_flow_pattern(section):
                ligation = extract_ligation_strategy(section)
                notes = extract_clinical_notes(section)

                example = {
                    "source_document": source_doc,
                    "page_or_section": f"Page {page_num}",
                    "instruction": section[:400].strip(),
                    "output": section.strip(),
                    "shunt_type": shunt_type or "Unknown",
                    "ligation_strategy": ligation,
                    "clinical_notes": notes
                }
                examples.append(example)

    return examples

def has_flow_pattern(text):
    """Check if text contains flow pattern descriptions (N1, N2, N3, EP, RP)."""
    flow_terms = ['N1', 'N2', 'N3', 'EP', 'RP', 'SFJ', 'flow', 'reflux', 'antegrade', 'retrograde', 'GSV', 'SSV']
    count = sum(1 for term in flow_terms if term.lower() in text.lower())
    return count >= 2

def extract_shunt_type(text):
    """Extract CHIVA shunt type from text."""
    text_lower = text.lower()

    # Order matters - check longer patterns first
    patterns = [
        (r'type\s*1\+2|type\s*1\s*\+\s*2|1\+2|chiva\s*1\+2', "1+2"),
        (r'type\s*5|chiva\s*5', "5"),
        (r'type\s*4|chiva\s*4', "4"),
        (r'type\s*3|chiva\s*3', "3"),
        (r'type\s*2c|2c|chiva\s*2c', "2C"),
        (r'type\s*2b|2b|chiva\s*2b', "2B"),
        (r'type\s*2a|2a|chiva\s*2a', "2A"),
        (r'type\s*2(?!\s*[abc])|chiva\s*2(?!\s*[abc])', "2"),
        (r'type\s*1(?!\+)|chiva\s*1(?!\+)', "1"),
        (r'no\s*shunt|no\s*leak', "No shunt"),
    ]

    for pattern, classification in patterns:
        if re.search(pattern, text_lower):
            return classification

    return None

def extract_ligation_strategy(text):
    """Extract ligation strategy recommendations."""
    text_lower = text.lower()

    ligation_keywords = [
        r'ligate.*?(?:sfj|hunterian|perforator|tributary|ep|endpoint|gsv|ssv)',
        r'(?:sfj|hunterian|perforator|tributary|ep|endpoint|gsv|ssv).*?ligate',
        r'ligation.*?(?:first|then|step|stage)',
        r'chiva\s*(?:1|2|3|4|5).*?(?:ligate|ablate|scleroth)',
        r'treatment.*?(?:ligate|ablate|sclerotherapy|compression)',
    ]

    for pattern in ligation_keywords:
        match = re.search(pattern, text_lower, re.DOTALL)
        if match:
            return match.group(0).strip()[:150]

    return None

def extract_clinical_notes(text):
    """Extract clinical findings, diameter info, outcomes."""
    notes = []

    # Diameter/caliber patterns
    diameter_match = re.search(r'(?:diameter|caliber|calibre|size).*?(?:\d+\.?\d*\s*(?:mm|cm))', text, re.IGNORECASE)
    if diameter_match:
        notes.append(diameter_match.group(0))

    # Flow state patterns
    flow_matches = re.findall(r'(?:reflux|flow|patency|incompetent|competent).*?(?:present|absent|normal|reduced)', text, re.IGNORECASE)
    notes.extend(flow_matches[:2])

    # Patient demographics/findings
    demo_match = re.search(r'(?:patient|age|female|male).*?(?:\d+|years)', text, re.IGNORECASE)
    if demo_match:
        notes.append(demo_match.group(0))

    return " | ".join(notes) if notes else ""

def main():
    all_examples = []
    total_by_doc = defaultdict(int)

    print("Extracting clinical examples from CHIVA PDFs using PyMuPDF...")
    print("=" * 70)

    for pdf_path, short_name in pdf_files.items():
        if not Path(pdf_path).exists():
            print(f"\nWARNING: {short_name} not found")
            continue

        print(f"\nProcessing: {short_name}")

        # Extract text
        text_by_page = extract_with_pymupdf(pdf_path)

        if not text_by_page:
            print(f"  No text extracted from this PDF")
            continue

        print(f"  Extracted {len(text_by_page)} pages")

        # Parse examples
        examples = extract_clinical_examples(text_by_page, short_name)
        all_examples.extend(examples)
        total_by_doc[short_name] = len(examples)

        print(f"  Found {len(examples)} clinical examples")

    # Save to JSON with UTF-8 encoding
    output_path = r"C:\Users\Krish\Downloads\LLM_Finetuning\extracted_chiva_examples_v2.json"

    # Remove duplicates based on output content
    unique_examples = []
    seen = set()
    for ex in all_examples:
        key = (ex['source_document'], ex['output'][:100])
        if key not in seen:
            seen.add(key)
            unique_examples.append(ex)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_examples, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"Total unique examples: {len(unique_examples)}")
    print(f"Saved to: {output_path}")

    print(f"\nExamples by document:")
    for doc, count in sorted(total_by_doc.items(), key=lambda x: x[1], reverse=True):
        print(f"  {doc}: {count}")

    # Print sample examples
    if unique_examples:
        print(f"\n{'=' * 70}")
        print("SAMPLE EXAMPLES:")
        for i, ex in enumerate(unique_examples[:3]):
            print(f"\n--- Example {i+1} ---")
            print(f"Source: {ex['source_document']}")
            print(f"Page: {ex['page_or_section']}")
            print(f"Type: {ex['shunt_type']}")
            print(f"Text: {ex['instruction'][:200]}")

if __name__ == "__main__":
    main()

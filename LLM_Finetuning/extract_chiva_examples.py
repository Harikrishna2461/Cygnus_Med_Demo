import pdfplumber
import json
import re
from pathlib import Path

# Define PDF files to extract from
pdf_files = [
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Task_1_Shunt_Classification_Knowledgebase.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Ligation_Knowledgebase_1.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Ligation_Knowledgebase_2.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Shunt_Book_8.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Shunt_Classification_Cheetsheat.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\0-Saphenous-Vein-Sparing-Strategies-in-Chronic-Venous-Disease.pdf",
]

all_examples = []
example_count = 0

def extract_text_from_pdf(pdf_path):
    """Extract full text from PDF."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    full_text += f"\n[PAGE {page_num + 1}]\n{text}\n"
            return full_text
    except Exception as e:
        print(f"Error extracting from {pdf_path}: {e}")
        return ""

def parse_clinical_examples(text, source_document):
    """
    Parse clinical examples from extracted text.
    Looks for patterns like case numbers, patient descriptions, classifications, etc.
    """
    global example_count
    examples = []

    # Split by page markers to maintain page tracking
    pages = text.split("\n[PAGE ")

    for page_section in pages:
        lines = page_section.split("\n")
        current_page = 1

        # Extract page number if present
        if lines and lines[0].strip().endswith("]"):
            try:
                current_page = int(lines[0].strip().rstrip("]"))
            except:
                pass

        # Join text for pattern matching
        section_text = "\n".join(lines)

        # Look for various patterns that indicate clinical cases
        # Pattern 1: "Case", "Patient", "Example" followed by descriptions
        case_patterns = [
            r"(?:Case|Patient|Example|Clinical\s+case)\s*[#:]*\s*(\d+)?.*?(?=(?:Case|Patient|Example|Clinical\s+case|^[A-Z]{2,}|$))",
        ]

        # Pattern 2: Flow pattern descriptions with classifications
        flow_patterns = [
            r"([A-Z].*?[Ff]low.*?(?:Type|Classification|CHIVA|shunt type|diagnosis).*?(?=\n\n|^[A-Z]{2,}|$))",
        ]

        # Pattern 3: Direct shunt classifications
        classification_patterns = [
            r"((?:Type\s*[1-5]|Type\s*1\+2|No\s*shunt|CHIVA\s*[1-5]).*?(?:(?:Type\s*[1-5]|Type\s*1\+2|No\s*shunt|CHIVA\s*[1-5])|Ligation|Treatment|Management).*?(?=\n\n|^[A-Z]{2,}|$))",
        ]

        # Look for clinical case descriptions
        if any(keyword in section_text.lower() for keyword in ['case', 'patient', 'example', 'clinical', 'presentation', 'diagnosis']):
            # This section likely contains clinical material
            # Extract meaningful chunks

            lines_in_section = section_text.split("\n")
            for i, line in enumerate(lines_in_section):
                # Look for lines that contain classifications or flow descriptions
                if any(term in line.lower() for term in ['type ', 'chiva', 'flow', 'shunt', 'classification', 'ligation', 'patient', 'case']):
                    # Capture context around this line
                    start = max(0, i - 3)
                    end = min(len(lines_in_section), i + 10)
                    context = "\n".join(lines_in_section[start:end]).strip()

                    if len(context) > 50:  # Only if substantial text
                        # Try to extract classification
                        shunt_type = extract_shunt_type(context)
                        ligation_strat = extract_ligation_strategy(context)

                        if shunt_type or ligation_strat:
                            example = {
                                "source_document": source_document,
                                "page_or_section": f"Page {current_page}",
                                "instruction": context[:500].strip(),
                                "output": context.strip(),
                                "shunt_type": shunt_type,
                                "ligation_strategy": ligation_strat,
                                "clinical_notes": extract_clinical_notes(context)
                            }
                            examples.append(example)
                            example_count += 1

    return examples

def extract_shunt_type(text):
    """Extract CHIVA shunt type classification from text."""
    text_lower = text.lower()

    type_mapping = [
        ("type 1+2", "1+2"),
        ("type 1 + 2", "1+2"),
        ("type 1\\+2", "1+2"),
        ("type 1 and 2", "1+2"),
        ("type 1 or 2", "1+2"),
        ("type 1/2", "1+2"),
        ("type 5", "5"),
        ("type 4", "4"),
        ("type 3", "3"),
        ("type 2c", "2C"),
        ("type 2b", "2B"),
        ("type 2a", "2A"),
        ("type 2", "2"),
        ("type 1", "1"),
        ("no shunt", "No shunt"),
        ("chiva 5", "5"),
        ("chiva 4", "4"),
        ("chiva 3", "3"),
        ("chiva 2c", "2C"),
        ("chiva 2b", "2B"),
        ("chiva 2a", "2A"),
        ("chiva 2", "2"),
        ("chiva 1", "1"),
    ]

    for pattern, classification in type_mapping:
        if re.search(pattern, text_lower):
            return classification

    return None

def extract_ligation_strategy(text):
    """Extract ligation strategy recommendations from text."""
    text_lower = text.lower()

    ligation_patterns = [
        r"ligate.*?(?:sfj|hunterian|perforator|tributary|branch|ep|endpoint)",
        r"(?:sfj|hunterian|perforator|tributary|branch|ep|endpoint).*?ligation",
        r"treatment.*?(?:ligate|surgery|ablation)",
        r"management.*?(?:ligate|surgery|ablation)",
        r"ligation strategy.*?(?:[a-z\s]+(?:ligate|step|first|then))+",
    ]

    for pattern in ligation_patterns:
        match = re.search(pattern, text_lower, re.DOTALL)
        if match:
            return match.group(0).strip()

    return None

def extract_clinical_notes(text):
    """Extract clinical findings and notes."""
    notes = []

    # Look for findings, observations, outcomes
    patterns = [
        r"(?:finding|observation|note|outcome|result).*?(?=\n|$)",
        r"(?:diameter|caliber|diameter|size).*?(?:mm|cm)?",
        r"(?:reflux|patency|flow).*?(?:present|absent|reduced|normal)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        notes.extend(matches)

    return " | ".join(list(set(notes))[:3]) if notes else ""

def main():
    """Main extraction function."""
    global all_examples, example_count

    print("Starting CHIVA clinical example extraction...")

    for pdf_path in pdf_files:
        if Path(pdf_path).exists():
            print(f"\nProcessing: {Path(pdf_path).name}")

            # Extract text from PDF
            text = extract_text_from_pdf(pdf_path)

            if text:
                # Parse clinical examples
                examples = parse_clinical_examples(text, Path(pdf_path).name)
                all_examples.extend(examples)
                print(f"  Found {len(examples)} examples")
            else:
                print(f"  No text extracted")
        else:
            print(f"File not found: {pdf_path}")

    print(f"\n\nTotal examples extracted: {example_count}")

    # Save to JSON
    output_path = r"C:\Users\Krish\Downloads\LLM_Finetuning\extracted_chiva_examples.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {output_path}")

    # Print summary statistics
    print("\nExamples by document:")
    doc_counts = {}
    for example in all_examples:
        doc = example['source_document']
        doc_counts[doc] = doc_counts.get(doc, 0) + 1

    for doc, count in sorted(doc_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {doc}: {count}")

if __name__ == "__main__":
    main()

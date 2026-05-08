import fitz  # PyMuPDF
import json
import re
from pathlib import Path
from collections import defaultdict

# Focus on the most productive PDFs
key_pdfs = {
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\0-Saphenous-Vein-Sparing-Strategies-in-Chronic-Venous-Disease.pdf": "Saphenous-Vein-Sparing.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Ligation_Knowledgebase_1.pdf": "Ligation_Knowledgebase_1.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Shunt_Book_8.pdf": "Shunt_Book_8.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Shunt_Classification_Cheetsheat.pdf": "Shunt_Classification_Cheetsheat.pdf",
}

class CHIVACaseExtractor:
    def __init__(self):
        self.cases = []

    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF by page."""
        try:
            doc = fitz.open(pdf_path)
            pages = {}
            for page_num in range(len(doc)):
                text = doc[page_num].get_text()
                if text and text.strip():
                    pages[page_num + 1] = text
            doc.close()
            return pages
        except Exception as e:
            print(f"    Error: {e}")
            return {}

    def identify_case_blocks(self, text):
        """
        Identify distinct clinical case blocks in text.
        Looks for:
        - Type X Shunt patterns
        - Clinical presentations
        - Flow patterns
        - Ligation strategies
        """
        # Look for explicit case markers
        explicit_patterns = [
            r'(?:Type|CHIVA)\s*[1-5].*?(?:(?:Type|CHIVA)\s*[1-5]|patient|case|clinical|figure)',
            r'(?:Case|Patient|Clinical case).*?(?:Type|Classification|diagnosis)',
            r'(?:Ligation|Treatment).*?(?:Type|strategy).*?(?:(?:Type|CHIVA)|ligation)',
        ]

        blocks = []

        # Try explicit patterns first
        for pattern in explicit_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                block = match.group(0).strip()
                if len(block) >= 100:  # Minimum meaningful length
                    blocks.append(block)

        # If few results, split by major delimiters
        if len(blocks) < 5:
            parts = re.split(r'\n(?=[A-Z]{2,}|\d{1,3}\.|•|\[)', text)
            for part in parts:
                part = part.strip()
                if (len(part) >= 100 and
                    any(term in part.lower() for term in ['type', 'chiva', 'shunt', 'ligation', 'flow', 'reflux'])):
                    blocks.append(part)

        return blocks

    def extract_case_info(self, block):
        """Extract structured information from a case block."""
        info = {
            "raw_text": block,
            "classification": None,
            "flow_patterns": [],
            "ligation_info": None,
            "clinical_markers": []
        }

        # Extract classification
        for pattern, classification in [
            (r'type\s*1\s*\+\s*2|1\+2', "1+2"),
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
            if re.search(pattern, block.lower()):
                info["classification"] = classification
                break

        # Extract flow patterns
        flow_patterns = re.findall(
            r'(?:EP|RP|antegrade|retrograde|flow)\s*(?:N[1-3]|GSV|SSV|saphenous|deep)',
            block, re.IGNORECASE
        )
        info["flow_patterns"] = flow_patterns[:5]

        # Extract ligation strategy
        ligation_match = re.search(
            r'(?:ligate|ligation|treatment).*?(?:at|sfj|hunterian|perforator|tributary|first|second)',
            block, re.IGNORECASE | re.DOTALL
        )
        if ligation_match:
            info["ligation_info"] = ligation_match.group(0)[:200]

        # Extract clinical markers (diameter, patient info, etc.)
        markers = re.findall(
            r'(?:diameter|size|caliber|patient|age|female|male|years|mm|cm)\s*(?:\d+\.?\d*)?',
            block, re.IGNORECASE
        )
        info["clinical_markers"] = markers[:3]

        return info

    def format_example(self, case_info, page_num, doc_name):
        """Format case into training example."""
        raw = case_info["raw_text"]

        return {
            "source_document": doc_name,
            "page_or_section": f"Page {page_num}",
            "instruction": raw[:500],
            "output": raw,
            "shunt_type": case_info["classification"] or "Unknown",
            "ligation_strategy": case_info["ligation_info"],
            "clinical_notes": " | ".join(case_info["flow_patterns"][:3]) if case_info["flow_patterns"] else ""
        }

    def process_pdf(self, pdf_path, doc_name):
        """Process a PDF and extract cases."""
        print(f"  Extracting text...")
        pages = self.extract_pdf_text(pdf_path)

        if not pages:
            print(f"    No pages extracted")
            return 0

        print(f"    Found {len(pages)} pages")

        case_count = 0

        for page_num, page_text in pages.items():
            # Identify case blocks in this page
            blocks = self.identify_case_blocks(page_text)

            for block in blocks:
                case_info = self.extract_case_info(block)

                # Only include if has clear classification or detailed flow info
                if case_info["classification"] or (len(case_info["flow_patterns"]) >= 2):
                    example = self.format_example(case_info, page_num, doc_name)
                    self.cases.append(example)
                    case_count += 1

        return case_count

    def save(self, output_path):
        """Save extracted cases."""
        # Deduplicate
        unique = []
        seen = set()

        for case in self.cases:
            key = (case['source_document'], case['output'][:200])
            if key not in seen:
                seen.add(key)
                unique.append(case)

        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(unique, f, indent=2, ensure_ascii=False)

        return len(unique)

def main():
    print("\n" + "="*70)
    print("REFINED CHIVA CLINICAL CASE EXTRACTION")
    print("="*70 + "\n")

    extractor = CHIVACaseExtractor()
    doc_stats = {}

    for pdf_path, doc_name in key_pdfs.items():
        if not Path(pdf_path).exists():
            print(f"SKIP: {doc_name} (not found)")
            continue

        print(f"\nProcessing: {doc_name}")
        count = extractor.process_pdf(pdf_path, doc_name)
        doc_stats[doc_name] = count
        print(f"  Extracted {count} clinical cases")

    # Save
    output_path = r"C:\Users\Krish\Downloads\LLM_Finetuning\chiva_refined_clinical_cases.json"
    total = extractor.save(output_path)

    print(f"\n{'='*70}")
    print(f"REFINED EXTRACTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total unique cases: {total}")

    # Statistics
    type_counts = defaultdict(int)
    for case in extractor.cases:
        stype = case['shunt_type']
        type_counts[stype] += 1

    print(f"\nCases by classification:")
    for stype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {stype}: {count}")

    print(f"\nCases by document:")
    for doc, count in sorted(doc_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {doc}: {count}")

    print(f"\nOutput: {output_path}\n")

    # Sample
    if extractor.cases:
        print(f"{'='*70}")
        print("SAMPLE CLINICAL CASES:")
        print(f"{'='*70}\n")

        for i, case in enumerate(extractor.cases[:5]):
            print(f"--- Case {i+1} ---")
            print(f"Type: {case['shunt_type']}")
            print(f"Source: {case['source_document']}, {case['page_or_section']}")
            if case['ligation_strategy']:
                print(f"Ligation: {case['ligation_strategy']}")
            if case['clinical_notes']:
                print(f"Flow patterns: {case['clinical_notes']}")
            print(f"Text: {case['instruction'][:250]}...\n")

if __name__ == "__main__":
    main()

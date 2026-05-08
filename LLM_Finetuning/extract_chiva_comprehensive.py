import fitz  # PyMuPDF
import json
import re
from pathlib import Path
from collections import defaultdict

# Define PDF files
pdf_files = {
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Task_1_Shunt_Classification_Knowledgebase.pdf": "Task_1_Shunt_Classification_Knowledgebase.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Ligation_Knowledgebase_1.pdf": "Ligation_Knowledgebase_1.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Ligation_Knowledgebase_2.pdf": "Ligation_Knowledgebase_2.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Shunt_Book_8.pdf": "Shunt_Book_8.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\Shunt_Classification_Cheetsheat.pdf": "Shunt_Classification_Cheetsheat.pdf",
    r"C:\Users\Krish\Downloads\LLM_Finetuning\Domain_Specific_Data\0-Saphenous-Vein-Sparing-Strategies-in-Chronic-Venous-Disease.pdf": "0-Saphenous-Vein-Sparing-Strategies.pdf",
}

class CHIVAExampleExtractor:
    def __init__(self):
        self.examples = []
        self.shunt_type_patterns = {
            r'type\s*1\+2|type\s*1\s*\+\s*2|1\+2|chiva.*?1\+2': "1+2",
            r'\btype\s*5\b|\bchiva.*?5\b': "5",
            r'\btype\s*4\b|\bchiva.*?4\b': "4",
            r'\btype\s*3\b|\bchiva.*?3\b': "3",
            r'\btype\s*2[cC]\b|\bchiva.*?2[cC]\b': "2C",
            r'\btype\s*2[bB]\b|\bchiva.*?2[bB]\b': "2B",
            r'\btype\s*2[aA]\b|\bchiva.*?2[aA]\b': "2A",
            r'\btype\s*2(?!\s*[a-c])\b|\bchiva.*?2(?!\s*[a-c])\b': "2",
            r'\btype\s*1(?!\+)\b|\bchiva.*?1(?!\+)\b': "1",
            r'no\s*shunt|no\s*venous': "No shunt",
        }

    def extract_from_pdf(self, pdf_path):
        """Extract text from PDF."""
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
            print(f"  ERROR extracting: {e}")
            return {}

    def extract_shunt_type(self, text):
        """Extract shunt type classification."""
        text_lower = text.lower()
        for pattern, shunt_type in self.shunt_type_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return shunt_type
        return None

    def extract_ligation(self, text):
        """Extract ligation-related content."""
        patterns = [
            r'ligate.*?(?:at|sfj|hunterian|perforator|tributary|branc|flush|below)',
            r'chiva\s*[1-5].*?(?:ligate|perforator|tributary)',
            r'high\s*(?:tie|ligation).*?sfj',
            r'flush\s*ligation.*?tributary',
            r'treatment.*?ligate.*?(?:first|second|step)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0)[:200].strip()
        return None

    def extract_flow_pattern(self, text):
        """Extract flow/hemodynamic information."""
        patterns = [
            r'(?:EP|retrograde|antegrade|reflux).*?(?:N[1-3]|GSV|SSV)',
            r'(?:N[1-3]).*?(?:EP|RP|reflux|flow)',
            r'SFJ.*?(?:incompetent|competent|diameter)',
            r'(?:diameter|caliber|size)\s*(?:\d+\.?\d*\s*(?:mm|cm))',
        ]

        findings = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            findings.extend(matches[:2])
        return " | ".join(findings) if findings else None

    def is_clinical_content(self, text):
        """Check if text contains clinical case information."""
        clinical_keywords = [
            'case', 'patient', 'type', 'shunt', 'chiva', 'ligation',
            'treatment', 'clinical', 'n1', 'n2', 'n3', 'ep', 'rp',
            'sfj', 'flow', 'reflux', 'diameter', 'incompetent'
        ]
        text_lower = text.lower()
        count = sum(1 for kw in clinical_keywords if kw in text_lower)
        return count >= 2 and len(text) >= 50

    def split_into_segments(self, text):
        """Split text into meaningful clinical segments."""
        # Split by common delimiters
        delimiters = [
            r'\n\s*(?=(?:Figure|Case|Patient|Example|Type|CHIVA|•|\d{1,3}\.|\[))',
            r'\n\s*(?=[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*:)',
            r'\n\n+',
        ]

        segments = []
        current = text

        for delimiter in delimiters:
            if re.search(delimiter, current):
                parts = re.split(delimiter, current)
                for part in parts:
                    if part and len(part.strip()) >= 30:
                        segments.append(part.strip())
                return segments

        # Fallback: split into chunks
        lines = text.split('\n')
        chunk = ""
        for line in lines:
            chunk += line + "\n"
            if len(chunk) > 200:
                if chunk.strip():
                    segments.append(chunk.strip())
                chunk = ""
        if chunk.strip():
            segments.append(chunk.strip())

        return segments

    def process_pdf(self, pdf_path, doc_name):
        """Process single PDF and extract examples."""
        print(f"  Extracting text...")
        pages = self.extract_from_pdf(pdf_path)

        if not pages:
            print(f"  No pages extracted")
            return 0

        print(f"  Found {len(pages)} pages, parsing content...")

        examples_in_doc = 0

        for page_num, page_text in pages.items():
            if not page_text.strip():
                continue

            # Split page into segments
            segments = self.split_into_segments(page_text)

            for segment in segments:
                if not self.is_clinical_content(segment):
                    continue

                shunt_type = self.extract_shunt_type(segment)
                flow_pattern = self.extract_flow_pattern(segment)
                ligation = self.extract_ligation(segment)

                # Create example
                example = {
                    "source_document": doc_name,
                    "page_or_section": f"Page {page_num}",
                    "instruction": segment[:400].strip(),
                    "output": segment.strip(),
                    "shunt_type": shunt_type or "Unknown",
                    "ligation_strategy": ligation,
                    "clinical_notes": flow_pattern or ""
                }

                self.examples.append(example)
                examples_in_doc += 1

        return examples_in_doc

    def deduplicate(self):
        """Remove near-duplicate examples."""
        unique = []
        seen = set()

        for ex in self.examples:
            # Create hash from content
            key = (ex['source_document'], ex['output'][:150])
            if key not in seen:
                seen.add(key)
                unique.append(ex)

        self.examples = unique

    def save(self, output_path):
        """Save examples to JSON."""
        self.deduplicate()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.examples, f, indent=2, ensure_ascii=False)
        return len(self.examples)

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE CHIVA CLINICAL EXAMPLE EXTRACTION")
    print("="*70 + "\n")

    extractor = CHIVAExampleExtractor()
    doc_stats = {}

    for pdf_path, doc_name in pdf_files.items():
        if not Path(pdf_path).exists():
            print(f"SKIP: {doc_name} (not found)")
            continue

        print(f"\nProcessing: {doc_name}")
        count = extractor.process_pdf(pdf_path, doc_name)
        doc_stats[doc_name] = count
        print(f"  Extracted {count} examples")

    # Save results
    output_path = r"C:\Users\Krish\Downloads\LLM_Finetuning\chiva_clinical_examples.json"
    total = extractor.save(output_path)

    # Print summary
    print(f"\n{'='*70}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total unique examples: {total}")
    print(f"Output file: {output_path}\n")

    print("Examples per document:")
    for doc, count in sorted(doc_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {doc}: {count}")

    # Show sample
    if extractor.examples:
        print(f"\n{'='*70}")
        print("SAMPLE EXAMPLES:")
        print(f"{'='*70}\n")

        for i, ex in enumerate(extractor.examples[:5]):
            print(f"--- Example {i+1} ---")
            print(f"Source: {ex['source_document']}, {ex['page_or_section']}")
            print(f"Shunt Type: {ex['shunt_type']}")
            if ex['ligation_strategy']:
                print(f"Ligation: {ex['ligation_strategy']}")
            if ex['clinical_notes']:
                print(f"Notes: {ex['clinical_notes']}")
            print(f"Text excerpt:\n{ex['instruction'][:300]}\n")

    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

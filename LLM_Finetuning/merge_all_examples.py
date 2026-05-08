import json
from pathlib import Path
from collections import defaultdict

def merge_json_files():
    """Merge all extracted example JSON files."""

    json_files = [
        r"C:\Users\Krish\Downloads\LLM_Finetuning\chiva_clinical_examples.json",
        r"C:\Users\Krish\Downloads\LLM_Finetuning\ocr_extracted_examples.json",
        r"C:\Users\Krish\Downloads\LLM_Finetuning\extracted_chiva_examples.json",
    ]

    all_examples = []
    doc_count = defaultdict(int)

    for json_file in json_files:
        if not Path(json_file).exists():
            print(f"Waiting for: {json_file}")
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                examples = json.load(f)
                all_examples.extend(examples)
                for ex in examples:
                    doc_count[ex['source_document']] += 1
                print(f"Loaded {len(examples)} from {Path(json_file).name}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    # Deduplicate
    print(f"\nDeduplicating {len(all_examples)} examples...")
    unique_examples = []
    seen = set()

    for ex in all_examples:
        # Create key from content
        key = (ex['source_document'], ex['output'][:150] if isinstance(ex.get('output'), str) else "")
        if key not in seen:
            seen.add(key)
            unique_examples.append(ex)

    # Sort by document name
    unique_examples.sort(key=lambda x: (x['source_document'], x.get('page_or_section', '')))

    # Save comprehensive output
    output_path = r"C:\Users\Krish\Downloads\LLM_Finetuning\CHIVA_Training_Examples_Complete.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "total_examples": len(unique_examples),
                "examples_by_document": dict(doc_count),
                "extraction_date": "2026-05-06",
                "description": "CHIVA clinical training examples extracted from medical PDFs"
            },
            "examples": unique_examples
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Total unique examples: {len(unique_examples)}")
    print(f"Saved to: {output_path}")

    print(f"\nBreakdown by document:")
    for doc, count in sorted(doc_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {doc}: {count}")

    # Print statistics
    type_counts = defaultdict(int)
    for ex in unique_examples:
        stype = ex.get('shunt_type', 'Unknown')
        type_counts[stype] += 1

    print(f"\nBreakdown by shunt type:")
    for stype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {stype}: {count}")

    # Print sample examples
    print(f"\n{'='*70}")
    print("SAMPLE EXAMPLES (first 3):")
    print(f"{'='*70}\n")

    for i, ex in enumerate(unique_examples[:3]):
        print(f"--- Example {i+1} ---")
        print(f"Source: {ex['source_document']}")
        print(f"Page: {ex['page_or_section']}")
        print(f"Shunt Type: {ex['shunt_type']}")
        if ex.get('ligation_strategy'):
            print(f"Ligation: {ex['ligation_strategy'][:100]}")
        if ex.get('clinical_notes'):
            print(f"Notes: {ex['clinical_notes'][:100]}")
        print(f"Text: {ex['instruction'][:200]}...\n")

if __name__ == "__main__":
    merge_json_files()

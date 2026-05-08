import json
from pathlib import Path
from collections import defaultdict
import re

def load_json_safe(path):
    """Load JSON with encoding handling."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, 'r', encoding='utf-16') as f:
            return json.load(f)

def consolidate_examples():
    """Create final consolidated dataset."""

    # Load all sources
    files_to_load = {
        r"C:\Users\Krish\Downloads\LLM_Finetuning\chiva_refined_clinical_cases.json": "refined",
        r"C:\Users\Krish\Downloads\LLM_Finetuning\chiva_clinical_examples.json": "comprehensive",
    }

    all_examples = []

    for filepath, source_type in files_to_load.items():
        if not Path(filepath).exists():
            print(f"Skip: {Path(filepath).name} not found")
            continue

        try:
            data = load_json_safe(filepath)

            # Handle both list and dict formats
            if isinstance(data, dict) and 'examples' in data:
                examples = data['examples']
            else:
                examples = data if isinstance(data, list) else []

            print(f"Loaded {len(examples)} from {source_type}")
            all_examples.extend(examples)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    print(f"\nTotal loaded: {len(all_examples)}")

    # Deduplicate intelligently
    print(f"Deduplicating...")
    unique_examples = []
    seen = set()

    for ex in all_examples:
        # Create unique key from output content (not instruction which is just truncated)
        output = ex.get('output', '')
        if isinstance(output, str):
            key_text = output[:200]
        else:
            key_text = str(output)[:200]

        key = (ex['source_document'], key_text)

        if key not in seen:
            seen.add(key)
            # Clean up the example
            clean_ex = {
                "source_document": ex.get('source_document', ''),
                "page_or_section": ex.get('page_or_section', ''),
                "instruction": ex.get('instruction', '')[:400],
                "output": ex.get('output', ''),
                "shunt_type": ex.get('shunt_type', 'Unknown'),
                "ligation_strategy": ex.get('ligation_strategy'),
                "clinical_notes": ex.get('clinical_notes', '')
            }
            unique_examples.append(clean_ex)

    print(f"Unique examples: {len(unique_examples)}")

    # Analyze
    doc_counts = defaultdict(int)
    type_counts = defaultdict(int)

    for ex in unique_examples:
        doc_counts[ex['source_document']] += 1
        type_counts[ex['shunt_type']] += 1

    # Create final output
    output_data = {
        "metadata": {
            "total_examples": len(unique_examples),
            "examples_by_document": dict(sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)),
            "examples_by_shunt_type": dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True)),
            "extraction_date": "2026-05-06",
            "description": "CHIVA venous shunt classification and ligation strategy training examples extracted from medical literature",
            "classification_types": ["1", "2", "2A", "2B", "2C", "3", "4", "5", "1+2", "No shunt", "Unknown"]
        },
        "examples": unique_examples
    }

    # Save
    output_path = r"C:\Users\Krish\Downloads\LLM_Finetuning\CHIVA_Training_Dataset_Final.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"FINAL DATASET CREATED")
    print(f"{'='*70}")
    print(f"Total examples: {len(unique_examples)}")
    print(f"Output file: {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / (1024*1024):.1f} MB")

    print(f"\nExamples by document:")
    for doc, count in sorted(doc_counts.items(), key=lambda x: x[1], reverse=True):
        short_name = doc.split('/')[-1] if '/' in doc else doc
        print(f"  {short_name}: {count}")

    print(f"\nExamples by shunt type:")
    for stype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {stype}: {count}")

    # Create CSV export for easier viewing
    create_csv_export(unique_examples)

def create_csv_export(examples):
    """Create CSV version for easier viewing."""
    import csv

    csv_path = r"C:\Users\Krish\Downloads\LLM_Finetuning\CHIVA_Training_Dataset_Final.csv"

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['source_document', 'page_or_section', 'shunt_type', 'ligation_strategy', 'clinical_notes', 'instruction_snippet']

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for ex in examples:
                writer.writerow({
                    'source_document': ex['source_document'],
                    'page_or_section': ex['page_or_section'],
                    'shunt_type': ex['shunt_type'],
                    'ligation_strategy': (ex['ligation_strategy'] or '')[:100],
                    'clinical_notes': (ex['clinical_notes'] or '')[:100],
                    'instruction_snippet': ex['instruction'][:200]
                })

        print(f"\nCSV export: {csv_path}")
        print(f"CSV file size: {Path(csv_path).stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"Could not create CSV: {e}")

if __name__ == "__main__":
    consolidate_examples()

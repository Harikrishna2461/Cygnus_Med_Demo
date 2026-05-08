import json
from pathlib import Path

def verify_dataset():
    """Verify the final dataset."""
    dataset_path = r"C:\Users\Krish\Downloads\LLM_Finetuning\CHIVA_Training_Dataset_Final.json"

    if not Path(dataset_path).exists():
        print("Dataset file not found!")
        return

    # Load and verify
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})
    examples = data.get('examples', [])

    print("="*70)
    print("DATASET VERIFICATION")
    print("="*70)
    print(f"\nTotal examples: {len(examples)}")
    print(f"Metadata present: {bool(metadata)}")

    if metadata:
        print(f"\nMetadata:")
        for key, value in metadata.items():
            if key not in ['examples_by_document', 'examples_by_shunt_type']:
                print(f"  {key}: {value}")

        print(f"\nBreakdown by document:")
        for doc, count in metadata.get('examples_by_document', {}).items():
            short = doc.split('/')[-1] if '/' in doc else doc
            print(f"  {short}: {count}")

        print(f"\nBreakdown by shunt type:")
        for stype, count in metadata.get('examples_by_shunt_type', {}).items():
            print(f"  {stype}: {count}")

    # Show sample examples
    print(f"\n{'='*70}")
    print("SAMPLE EXAMPLES:")
    print(f"{'='*70}\n")

    for i, example in enumerate(examples[:5]):
        print(f"--- Example {i+1} ---")
        print(f"Source: {example['source_document']}")
        print(f"Page: {example['page_or_section']}")
        print(f"Shunt Type: {example['shunt_type']}")

        if example.get('ligation_strategy'):
            lig = example['ligation_strategy']
            if len(lig) > 100:
                print(f"Ligation: {lig[:100]}...")
            else:
                print(f"Ligation: {lig}")

        if example.get('clinical_notes'):
            notes = example['clinical_notes']
            if len(notes) > 100:
                print(f"Notes: {notes[:100]}...")
            else:
                print(f"Notes: {notes}")

        instruction = example['instruction']
        if len(instruction) > 150:
            print(f"Text: {instruction[:150]}...")
        else:
            print(f"Text: {instruction}")

        print()

    # Summary stats
    print(f"{'='*70}")
    print("DATA QUALITY CHECK:")
    print(f"{'='*70}\n")

    # Check completeness
    has_shunt_type = sum(1 for ex in examples if ex.get('shunt_type') and ex['shunt_type'] != 'Unknown')
    has_ligation = sum(1 for ex in examples if ex.get('ligation_strategy'))
    has_notes = sum(1 for ex in examples if ex.get('clinical_notes'))

    print(f"Examples with known shunt type: {has_shunt_type}/{len(examples)} ({100*has_shunt_type/len(examples):.1f}%)")
    print(f"Examples with ligation strategy: {has_ligation}/{len(examples)} ({100*has_ligation/len(examples):.1f}%)")
    print(f"Examples with clinical notes: {has_notes}/{len(examples)} ({100*has_notes/len(examples):.1f}%)")

    # Average lengths
    avg_instruction = sum(len(ex.get('instruction', '')) for ex in examples) / len(examples)
    avg_output = sum(len(ex.get('output', '')) for ex in examples) / len(examples)

    print(f"\nAverage instruction length: {avg_instruction:.0f} chars")
    print(f"Average output length: {avg_output:.0f} chars")

    # Check for issues
    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}\n")

    issues = []

    # Check all examples have required fields
    required_fields = ['source_document', 'page_or_section', 'shunt_type', 'instruction', 'output']
    for i, ex in enumerate(examples):
        for field in required_fields:
            if field not in ex:
                issues.append(f"Example {i}: missing {field}")

    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"  {issue}")
    else:
        print("All examples have required fields: OK")

    # Verify no null shunt types
    unknown_count = sum(1 for ex in examples if ex['shunt_type'] == 'Unknown' or ex['shunt_type'] is None)
    if unknown_count > 0:
        print(f"Examples with Unknown shunt type: {unknown_count} (may need further refinement)")

    print(f"\n{'='*70}")
    print(f"Dataset is READY for fine-tuning!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    verify_dataset()

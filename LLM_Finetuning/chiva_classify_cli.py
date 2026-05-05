#!/usr/bin/env python3
"""
CHIVA Shunt Classifier - Command Line Interface
Quick classification from the command line
"""

import json
import sys
import argparse
from pathlib import Path
from chiva_classifier_api import CHIVAShuntClassifier


def parse_clips_json(json_str: str) -> list:
    """Parse JSON string into clips list."""
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            print(f"Error: Expected list or dict, got {type(data)}")
            sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)


def parse_clips_file(filepath: str) -> list:
    """Parse clips from JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "clips" in data:
            return data["clips"]
        else:
            print(f"Error: File should contain list of clips or dict with 'clips' key")
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        sys.exit(1)


def example_types() -> str:
    """Show example clip formats."""
    examples = {
        "Type 1": [
            {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.080},
            {"flow": "RP", "fromType": "N2", "toType": "N1", "posYRatio": 0.300},
        ],
        "Type 2A": [
            {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.200},
            {"flow": "RP", "fromType": "N3", "toType": "N2", "posYRatio": 0.470},
        ],
        "Type 3": [
            {"flow": "EP", "fromType": "N1", "toType": "N2", "posYRatio": 0.050},
            {"flow": "EP", "fromType": "N2", "toType": "N3", "posYRatio": 0.132},
            {"flow": "RP", "fromType": "N3", "toType": "N1", "posYRatio": 0.212},
        ],
    }
    return json.dumps(examples, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="CHIVA Shunt Classifier - Classify venous shunts from ultrasound clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Classify from JSON string
  python chiva_classify_cli.py --json '[{"flow":"EP","fromType":"N1","toType":"N2","posYRatio":0.08},{"flow":"RP","fromType":"N2","toType":"N1","posYRatio":0.3}]'

  # Classify from JSON file
  python chiva_classify_cli.py --file clips.json --leg Left

  # Show example formats
  python chiva_classify_cli.py --examples

  # Use LoRA model
  python chiva_classify_cli.py --json '[...]' --lora

  # Output as JSON
  python chiva_classify_cli.py --json '[...]' --output json
        """
    )

    parser.add_argument(
        "--json", "-j",
        help="Clip data as JSON string"
    )
    parser.add_argument(
        "--file", "-f",
        help="Path to JSON file with clips"
    )
    parser.add_argument(
        "--leg", "-l",
        default="Assessment",
        help="Leg label (Left/Right/Assessment, default: Assessment)"
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use LoRA fine-tuned model"
    )
    parser.add_argument(
        "--examples", "-e",
        action="store_true",
        help="Show example clip formats"
    )
    parser.add_argument(
        "--output", "-o",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )

    args = parser.parse_args()

    # Show examples
    if args.examples:
        print("Example clip formats:\n")
        print(example_types())
        return

    # Get clips
    if args.json:
        clips = parse_clips_json(args.json)
    elif args.file:
        clips = parse_clips_file(args.file)
    else:
        parser.print_help()
        print("\nError: Provide either --json or --file")
        sys.exit(1)

    if not clips:
        print("Error: No clips provided")
        sys.exit(1)

    # Initialize classifier
    print(f"Loading model {'(with LoRA)' if args.lora else '(base model)'}...", file=sys.stderr)
    classifier = CHIVAShuntClassifier(use_lora=args.lora)

    # Classify
    print(f"Classifying {len(clips)} clip(s)...", file=sys.stderr)
    result = classifier.classify(clips, leg_label=args.leg)

    # Output
    if args.output == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"\nLeg: {args.leg}")
        print(f"Type: {result['shunt_type']}")
        print(f"Confidence: {result['confidence']:.1%}")
        if result['status'] == 'success':
            print(f"Reasoning: {result['reasoning']}")
        else:
            print(f"Error: {result['reasoning']}")

        if result['raw_response']:
            print(f"\nRaw response:\n{result['raw_response']}")


if __name__ == "__main__":
    main()

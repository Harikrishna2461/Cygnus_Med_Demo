"""
Synthetic Training Data Generator for Cross-Encoder Fine-tuning
================================================================
Generates query-document pairs from CHIVA ligation knowledge base.
Creates positive pairs (query matches document),
negative pairs (query doesn't match), and hard negatives (related but different).
"""

import json
import sys
from pathlib import Path

# Add parent backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "cmed_demo" / "backend"))

from qdrant_client import QdrantClient
from config import QDRANT_PATH, QDRANT_COLLECTION


# Query templates for synthetic data generation
QUERY_TEMPLATES = {
    "classification": [
        "What is {type} shunt classification?",
        "How to classify {type} venous shunt?",
        "What defines {type} shunt?",
        "Characteristics of {type} classification",
        "{type} shunt: what are the key features?",
        "Diagnostic criteria for {type}",
    ],
    "ligation": [
        "How do I ligate {type} shunt?",
        "{type} ligation technique and approach",
        "What is the ligation strategy for {type}?",
        "Where is the ligation point for {type}?",
        "Ligation management for {type} shunt",
        "{type}: primary and secondary ligation sites",
        "Best approach for {type} ligation",
    ],
    "surgical": [
        "Surgical steps for {type} ligation",
        "{type} ligation: step-by-step procedure",
        "How to perform {type} ligation surgery?",
        "Surgical technique for {type} shunt",
        "{type} ligation: incision and dissection approach",
        "Operating room steps for {type} ligation",
    ],
    "follow_up": [
        "{type} ligation: follow-up management",
        "Post-operative follow-up for {type}",
        "What to expect after {type} ligation?",
        "Long-term management of {type} ligation",
        "Re-assessment after {type} ligation",
    ],
    "anatomy": [
        "Anatomical understanding of {type} shunt",
        "N1, N2, N3 pathways in {type}",
        "Blood flow patterns in {type}",
        "Hemodynamic principles of {type}",
    ],
}

KEYWORDS_BY_TYPE = {
    "Type 1": ["SFJ", "N1->N2", "N2->N1", "groin", "high ligation", "circular reflux"],
    "Type 2A": ["EP N2->N3", "tributary", "GSV to branch", "escape point"],
    "Type 2B": ["perforator", "N2->N2", "SFJ competent", "selective"],
    "Type 2C": ["perforator", "N2->N2", "RP N2->N1", "GSV reflux"],
    "Type 3": ["SFJ incompetent", "N1->N2", "N2->N3", "staged", "tributary"],
    "Type 1+2": ["dual entry", "SFJ incompetent", "tributary involvement", "CHIVA 2"],
}


def extract_type_from_doc(doc_text: str) -> str:
    """Extract shunt type from document text."""
    for shunt_type in KEYWORDS_BY_TYPE.keys():
        if shunt_type in doc_text:
            return shunt_type
    return "Unknown"


def generate_queries_from_doc(doc_text: str, shunt_type: str, num_queries: int = 50) -> list[str]:
    """Generate synthetic queries from a document."""
    queries = []
    templates_to_use = []

    # Select appropriate templates based on document content
    if "ligation" in doc_text.lower() and "surgical" in doc_text.lower():
        templates_to_use = QUERY_TEMPLATES["surgical"] + QUERY_TEMPLATES["ligation"]
    elif "ligation" in doc_text.lower():
        templates_to_use = QUERY_TEMPLATES["ligation"] + QUERY_TEMPLATES["classification"]
    elif "surgical" in doc_text.lower():
        templates_to_use = QUERY_TEMPLATES["surgical"] + QUERY_TEMPLATES["follow_up"]
    else:
        templates_to_use = QUERY_TEMPLATES["classification"] + QUERY_TEMPLATES["anatomy"]

    # Generate queries by filling templates
    for i in range(num_queries):
        template = templates_to_use[i % len(templates_to_use)]
        query = template.format(type=shunt_type)
        queries.append(query)

    return queries


def load_documents_from_qdrant() -> list[dict]:
    """Load all documents from Qdrant collection."""
    client = QdrantClient(path=QDRANT_PATH)
    documents = []
    offset = None

    while True:
        result, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in result:
            text = point.payload.get("text", "")
            if text:
                documents.append({"id": point.id, "text": text})

        if next_offset is None:
            break
        offset = next_offset

    return documents


def generate_training_data(documents: list[dict], output_file: str = "synthetic_training_data.json"):
    """Generate training data with positive and negative pairs."""
    training_data = []

    print("\n[INFO] Generating synthetic training data from {} documents...\n".format(len(documents)))

    # Generate positive and negative pairs
    for idx, doc in enumerate(documents):
        doc_text = doc["text"]
        shunt_type = extract_type_from_doc(doc_text)

        print("Processing Doc {}/{}: {}".format(idx + 1, len(documents), shunt_type))
        print("  Text length: {} chars".format(len(doc_text)))

        # Generate queries for this document
        queries = generate_queries_from_doc(doc_text, shunt_type, num_queries=50)
        print("  Generated {} queries".format(len(queries)))

        # Create positive pairs (query matches this document)
        for query in queries:
            training_data.append(
                {
                    "query": query,
                    "document": doc_text,
                    "relevance_score": 1.0,
                    "label": "positive",
                    "source_doc_id": idx,
                    "shunt_type": shunt_type,
                }
            )

        # Create negative pairs (query from this doc, but match with other docs)
        for other_idx, other_doc in enumerate(documents):
            if other_idx == idx:
                continue

            other_text = other_doc["text"]
            other_type = extract_type_from_doc(other_text)

            # Select 2-3 queries from current doc
            selected_queries = queries[: min(3, len(queries))]

            for query in selected_queries:
                # Hard negative: related type but different approach
                if shunt_type in ["Type 1", "Type 1+2"] and other_type in ["Type 2A", "Type 2B", "Type 2C"]:
                    relevance = 0.3  # Related but different (SFJ vs perforator)
                elif shunt_type in ["Type 2A", "Type 2B", "Type 2C"] and other_type in ["Type 1", "Type 1+2"]:
                    relevance = 0.3  # Related but different
                elif "Type" in shunt_type and "Type" in other_type and shunt_type != other_type:
                    relevance = 0.2  # Different type, lower relevance
                elif "surgical" in doc_text.lower() != "surgical" in other_text.lower():
                    relevance = 0.1  # Different context (classification vs surgical)
                else:
                    relevance = 0.0  # Not relevant

                training_data.append(
                    {
                        "query": query,
                        "document": other_text,
                        "relevance_score": relevance,
                        "label": "negative" if relevance < 0.5 else "hard_negative",
                        "source_doc_id": idx,
                        "target_doc_id": other_idx,
                        "shunt_type": shunt_type,
                        "target_shunt_type": other_type,
                    }
                )

        print()

    # Split into train, validation, test
    total = len(training_data)
    train_split = int(0.7 * total)
    val_split = int(0.85 * total)

    train_data = training_data[:train_split]
    val_data = training_data[train_split:val_split]
    test_data = training_data[val_split:]

    # Save all data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "total_samples": len(training_data),
                    "train_size": len(train_data),
                    "val_size": len(val_data),
                    "test_size": len(test_data),
                    "num_source_documents": len(documents),
                },
                "train": train_data,
                "val": val_data,
                "test": test_data,
            },
            f,
            indent=2,
        )

    print("=" * 80)
    print("[OK] Training data saved to: {}".format(output_path))
    print("=" * 80)
    print("\n[STATS] Data Statistics:")
    print("  Total samples: {}".format(len(training_data)))
    print("  Training set: {} ({:.1f}%)".format(len(train_data), len(train_data) / total * 100))
    print("  Validation set: {} ({:.1f}%)".format(len(val_data), len(val_data) / total * 100))
    print("  Test set: {} ({:.1f}%)".format(len(test_data), len(test_data) / total * 100))
    print("\n  Label distribution:")

    label_counts = {}
    for item in training_data:
        label = item["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    for label, count in sorted(label_counts.items()):
        print("    {}: {} ({:.1f}%)".format(label, count, count / total * 100))

    print("\n  Relevance score distribution:")
    scores = [item["relevance_score"] for item in training_data]
    print("    0.0 (not relevant): {}".format(sum(1 for s in scores if s == 0.0)))
    print("    0.2 (low relevance): {}".format(sum(1 for s in scores if s == 0.2)))
    print("    0.3 (hard negative): {}".format(sum(1 for s in scores if s == 0.3)))
    print("    1.0 (positive): {}".format(sum(1 for s in scores if s == 1.0)))

    return output_path


if __name__ == "__main__":
    # Load documents from Qdrant
    documents = load_documents_from_qdrant()
    print("\n[OK] Loaded {} documents from Qdrant\n".format(len(documents)))

    # Generate training data
    output_file = Path(__file__).parent / "synthetic_training_data.json"
    generate_training_data(documents, str(output_file))

    print("\n[OK] Data generation complete!")
    print("   Ready for fine-tuning notebook\n")

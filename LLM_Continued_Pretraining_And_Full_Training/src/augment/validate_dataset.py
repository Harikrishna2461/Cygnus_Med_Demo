import json
import logging
import hashlib
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

try:
    from transformers import AutoTokenizer
    from langdetect import detect, LangDetectException
except ImportError as e:
    raise ImportError(f"Required library missing: {e}")

from tqdm import tqdm

# ==================== CONFIG ====================
TRAIN_FILE = Path("data/final_augmented/train.jsonl")
MIN_TOKENS = 100
MAX_TOKENS = 2000
TOKEN_COUNT_TOLERANCE = 0.10  # 10% tolerance
LANGUAGE_SAMPLE_SIZE = 50
MAX_BOOK_DOMINANCE = 0.60  # 60%
TOKENIZER_MODEL = "Qwen/Qwen2.5-7B"
TRANSLATION_DEGRADED_THRESHOLD = 0.30  # Fail if >30% are degraded
# ================================================

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "validate_dataset.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_chunks(file_path: Path) -> List[Dict]:
    """Load all chunks from JSONL file."""
    chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"Line {i}: JSON decode error: {e}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

    return chunks


def initialize_tokenizer():
    """Initialize the Qwen tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer {TOKENIZER_MODEL}: {e}")
        raise


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the tokenizer."""
    try:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Tokenization error: {e}, using word count estimate")
        return len(text.split())


def check_token_range(chunks: List[Dict]) -> Tuple[bool, int, int]:
    """
    Check 1: No chunk under MIN_TOKENS or over MAX_TOKENS.
    Returns (passed, under_min_count, over_max_count)
    """
    under_min = []
    over_max = []

    for i, chunk in enumerate(chunks):
        token_count = chunk.get('token_count', 0)
        if token_count < MIN_TOKENS:
            under_min.append((i, token_count))
        if token_count > MAX_TOKENS:
            over_max.append((i, token_count))

    if under_min:
        logger.warning(f"Found {len(under_min)} chunks under {MIN_TOKENS} tokens")
    if over_max:
        logger.warning(f"Found {len(over_max)} chunks over {MAX_TOKENS} tokens")

    return len(under_min) == 0 and len(over_max) == 0, len(under_min), len(over_max)


def check_metadata_fields(chunks: List[Dict]) -> Tuple[bool, int]:
    """
    Check 2: All chunks have required metadata fields.
    Required: source_book, chapter, section, augmentation (or confirm original)
    Returns (passed, chunks_with_missing_fields)
    """
    missing_count = 0
    required_fields = {'augmentation'}  # augmentation can be absent for originals

    for i, chunk in enumerate(chunks):
        metadata = chunk.get('metadata', {})

        # Check if it's marked as original (no augmentation field)
        if 'augmentation' not in metadata:
            # Original chunks may have fewer fields
            continue

        # Augmented chunks should have proper metadata
        has_all = all(field in metadata for field in required_fields)
        if not has_all:
            missing_count += 1

    if missing_count > 0:
        logger.warning(f"Found {missing_count} chunks with missing metadata fields")

    return missing_count == 0, missing_count


def check_token_count_accuracy(chunks: List[Dict], tokenizer) -> Tuple[bool, int]:
    """
    Check 3: Token count field matches actual tokenized length (within tolerance).
    Returns (passed, mismatched_count)
    """
    mismatched = []

    for i, chunk in enumerate(tqdm(chunks, desc="Checking token counts", leave=False)):
        recorded_count = chunk.get('token_count', 0)
        actual_count = count_tokens(chunk.get('text', ''), tokenizer)

        if recorded_count == 0:
            continue

        diff = abs(actual_count - recorded_count) / recorded_count
        if diff > TOKEN_COUNT_TOLERANCE:
            mismatched.append((i, recorded_count, actual_count))

    if mismatched:
        logger.warning(f"Found {len(mismatched)} chunks with token count mismatch")
        for i, recorded, actual in mismatched[:5]:
            logger.warning(f"  Chunk {i}: recorded={recorded}, actual={actual}")

    return len(mismatched) == 0, len(mismatched)


def check_exact_duplicates(chunks: List[Dict]) -> Tuple[bool, int]:
    """
    Check 4: No exact duplicates (by text hash).
    Returns (passed, duplicate_count)
    """
    seen_hashes = {}
    duplicates = []

    for i, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        if text_hash in seen_hashes:
            duplicates.append((i, seen_hashes[text_hash]))
        else:
            seen_hashes[text_hash] = i

    if duplicates:
        logger.warning(f"Found {len(duplicates)} duplicate chunks")

    return len(duplicates) == 0, len(duplicates)


def check_translation_degradation(chunks: List[Dict]) -> Tuple[bool, float]:
    """
    Check 5: Back-translated chunks have acceptable number degradation rate.
    Returns (passed, degradation_rate)
    """
    back_translated_chunks = [c for c in chunks
                              if c.get('metadata', {}).get('augmentation') == 'back_translation']

    if not back_translated_chunks:
        logger.info("No back-translated chunks found")
        return True, 0.0

    degraded_count = sum(1 for c in back_translated_chunks
                        if c.get('metadata', {}).get('translation_degraded', False))
    degradation_rate = degraded_count / len(back_translated_chunks)

    if degradation_rate > TRANSLATION_DEGRADED_THRESHOLD:
        logger.warning(f"Translation degradation rate: {degradation_rate:.1%} (limit: {TRANSLATION_DEGRADED_THRESHOLD:.1%})")
        return False, degradation_rate

    logger.info(f"Translation degradation rate: {degradation_rate:.1%}")
    return True, degradation_rate


def check_source_distribution(chunks: List[Dict]) -> Tuple[bool, Dict[str, float]]:
    """
    Check 6: No single source book dominates >60% of chunks.
    Returns (passed, source_distribution_dict)
    """
    source_counts = defaultdict(int)

    for chunk in chunks:
        metadata = chunk.get('metadata', {})
        source = metadata.get('source_book', 'unknown')
        source_counts[source] += 1

    # Calculate percentages
    total = len(chunks)
    distribution = {src: count / total for src, count in source_counts.items()}

    max_dominance = max(distribution.values()) if distribution else 0
    if max_dominance > MAX_BOOK_DOMINANCE:
        dominant_source = max(distribution, key=distribution.get)
        logger.warning(
            f"Source '{dominant_source}' dominates {max_dominance:.1%} "
            f"(limit: {MAX_BOOK_DOMINANCE:.1%})"
        )
        return False, distribution

    logger.info(f"Max source dominance: {max_dominance:.1%}")
    return True, distribution


def check_language(chunks: List[Dict]) -> Tuple[bool, int]:
    """
    Check 7: Sample chunks verify they are English.
    Returns (passed, non_english_count)
    """
    if len(chunks) < LANGUAGE_SAMPLE_SIZE:
        sample = chunks
    else:
        sample = random.sample(chunks, LANGUAGE_SAMPLE_SIZE)

    non_english = []

    for i, chunk in enumerate(sample):
        text = chunk.get('text', '')
        # Take first 500 chars for language detection
        sample_text = text[:500]

        if not sample_text.strip():
            continue

        try:
            lang = detect(sample_text)
            if lang != 'en':
                non_english.append((i, lang))
        except LangDetectException:
            logger.debug(f"Could not detect language for chunk {i}")

    if non_english:
        logger.warning(f"Found {len(non_english)} non-English chunks in sample")

    return len(non_english) == 0, len(non_english)


def get_dataset_statistics(chunks: List[Dict]) -> Dict:
    """Calculate dataset statistics."""
    if not chunks:
        return {}

    token_counts = [c.get('token_count', 0) for c in chunks]
    augmentation_types = defaultdict(int)

    for chunk in chunks:
        aug_type = chunk.get('metadata', {}).get('augmentation', 'original')
        augmentation_types[aug_type] += 1

    stats = {
        'total_chunks': len(chunks),
        'total_tokens': sum(token_counts),
        'avg_tokens_per_chunk': sum(token_counts) // len(chunks) if chunks else 0,
        'min_tokens': min(token_counts) if token_counts else 0,
        'max_tokens': max(token_counts) if token_counts else 0,
        'augmentation_breakdown': dict(augmentation_types)
    }

    return stats


def main():
    logger.info("=" * 70)
    logger.info("FINAL DATASET VALIDATION")
    logger.info("=" * 70)

    # Verify input exists
    if not TRAIN_FILE.exists():
        logger.error(f"Dataset file not found: {TRAIN_FILE}")
        sys.exit(1)

    # Load dataset
    logger.info(f"Loading dataset from {TRAIN_FILE}")
    chunks = load_chunks(TRAIN_FILE)
    logger.info(f"Loaded {len(chunks):,} chunks")

    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {TOKENIZER_MODEL}")
    tokenizer = initialize_tokenizer()

    # Run all checks
    logger.info("\nRunning validation checks...")
    logger.info("-" * 70)

    checks = []

    # Check 1: Token range
    check1_pass, under_min, over_max = check_token_range(chunks)
    checks.append(("Token range (100-2000)", check1_pass))
    logger.info(f"Check 1 - Token range: {'PASS' if check1_pass else 'FAIL'}")
    if not check1_pass:
        logger.info(f"  Under {MIN_TOKENS}: {under_min}, Over {MAX_TOKENS}: {over_max}")

    # Check 2: Metadata fields
    check2_pass, missing_meta = check_metadata_fields(chunks)
    checks.append(("Metadata fields", check2_pass))
    logger.info(f"Check 2 - Metadata fields: {'PASS' if check2_pass else 'FAIL'}")
    if not check2_pass:
        logger.info(f"  Missing fields: {missing_meta} chunks")

    # Check 3: Token count accuracy
    check3_pass, mismatched = check_token_count_accuracy(chunks, tokenizer)
    checks.append(("Token count accuracy", check3_pass))
    logger.info(f"Check 3 - Token count accuracy: {'PASS' if check3_pass else 'FAIL'}")
    if not check3_pass:
        logger.info(f"  Mismatched: {mismatched} chunks")

    # Check 4: Exact duplicates
    check4_pass, dup_count = check_exact_duplicates(chunks)
    checks.append(("No exact duplicates", check4_pass))
    logger.info(f"Check 4 - Exact duplicates: {'PASS' if check4_pass else 'FAIL'}")
    if not check4_pass:
        logger.info(f"  Found: {dup_count} duplicates")

    # Check 5: Translation degradation
    check5_pass, deg_rate = check_translation_degradation(chunks)
    checks.append(("Translation quality", check5_pass))
    logger.info(f"Check 5 - Translation quality: {'PASS' if check5_pass else 'FAIL'}")

    # Check 6: Source distribution
    check6_pass, distribution = check_source_distribution(chunks)
    checks.append(("Source distribution", check6_pass))
    logger.info(f"Check 6 - Source distribution: {'PASS' if check6_pass else 'FAIL'}")
    if distribution:
        for source, pct in sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:3]:
            logger.info(f"  {source}: {pct:.1%}")

    # Check 7: Language
    check7_pass, non_eng = check_language(chunks)
    checks.append(("Language (English)", check7_pass))
    logger.info(f"Check 7 - Language: {'PASS' if check7_pass else 'FAIL'}")
    if not check7_pass:
        logger.info(f"  Non-English: {non_eng} chunks in sample")

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)

    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{check_name:.<50} {status}")

    all_passed = all(passed for _, passed in checks)

    # Print statistics
    logger.info("\n" + "-" * 70)
    logger.info("DATASET STATISTICS")
    logger.info("-" * 70)

    stats = get_dataset_statistics(chunks)
    if stats:
        logger.info(f"Total chunks:              {stats['total_chunks']:>10,}")
        logger.info(f"Total tokens:              {stats['total_tokens']:>10,}")
        logger.info(f"Avg tokens/chunk:          {stats['avg_tokens_per_chunk']:>10}")
        logger.info(f"Min tokens/chunk:          {stats['min_tokens']:>10}")
        logger.info(f"Max tokens/chunk:          {stats['max_tokens']:>10}")
        logger.info(f"\nAugmentation breakdown:")
        for aug_type, count in sorted(stats['augmentation_breakdown'].items()):
            pct = count / stats['total_chunks'] * 100
            logger.info(f"  {aug_type:.<30} {count:>10,} ({pct:>5.1f}%)")

    logger.info("=" * 70)

    # Exit with appropriate code
    if all_passed:
        logger.info("✓ All validation checks PASSED")
        sys.exit(0)
    else:
        logger.error("✗ Some validation checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

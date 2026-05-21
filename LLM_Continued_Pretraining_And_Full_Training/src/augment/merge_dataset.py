import json
import logging
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

from tqdm import tqdm

# ==================== CONFIG ====================
ORIGINAL_TRAIN = Path("data/final/train.jsonl")
ORIGINAL_EVAL = Path("data/final/eval.jsonl")
BACK_TRANSLATED = Path("data/augmented/back_translated.jsonl")
KEY_POINTS = Path("data/augmented/key_points.jsonl")
OUTPUT_TRAIN = Path("data/final_augmented/train.jsonl")
OUTPUT_EVAL = Path("data/final_augmented/eval.jsonl")
DEDUP_THRESHOLD = 0.85  # MinHash similarity threshold
MIN_CHUNK_TOKENS = 100
RANDOM_SEED = 42
NUM_HASH_FUNCS = 128
# ================================================

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "merge_dataset.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load chunks from JSONL file."""
    chunks = []
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return chunks

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"{file_path} Line {i}: JSON decode error: {e}")
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")

    return chunks


def shingle_text(text: str, k: int = 5) -> Set[str]:
    """
    Create k-shingles from text for MinHash.
    Shingles are contiguous substrings of k words.
    """
    words = text.lower().split()
    shingles = set()

    for i in range(len(words) - k + 1):
        shingle = ' '.join(words[i:i + k])
        shingles.add(shingle)

    if len(words) < k:
        shingles.add(' '.join(words))

    return shingles


def hash_shingle(shingle: str, seed: int) -> int:
    """Hash a shingle with a given seed."""
    h = hashlib.md5((shingle + str(seed)).encode())
    return int(h.hexdigest(), 16)


def minhash_signature(text: str, num_hashes: int = NUM_HASH_FUNCS) -> List[int]:
    """
    Generate MinHash signature for text.
    Returns list of minimum hash values for each hash function.
    """
    shingles = shingle_text(text)

    if not shingles:
        return [float('inf')] * num_hashes

    signature = []
    for seed in range(num_hashes):
        min_hash = min(hash_shingle(shingle, seed) for shingle in shingles)
        signature.append(min_hash)

    return signature


def jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
    """
    Estimate Jaccard similarity from MinHash signatures.
    Returns similarity between 0 and 1.
    """
    if not sig1 or not sig2:
        return 0.0

    matching = sum(1 for h1, h2 in zip(sig1, sig2) if h1 == h2)
    return matching / len(sig1)


def deduplicate_chunks(chunks: List[Dict],
                       threshold: float = DEDUP_THRESHOLD) -> Tuple[List[Dict], int]:
    """
    Remove near-duplicate chunks using MinHash.
    Returns (deduplicated_chunks, num_removed).
    """
    logger.info(f"Computing MinHash signatures for {len(chunks)} chunks...")
    signatures = []
    for chunk in tqdm(chunks, desc="Computing signatures"):
        text = chunk.get('text', '')
        sig = minhash_signature(text)
        signatures.append(sig)

    logger.info("Identifying duplicates...")
    kept_indices = set()
    removed_count = 0

    for i in tqdm(range(len(chunks)), desc="Deduplicating"):
        if i in kept_indices:
            continue

        kept_indices.add(i)

        for j in range(i + 1, len(chunks)):
            if j in kept_indices:
                continue

            similarity = jaccard_similarity(signatures[i], signatures[j])
            if similarity >= threshold:
                kept_indices.add(j)  # Mark as duplicate (keep first occurrence)
                removed_count += 1

    # Return unique chunks
    deduplicated = [chunks[i] for i in sorted(kept_indices)]
    return deduplicated, len(chunks) - len(deduplicated)


def filter_short_chunks(chunks: List[Dict],
                       min_tokens: int = MIN_CHUNK_TOKENS) -> Tuple[List[Dict], int]:
    """Remove chunks with fewer than min_tokens."""
    filtered = []
    removed_count = 0

    for chunk in chunks:
        token_count = chunk.get('token_count', 0)
        if token_count >= min_tokens:
            filtered.append(chunk)
        else:
            removed_count += 1

    return filtered, removed_count


def main():
    logger.info("=" * 60)
    logger.info("Starting dataset merge and deduplication")
    logger.info("=" * 60)

    # Load all datasets
    logger.info("Loading datasets...")
    original_train = load_jsonl(ORIGINAL_TRAIN)
    original_eval = load_jsonl(ORIGINAL_EVAL)
    back_translated = load_jsonl(BACK_TRANSLATED)
    key_points = load_jsonl(KEY_POINTS)

    logger.info(f"  Original train:   {len(original_train)} chunks")
    logger.info(f"  Original eval:    {len(original_eval)} chunks")
    logger.info(f"  Back-translated:  {len(back_translated)} chunks")
    logger.info(f"  Key points:       {len(key_points)} chunks")

    # Prepare eval dataset (unchanged)
    output_eval_dir = OUTPUT_EVAL.parent
    output_eval_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval dataset to {OUTPUT_EVAL}")
    with open(OUTPUT_EVAL, 'w', encoding='utf-8') as f:
        for chunk in original_eval:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    # Combine train datasets
    combined_train = original_train + back_translated + key_points
    logger.info(f"Combined train size: {len(combined_train)} chunks")

    # Filter short chunks
    logger.info(f"Filtering chunks under {MIN_CHUNK_TOKENS} tokens...")
    filtered_train, short_removed = filter_short_chunks(combined_train, MIN_CHUNK_TOKENS)
    logger.info(f"  Removed {short_removed} short chunks")

    # Deduplicate
    logger.info(f"Running MinHash deduplication (threshold={DEDUP_THRESHOLD})...")
    deduplicated_train, dups_removed = deduplicate_chunks(filtered_train, DEDUP_THRESHOLD)
    logger.info(f"  Removed {dups_removed} duplicate chunks")

    # Shuffle with fixed seed
    logger.info(f"Shuffling with seed {RANDOM_SEED}...")
    random.seed(RANDOM_SEED)
    random.shuffle(deduplicated_train)

    # Calculate statistics
    original_tokens = sum(c.get('token_count', 0) for c in original_train)
    final_tokens = sum(c.get('token_count', 0) for c in deduplicated_train)
    increase_pct = ((final_tokens - original_tokens) / original_tokens * 100) if original_tokens > 0 else 0

    # Save final training dataset
    output_train_dir = OUTPUT_TRAIN.parent
    output_train_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving final training dataset to {OUTPUT_TRAIN}")
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        for chunk in deduplicated_train:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    # Print final statistics
    logger.info("=" * 60)
    logger.info("DATASET MERGE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Original train chunks:       {len(original_train):,}")
    logger.info(f"After back-translation:      {len(combined_train):,}")
    logger.info(f"After key points:            {len(combined_train):,}")
    logger.info(f"After filtering (<100 tok):  {len(filtered_train):,}")
    logger.info(f"After deduplication:         {len(deduplicated_train):,}")
    logger.info(f"")
    logger.info(f"Final train chunks:          {len(deduplicated_train):,}")
    logger.info(f"Final eval chunks:           {len(original_eval):,}")
    logger.info(f"")
    logger.info(f"Original train tokens:       {original_tokens:,}")
    logger.info(f"Final train tokens:          {final_tokens:,}")
    logger.info(f"Dataset size increase:       {increase_pct:+.1f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

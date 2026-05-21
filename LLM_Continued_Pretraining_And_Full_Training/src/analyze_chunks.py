"""
analyze_chunks.py
=================
Loads all chunks from data/chunks/all_chunks.jsonl, analyses their quality,
removes near-duplicates and outliers, then writes:

  data/final/train.jsonl  (95 % of books)
  data/final/eval.jsonl   (5 % of books – held-out books, not random rows)

Analysis plots saved to data/analysis/.

Quality filters applied:
  1.  Near-duplicate removal via MinHash LSH (Jaccard threshold = 0.85)
  2.  Chunks that are mostly numbers / billing codes
  3.  Perplexity outlier removal:
       - Extremely HIGH perplexity  → garbled OCR text
       - Extremely LOW  perplexity  → repetitive boilerplate (headers repeated)
  The perplexity proxy is character-level bigram entropy (fast, no model needed).
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

# ── Configuration constants ───────────────────────────────────────────────────
CHUNKS_DIR       = Path(__file__).parent.parent / "data" / "chunks"
FINAL_DIR        = Path(__file__).parent.parent / "data" / "final"
ANALYSIS_DIR     = Path(__file__).parent.parent / "data" / "analysis"
LOG_DIR          = Path(__file__).parent.parent / "logs"
LOG_FILE         = LOG_DIR / "analyze_chunks.log"

ALL_CHUNKS_FILE  = CHUNKS_DIR / "all_chunks.jsonl"

# Near-duplicate detection
MINHASH_NUM_PERM       = 128          # permutations for MinHash
MINHASH_THRESHOLD      = 0.85         # Jaccard similarity threshold
MINHASH_NGRAM_SIZE     = 3            # character n-gram size for shingles

# Perplexity-proxy thresholds (character-bigram entropy)
ENTROPY_LOW_PERCENTILE  =  2          # % – remove bottom  2 % (boilerplate)
ENTROPY_HIGH_PERCENTILE = 98          # % – remove top     2 % (garbled OCR)

# Train / eval split – one held-out book per ~20 books  (≥5% eval)
EVAL_FRACTION            = 0.05       # target fraction

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(str(LOG_FILE), level="DEBUG", rotation="10 MB",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD CHUNKS
# ═══════════════════════════════════════════════════════════════════════════════

def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        logger.error(f"Chunks file not found: {path}. Run chunk_documents.py first.")
        sys.exit(1)
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info(f"Loaded {len(chunks):,} chunks from {path}")
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# PERPLEXITY PROXY  –  character-level bigram entropy
# ═══════════════════════════════════════════════════════════════════════════════

def _char_bigram_entropy(text: str) -> float:
    """
    Compute character-level bigram entropy as a perplexity proxy.
    Low entropy  → highly repetitive text (boilerplate).
    High entropy → very random text (garbled OCR / gibberish).
    """
    text = text.lower()
    bigrams = [text[i:i + 2] for i in range(len(text) - 1)]
    if not bigrams:
        return 0.0
    counts = Counter(bigrams)
    total  = len(bigrams)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return entropy


def compute_entropies(chunks: list[dict]) -> list[float]:
    from tqdm import tqdm
    return [
        _char_bigram_entropy(c["text"])
        for c in tqdm(chunks, desc="Computing entropy", unit="chunk")
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# VOCABULARY RICHNESS
# ═══════════════════════════════════════════════════════════════════════════════

def _vocab_richness(text: str) -> float:
    """Type-token ratio: unique_words / total_words."""
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


# ═══════════════════════════════════════════════════════════════════════════════
# NEAR-DUPLICATE REMOVAL  –  MinHash LSH
# ═══════════════════════════════════════════════════════════════════════════════

def _get_shingles(text: str, n: int = MINHASH_NGRAM_SIZE) -> set[str]:
    """Character n-gram shingles for MinHash."""
    text = re.sub(r"\s+", " ", text.lower())
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def remove_near_duplicates(chunks: list[dict]) -> tuple[list[dict], int]:
    """
    Remove near-duplicate chunks using MinHash LSH.
    Returns (deduplicated_chunks, num_removed).
    """
    from datasketch import MinHash, MinHashLSH
    from tqdm import tqdm

    logger.info(f"Building MinHash LSH with threshold={MINHASH_THRESHOLD}, "
                f"num_perm={MINHASH_NUM_PERM}…")

    lsh     = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    minhashes: list[MinHash] = []

    for i, chunk in enumerate(tqdm(chunks, desc="Building MinHashes", unit="chunk")):
        m = MinHash(num_perm=MINHASH_NUM_PERM)
        for shingle in _get_shingles(chunk["text"]):
            m.update(shingle.encode("utf-8"))
        minhashes.append(m)
        try:
            lsh.insert(str(i), m)
        except ValueError:
            # Duplicate key (shouldn't happen but guard anyway)
            pass

    # Find duplicates: for each chunk, query LSH and mark later ones as removed
    keep   = [True] * len(chunks)
    removed = 0

    for i in tqdm(range(len(chunks)), desc="Deduplicating", unit="chunk"):
        if not keep[i]:
            continue
        results = lsh.query(minhashes[i])
        for result_key in results:
            j = int(result_key)
            if j != i and keep[j]:
                keep[j] = False
                removed += 1

    deduped = [c for c, k in zip(chunks, keep) if k]
    logger.info(f"Removed {removed:,} near-duplicate chunks ({removed/len(chunks)*100:.1f}%)")
    return deduped, removed


# ═══════════════════════════════════════════════════════════════════════════════
# NUMERIC / BILLING CODE FILTER
# ═══════════════════════════════════════════════════════════════════════════════

_BILLING_CODE_PAT = re.compile(
    r"(?:ICD[-\s]?\d{1,2}|CPT|DRG|NDC|HCPCS|[A-Z]\d{2}\.?\d{0,2})"
)

def _is_billing_code_chunk(text: str) -> bool:
    """True if chunk is dominated by medical billing / classification codes."""
    billing_hits = len(_BILLING_CODE_PAT.findall(text))
    words        = len(text.split())
    if words < 10:
        return False
    # > 15 % of words are billing codes
    return billing_hits / words > 0.15


def filter_numeric_chunks(
    chunks: list[dict],
    entropies: list[float] | None = None,
) -> tuple[list[dict], list[float] | None, int]:
    """
    Remove billing-code / numeric chunks.
    When entropies is provided, keeps it in sync so the two lists stay aligned.
    """
    before   = len(chunks)
    if entropies is not None:
        pairs   = [(c, e) for c, e in zip(chunks, entropies)
                   if not _is_billing_code_chunk(c["text"])]
        cleaned = [p[0] for p in pairs]
        ents    = [p[1] for p in pairs]
    else:
        cleaned = [c for c in chunks if not _is_billing_code_chunk(c["text"])]
        ents    = None
    removed = before - len(cleaned)
    if removed:
        logger.info(f"Removed {removed:,} billing-code / numeric chunks")
    return cleaned, ents, removed


# ═══════════════════════════════════════════════════════════════════════════════
# ENTROPY OUTLIER FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def filter_entropy_outliers(
    chunks: list[dict],
    entropies: list[float],
) -> tuple[list[dict], list[float], int]:
    import numpy as np

    arr     = np.array(entropies)
    lo_cut  = float(np.percentile(arr, ENTROPY_LOW_PERCENTILE))
    hi_cut  = float(np.percentile(arr, ENTROPY_HIGH_PERCENTILE))

    logger.info(f"Entropy range: [{arr.min():.3f}, {arr.max():.3f}]")
    logger.info(f"Keeping chunks with entropy in [{lo_cut:.3f}, {hi_cut:.3f}]")

    kept_chunks    = []
    kept_entropies = []
    removed        = 0

    for chunk, ent in zip(chunks, entropies):
        if lo_cut <= ent <= hi_cut:
            kept_chunks.append(chunk)
            kept_entropies.append(ent)
        else:
            removed += 1

    logger.info(f"Removed {removed:,} entropy-outlier chunks ({removed/(len(chunks) or 1)*100:.1f}%)")
    return kept_chunks, kept_entropies, removed


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL SPLIT  –  by book
# ═══════════════════════════════════════════════════════════════════════════════

def split_train_eval(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Allocate roughly EVAL_FRACTION of *books* (not chunks) to eval.
    This ensures eval contains unseen books.
    """
    from collections import defaultdict

    book_to_chunks: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        book_to_chunks[c["metadata"]["source_book"]].append(c)

    books = sorted(book_to_chunks.keys())
    n_eval_books = max(1, round(len(books) * EVAL_FRACTION))
    # Pick the last N books alphabetically as eval (deterministic)
    eval_books  = set(books[-n_eval_books:])
    train_books = set(books) - eval_books

    logger.info(f"Train books ({len(train_books)}): {sorted(train_books)}")
    logger.info(f"Eval books  ({len(eval_books)}):  {sorted(eval_books)}")

    train_chunks = [c for c in chunks if c["metadata"]["source_book"] in train_books]
    eval_chunks  = [c for c in chunks if c["metadata"]["source_book"] in eval_books]

    return train_chunks, eval_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def save_plots(chunks: list[dict], entropies: list[float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from collections import Counter

    sns.set_theme(style="whitegrid", context="talk")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    token_counts = [c["token_count"] for c in chunks]
    book_names   = [c["metadata"]["source_book"] for c in chunks]
    vocab_scores = [_vocab_richness(c["text"]) for c in chunks]

    # ── 1. Token length distribution ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(token_counts, bins=50, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(token_counts), color="#F44336", linestyle="--",
               label=f"Mean = {np.mean(token_counts):.0f}")
    ax.axvline(np.median(token_counts), color="#4CAF50", linestyle="--",
               label=f"Median = {np.median(token_counts):.0f}")
    ax.set_xlabel("Token count per chunk")
    ax.set_ylabel("Frequency")
    ax.set_title("Token Length Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "token_length_dist.png", dpi=150)
    plt.close()

    # ── 2. Per-book chunk counts ──────────────────────────────────────────────
    book_counts  = Counter(book_names)
    books_sorted = sorted(book_counts.keys(), key=lambda b: book_counts[b], reverse=True)
    counts       = [book_counts[b] for b in books_sorted]
    short_names  = [b[:30] for b in books_sorted]

    fig, ax = plt.subplots(figsize=(14, max(5, len(books_sorted) * 0.5)))
    bars = ax.barh(range(len(books_sorted)), counts, color="#7E57C2")
    ax.set_yticks(range(len(books_sorted)))
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Number of chunks")
    ax.set_title("Chunks per Book")
    for bar, count in zip(bars, counts):
        ax.text(count + 1, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "chunks_per_book.png", dpi=150)
    plt.close()

    # ── 3. Vocabulary richness distribution ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vocab_scores, bins=40, color="#26A69A", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(vocab_scores), color="#F44336", linestyle="--",
               label=f"Mean = {np.mean(vocab_scores):.3f}")
    ax.set_xlabel("Vocabulary richness (unique/total tokens)")
    ax.set_ylabel("Frequency")
    ax.set_title("Vocabulary Richness per Chunk")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "vocab_richness.png", dpi=150)
    plt.close()

    # ── 4. Entropy distribution ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(entropies, bins=50, color="#FF7043", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Character-bigram entropy")
    ax.set_ylabel("Frequency")
    ax.set_title("Entropy Distribution (perplexity proxy)")
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "entropy_distribution.png", dpi=150)
    plt.close()

    logger.info(f"Saved 4 analysis plots to {ANALYSIS_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
# WRITE JSONL
# ═══════════════════════════════════════════════════════════════════════════════

def write_jsonl(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(chunks):,} chunks to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    from tqdm import tqdm
    import numpy as np

    # ── 1. Load ───────────────────────────────────────────────────────────────
    chunks = load_chunks(ALL_CHUNKS_FILE)
    original_count = len(chunks)

    # ── 2. Compute entropies for all chunks BEFORE filtering ──────────────────
    entropies = compute_entropies(chunks)

    # ── 3. Save pre-filter plots ──────────────────────────────────────────────
    logger.info("Saving pre-filter analysis plots…")
    save_plots(chunks, entropies)

    # ── 4. Entropy outlier filter ─────────────────────────────────────────────
    chunks, entropies, n_entropy = filter_entropy_outliers(chunks, entropies)

    # ── 5. Billing-code / numeric filter ─────────────────────────────────────
    chunks, entropies, n_numeric = filter_numeric_chunks(chunks, entropies)

    # ── 6. Near-duplicate removal ─────────────────────────────────────────────
    chunks, n_dedup = remove_near_duplicates(chunks)

    # ── 7. Train / eval split by book ────────────────────────────────────────
    train_chunks, eval_chunks = split_train_eval(chunks)

    # ── 8. Write final datasets ───────────────────────────────────────────────
    write_jsonl(train_chunks, FINAL_DIR / "train.jsonl")
    write_jsonl(eval_chunks,  FINAL_DIR / "eval.jsonl")

    # ── 9. Summary ────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("ANALYSIS & FILTERING COMPLETE")
    logger.info(f"  Original chunks    : {original_count:,}")
    logger.info(f"  Entropy outliers   : -{n_entropy:,}")
    logger.info(f"  Numeric/code chunks: -{n_numeric:,}")
    logger.info(f"  Near-duplicates    : -{n_dedup:,}")
    logger.info(f"  Final chunks       : {len(chunks):,}")
    logger.info(f"  Train chunks       : {len(train_chunks):,}")
    logger.info(f"  Eval  chunks       : {len(eval_chunks):,}")
    logger.info(f"  Eval fraction      : {len(eval_chunks)/len(chunks)*100:.1f}%")
    logger.info("=" * 65)

    all_toks  = [c["token_count"] for c in chunks]
    print(f"\nFinal dataset statistics:")
    print(f"  Total chunks : {len(chunks):,}")
    print(f"  Avg tokens   : {np.mean(all_toks):.1f}")
    print(f"  Median tokens: {np.median(all_toks):.1f}")
    print(f"  Total tokens : {sum(all_toks):,}")
    print(f"  Train / Eval : {len(train_chunks):,} / {len(eval_chunks):,}")
    print(f"\nPlots saved to: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()

"""
chunk_documents.py
==================
Reads all extracted .txt files from data/raw/ and splits them into
semantically coherent chunks suitable for continued pre-training.

Chunking priority:
  1. Chapter boundaries (never merge across chapters)
  2. Section boundaries (prefer to break here)
  3. Paragraph boundaries (double newlines)
  4. Sentence boundaries (NLTK punkt tokeniser) – last resort
  5. Hard token limit (1024 tokens, 128-token overlap window)

Each chunk is formatted as:
  ### Medical Reference
  Source: {book_name} | Chapter: {chapter} | Section: {section}

  {chunk_text}

Filters applied:
  - Discard chunks < 100 tokens
  - Discard chunks with > 30 % non-alphabetic characters
  - Chapter-summary / key-point sections are duplicated (upweighted)

Output:
  data/chunks/<book_stem>.jsonl  (per-book, for debugging)
  data/chunks/all_chunks.jsonl   (merged)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Generator

# ── Configuration constants ───────────────────────────────────────────────────
RAW_DIR      = Path(__file__).parent.parent / "data" / "raw"
CHUNKS_DIR   = Path(__file__).parent.parent / "data" / "chunks"
LOG_DIR      = Path(__file__).parent.parent / "logs"
LOG_FILE     = LOG_DIR / "chunk_documents.log"

# Tokeniser model (must match pre-training model)
TOKENIZER_MODEL     = "Qwen/Qwen2.5-7B-Instruct"

# Chunking parameters
TARGET_TOKENS       = 1024
OVERLAP_TOKENS      = 128
MIN_CHUNK_TOKENS    = 100
MAX_NON_ALPHA_RATIO = 0.30   # chunks with > 30 % non-alpha chars are discarded

# High-value section keywords – these chunks get duplicated in the output
HIGH_VALUE_PATTERNS = re.compile(
    r"(key\s+point|summary|clinical\s+pearl|important\s+note|take[\s-]?away"
    r"|learning\s+objective|clinical\s+tip|remember|must\s+know)",
    re.IGNORECASE,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(str(LOG_FILE), level="DEBUG", rotation="10 MB",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENISER
# ═══════════════════════════════════════════════════════════════════════════════

class TokenCounter:
    """Wraps a HuggingFace tokenizer for token counting."""

    def __init__(self, model_name: str = TOKENIZER_MODEL):
        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        if text in self.cache:
            return self.cache[text]
        n = len(self.tokenizer.encode(text, add_special_tokens=False))
        if len(self.cache) < 50_000:
            self.cache[text] = n
        return n

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    text:        str
    source_book: str
    chapter:     str
    section:     str
    position:    int   # chunk index within the book (0-based)
    token_count: int
    high_value:  bool  # True if the chunk is from a summary/key-points section


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT PARSER  –  splits raw .txt into (chapter, section, paragraphs)
# ═══════════════════════════════════════════════════════════════════════════════

_CHAPTER_MARKER = re.compile(r"^\[CHAPTER:\s*(.*?)\]$")
_SECTION_MARKER = re.compile(r"^\[SECTION:\s*(.*?)\]$")


def parse_document(text: str) -> Generator[tuple[str, str, str], None, None]:
    """
    Yields (chapter, section, paragraph_text) tuples.
    Paragraphs are separated by blank lines within a section.
    """
    current_chapter = "Preamble"
    current_section = "General"
    buffer: list[str] = []

    def flush():
        nonlocal buffer
        para = " ".join(buffer).strip()
        if para:
            yield (current_chapter, current_section, para)
        buffer = []

    lines = text.split("\n")
    for line in lines:
        ch_match = _CHAPTER_MARKER.match(line.strip())
        sc_match = _SECTION_MARKER.match(line.strip())

        if ch_match:
            yield from flush()
            current_chapter = ch_match.group(1).strip() or "Unknown Chapter"
            current_section = "General"
        elif sc_match:
            yield from flush()
            current_section = sc_match.group(1).strip() or "General"
        elif line.strip() == "":
            yield from flush()
        else:
            buffer.append(line.strip())

    yield from flush()


# ═══════════════════════════════════════════════════════════════════════════════
# SENTENCE SPLITTER (NLTK)
# ═══════════════════════════════════════════════════════════════════════════════

_SENT_TOKENIZER = None

def _get_sent_tokenizer():
    global _SENT_TOKENIZER
    if _SENT_TOKENIZER is None:
        import nltk
        try:
            _SENT_TOKENIZER = nltk.data.load("tokenizers/punkt/english.pickle")
        except LookupError:
            logger.info("Downloading NLTK punkt tokeniser…")
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            _SENT_TOKENIZER = nltk.data.load("tokenizers/punkt/english.pickle")
    return _SENT_TOKENIZER


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences without cutting medical abbreviations badly."""
    tokenizer = _get_sent_tokenizer()
    return tokenizer.tokenize(text)


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

def _non_alpha_ratio(text: str) -> float:
    """Fraction of characters that are NOT letters or whitespace."""
    chars = re.sub(r"\s", "", text)
    if not chars:
        return 1.0
    non_alpha = sum(1 for c in chars if not c.isalpha())
    return non_alpha / len(chars)


def _is_valid_chunk(text: str, token_count: int) -> bool:
    """Return True if the chunk passes quality filters."""
    if token_count < MIN_CHUNK_TOKENS:
        return False
    if _non_alpha_ratio(text) > MAX_NON_ALPHA_RATIO:
        return False
    # Reject pure-numeric / code blocks (billing codes, lab reference ranges)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    numeric_lines = sum(
        1 for l in lines
        if re.fullmatch(r"[\d\s.,;:\-/\\|()[\]{}%+<>=*#@!?'\"~^&_]{3,}", l)
    )
    if lines and numeric_lines / len(lines) > 0.50:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNK FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

def _format_chunk(text: str, source_book: str, chapter: str, section: str) -> str:
    """Prepend the required header to a chunk."""
    book_display = source_book.replace("_", " ")
    header = (
        f"### Medical Reference\n"
        f"Source: {book_display} | Chapter: {chapter} | Section: {section}\n\n"
    )
    return header + text


# ═══════════════════════════════════════════════════════════════════════════════
# CORE CHUNKER  –  token-aware sliding window
# ═══════════════════════════════════════════════════════════════════════════════

def _chunk_paragraph(
    para: str,
    counter: TokenCounter,
    chapter: str,
    section: str,
    source_book: str,
    position_start: int,
) -> list[Chunk]:
    """
    Splits a single paragraph into ≤ TARGET_TOKENS chunks with OVERLAP_TOKENS
    overlap, always cutting at sentence boundaries.

    The window is tracked as a list of (sent_text, sent_token_ids) pairs so
    the overlap look-back always uses the correct token ids for the sentences
    that are actually in the current window (not the wrong positional slice of
    the full sentence list, which is a common off-by-one bug).
    """
    sentences = _split_sentences(para)
    if not sentences:
        return []

    # Pre-encode every sentence once
    encoded: list[tuple[str, list[int]]] = [
        (s, counter.encode(s)) for s in sentences
    ]

    chunks: list[Chunk] = []
    # window = list of (sent_text, sent_token_ids) for the current sliding window
    window: list[tuple[str, list[int]]] = []
    window_len = 0          # total token count in current window
    pos = position_start

    def _flush_window() -> None:
        nonlocal window, window_len, pos
        if not window:
            return
        chunk_text = " ".join(s for s, _ in window).strip()
        tok_count  = window_len
        is_hv      = bool(HIGH_VALUE_PATTERNS.search(
            chapter + " " + section + " " + chunk_text
        ))
        formatted = _format_chunk(chunk_text, source_book, chapter, section)
        final_tok = counter.count(formatted)
        if _is_valid_chunk(chunk_text, tok_count):
            chunks.append(Chunk(
                text=formatted,
                source_book=source_book,
                chapter=chapter,
                section=section,
                position=pos,
                token_count=final_tok,
                high_value=is_hv,
            ))
            pos += 1

        # Build overlap: walk the window backwards, collecting sentences until
        # we reach OVERLAP_TOKENS.  These are the actual token ids for the
        # sentences in this window, so the overlap is always accurate.
        overlap: list[tuple[str, list[int]]] = []
        overlap_len = 0
        for sent_text, sent_ids in reversed(window):
            if overlap_len + len(sent_ids) <= OVERLAP_TOKENS:
                overlap.insert(0, (sent_text, sent_ids))
                overlap_len += len(sent_ids)
            else:
                break
        window     = overlap
        window_len = overlap_len

    for sent_text, sent_ids in encoded:
        # Overflow: flush the current window before adding this sentence
        if window and window_len + len(sent_ids) > TARGET_TOKENS:
            _flush_window()

        window.append((sent_text, sent_ids))
        window_len += len(sent_ids)

    # Flush the final (possibly partial) window
    _flush_window()

    return chunks


def chunk_document(text: str, source_book: str, counter: TokenCounter) -> list[Chunk]:
    """
    Full chunking pipeline for one book.
    Groups paragraphs by section and ensures we never split mid-sentence.
    """
    all_chunks: list[Chunk] = []
    pos = 0

    # Group paragraphs into sections to respect semantic boundaries
    current_chapter = ""
    current_section = ""
    section_paras:  list[str] = []

    def flush_section(ch: str, sc: str, paras: list[str]) -> list[Chunk]:
        nonlocal pos
        sc_chunks = []
        # Concatenate small paragraphs to approach TARGET_TOKENS before splitting
        merged_paras: list[str] = []
        running      = ""
        running_toks = 0

        for para in paras:
            tok = counter.count(para)
            if running_toks + tok <= TARGET_TOKENS * 0.85 and running:
                running += "\n\n" + para
                running_toks += tok
            else:
                if running:
                    merged_paras.append(running)
                running      = para
                running_toks = tok
        if running:
            merged_paras.append(running)

        for merged_para in merged_paras:
            new_chunks = _chunk_paragraph(merged_para, counter, ch, sc, source_book, pos)
            pos += len(new_chunks)
            sc_chunks.extend(new_chunks)

        return sc_chunks

    for ch, sc, para in parse_document(text):
        if (ch != current_chapter or sc != current_section):
            if section_paras:
                all_chunks.extend(flush_section(current_chapter, current_section, section_paras))
                section_paras = []
            current_chapter = ch
            current_section = sc
        section_paras.append(para)

    if section_paras:
        all_chunks.extend(flush_section(current_chapter, current_section, section_paras))

    # Duplicate high-value chunks (chapter summaries, key points)
    hv_chunks = [c for c in all_chunks if c.high_value]
    if hv_chunks:
        logger.info(f"  {source_book}: duplicating {len(hv_chunks)} high-value chunks")
        # Give them a new position beyond the last
        next_pos = max(c.position for c in all_chunks) + 1
        for hvc in hv_chunks:
            dup = Chunk(
                text=hvc.text,
                source_book=hvc.source_book,
                chapter=hvc.chapter,
                section=hvc.section,
                position=next_pos,
                token_count=hvc.token_count,
                high_value=True,
            )
            all_chunks.append(dup)
            next_pos += 1

    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# JSONL WRITER
# ═══════════════════════════════════════════════════════════════════════════════

def _write_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "text":        chunk.text,
                "token_count": chunk.token_count,
                "metadata": {
                    "source_book": chunk.source_book,
                    "chapter":     chunk.chapter,
                    "section":     chunk.section,
                    "position":    chunk.position,
                    "high_value":  chunk.high_value,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    from tqdm import tqdm
    import statistics

    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        logger.error(f"No .txt files found in {RAW_DIR}. Run extract_books.py first.")
        sys.exit(1)

    logger.info(f"Found {len(raw_files)} raw text file(s)")
    logger.info(f"Loading tokenizer (this may take a moment on first run)…")

    counter = TokenCounter()

    all_chunks:  list[Chunk] = []
    per_book_stats: dict[str, dict] = {}

    for txt_path in tqdm(raw_files, desc="Chunking books", unit="book"):
        logger.info(f"Processing: {txt_path.name}")
        text        = txt_path.read_text(encoding="utf-8")
        source_book = txt_path.stem

        try:
            book_chunks = chunk_document(text, source_book, counter)
        except Exception as exc:
            logger.error(f"Chunking failed for {txt_path.name}: {exc}")
            continue

        # Per-book JSONL
        book_out = CHUNKS_DIR / f"{source_book}.jsonl"
        _write_jsonl(book_chunks, book_out)

        token_counts = [c.token_count for c in book_chunks]
        per_book_stats[source_book] = {
            "num_chunks":  len(book_chunks),
            "avg_tokens":  round(statistics.mean(token_counts), 1) if token_counts else 0,
            "min_tokens":  min(token_counts) if token_counts else 0,
            "max_tokens":  max(token_counts) if token_counts else 0,
        }
        logger.info(
            f"  {source_book}: {len(book_chunks)} chunks | "
            f"avg_tokens={per_book_stats[source_book]['avg_tokens']}"
        )
        all_chunks.extend(book_chunks)

    if not all_chunks:
        logger.error("No chunks produced. Check raw text files.")
        sys.exit(1)

    # Write merged file
    merged_out = CHUNKS_DIR / "all_chunks.jsonl"
    _write_jsonl(all_chunks, merged_out)

    # ── Stats ─────────────────────────────────────────────────────────────────
    all_token_counts = [c.token_count for c in all_chunks]
    logger.info("=" * 65)
    logger.info("CHUNKING COMPLETE")
    logger.info(f"  Total chunks   : {len(all_chunks):,}")
    logger.info(f"  Avg token count: {statistics.mean(all_token_counts):.1f}")
    logger.info(f"  Median tokens  : {statistics.median(all_token_counts):.1f}")
    logger.info(f"  Min / Max      : {min(all_token_counts)} / {max(all_token_counts)}")
    logger.info(f"  High-value dup : {sum(1 for c in all_chunks if c.high_value)}")
    logger.info(f"  Output         : {merged_out}")
    logger.info("=" * 65)

    print(f"\n{'Book':<50} {'Chunks':>7} {'AvgTok':>8} {'Min':>6} {'Max':>6}")
    print("-" * 82)
    for book, s in per_book_stats.items():
        print(f"{book[:50]:<50} {s['num_chunks']:>7} {s['avg_tokens']:>8} "
              f"{s['min_tokens']:>6} {s['max_tokens']:>6}")


if __name__ == "__main__":
    main()

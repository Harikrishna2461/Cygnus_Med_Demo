import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

from deep_translator import GoogleTranslator
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("transformers library required for tokenization")

# ==================== CONFIG ====================
SOURCE_LANG = "en"
BRIDGE_LANG = "de"
INPUT_FILE = Path("data/final/train.jsonl")
OUTPUT_FILE = Path("data/augmented/back_translated.jsonl")
BATCH_SIZE = 5
SLEEP_BETWEEN_BATCHES = 1.0  # seconds
MAX_CHUNK_LENGTH = 4000  # characters
TOKENIZER_MODEL = "Qwen/Qwen2.5-7B"
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds
# ================================================

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "back_translate.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def split_header_and_body(text: str) -> Tuple[str, str]:
    """
    Extract header and body from chunk text.
    Header is "### Medical Reference\nSource: ..." until first double newline.
    """
    lines = text.split('\n')
    header_end = 0

    for i in range(len(lines) - 1):
        if lines[i] == '' and lines[i + 1] != '':
            header_end = i + 1
            break

    if header_end == 0:
        return text, ""

    header = '\n'.join(lines[:header_end])
    body = '\n'.join(lines[header_end:])
    return header, body


def extract_numbers(text: str) -> set:
    """Extract all numbers from text using regex."""
    pattern = r'\d+\.?\d*'
    numbers = set(re.findall(pattern, text))
    return numbers


def translate_with_backoff(text: str, from_lang: str, to_lang: str,
                          chunk_id: str, attempt: int = 0) -> Optional[str]:
    """
    Translate text with exponential backoff for rate limiting.
    Returns None if all retries exhausted.
    """
    if attempt >= MAX_RETRIES:
        logger.error(f"Chunk {chunk_id}: Max retries ({MAX_RETRIES}) exhausted")
        return None

    try:
        translator = GoogleTranslator(source_language=from_lang,
                                     target_language=to_lang)
        result = translator.translate(text)
        return result

    except Exception as e:
        if "429" in str(e) or "rate" in str(e).lower():
            backoff_time = INITIAL_BACKOFF * (2 ** attempt)
            logger.warning(
                f"Chunk {chunk_id}: Rate limit (attempt {attempt + 1}/{MAX_RETRIES}), "
                f"backing off {backoff_time:.1f}s"
            )
            time.sleep(backoff_time)
            return translate_with_backoff(text, from_lang, to_lang,
                                         chunk_id, attempt + 1)
        else:
            logger.error(f"Chunk {chunk_id}: Translation error: {e}")
            return None


def split_long_text(text: str, max_length: int) -> List[str]:
    """Split text into chunks not exceeding max_length characters."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    current_chunk = ""

    sentences = text.split('. ')
    for i, sentence in enumerate(sentences):
        sentence = sentence if i == len(sentences) - 1 else sentence + '. '

        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def validate_number_preservation(original: str, translated: str) -> bool:
    """
    Check if numbers survived translation.
    Returns True if at least 80% of numbers are preserved.
    """
    original_nums = extract_numbers(original)
    translated_nums = extract_numbers(translated)

    if not original_nums:
        return True  # No numbers to check

    preserved = len(original_nums & translated_nums)
    preservation_rate = preserved / len(original_nums)

    return preservation_rate >= 0.8


def load_chunks(input_file: Path) -> List[Dict]:
    """Load all chunks from JSONL file."""
    chunks = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"Line {i}: JSON decode error: {e}")
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
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


def back_translate_chunk(chunk: Dict, chunk_idx: int, tokenizer) -> Tuple[Optional[Dict], bool]:
    """
    Back-translate a single chunk. Returns (translated_chunk, success).
    success=False means chunk should be skipped.
    """
    try:
        text = chunk.get('text', '')
        header, body = split_header_and_body(text)

        if not body.strip():
            logger.warning(f"Chunk {chunk_idx}: No body text found")
            return None, False

        # Split long texts
        body_chunks = split_long_text(body, MAX_CHUNK_LENGTH)
        translated_parts = []
        degraded = False

        for part in body_chunks:
            # English -> German
            de_text = translate_with_backoff(part, SOURCE_LANG, BRIDGE_LANG,
                                            f"{chunk_idx}")
            if de_text is None:
                return None, False

            # German -> English
            en_text = translate_with_backoff(de_text, BRIDGE_LANG, SOURCE_LANG,
                                            f"{chunk_idx}")
            if en_text is None:
                return None, False

            # Check number preservation
            if not validate_number_preservation(part, en_text):
                degraded = True
                logger.warning(f"Chunk {chunk_idx}: Number degradation detected")

            translated_parts.append(en_text)
            time.sleep(0.1)  # Small delay between parts

        translated_body = ' '.join(translated_parts)
        new_text = header + '\n' + translated_body

        # Prepare output chunk
        new_chunk = chunk.copy()
        new_chunk['text'] = new_text
        new_chunk['token_count'] = count_tokens(new_text, tokenizer)

        if 'metadata' not in new_chunk:
            new_chunk['metadata'] = {}
        new_chunk['metadata']['augmentation'] = 'back_translation'
        if degraded:
            new_chunk['metadata']['translation_degraded'] = True

        return new_chunk, True

    except Exception as e:
        logger.error(f"Chunk {chunk_idx}: Unexpected error: {e}")
        return None, False


def main():
    logger.info("=" * 60)
    logger.info("Starting back-translation augmentation pipeline")
    logger.info("=" * 60)

    # Verify input exists
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        return

    # Create output directory
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {TOKENIZER_MODEL}")
    tokenizer = initialize_tokenizer()

    # Load chunks
    logger.info(f"Loading chunks from {INPUT_FILE}")
    chunks = load_chunks(INPUT_FILE)
    logger.info(f"Loaded {len(chunks)} chunks")

    # Process chunks
    translated_chunks = []
    failed_chunks = []
    degraded_count = 0
    total_tokens = 0

    logger.info("Starting back-translation process...")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Processing batches"):
        batch = chunks[i:i + BATCH_SIZE]

        for j, chunk in enumerate(batch):
            chunk_idx = i + j
            new_chunk, success = back_translate_chunk(chunk, chunk_idx, tokenizer)

            if success:
                translated_chunks.append(new_chunk)
                total_tokens += new_chunk['token_count']
                if new_chunk.get('metadata', {}).get('translation_degraded', False):
                    degraded_count += 1
            else:
                failed_chunks.append(chunk_idx)

        if i + BATCH_SIZE < len(chunks):
            time.sleep(SLEEP_BETWEEN_BATCHES)

    # Save results
    logger.info(f"Saving {len(translated_chunks)} chunks to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for chunk in translated_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    # Print statistics
    logger.info("=" * 60)
    logger.info("BACK-TRANSLATION PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total chunks attempted:    {len(chunks)}")
    logger.info(f"Successfully translated:   {len(translated_chunks)}")
    logger.info(f"Failed chunks:             {len(failed_chunks)}")
    logger.info(f"Chunks with degradation:   {degraded_count}")
    logger.info(f"Total output tokens:       {total_tokens:,}")
    logger.info(f"Average tokens per chunk:  {total_tokens // len(translated_chunks) if translated_chunks else 0}")
    logger.info("=" * 60)

    if failed_chunks:
        logger.info(f"Failed chunk indices: {failed_chunks[:10]}")


if __name__ == "__main__":
    main()

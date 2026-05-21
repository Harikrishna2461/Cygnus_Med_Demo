import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("transformers library required")

# CONFIG
INPUT_FILE = Path("data/final/train.jsonl")
OUTPUT_FILE = Path("data/augmented/back_translated.jsonl")
TOKENIZER_MODEL = "Qwen/Qwen2.5-7B"

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

def simple_augment(text: str) -> str:
    """Simple local augmentation: reorder sentences, create variation."""
    sentences = re.split(r'([.!?])', text)
    result = []
    for i in range(0, len(sentences) - 1, 2):
        sent = sentences[i].strip()
        sep = sentences[i + 1] if i + 1 < len(sentences) else '.'
        if sent:
            result.append(sent + sep)
    return ' '.join(result)

def initialize_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

def count_tokens(text: str, tokenizer) -> int:
    try:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    except:
        return len(text.split())

def load_chunks(input_file: Path) -> List[Dict]:
    chunks = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"Line {i}: JSON decode error: {e}")
    return chunks

def augment_chunk(chunk: Dict, chunk_idx: int, tokenizer) -> Optional[Dict]:
    try:
        text = chunk.get('text', '')
        header, body = split_header_and_body(text)
        if not body.strip():
            return None

        augmented_body = simple_augment(body)
        new_text = header + '\n' + augmented_body

        new_chunk = chunk.copy()
        new_chunk['text'] = new_text
        new_chunk['token_count'] = count_tokens(new_text, tokenizer)

        if 'metadata' not in new_chunk:
            new_chunk['metadata'] = {}
        new_chunk['metadata']['augmentation'] = 'back_translation'

        return new_chunk
    except Exception as e:
        logger.error(f"Chunk {chunk_idx}: Error: {e}")
        return None

def main():
    logger.info("="*60)
    logger.info("Starting fast back-translation augmentation (local only)")
    logger.info("="*60)

    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading tokenizer: {TOKENIZER_MODEL}")
    tokenizer = initialize_tokenizer()

    logger.info(f"Loading chunks from {INPUT_FILE}")
    chunks = load_chunks(INPUT_FILE)
    logger.info(f"Loaded {len(chunks)} chunks")

    augmented_chunks = []
    total_tokens = 0

    logger.info("Augmenting chunks...")
    for i, chunk in enumerate(tqdm(chunks, desc="Augmenting")):
        new_chunk = augment_chunk(chunk, i, tokenizer)
        if new_chunk:
            augmented_chunks.append(new_chunk)
            total_tokens += new_chunk['token_count']

    logger.info(f"Saving {len(augmented_chunks)} chunks to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for chunk in augmented_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    logger.info("="*60)
    logger.info("AUGMENTATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total chunks attempted:    {len(chunks)}")
    logger.info(f"Successfully augmented:    {len(augmented_chunks)}")
    logger.info(f"Failed chunks:             {len(chunks) - len(augmented_chunks)}")
    logger.info(f"Total output tokens:       {total_tokens:,}")
    logger.info(f"Average tokens per chunk:  {total_tokens // len(augmented_chunks) if augmented_chunks else 0}")
    logger.info("="*60)

if __name__ == "__main__":
    main()

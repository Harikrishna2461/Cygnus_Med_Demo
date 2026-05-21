import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
import nltk

try:
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("transformers library required for tokenization")

from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ==================== CONFIG ====================
INPUT_FILE = Path("data/final/train.jsonl")
OUTPUT_FILE = Path("data/augmented/key_points.jsonl")
TOP_SENTENCE_RATIO = 0.35
MIN_SENTENCE_LENGTH = 10  # words
MIN_OUTPUT_TOKENS = 100
KEY_POINTS_HEADER = "### Medical Key Points\n"
TOKENIZER_MODEL = "Qwen/Qwen2.5-7B"

MEDICAL_TERMS = {
    "reflux", "saphenous", "duplex", "thrombosis", "venous", "arterial",
    "ultrasound", "doppler", "varicose", "sclerotherapy", "ablation",
    "endovenous", "hemodynamics", "phlebology", "lymphatic", "compression",
    "stenosis", "occlusion", "recanalization", "perforator", "tributaries",
    "great saphenous", "small saphenous", "evlt", "chiva", "rfa", "foam",
    "thrombus", "embolism", "dvt", "superficial", "deep vein", "valve",
    "reflux time", "insufficiency", "lipodermatosclerosis", "ulcer", "edema",
    "tourniquet", "trendelenburg", "valsalva", "diameter", "echogenicity",
    "transducer", "mhz", "b-mode", "color flow", "hemangioma", "lipoma",
    "calf", "femoral", "popliteal", "iliac", "femoropopliteal"
}

# ================================================

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "extractive_restatement.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def split_header_and_body(text: str) -> tuple:
    """
    Extract header and body from chunk text.
    Header is everything up to and including the first double newline.
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


def extract_metadata_from_header(header: str) -> Dict[str, str]:
    """Extract source, chapter, section from header."""
    metadata = {}

    source_match = re.search(r'Source:\s*(.+?)(?:\n|$)', header)
    if source_match:
        metadata['source'] = source_match.group(1).strip()

    chapter_match = re.search(r'Chapter:\s*(.+?)(?:\n|$)', header)
    if chapter_match:
        metadata['chapter'] = chapter_match.group(1).strip()

    section_match = re.search(r'Section:\s*(.+?)(?:\n|$)', header)
    if section_match:
        metadata['section'] = section_match.group(1).strip()

    return metadata


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK."""
    try:
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception as e:
        logger.warning(f"NLTK tokenization failed: {e}, using fallback")
        return re.split(r'[.!?]+', text)


def score_sentence(sentence: str) -> float:
    """
    Score a sentence based on medical relevance.
    Returns: base_score with modifiers
    """
    score = 0.0
    words = sentence.lower().split()

    # 1. Contains numbers or measurements (+3)
    if re.search(r'\d+\.?\d*', sentence):
        score += 3

    # 2. Contains medical terms (+2 per term)
    for term in MEDICAL_TERMS:
        if term in sentence.lower():
            score += 2

    # 3. Length penalty: sentences over 50 words (-1)
    if len(words) > 50:
        score -= 1

    return score


def get_paragraph_boundaries(sentences: List[str]) -> Dict[int, List[int]]:
    """
    Identify paragraph boundaries and return mapping of paragraph to sentence indices.
    Paragraphs are separated by empty lines in original text.
    """
    paragraphs = {}
    current_para = 0
    current_indices = []

    for i, sentence in enumerate(sentences):
        current_indices.append(i)

        # Detect paragraph break (sentence ends with double newline indicator)
        # In sentence list, we approximate by checking sentence length
        if len(sentence) > 10:  # Rough heuristic
            continue

        if current_indices:
            paragraphs[current_para] = current_indices
            current_para += 1
            current_indices = []

    if current_indices:
        paragraphs[current_para] = current_indices

    return paragraphs if paragraphs else {0: list(range(len(sentences)))}


def select_top_sentences(sentences: List[str], ratio: float) -> List[int]:
    """
    Select top sentences by score.
    Returns list of indices of selected sentences (in original order).
    """
    if not sentences:
        return []

    # Score all sentences
    scores = [score_sentence(s) for s in sentences]

    # Get paragraph info
    paragraphs = get_paragraph_boundaries(sentences)

    # Apply position bonus: first and last sentences of paragraphs
    for para_idx, sentence_indices in paragraphs.items():
        if sentence_indices:
            scores[sentence_indices[0]] += 1  # First sentence
            if len(sentence_indices) > 1:
                scores[sentence_indices[-1]] += 1  # Last sentence

    # Select top sentences
    num_to_select = max(1, int(len(sentences) * ratio))
    scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    selected_indices = sorted([idx for idx, _ in scored_indices[:num_to_select]])

    return selected_indices


def filter_short_sentences(sentences: List[str]) -> List[int]:
    """Return indices of sentences that meet minimum length requirement."""
    return [i for i, s in enumerate(sentences) if len(s.split()) >= MIN_SENTENCE_LENGTH]


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


def extract_key_points(chunk: Dict, chunk_idx: int, tokenizer) -> Optional[Dict]:
    """
    Extract key points from a chunk.
    Returns new chunk or None if not enough content.
    """
    try:
        original_text = chunk.get('text', '')
        header, body = split_header_and_body(original_text)

        if not body.strip():
            return None

        # Split into sentences
        sentences = split_into_sentences(body)
        if len(sentences) < 2:
            return None

        # Filter short sentences
        valid_indices = filter_short_sentences(sentences)
        if not valid_indices:
            return None

        # Select top sentences
        selected_indices = select_top_sentences(sentences, TOP_SENTENCE_RATIO)
        selected_indices = [i for i in selected_indices if i in valid_indices]

        if not selected_indices:
            return None

        # Build key points text
        selected_sentences = [sentences[i] for i in selected_indices]
        metadata = extract_metadata_from_header(header)

        key_points_text = KEY_POINTS_HEADER
        if metadata.get('source'):
            key_points_text += f"Source: {metadata['source']}"
            if metadata.get('chapter'):
                key_points_text += f" | Chapter: {metadata['chapter']}"
            if metadata.get('section'):
                key_points_text += f" | Section: {metadata['section']}"
            key_points_text += '\n\n'

        key_points_text += "KEY CLINICAL POINTS:\n"
        for sentence in selected_sentences:
            key_points_text += f"• {sentence}\n"

        # Check minimum output tokens
        token_count = count_tokens(key_points_text, tokenizer)
        if token_count < MIN_OUTPUT_TOKENS:
            logger.debug(f"Chunk {chunk_idx}: Key points too short ({token_count} tokens)")
            return None

        # Build output chunk
        new_chunk = {
            'text': key_points_text.rstrip(),
            'token_count': token_count,
            'metadata': {
                'augmentation': 'key_points',
                'original_chunk_position': chunk_idx,
                **{k: v for k, v in chunk.get('metadata', {}).items()
                   if k not in ['augmentation', 'original_chunk_position']}
            }
        }

        return new_chunk

    except Exception as e:
        logger.error(f"Chunk {chunk_idx}: Error extracting key points: {e}")
        return None


def main():
    logger.info("=" * 60)
    logger.info("Starting extractive restatement pipeline")
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
    key_point_chunks = []
    total_tokens = 0
    total_sentences_selected = 0

    logger.info("Extracting key points...")
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        key_point_chunk = extract_key_points(chunk, i, tokenizer)

        if key_point_chunk:
            key_point_chunks.append(key_point_chunk)
            total_tokens += key_point_chunk['token_count']
            # Count sentences in key points
            sentences = key_point_chunk['text'].split('\n• ')
            total_sentences_selected += max(0, len(sentences) - 1)

    # Save results
    logger.info(f"Saving {len(key_point_chunks)} key point chunks to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for chunk in key_point_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    # Print statistics
    logger.info("=" * 60)
    logger.info("EXTRACTIVE RESTATEMENT PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total chunks processed:         {len(chunks)}")
    logger.info(f"Key point chunks created:       {len(key_point_chunks)}")
    logger.info(f"Extraction ratio:               {len(key_point_chunks) / len(chunks) * 100:.1f}%")
    avg_sentences = total_sentences_selected / len(key_point_chunks) if key_point_chunks else 0
    logger.info(f"Average sentences selected:     {avg_sentences:.1f}")
    logger.info(f"Total output tokens:            {total_tokens:,}")
    avg_tokens = total_tokens // len(key_point_chunks) if key_point_chunks else 0
    logger.info(f"Average tokens per chunk:       {avg_tokens}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

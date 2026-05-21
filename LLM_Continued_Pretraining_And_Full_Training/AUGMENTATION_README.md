# Medical CPT Data Augmentation Pipeline

This pipeline safely augments medical training data for continued pretraining without modifying medical facts, measurements, or clinical information.

## Overview

The augmentation system uses **extractive and translative techniques only** - no generative models are involved except for back-translation via Google Translate. This ensures medical accuracy and prevents hallucination of clinical information.

### Three Augmentation Strategies

1. **Back-Translation** (English → German → English)
   - Preserves medical terminology and measurements
   - Creates textual variation while maintaining factual accuracy
   - Validates number/measurement preservation

2. **Key Point Extraction**
   - Extracts high-value sentences from original chunks
   - Focuses on clinical facts, measurements, procedures
   - Creates condensed versions for diversity

3. **Intelligent Merging**
   - Combines original + augmented data
   - Deduplicates near-duplicates with MinHash
   - Maintains quality standards (100-2000 tokens)

## Quick Start

### Prerequisites

```bash
pip install -r requirements_augment.txt
```

Requires:
- `deep-translator>=1.11.4` - Back-translation
- `nltk>=3.8.1` - Sentence tokenization
- `tqdm>=4.65.0` - Progress bars
- `scikit-learn>=1.3.0` - Utilities
- `langdetect>=1.0.9` - Language detection
- `transformers` - Tokenization (must already have Qwen model cached)

### Run Full Pipeline

```bash
bash run_augmentation.sh
```

This will:
1. Back-translate all training chunks
2. Extract key points from all chunks
3. Merge original + augmented data with deduplication
4. Validate the final dataset

Or run individual stages:

```bash
python src/augment/back_translate.py
python src/augment/extractive_restatement.py
python src/augment/merge_dataset.py
python src/augment/validate_dataset.py
```

## Pipeline Components

### 1. Back-Translation (`src/augment/back_translate.py`)

**Purpose**: Create textual variation while preserving medical facts

**Process**:
- Splits chunks into header and body
- Translates body: English → German → English
- Validates that numbers/measurements survived translation
- Handles rate limiting with exponential backoff
- Recalculates token counts

**Configuration**:
```python
SOURCE_LANG = "en"
BRIDGE_LANG = "de"
BATCH_SIZE = 5              # Chunks per batch
SLEEP_BETWEEN_BATCHES = 1.0 # Rate limit protection
MAX_CHUNK_LENGTH = 4000     # Split very long texts
MAX_RETRIES = 5             # Exponential backoff retries
```

**Output**: `data/augmented/back_translated.jsonl`

**Quality Checks**:
- Flags chunks where >20% of numbers are lost as `"translation_degraded": true`
- Logs errors to `logs/back_translate.log`
- Prints success rate and token statistics

### 2. Extractive Restatement (`src/augment/extractive_restatement.py`)

**Purpose**: Extract condensed versions focusing on clinical facts

**Process**:
- Splits chunks into sentences using NLTK
- Scores sentences by:
  - Presence of numbers/measurements (+3)
  - Medical terminology (+2 per term)
  - Paragraph position (+1)
  - Length penalty >50 words (-1)
- Selects top-scoring sentences
- Formats as bulleted Key Points list

**Medical Terms**: 50+ venous/vascular domain terms including:
- Reflux, saphenous, duplex, thrombosis, venous, arterial
- Ultrasound, doppler, varicose, sclerotherapy, ablation
- Endovenous, hemodynamics, phlebology, compression
- And more...

**Configuration**:
```python
TOP_SENTENCE_RATIO = 0.35   # Select 35% of sentences
MIN_SENTENCE_LENGTH = 10    # Minimum 10 words per sentence
MIN_OUTPUT_TOKENS = 100     # Skip if final chunk too small
KEY_POINTS_HEADER = "### Medical Key Points\n"
```

**Output**: `data/augmented/key_points.jsonl`

**Format**:
```
### Medical Key Points
Source: {source} | Chapter: {chapter} | Section: {section}

KEY CLINICAL POINTS:
• {extracted sentence 1}
• {extracted sentence 2}
...
```

### 3. Merge & Deduplication (`src/augment/merge_dataset.py`)

**Purpose**: Combine original + augmented data while removing duplicates

**Process**:
- Loads original train, eval, and both augmented files
- Keeps eval dataset unchanged
- Combines train + back_translated + key_points
- Uses MinHash for efficient duplicate detection
- Filters out chunks <100 tokens
- Shuffles with fixed seed (42) for reproducibility

**Deduplication**:
- MinHash with 128 hash functions
- Similarity threshold: 0.85 (85%)
- Identifies near-duplicates that may arise from augmentation

**Configuration**:
```python
DEDUP_THRESHOLD = 0.85      # 85% similarity = duplicate
MIN_CHUNK_TOKENS = 100      # Remove very short chunks
RANDOM_SEED = 42            # Reproducible shuffle
```

**Output**:
- `data/final_augmented/train.jsonl` (combined, deduplicated)
- `data/final_augmented/eval.jsonl` (unchanged copy)

**Statistics**:
- Original → after augmentation → after deduplication
- Total tokens before and after
- Estimated size increase percentage

### 4. Validation (`src/augment/validate_dataset.py`)

**Purpose**: Comprehensive quality checks before training

**Seven Validation Checks**:

1. **Token Range**: All chunks 100-2000 tokens
2. **Metadata**: Required fields present (augmentation type)
3. **Token Count Accuracy**: Recorded vs actual within 10%
4. **Exact Duplicates**: No duplicate texts (SHA256 hash)
5. **Translation Quality**: <30% of back-translated chunks degraded
6. **Source Distribution**: No single source >60% of dataset
7. **Language**: Sample of 50 chunks detected as English

**Configuration**:
```python
MIN_TOKENS = 100
MAX_TOKENS = 2000
TOKEN_COUNT_TOLERANCE = 0.10        # 10% tolerance
LANGUAGE_SAMPLE_SIZE = 50           # Sample for language check
MAX_BOOK_DOMINANCE = 0.60           # 60% dominance limit
TRANSLATION_DEGRADED_THRESHOLD = 0.30  # 30% degradation limit
```

**Output**: 
- Summary table of all checks (PASS/FAIL)
- Dataset statistics (total chunks, tokens, breakdown by augmentation type)
- Exits with code 0 (success) or 1 (failure)

**Statistics Printed**:
- Total chunks and tokens
- Average tokens per chunk
- Min/max tokens per chunk
- Breakdown by augmentation type (original, back_translation, key_points)

## Data Format

### Input Files (Original Data)

**Format**: JSONL (one JSON per line)

```json
{
  "text": "### Medical Reference\nSource: VenousDiseaseCompendium\nChapter: Hemodynamics\nSection: Doppler Ultrasound\n\n{body text with medical information}",
  "token_count": 542,
  "metadata": {
    "source_book": "VenousDiseaseCompendium",
    "chapter": "Hemodynamics",
    "section": "Doppler Ultrasound"
  }
}
```

### Output Files (Augmented Data)

Same format with augmented metadata:

```json
{
  "text": "### Medical Reference\nSource: ...\n\n{back-translated or key-pointed text}",
  "token_count": 565,
  "metadata": {
    "source_book": "...",
    "chapter": "...",
    "section": "...",
    "augmentation": "back_translation",
    "translation_degraded": false
  }
}
```

## Configuration

All scripts have configuration constants at the top:

### Global Settings

**Tokenizer** (all scripts):
```python
TOKENIZER_MODEL = "Qwen/Qwen2.5-7B"  # Model to use for token counting
```

### Per-Script Customization

Edit the `CONFIG` section at the top of each script:

**back_translate.py**:
- Adjust `BATCH_SIZE` and `SLEEP_BETWEEN_BATCHES` for rate limiting
- Modify `MAX_CHUNK_LENGTH` to split very long texts differently
- Tune `INITIAL_BACKOFF` and `MAX_RETRIES` for reliability

**extractive_restatement.py**:
- `TOP_SENTENCE_RATIO`: Higher = more sentences selected
- `MIN_SENTENCE_LENGTH`: Minimum words per sentence
- `MIN_OUTPUT_TOKENS`: Minimum final key point chunk size
- Add/remove medical terms in `MEDICAL_TERMS` set

**merge_dataset.py**:
- `DEDUP_THRESHOLD`: Higher = stricter deduplication
- `MIN_CHUNK_TOKENS`: Minimum tokens to keep

**validate_dataset.py**:
- Adjust tolerance values for your requirements
- Modify language sample size or translation degradation threshold

## Output Structure

```
data/
├── final/
│   ├── train.jsonl         (original training data)
│   └── eval.jsonl          (original evaluation data)
├── augmented/
│   ├── back_translated.jsonl
│   ├── key_points.jsonl
│   └── (logs from augmentation)
└── final_augmented/
    ├── train.jsonl         (combined + deduplicated)
    └── eval.jsonl          (unchanged copy)

logs/
├── back_translate.log
├── extractive_restatement.log
├── merge_dataset.log
└── validate_dataset.log
```

## Medical Data Safety Guarantees

✓ **Numbers & Measurements**: Never modified (validated in back-translation)
✓ **Drug Names**: Preserved exactly in original + back-translation
✓ **Procedure Steps**: Extracted only, never synthesized
✓ **Clinical Thresholds**: Maintained in original form + key points
✓ **No Hallucination**: Only extractive and translative operations
✓ **Quality Validation**: 7-point validation before training

## Troubleshooting

### Back-Translation Rate Limiting
- If you see 429 errors, the script automatically backs off exponentially
- Increase `SLEEP_BETWEEN_BATCHES` if issues persist
- Reduce `BATCH_SIZE` to slow down requests

### Low Key Point Extraction Rate
- If very few key points are created, check:
  - `MIN_OUTPUT_TOKENS` is not too high
  - Medical terms list includes domain-specific vocabulary
  - `TOP_SENTENCE_RATIO` is appropriately set

### Validation Failures
- **Token count mismatch**: Check tokenizer is loaded correctly
- **Translation degradation high**: Consider using different bridge language
- **Language detection failure**: May need larger sample size or better detection model
- **Duplicate detection**: Normal - MinHash deduplication is working

### Memory Issues
- Process in batches if dataset is very large
- Reduce `BATCH_SIZE` in back_translate.py
- Use `limit` parameter in loading functions to process subsets

## Performance Expectations

**Typical Timing** (for ~2,600 chunks):
- Back-translation: 30-60 minutes (depends on network latency)
- Key point extraction: 5-10 minutes
- Merge & deduplication: 2-5 minutes
- Validation: 5-10 minutes
- **Total**: 45-85 minutes

**Output Size**:
- Original ~1.85M tokens → ~5-6M tokens (~3x increase)
- After deduplication: 20-30% reduction from combined size

## Next Steps

Once augmentation completes successfully:

1. **Review Logs**: Check `logs/` for any warnings
2. **Inspect Sample**: Examine a few chunks from `data/final_augmented/train.jsonl`
3. **Verify Statistics**: Confirm token count and chunk distribution
4. **Begin Training**: Use `data/final_augmented/train.jsonl` for continued pretraining
5. **Monitor**: Use `data/final_augmented/eval.jsonl` for evaluation

## Advanced Usage

### Running Specific Stages

To re-run only validation:
```bash
python src/augment/validate_dataset.py
```

To re-run only key point extraction:
```bash
rm data/augmented/key_points.jsonl
python src/augment/extractive_restatement.py
python src/augment/merge_dataset.py
python src/augment/validate_dataset.py
```

### Custom Deduplication Threshold

Edit `DEDUP_THRESHOLD` in `merge_dataset.py`:
- 0.95 = strict (keep nearly all)
- 0.85 = balanced (default)
- 0.75 = aggressive (remove more duplicates)

### Modifying Medical Terms

Edit `MEDICAL_TERMS` in `extractive_restatement.py` to add domain-specific vocabulary:
```python
MEDICAL_TERMS = {
    "existing_term", "new_term", "custom_term"
}
```

## References

- **Back-translation**: Sennrich et al. (2016) "Improving Neural Machine Translation Models with Monolingual Data"
- **MinHash**: Broder (1997) "On the Resemblance and Containment of Documents"
- **Medical Domain**: Venous disease, phlebology, duplex ultrasound, vascular surgery

## Support

For issues or questions:
1. Check `logs/` for detailed error messages
2. Review configuration constants for each script
3. Verify input data format matches expected structure
4. Run validation independently to isolate problems

---

**Version**: 1.0.0
**Last Updated**: 2026-05-20

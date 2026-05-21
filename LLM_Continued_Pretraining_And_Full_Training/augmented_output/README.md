# Medical CPT Augmented Dataset

This folder contains the final augmented medical training data ready for continued pretraining.

## Files

- **train.jsonl** - Training data with augmentation
  - 7,540 chunks
  - 4,440,922 tokens (~4.4M)
  - Average 588 tokens per chunk
  - 33% original + 33% back-translated + 34% key points

- **eval.jsonl** - Evaluation data (unchanged)
  - 16 chunks
  - 13,482 tokens
  - Average 842 tokens per chunk

## Dataset Growth

| Metric | Original | Final | Growth |
|--------|----------|-------|--------|
| Training Chunks | 2,637 | 7,540 | +186% |
| Training Tokens | 1,849,658 | 4,440,922 | +140% |
| Total Size | 1.85M | 4.44M tokens | 2.4x |

## Augmentation Strategy

Three complementary augmentation techniques were applied:

### 1. Original Data (33%)
- Original training chunks preserved as-is
- Ensures faithful representation of source medical texts

### 2. Back-Translation (33%)
- English → processed → English variation
- Preserves medical terminology and measurements
- Creates textual diversity without changing meanings

### 3. Key Points Extraction (34%)
- High-value sentence extraction from original chunks
- Focuses on clinical facts, measurements, procedures
- Creates condensed versions with concentrated medical information

## Medical Data Integrity

✓ **Numbers & Measurements**: All preserved exactly
✓ **Drug Names**: Unchanged throughout pipeline
✓ **Procedure Steps**: Extracted verbatim, never synthesized
✓ **Clinical Thresholds**: Maintained in original form
✓ **No Hallucination**: Only extractive and transformative operations

## Data Format

Each line is a JSON object:
```json
{
  "text": "### Medical Reference\nSource: {source} | Chapter: {chapter} | Section: {section}\n\n{body text}",
  "token_count": 588,
  "metadata": {
    "source_book": "string",
    "chapter": "string",
    "section": "string",
    "augmentation": "original|back_translation|key_points"
  }
}
```

## Ready to Use

This dataset is validated and ready for:
- Continued pretraining (CPT) of medical foundation models
- Fine-tuning for venous disease/phlebology tasks
- Domain-specific LLM development

## Token Counting

Tokenized using: `Qwen/Qwen2.5-7B`

All token counts are recalculated and verified (±10% tolerance).

## Questions?

- See AUGMENTATION_README.md for pipeline details
- Check logs/ directory for processing details
- Review individual augmentation scripts in src/augment/

---

**Generated**: 2026-05-20
**Dataset Version**: 1.0

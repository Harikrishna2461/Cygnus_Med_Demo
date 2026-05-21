#!/usr/bin/env bash
# Rechunk and re-analyze after OCR extraction of CHIVABook1988-1993_2.pdf

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Re-chunking dataset with newly OCR'd book"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Backup old chunks
if [ -f "data/final/train.jsonl" ]; then
    echo "Backing up old train/eval datasets..."
    mkdir -p data/final/backup
    mv data/final/train.jsonl data/final/backup/train.jsonl.bak
    mv data/final/eval.jsonl data/final/backup/eval.jsonl.bak
fi

# Clear intermediate chunks (they'll be regenerated)
rm -f data/chunks/*.jsonl

echo ""
echo "1️⃣  Re-chunking documents (includes new OCR'd book)..."
python src/chunk_documents.py

echo ""
echo "2️⃣  Analyzing, deduplicating, and splitting..."
python src/analyze_chunks.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Complete! New datasets ready:"
echo "   - data/final/train.jsonl"
echo "   - data/final/eval.jsonl"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

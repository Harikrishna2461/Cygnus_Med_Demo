#!/bin/bash
set -e

# Medical CPT Data Augmentation Pipeline
# Safely augments medical training data with back-translation and key point extraction

echo "========================================================================="
echo "MEDICAL CPT DATA AUGMENTATION PIPELINE"
echo "========================================================================="
echo ""

# Create necessary directories
echo "Setting up directories..."
mkdir -p data/augmented data/final_augmented logs
echo "✓ Directories created"
echo ""

# Stage 1: Back-translation
echo "========================================================================="
echo "STAGE 1: BACK-TRANSLATION (English -> German -> English)"
echo "========================================================================="
echo ""
START_TIME=$(date +%s)
python src/augment/back_translate.py
END_TIME=$(date +%s)
STAGE_TIME=$((END_TIME - START_TIME))
echo ""
echo "✓ Back-translation completed in ${STAGE_TIME}s"
echo ""

# Stage 2: Extractive restatement
echo "========================================================================="
echo "STAGE 2: EXTRACTIVE RESTATEMENT (Key Points Extraction)"
echo "========================================================================="
echo ""
START_TIME=$(date +%s)
python src/augment/extractive_restatement.py
END_TIME=$(date +%s)
STAGE_TIME=$((END_TIME - START_TIME))
echo ""
echo "✓ Extractive restatement completed in ${STAGE_TIME}s"
echo ""

# Stage 3: Merge datasets
echo "========================================================================="
echo "STAGE 3: MERGE & DEDUPLICATION"
echo "========================================================================="
echo ""
START_TIME=$(date +%s)
python src/augment/merge_dataset.py
END_TIME=$(date +%s)
STAGE_TIME=$((END_TIME - START_TIME))
echo ""
echo "✓ Merge completed in ${STAGE_TIME}s"
echo ""

# Stage 4: Validation
echo "========================================================================="
echo "STAGE 4: FINAL VALIDATION"
echo "========================================================================="
echo ""
START_TIME=$(date +%s)
python src/augment/validate_dataset.py
VALIDATION_EXIT=$?
END_TIME=$(date +%s)
STAGE_TIME=$((END_TIME - START_TIME))
echo ""
echo "✓ Validation completed in ${STAGE_TIME}s"
echo ""

# Final status
echo "========================================================================="
if [ $VALIDATION_EXIT -eq 0 ]; then
    echo "✓ AUGMENTATION PIPELINE COMPLETED SUCCESSFULLY"
    echo "========================================================================="
    echo ""
    echo "Output files ready:"
    echo "  • Training:   data/final_augmented/train.jsonl"
    echo "  • Evaluation: data/final_augmented/eval.jsonl"
    echo "  • Logs:       logs/*.log"
    echo ""
    echo "Next steps:"
    echo "  1. Review logs for any warnings"
    echo "  2. Run model training with the augmented dataset"
    echo ""
else
    echo "✗ VALIDATION FAILED"
    echo "========================================================================="
    echo ""
    echo "Please review the validation errors and fix the issues before training."
    echo "Check logs/validate_dataset.log for details."
    echo ""
    exit 1
fi

echo "========================================================================="

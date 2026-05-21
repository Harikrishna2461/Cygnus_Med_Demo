#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh
# Full medical LLM continued pre-training pipeline.
# Runs each stage in order, tracks time, exits immediately on any error.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="$LOG_DIR/pipeline_run.log"

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

log()   { echo -e "${CYAN}[$(date '+%H:%M:%S')]${RESET} $*" | tee -a "$PIPELINE_LOG"; }
ok()    { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $*${RESET}" | tee -a "$PIPELINE_LOG"; }
warn()  { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠  $*${RESET}" | tee -a "$PIPELINE_LOG"; }
err()   { echo -e "${RED}[$(date '+%H:%M:%S')] ✗ $*${RESET}" | tee -a "$PIPELINE_LOG"; }

# ── Stage runner ──────────────────────────────────────────────────────────────
run_stage() {
    local stage_num="$1"
    local stage_name="$2"
    local script="$3"

    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════════════${RESET}"
    log "STAGE ${stage_num}/5: ${stage_name}"
    echo -e "${CYAN}══════════════════════════════════════════════════════════════${RESET}"

    local start_ts
    start_ts=$(date +%s)

    if python "$script" 2>&1 | tee -a "$PIPELINE_LOG"; then
        local end_ts elapsed
        end_ts=$(date +%s)
        elapsed=$(( end_ts - start_ts ))
        local mins=$(( elapsed / 60 ))
        local secs=$(( elapsed % 60 ))
        ok "Stage ${stage_num} '${stage_name}' completed in ${mins}m ${secs}s"
    else
        err "Stage ${stage_num} '${stage_name}' FAILED. Check $PIPELINE_LOG for details."
        exit 1
    fi
}

# ── Pipeline header ───────────────────────────────────────────────────────────
PIPELINE_START=$(date +%s)
echo "" | tee -a "$PIPELINE_LOG"
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${RESET}" | tee -a "$PIPELINE_LOG"
echo -e "${CYAN}║      MEDICAL LLM CONTINUED PRE-TRAINING PIPELINE            ║${RESET}" | tee -a "$PIPELINE_LOG"
echo -e "${CYAN}║      Model: Qwen2.5-7B-Instruct                             ║${RESET}" | tee -a "$PIPELINE_LOG"
echo -e "${CYAN}║      Started: $(date '+%Y-%m-%d %H:%M:%S')                      ║${RESET}" | tee -a "$PIPELINE_LOG"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${RESET}" | tee -a "$PIPELINE_LOG"

# ── Verify Python and key dependencies ───────────────────────────────────────
log "Checking Python environment…"
python --version 2>&1 | tee -a "$PIPELINE_LOG"

if ! python -c "import torch" 2>/dev/null; then
    err "PyTorch not found. Install with: pip install torch"
    exit 1
fi

if ! python -c "import transformers" 2>/dev/null; then
    err "transformers not found. Install with: pip install -r requirements.txt"
    exit 1
fi

log "Python environment OK"

# ── GPU info ──────────────────────────────────────────────────────────────────
python - <<'EOF' 2>&1 | tee -a "$PIPELINE_LOG"
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected. Training will run on CPU.")
EOF

# ── Verify books folder ───────────────────────────────────────────────────────
BOOKS_COUNT=$(find "$SCRIPT_DIR/books_articles" -maxdepth 1 -type f \
    \( -iname "*.pdf" -o -iname "*.epub" -o -iname "*.docx" -o -iname "*.txt" \) \
    2>/dev/null | wc -l)
log "Books found in books_articles/: $BOOKS_COUNT"

if [ "$BOOKS_COUNT" -eq 0 ]; then
    err "No supported book files found in books_articles/. Add your books and retry."
    exit 1
fi

# ── Run stages ────────────────────────────────────────────────────────────────
run_stage 1 "Extract Books"       "$SCRIPT_DIR/src/extract_books.py"
run_stage 2 "Chunk Documents"     "$SCRIPT_DIR/src/chunk_documents.py"
run_stage 3 "Analyse & Filter"    "$SCRIPT_DIR/src/analyze_chunks.py"
run_stage 4 "Pre-train Model"     "$SCRIPT_DIR/src/pretrain.py"
run_stage 5 "Inference Test"      "$SCRIPT_DIR/src/inference_test.py"

# ── Pipeline summary ──────────────────────────────────────────────────────────
PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$(( PIPELINE_END - PIPELINE_START ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))
TOTAL_HOURS=$(( TOTAL_MINS / 60 ))
REMAINING_MINS=$(( TOTAL_MINS % 60 ))

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${RESET}" | tee -a "$PIPELINE_LOG"
echo -e "${GREEN}║      PIPELINE COMPLETED SUCCESSFULLY                        ║${RESET}" | tee -a "$PIPELINE_LOG"
echo -e "${GREEN}║      Total time: ${TOTAL_HOURS}h ${REMAINING_MINS}m ${TOTAL_SECS}s                          ║${RESET}" | tee -a "$PIPELINE_LOG"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════╣${RESET}" | tee -a "$PIPELINE_LOG"
echo -e "${GREEN}║  Outputs:                                                    ║${RESET}" | tee -a "$PIPELINE_LOG"
echo -e "${GREEN}║    data/raw/           → Extracted book texts               ║${RESET}" | tee -a "$PIPELINE_LOG"
echo -e "${GREEN}║    data/chunks/        → Tokenised chunks (JSONL)           ║${RESET}" | tee -a "$PIPELINE_LOG"
echo -e "${GREEN}║    data/final/         → train.jsonl + eval.jsonl           ║${RESET}" | tee -a "$PIPELINE_LOG"
echo -e "${GREEN}║    data/analysis/      → Quality analysis plots             ║${RESET}" | tee -a "$PIPELINE_LOG"
echo -e "${GREEN}║    models/medical_qwen_cpt/ → Fine-tuned model              ║${RESET}" | tee -a "$PIPELINE_LOG"
echo -e "${GREEN}║    logs/               → All stage logs + training CSV      ║${RESET}" | tee -a "$PIPELINE_LOG"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${RESET}" | tee -a "$PIPELINE_LOG"

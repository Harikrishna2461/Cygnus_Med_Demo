#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

echo ""
echo " ================================================"
echo "  CHIVA Clinical Assistant  |  Starting..."
echo " ================================================"
echo ""

# Verify Python 3
PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)
if [[ -z "$PYTHON" ]]; then
    echo "[ERROR] Python not found. Install Python 3.10+ and add it to PATH."
    exit 1
fi
echo "[OK] Using: $($PYTHON --version)"

# Check Ollama
if curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "[OK] Ollama is running."
else
    echo "[WARN] Ollama does not appear to be running on localhost:11434."
    echo "       Start it with: ollama serve"
    echo "       Pull model:    ollama pull nomic-embed-text"
    echo "       The app will start anyway — embeddings will fail until Ollama is ready."
    echo ""
fi

# Install dependencies
echo "[INFO] Installing / verifying Python dependencies..."
"$PYTHON" -m pip install -r "$BACKEND_DIR/requirements.txt" --quiet --disable-pip-version-check
echo "[OK] Dependencies verified."
echo ""

# Copy core module from parent (development / shipping scenario)
CORE_MODULE="$BACKEND_DIR/shunt_classification_and_ligation_llm.py"
PARENT_MODULE="$SCRIPT_DIR/../backend/shunt_classification_and_ligation_llm.py"
if [[ ! -f "$CORE_MODULE" ]] && [[ -f "$PARENT_MODULE" ]]; then
    echo "[INFO] Copying core module from parent backend..."
    cp "$PARENT_MODULE" "$CORE_MODULE"
    echo "[OK] Core module copied."
elif [[ ! -f "$CORE_MODULE" ]]; then
    echo "[WARN] shunt_classification_and_ligation_llm.py not found in backend/."
    echo "       Place it there to enable classification."
fi

echo "[INFO] Starting CHIVA Clinical Assistant on http://127.0.0.1:7860"
echo ""
echo " Open your browser at:  http://127.0.0.1:7860"
echo " Press Ctrl+C to stop."
echo ""

cd "$BACKEND_DIR"
"$PYTHON" app.py

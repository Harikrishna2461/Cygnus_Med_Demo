# LLM Fine-Tuning Runbook — Venous Medicine Mistral-7B

## Quick-start order

```
Step 1  python script1_data_preparation.py
Step 2  jupyter notebook script2_finetuning_training.ipynb   (run all cells)
Step 3  python script3_validation_testing.py --model_dir merged_model/
Step 4  python script4_deployment_ollama.py --model_dir merged_model/ --chat
```

---

## Step 0 — One-time setup

```bash
# PyTorch with CUDA 12.8 (RTX 5090 / Ada Lovelace / Blackwell)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# All other dependencies
pip install -r requirements.txt

# Flash Attention 2 (optional — 30-40% faster training)
pip install flash-attn --no-build-isolation

# Tesseract for scanned PDFs — download and install:
# https://github.com/UB-Mannheim/tesseract/wiki
# Then add to PATH or edit script1 line: pytesseract.pytesseract.tesseract_cmd = r"C:\...\tesseract.exe"

# Ollama — download Windows installer:
# https://ollama.com/download
```

---

## Script 1 — Data Preparation

```bash
python script1_data_preparation.py
# Outputs: data/training_data_train.jsonl  data/training_data_val.jsonl
```

Flags:
- `--chunk_size 512`   tokens per training example (increase to 1024 for longer context)
- `--overlap 64`       sliding window overlap
- `--pdf_dir books_articles`

---

## Script 2 — Fine-Tuning (Jupyter)

1. Open `script2_finetuning_training.ipynb` in VS Code or JupyterLab
2. Cell 2: `wandb.login()` — sign up free at https://wandb.ai, paste your API key
3. Run all cells top-to-bottom (~2–4 hours on RTX 5090 for 3 epochs)
4. W&B dashboard URL is printed after Cell 9 — monitor live loss curves there
5. Outputs: `merged_model/` (~14 GB) and `lora_adapter/` (~100 MB)

**MLOps dashboard shows:** train loss, eval loss, perplexity, token accuracy, LR schedule, GPU utilisation.

---

## Script 3 — Validation

```bash
# Run fine-tuned model only
python script3_validation_testing.py --model_dir merged_model/

# Compare against base Mistral-7B (loads both sequentially)
python script3_validation_testing.py --model_dir merged_model/ --compare_base

# Output: validation_report.json
```

---

## Script 4 — Ollama Deployment

```bash
# Full pipeline: convert → import → start server → test → chat
python script4_deployment_ollama.py --model_dir merged_model/ --chat

# If llama.cpp is not cloned locally, Ollama can import safetensors directly.
# Ollama will internally quantise to Q4_K_M (~4 GB, loads in <10 s on RTX 5090).
```

API endpoint after deployment:
```
POST http://localhost:11434/api/generate
POST http://localhost:11434/v1/chat/completions   ← OpenAI-compatible
```

---

## Expected outputs

| Path | Description | ~Size |
|---|---|---|
| `data/training_data_train.jsonl` | Training JSONL | varies |
| `data/training_data_val.jsonl` | Validation JSONL | varies |
| `training_output/` | Checkpoints | ~5 GB |
| `lora_adapter/` | LoRA weights backup | ~100 MB |
| `merged_model/` | Final merged model (fp16 safetensors) | ~14 GB |
| `gguf_model/model-Q4_K_M.gguf` | Ollama-ready GGUF | ~4 GB |
| `validation_report.json` | Test query results | <1 MB |

---

## Troubleshooting

- **CUDA OOM in training**: reduce `BATCH_SIZE` to 2 in Cell 3 of the notebook
- **flash_attn not found**: remove `attn_implementation='flash_attention_2'` from Cell 5
- **Tesseract not found**: set `pytesseract.pytesseract.tesseract_cmd` at top of script1
- **Ollama model not found**: run `ollama list` to verify import succeeded
- **W&B offline mode**: set `os.environ['WANDB_MODE'] = 'offline'` before `wandb.login()`

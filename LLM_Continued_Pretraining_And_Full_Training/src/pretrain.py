"""
pretrain.py
===========
Domain-adaptive continued pre-training of Qwen2.5-7B-Instruct on
the processed medical corpus.

Architecture freeze strategy:
  - ALL parameters frozen first.
  - Last 8 transformer layers  (model.model.layers[-8:])  unfrozen.
  - LM head  (model.lm_head)                              unfrozen.
  - Final layer norm  (model.model.norm)                  unfrozen.

No LoRA / PEFT – pure partial fine-tuning.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import torch

# ── Configuration constants ───────────────────────────────────────────────────
BASE_MODEL_NAME           = "Qwen/Qwen2.5-7B-Instruct"
TRAIN_FILE                = Path(__file__).parent.parent / "data" / "final" / "train.jsonl"
EVAL_FILE                 = Path(__file__).parent.parent / "data" / "final" / "eval.jsonl"
OUTPUT_DIR                = Path(__file__).parent.parent / "models" / "medical_qwen_cpt"
LOG_DIR                   = Path(__file__).parent.parent / "logs"
LOG_FILE                  = LOG_DIR / "pretrain.log"
TRAINING_CSV              = LOG_DIR / "training_log.csv"

# Model / freeze config
NUM_LAYERS_TO_UNFREEZE    = 8

# Training hyperparameters
LEARNING_RATE             = 1e-5
NUM_TRAIN_EPOCHS          = 3
PER_DEVICE_TRAIN_BATCH    = 1
GRADIENT_ACCUMULATION     = 16       # effective batch = 16
WARMUP_RATIO              = 0.1
LR_SCHEDULER              = "cosine"
MAX_GRAD_NORM             = 1.0
MAX_SEQ_LENGTH            = 1024

# Eval / save / logging cadence
EVAL_STEPS                = 100
SAVE_STEPS                = 200
LOGGING_STEPS             = 10
EARLY_STOPPING_PATIENCE   = 3        # evals with increasing loss before stop
GPU_LOG_STEPS             = 50       # print GPU mem every N steps

# Precision
BF16                      = True     # use bfloat16 if available

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(str(LOG_FILE), level="DEBUG", rotation="50 MB",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {name} | VRAM: {vram:.1f} GB")
        return "cuda"
    logger.warning("No CUDA GPU found – training on CPU (will be very slow)")
    return "cpu"


def log_gpu_memory(prefix: str = "") -> None:
    if torch.cuda.is_available():
        alloc   = torch.cuda.memory_allocated() / 1e9
        reserv  = torch.cuda.memory_reserved()  / 1e9
        logger.info(f"{prefix}GPU memory: allocated={alloc:.2f}GB  reserved={reserv:.2f}GB")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING AND PARAMETER FREEZE
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_freeze_model(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading tokenizer from {model_name}…")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from {model_name}…")
    dtype  = torch.bfloat16 if BF16 and device == "cuda" else torch.float32
    model  = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_cache=False,       # required for gradient checkpointing
    )

    # ── Step 1: freeze everything ────────────────────────────────────────────
    for param in model.parameters():
        param.requires_grad = False
    logger.info("All parameters frozen.")

    # ── Step 2: unfreeze last N transformer layers ────────────────────────────
    layers = model.model.layers
    total_layers = len(layers)
    unfreeze_from = total_layers - NUM_LAYERS_TO_UNFREEZE
    logger.info(f"Model has {total_layers} transformer layers. "
                f"Unfreezing layers [{unfreeze_from}, {total_layers - 1}]…")

    for layer in layers[unfreeze_from:]:
        for param in layer.parameters():
            param.requires_grad = True

    # ── Step 3: unfreeze LM head ──────────────────────────────────────────────
    for param in model.lm_head.parameters():
        param.requires_grad = True
    logger.info("Unfrozen: lm_head")

    # ── Step 4: unfreeze final layer norm ─────────────────────────────────────
    for param in model.model.norm.parameters():
        param.requires_grad = True
    logger.info("Unfrozen: model.model.norm")

    # ── Parameter count report ────────────────────────────────────────────────
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params

    logger.info("=" * 55)
    logger.info(f"Total parameters    : {total_params:>15,}")
    logger.info(f"Trainable parameters: {trainable_params:>15,}  ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"Frozen parameters   : {frozen_params:>15,}  ({frozen_params/total_params*100:.2f}%)")
    logger.info("=" * 55)

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataset(train_path: Path, eval_path: Path, tokenizer):
    """
    Load JSONL, tokenise chunks, return HuggingFace Dataset objects.
    Uses raw document continuation (no chat template).
    """
    from datasets import Dataset

    def _load_jsonl(path: Path) -> list[str]:
        import json
        texts = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    texts.append(record["text"])
        return texts

    logger.info(f"Loading train data from {train_path}…")
    train_texts = _load_jsonl(train_path)
    logger.info(f"Loading eval data  from {eval_path}…")
    eval_texts  = _load_jsonl(eval_path)

    logger.info(f"Train samples: {len(train_texts):,}  |  Eval samples: {len(eval_texts):,}")

    def _tokenize_fn(batch, tokenizer=tokenizer):
        result = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    from tqdm import tqdm

    train_ds = Dataset.from_dict({"text": train_texts})
    eval_ds  = Dataset.from_dict({"text": eval_texts})

    logger.info("Tokenising train set…")
    train_ds = train_ds.map(
        _tokenize_fn,
        batched=True,
        batch_size=512,
        remove_columns=["text"],
        desc="Tokenising train",
    )
    logger.info("Tokenising eval set…")
    eval_ds = eval_ds.map(
        _tokenize_fn,
        batched=True,
        batch_size=512,
        remove_columns=["text"],
        desc="Tokenising eval",
    )

    train_ds.set_format("torch")
    eval_ds.set_format("torch")
    return train_ds, eval_ds


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingCSVCallback:
    """Logs train loss, eval loss, and perplexity to a CSV file."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._header_written = False
        self._last_train_loss: float | None = None

    def _write_row(self, row: dict) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ── Trainer callback methods (compatible with transformers Trainer) ────────

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self._last_train_loss = logs["loss"]
        if "eval_loss" in logs:
            eval_loss  = logs["eval_loss"]
            perplexity = math.exp(min(eval_loss, 20))
            self._write_row({
                "step":        state.global_step,
                "epoch":       round(state.epoch or 0, 3),
                "train_loss":  self._last_train_loss,
                "eval_loss":   eval_loss,
                "perplexity":  round(perplexity, 4),
            })
            logger.info(
                f"[Step {state.global_step}] "
                f"train_loss={self._last_train_loss}  "
                f"eval_loss={eval_loss:.4f}  "
                f"perplexity={perplexity:.2f}"
            )

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % GPU_LOG_STEPS == 0:
            log_gpu_memory(prefix=f"[Step {state.global_step}] ")


class EarlyStoppingCallback:
    """Stops training if eval loss increases for EARLY_STOPPING_PATIENCE consecutive evals."""

    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE):
        self.patience     = patience
        self.best_loss    = float("inf")
        self.bad_evals    = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss", float("inf"))
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.bad_evals = 0
        else:
            self.bad_evals += 1
            logger.warning(
                f"Eval loss did not improve ({self.bad_evals}/{self.patience}). "
                f"Current={eval_loss:.4f}  Best={self.best_loss:.4f}"
            )
            if self.bad_evals >= self.patience:
                logger.warning("Early stopping triggered.")
                control.should_training_stop = True


# We need math for the CSV callback
import math


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train(model, tokenizer, train_ds, eval_ds, device: str) -> None:
    from transformers import (
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        TrainerCallback,
    )

    # ── Wrap our custom callbacks as TrainerCallback subclasses ───────────────

    csv_cb = TrainingCSVCallback(TRAINING_CSV)
    es_cb  = EarlyStoppingCallback(EARLY_STOPPING_PATIENCE)

    class _CsvTrainerCB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            csv_cb.on_log(args, state, control, logs=logs, **kwargs)

        def on_step_end(self, args, state, control, **kwargs):
            csv_cb.on_step_end(args, state, control, **kwargs)

    class _EsTrainerCB(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            es_cb.on_evaluate(args, state, control, metrics=metrics, **kwargs)
            return control

    # ── TrainingArguments ─────────────────────────────────────────────────────
    bf16_available = BF16 and device == "cuda" and torch.cuda.is_bf16_supported()
    fp16_fallback  = (not bf16_available) and (device == "cuda")

    training_args = TrainingArguments(
        output_dir                  = str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs            = NUM_TRAIN_EPOCHS,
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH,
        per_device_eval_batch_size  = 1,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION,
        learning_rate               = LEARNING_RATE,
        warmup_ratio                = WARMUP_RATIO,
        lr_scheduler_type           = LR_SCHEDULER,
        bf16                        = bf16_available,
        fp16                        = fp16_fallback,
        gradient_checkpointing      = True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm               = MAX_GRAD_NORM,
        evaluation_strategy         = "steps",
        eval_steps                  = EVAL_STEPS,
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,
        logging_steps               = LOGGING_STEPS,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        dataloader_num_workers      = 4,
        report_to                   = "none",   # no wandb / tensorboard
        remove_unused_columns       = False,
        prediction_loss_only        = True,
        optim                       = "adamw_torch_fused" if device == "cuda" else "adamw_torch",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
        data_collator = data_collator,
        callbacks     = [_CsvTrainerCB(), _EsTrainerCB()],
    )

    # ── Pre-training info ─────────────────────────────────────────────────────
    log_gpu_memory("Pre-training: ")
    total_steps = (
        len(train_ds)
        // (PER_DEVICE_TRAIN_BATCH * GRADIENT_ACCUMULATION)
        * NUM_TRAIN_EPOCHS
    )
    logger.info(f"Estimated training steps : {total_steps:,}")
    logger.info(f"Effective batch size     : {PER_DEVICE_TRAIN_BATCH * GRADIENT_ACCUMULATION}")
    logger.info(f"BF16 enabled             : {bf16_available}")

    # ── Train ─────────────────────────────────────────────────────────────────
    start_time = time.time()
    logger.info("Starting training…")
    trainer.train()
    elapsed = time.time() - start_time

    logger.info(f"Training completed in {elapsed/3600:.2f} hours")
    log_gpu_memory("Post-training: ")

    # ── Save final model ──────────────────────────────────────────────────────
    logger.info(f"Saving final model to {OUTPUT_DIR}…")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    logger.info("Model and tokenizer saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    for path in (TRAIN_FILE, EVAL_FILE):
        if not path.exists():
            logger.error(f"Required file not found: {path}. Run analyze_chunks.py first.")
            sys.exit(1)

    device = get_device()

    model, tokenizer = load_and_freeze_model(BASE_MODEL_NAME, device)

    logger.info("Enabling gradient checkpointing…")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    train_ds, eval_ds = build_dataset(TRAIN_FILE, EVAL_FILE, tokenizer)

    train(model, tokenizer, train_ds, eval_ds, device)

    logger.info("=" * 55)
    logger.info("PRE-TRAINING COMPLETE")
    logger.info(f"Model saved to : {OUTPUT_DIR}")
    logger.info(f"Training log   : {TRAINING_CSV}")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()

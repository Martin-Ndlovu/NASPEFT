#!/usr/bin/env python
# coding=utf-8
"""
Full fine-tuning of LLaMA-3.2-1B on WikiText-2
(All model parameters are trainable)
"""

# ------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------
import os
import re
import math
import json
import torch
import logging
import statistics
import datasets

from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
    TrainerCallback,
)

# -------------------------------------------------------------
# Environment & Logging
# -------------------------------------------------------------
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
accelerator = Accelerator()

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
MODEL_NAME = "models/Llama-3.2-1B"
OUTPUT_DIR = "output/full_finetune"

MAX_SEQ_LEN = 512
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# Dataset Utilities
# -------------------------------------------------------------
def clean_text(example):
    text = example["text"]
    text = re.sub(r'@-@', '-', text)
    text = re.sub(r'@,@', ',', text)
    text = re.sub(r'@.@', '.', text)
    return {"text": text}

def load_wikitext():
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    for split in ["train", "validation"]:
        dataset[split] = dataset[split].filter(lambda x: len(x["text"]) > 5)
        dataset[split] = dataset[split].map(clean_text)
    return dataset

# --------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------
def tokenize_fn(examples, tokenizer):
    out = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    out["labels"] = out["input_ids"].copy()
    return out

def group_texts(examples):
    concat = {k: sum(examples[k], []) for k in examples}
    total_len = (len(concat["input_ids"]) // MAX_SEQ_LEN) * MAX_SEQ_LEN
    return {
        k: [t[i:i + MAX_SEQ_LEN] for i in range(0, total_len, MAX_SEQ_LEN)]
        for k, t in concat.items()
    }

# ---------------------------------------------------------------
# Perplexity Callback
# ---------------------------------------------------------------
class PerplexityCallback(TrainerCallback):
    def __init__(self):
        self.last_eval_metrics = {}

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics")
        if metrics is None:
            return control

        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            perplexity = math.exp(eval_loss)
            metrics["eval_perplexity"] = perplexity

            # ------------------------------------------------------
            # Explicitly push into log_history (CRITICAL FIX)
            # ------------------------------------------------------
            state.log_history.append({
                "epoch": state.epoch,
                "step": state.global_step,
                "eval_loss": eval_loss,
                "eval_perplexity": perplexity,
                "eval_runtime": metrics.get("eval_runtime"),
                "eval_samples_per_second": metrics.get("eval_samples_per_second"),
                "eval_steps_per_second": metrics.get("eval_steps_per_second"),
            })

        # Store for later retrieval
        self.last_eval_metrics = metrics.copy()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control

        # ------------------------------------------------------
        # Ensure perplexity always appears in console logs
        # ------------------------------------------------------
        if "eval_loss" in logs and "eval_perplexity" not in logs:
            eval_loss = logs.get("eval_loss")
            if eval_loss is not None:
                logs["eval_perplexity"] = math.exp(eval_loss)

        return control

# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    set_seed(SEED)

    logger.info("Loading dataset...")
    dataset = load_wikitext()

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(
        lambda x: tokenize_fn(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    tokenized = tokenized.map(group_texts, batched=True)

    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    logger.info("Loading model for full fine-tuning...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        learning_rate=1e-5,
        max_steps=25000,
        warmup_ratio=0.05,
        logging_steps=500,
        eval_steps=500,
        save_steps=1000,
        save_total_limit=3,
        eval_strategy="steps",   
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_perplexity",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            PerplexityCallback(),
            # EarlyStoppingCallback(early_stopping_patience=5),
        ],
    )

    logger.info("Training (full fine-tuning)...")
    trainer.train()

    logger.info("Evaluating...")
    eval_metrics = trainer.evaluate()

    if "eval_loss" in eval_metrics:
        eval_metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])

    with open(os.path.join(OUTPUT_DIR, "eval_results.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)

    if accelerator.is_main_process:
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

    logger.info("Full fine-tuning completed successfully.")

# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()

"""
Layer Selection with NASPEFT reference adapter (LoRA + ParBn + PrefixTuning union).

Implements §3.3 "Layer-Sensitivity-Based Search Space Reduction" with three
stages, all using the union adapter (no full-model fine-tuning anywhere).

  Stage A — Per-layer probe
    Fine-tune each layer i in isolation under the reference NASPEFT config.
    P_i (lower = better, paper's minimization convention) is recorded.

  Stage B — Candidate-threshold evaluation
    For each candidate percentile in {25, 50, 75, 100}, build
        G1 = {i | P_i <= tau_p}
    and run one union-adapter fine-tune with only G1 active. The tau_100
    candidate is mathematically equivalent to leave_out=[] (all layers active),
    so it serves as the BASELINE for the gain formulas — not a full-model
    fine-tune, but an all-layers union-adapter fine-tune. This is the apples-
    to-apples baseline: it isolates "did being selective help?" from "did
    adapters help?".

  Stage C — Gain-based auto-selection (eqs. 3.7 / 3.8 from §3.3)
    perf_gain  = (baseline_perf  - candidate_perf ) / |baseline_perf|  * 100  (lower-is-better metrics)
    perf_gain  = (candidate_perf - baseline_perf ) / |baseline_perf|  * 100  (higher-is-better metrics)
    param_gain = (baseline_param - candidate_param) / baseline_param * 100

    Selection: argmax (perf_gain + param_gain) over {25, 50, 75}.
    tau_100 is excluded from selection because it IS the baseline (its gains
    are 0 by definition).

All runs use the SAME number of epochs (default 10) to keep the comparison
fair. The only knob you typically need to set is --epochs.

Public entry point:
    run_layer_selection(model_name_or_path, dataset_name, **kwargs) -> dict
"""

import sys
import os
import json
import math
import logging
from typing import Optional, List, Dict, Any, Tuple
from xml.parsers.expat import model

import numpy as np
import torch

# Path bootstrap — adjust if your project layout differs
sys.path.append('/root/Martin/NasPEFT/naspeft')

from transformers import (
    AutoConfig,
    AutoTokenizer,
    set_seed,
    TrainingArguments,
    EvalPrediction,
    DataCollatorForLanguageModeling,
    default_data_collator,
)
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback

import adapters
from adapters import (
    AutoAdapterModel,
    AdapterTrainer,
    ConfigUnion,
    LoRAConfig,
    ParBnConfig,
    PrefixTuningConfig,
)

from datasets import load_dataset, load_from_disk
from evaluate import load as load_metric

logger = logging.getLogger(__name__)


# =============================================================================
# Task registry
# =============================================================================
TASK_TO_KEYS = {
    "cola":    ("sentence", None),
    "mnli":    ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc":    ("sentence1", "sentence2"),
    "qnli":    ("question", "sentence"),
    "qqp":     ("question1", "question2"),
    "rte":     ("sentence1", "sentence2"),
    "sst2":    ("sentence", None),
    "stsb":    ("sentence1", "sentence2"),
    "wnli":    ("sentence1", "sentence2"),
}

# (metric_key_in_trainer_output, direction)
#   direction = +1  --> higher is better (accuracy, F1, MCC, Spearman)
#   direction = -1  --> lower is better  (loss, perplexity)
#
# Reporting matrix (matches main-results convention):
#   - CoLA  : Matthews correlation
#   - STS-B : Spearman correlation
#   - MNLI  : matched accuracy (handled by selecting validation_matched split)
#   - all other GLUE tasks: accuracy
# The probe metric must equal the reported metric so threshold selection
# optimizes the same quantity we report.
GLUE_METRIC_SPEC = {
    "cola": ("eval_matthews_correlation", +1),
    "stsb": ("eval_spearmanr",            +1),
    "mrpc": ("eval_accuracy",             +1),
    "qqp":  ("eval_accuracy",             +1),
    "mnli": ("eval_accuracy",             +1),
    "qnli": ("eval_accuracy",             +1),
    "rte":  ("eval_accuracy",             +1),
    "sst2": ("eval_accuracy",             +1),
    "wnli": ("eval_accuracy",             +1),
}


# =============================================================================
# Reference NASPEFT adapter (union of LoRA + ParBn + PrefixTuning)
#   - "fixed reference configuration that activates all three modules at
#      default capacity" (paper §3.3)
# =============================================================================
def _build_reference_adapter(num_layers: int, leave_out: List[int]) -> ConfigUnion:
    # Reference capacities chosen to match AutoPEFT's economical baseline so
    # the probe parameter % is comparable to what their main results report
    # (~0.76% on BERT-base / RTE with rf=96 parallel, prefix_length=1).
    # The dominant cost is prefix length: dropping from 128 to 1 alone takes
    # us from ~3% to ~1% of model params.
    return ConfigUnion(
        LoRAConfig(
            r=16,
            alpha=32,
            attn_matrices=["q", "v"],          # LoRA paper default; AutoPEFT-style
            leave_out= leave_out, #[3,4,8,11],
            dropout=0.1,
        ),
        ParBnConfig(
            reduction_factor=16,               
            leave_out= leave_out, #[3,4,8,11],
            non_linearity="relu",
            output_adapter=True,
            dropout=0.1,
        ),
        PrefixTuningConfig(
            prefix_length=2,    
            bottleneck_size=1024, 
            flat=True,                                     
            leave_out= leave_out,  # leave_out, #[3,4,8,11],
            non_linearity="tanh",
        ),
    )


# =============================================================================
# Perplexity callback
# =============================================================================
class _PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics")
        if metrics is None:
            return control
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and "eval_perplexity" not in metrics:
            try:
                metrics["eval_perplexity"] = math.exp(eval_loss)
            except OverflowError:
                metrics["eval_perplexity"] = float("inf")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control
        if "eval_loss" in logs and "eval_perplexity" not in logs:
            eval_loss = logs.get("eval_loss")
            if eval_loss is not None:
                try:
                    logs["eval_perplexity"] = math.exp(eval_loss)
                except OverflowError:
                    logs["eval_perplexity"] = float("inf")
        return control


# =============================================================================
# Dataset loading & preprocessing
# =============================================================================
def _load_and_preprocess(
    dataset_name: str,
    tokenizer,
    max_seq_length: int,
    glue_local_root: Optional[str] = None,
    dataset_config_name: Optional[str] = None,
):
    """Returns (train_ds, eval_ds, test_ds, task_name, is_causal_lm, num_labels, is_regression)."""
    ds_lower = dataset_name.lower()

    # ------------------------------- Causal LM -------------------------------
    if "wikitext" in ds_lower:
        cfg = dataset_config_name or "wikitext-103-raw-v1"
        raw = load_dataset(dataset_name, cfg)
        task_name = "wikitext"

        import re

        def filter_texts(example):
            t = example.get("text")
            return t is not None and isinstance(t, str) and len(t) > 5

        def clean_special(example):
            t = example["text"]
            t = re.sub(r'@-@', '-', t)
            t = re.sub(r'@,@', ',', t)
            t = re.sub(r'@\.@', '.', t)
            return {"text": t}

        for split in ["train", "validation", "test"]:
            raw[split] = raw[split].filter(filter_texts, batched=False)
            raw[split] = raw[split].map(clean_special, batched=False)

        def tok_fn(examples):
            out = tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_length,
            )
            out["labels"] = [list(ids) for ids in out["input_ids"]]
            return out

        tokenized = raw.map(
            tok_fn,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing",
        )

        def group_texts(examples):
            concat = {k: sum(examples[k], []) for k in examples.keys()}
            total = len(concat[list(examples.keys())[0]])
            total = (total // max_seq_length) * max_seq_length
            return {
                k: [t[i:i + max_seq_length] for i in range(0, total, max_seq_length)]
                for k, t in concat.items()
            }

        tokenized = tokenized.map(
            group_texts,
            batched=True,
            desc=f"Grouping into {max_seq_length}-token chunks",
        )

        return (
            tokenized["train"],
            tokenized["validation"],
            tokenized.get("test", tokenized["validation"]),
            task_name, True, None, False,
        )

    # ------------------------------- GLUE -------------------------------
    if ds_lower in TASK_TO_KEYS:
        task_name = ds_lower
        if glue_local_root is not None:
            raw = load_from_disk(os.path.join(glue_local_root, task_name))
        else:
            raw = load_dataset("glue", task_name)

        is_regression = (task_name == "stsb")
        num_labels = 1 if is_regression else len(raw["train"].features["label"].names)

        s1, s2 = TASK_TO_KEYS[task_name]

        def tok_fn(examples):
            args = (examples[s1],) if s2 is None else (examples[s1], examples[s2])
            out = tokenizer(*args, padding="max_length", max_length=max_seq_length, truncation=True)
            if "label" in examples:
                out["labels"] = examples["label"]
            return out

        tokenized = raw.map(
            tok_fn,
            batched=True,
            remove_columns=[c for c in raw["train"].column_names if c not in ("label",)],
            desc="Tokenizing",
        )

        train_ds = tokenized["train"]
        eval_key = "validation_matched" if task_name == "mnli" else "validation"
        test_key = "test_matched" if task_name == "mnli" else "test"
        eval_ds = tokenized[eval_key]
        test_ds = tokenized.get(test_key, eval_ds)

        # GLUE test sets ship without labels: fall back to validation for scoring.
        if "label" in test_ds.column_names:
            sample_label = test_ds[0].get("label", -1)
            if sample_label == -1:
                logger.info(
                    f"[{task_name}] test split has no labels; using validation set for scoring."
                )
                test_ds = eval_ds

        return train_ds, eval_ds, test_ds, task_name, False, num_labels, is_regression

    raise ValueError(f"Unsupported dataset: {dataset_name}")


# =============================================================================
# Model + adapter construction
# =============================================================================
def _build_model_with_adapter(
    model_name_or_path: str,
    is_causal_lm: bool,
    num_labels: Optional[int],
    task_name: str,
    tokenizer,
    leave_out: List[int],
    num_layers: int,
    adapter_name: str = "naspeft_ref",
):
    if is_causal_lm:
        model = AutoAdapterModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.use_cache = False
    else:
        config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels, finetuning_task=task_name
        )
        model = AutoAdapterModel.from_pretrained(model_name_or_path, config=config)
        if "t5" in model_name_or_path.lower():
            try:
                model.delete_head("default")
            except Exception:
                pass
        model.add_classification_head(task_name, num_labels=num_labels)

    adapters.init(model)
    try:
        existing = list(model.adapters_config.adapters.keys())
    except Exception:
        existing = []
    for ad in existing:
        try:
            model.delete_adapter(ad)
        except Exception:
            pass

    adapter_config = _build_reference_adapter(num_layers, leave_out=leave_out)
    model.add_adapter(adapter_name, config=adapter_config)
    model.train_adapter([adapter_name])
    return model, adapter_name


def _count_param_percentage(model) -> float:
    """Return adapter_params / pretrained_params * 100.

    "The percentage of parameters is
    the ratio of the number of additional parameters to the pretrained
    parameters." Concretely:
      - adapter_params  = sum of trainable params (everything with requires_grad=True
                          after train_adapter() was called, i.e. the adapter modules).
      - pretrained_params = the original backbone size (everything that is NOT
                          trainable, i.e. requires_grad=False). This excludes the
                          adapter modules and excludes any added prediction heads
                          (which are also frozen when only the adapter is trained).
    Using this convention produces the same number that model.adapter_summary()
    reports under %Param,.
    """
    trainable = 0
    pretrained = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            pretrained += p.numel()
    if pretrained == 0:
        return 0.0
    return 100.0 * trainable / pretrained


# =============================================================================
# One union-adapter fine-tune run
#   - Used identically for the per-layer probe and for every threshold candidate.
#   - All runs share the same number of epochs (paper requirement: fairness).
# =============================================================================
def _fine_tune_run(
    *,
    run_tag: str,
    leave_out: List[int],
    num_layers: int,
    model_name_or_path: str,
    is_causal_lm: bool,
    num_labels: Optional[int],
    is_regression: bool,
    task_name: str,
    tokenizer,
    train_ds, eval_ds, test_ds,
    output_dir: str,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
    max_train_samples: Optional[int],
    glue_metrics_path: Optional[str],
) -> Dict[str, Any]:
    """Train the union adapter under the given leave_out mask; return metrics + param%."""

    run_dir = os.path.join(output_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)

    model, adapter_name = _build_model_with_adapter(
        model_name_or_path=model_name_or_path,
        is_causal_lm=is_causal_lm,
        num_labels=num_labels,
        task_name=task_name,
        tokenizer=tokenizer,
        leave_out=leave_out,
        num_layers=num_layers,
    )
    adapter_summary = None
    summary_param_pct = None
    try:
        adapter_summary = model.adapter_summary(as_dict=True)
        logger.info(f"[{run_tag}] Adapter summary:\n{model.adapter_summary()}")
        # Pull the official param% from the adapter library's summary so the
        # number in our JSON matches the number printed in the training logs.
        # The "Full model" row is always last; everything before it is an
        # adapter/head row whose %param is on the same denominator.
        if isinstance(adapter_summary, list):
            for row in adapter_summary:
                if row.get("name") == "naspeft_ref":
                    summary_param_pct = float(row.get("%param", 0.0))
                    break
    except Exception:
        adapter_summary = None
    # Prefer the library's own number; fall back to our counter if it's missing.
    param_pct = summary_param_pct if summary_param_pct is not None else _count_param_percentage(model)

    train_used = train_ds
    if max_train_samples is not None and max_train_samples < len(train_ds):
        train_used = train_ds.select(range(max_train_samples))

    if is_causal_lm:
        metric_for_best = "eval_perplexity"
        greater_is_better = False
    else:
        m_name, direction = GLUE_METRIC_SPEC.get(task_name, ("eval_accuracy", +1))
        metric_for_best = m_name
        greater_is_better = (direction == +1)

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=greater_is_better,
        seed=seed,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=None,
        report_to="none",
        fp16=False,
        bf16=torch.cuda.is_available(),
        overwrite_output_dir=True,
        remove_unused_columns=False,
    )

    if is_causal_lm:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            if not isinstance(logits, torch.Tensor):
                logits = torch.from_numpy(logits)
            if not isinstance(labels, torch.Tensor):
                labels = torch.from_numpy(labels)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            try:
                ppl = math.exp(loss.item())
            except OverflowError:
                ppl = float("inf")
            return {"perplexity": ppl}

        callbacks = [
            _PerplexityCallback(),
            EarlyStoppingCallback(early_stopping_patience=patience),
        ]
    else:
        data_collator = default_data_collator
        metric = None
        if glue_metrics_path is not None and os.path.exists(glue_metrics_path):
            try:
                metric = load_metric(glue_metrics_path, task_name)
            except Exception as e:
                logger.warning(f"Failed to load local GLUE metric from {glue_metrics_path}: {e}")
        if metric is None:
            try:
                m = load_metric("glue", task_name)
                # evaluate.load may return None on registry/download failure
                if m is not None and hasattr(m, "compute"):
                    metric = m
            except Exception as e:
                logger.warning(f"evaluate.load('glue', {task_name!r}) failed: {e}")

        if metric is not None:
            def compute_metrics(p: EvalPrediction):
                preds = p.predictions
                if isinstance(preds, tuple):
                    preds = preds[0]
                preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
                result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = float(np.mean(list(result.values())))
                return result
        else:
            # Inline fallback — no network or registry needed.
            # Mirrors the metrics the HF GLUE module would compute per task.
            logger.info(f"Using inline GLUE metric fallback for task={task_name!r}.")
            from sklearn.metrics import (
                accuracy_score, f1_score, matthews_corrcoef
            )
            from scipy.stats import pearsonr, spearmanr

            def compute_metrics(p: EvalPrediction):
                preds = p.predictions
                if isinstance(preds, tuple):
                    preds = preds[0]
                preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
                refs = p.label_ids
                out = {}
                if task_name == "stsb":
                    # Regression — Spearman is primary, Pearson for completeness
                    out["pearson"]   = float(pearsonr(preds, refs)[0])
                    out["spearmanr"] = float(spearmanr(preds, refs)[0])
                elif task_name == "cola":
                    # CoLA primary metric is Matthews correlation
                    out["matthews_correlation"] = float(matthews_corrcoef(refs, preds))
                    out["accuracy"] = float(accuracy_score(refs, preds))
                elif task_name in ("mrpc", "qqp"):
                    out["accuracy"] = float(accuracy_score(refs, preds))
                    out["f1"]       = float(f1_score(refs, preds))
                else:  # sst2, mnli, qnli, rte, wnli
                    out["accuracy"] = float(accuracy_score(refs, preds))
                if len(out) > 1:
                    out["combined_score"] = float(np.mean(list(out.values())))
                return out

        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_used,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    train_result = trainer.train()
    test_metrics = trainer.evaluate(test_ds)

    out = {
        "run_tag": run_tag,
        "leave_out": leave_out,
        "active_layers": [i for i in range(num_layers) if i not in leave_out],
        "num_active_layers": num_layers - len(leave_out),
        "param_pct": param_pct,
        "train_metrics": train_result.metrics,
        "test_metrics": test_metrics,
        "adapter_summary": adapter_summary,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


# =============================================================================
# Score / gain helpers
# =============================================================================
def _extract_score(metrics: Dict[str, float], is_causal_lm: bool, task_name: str) -> Tuple[float, float]:
    """Return (P_i_minimization, raw_value).

      - P_i_minimization : lower-is-better, for ranking layers (stage A).
      - raw_value        : metric as reported (perplexity, or accuracy in [0,1]),
                           used for gain computation (stage C).
    """
    if is_causal_lm:
        ppl = metrics.get("eval_perplexity")
        if ppl is None:
            loss = metrics.get("eval_loss")
            if loss is None:
                return float("inf"), float("inf")
            try:
                ppl = math.exp(loss)
            except OverflowError:
                ppl = float("inf")
        return float(ppl), float(ppl)

    m_name, direction = GLUE_METRIC_SPEC.get(task_name, ("eval_accuracy", +1))
    val = metrics.get(m_name)
    if val is None:
        for fb in ("eval_accuracy", "eval_f1", "eval_combined_score", "eval_loss"):
            if fb in metrics:
                val = metrics[fb]
                if fb == "eval_loss":
                    direction = -1
                break
    if val is None:
        return float("inf"), float("nan")
    raw = float(val)
    p_min = -raw if direction == +1 else raw
    return p_min, raw


def _perf_gain(baseline_raw: float, candidate_raw: float, *, lower_is_better: bool) -> float:
    """Eq. 3.7 generalised to both directions.

    Lower-is-better (perplexity, loss):
        gain = (baseline - candidate) / |baseline| * 100

    Higher-is-better (accuracy, F1, MCC, Spearman)
    """
    if baseline_raw == 0 or not math.isfinite(baseline_raw) or not math.isfinite(candidate_raw):
        return float("nan")
    if lower_is_better:
        return (baseline_raw - candidate_raw) / abs(baseline_raw) * 100.0
    return (candidate_raw - baseline_raw) / abs(baseline_raw) * 100.0


def _param_gain(baseline_param: float, candidate_param: float) -> float:
    """Eq. 3.8 — params are lower-is-better."""
    if baseline_param == 0:
        return float("nan")
    return (baseline_param - candidate_param) / baseline_param * 100.0


# =============================================================================
# Public entry point
# =============================================================================
def run_layer_selection(
    model_name_or_path: str,
    dataset_name: str,
    output_root: str = "output/Ex1",
    epochs: int = 5,
    batch_size: int = 8,
    eval_batch_size: int = 8,
    max_seq_length: int = 128,
    learning_rate: float = 1e-4,
    patience: int = 3,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    glue_local_root: Optional[str] = None,
    glue_metrics_path: Optional[str] = None,
    dataset_config_name: Optional[str] = None,
    candidate_percentiles: Tuple[float, ...] = (25.0, 50.0, 75.0, 100.0),
) -> Dict[str, Any]:
    """Run the full layer-sensitivity pipeline + percentile auto-selection.

    All runs (per-layer probe and every threshold candidate) use the same
    number of epochs for a fair comparison. Only the union adapter is trained
    in every run — the backbone weights are always frozen.

    Args:
        model_name_or_path     : backbone (e.g. "roberta-base", "meta-llama/Llama-3.2-1B").
        dataset_name           : "wikitext" or a GLUE task name.
        epochs                 : single epoch budget for every run (default 10).
        candidate_percentiles  : thresholds to evaluate. Must include 100, which
                                 is used as the baseline (all-layers union adapter).
                                 Default: (25, 50, 75, 100).

    Returns:
        dict with `searchable_layers` (G1), `frozen_layers` (G2), `num_searchable`,
        the chosen percentile/threshold, and full per-candidate diagnostics.
    """
    set_seed(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    output_dir = os.path.join(
        output_root,
        f"{os.path.basename(model_name_or_path)}_{dataset_name}",
    )
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    logger.info(f"Layer selection: model={model_name_or_path} dataset={dataset_name}")
    logger.info(f"Output dir: {output_dir}")

    # Ensure the baseline percentile (100) is present; it's the gain denominator.
    if 100.0 not in candidate_percentiles:
        candidate_percentiles = tuple(list(candidate_percentiles) + [100.0])
        logger.info("Added tau_100 to candidate_percentiles (required as baseline).")

    # ---- Tokenizer ----
    is_causal_hint = any(
        kw in model_name_or_path.lower()
        for kw in ["llama", "mistral", "mixtral", "gemma", "phi", "qwen", "falcon", "deepseek", "yi"]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=is_causal_hint
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Dataset ----
    (train_ds, eval_ds, test_ds,
     task_name, is_causal_lm, num_labels, is_regression) = _load_and_preprocess(
        dataset_name, tokenizer, max_seq_length,
        glue_local_root=glue_local_root,
        dataset_config_name=dataset_config_name,
    )
    logger.info(
        f"Loaded — task={task_name} causal_lm={is_causal_lm} "
        f"train={len(train_ds)} eval={len(eval_ds)} test={len(test_ds)}"
    )

    # ---- Layer count ----
    model_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=is_causal_hint)
    num_layers = getattr(model_cfg, "num_hidden_layers", None) or getattr(model_cfg, "num_layers", 12)
    logger.info(f"Backbone has {num_layers} layers")

    lower_is_better = is_causal_lm  # perplexity vs higher-is-better GLUE metrics

    # =========================================================================
    # STAGE A: per-layer probe → P_i
    # =========================================================================
    logger.info("=" * 60)
    logger.info(f"STAGE A — Per-layer probe ({num_layers} runs, {epochs} epochs each)")
    logger.info("=" * 60)

    per_layer_runs: List[Dict[str, Any]] = []
    scores_min: List[float] = []

    for layer_idx in range(num_layers):
        logger.info(f"--- Probing layer {layer_idx + 1}/{num_layers} ---")
        leave_out = [i for i in range(num_layers) if i != layer_idx]
        run = _fine_tune_run(
            run_tag=f"probe/layer_{layer_idx}",
            leave_out=leave_out,
            num_layers=num_layers,
            model_name_or_path=model_name_or_path,
            is_causal_lm=is_causal_lm,
            num_labels=num_labels,
            is_regression=is_regression,
            task_name=task_name,
            tokenizer=tokenizer,
            train_ds=train_ds, eval_ds=eval_ds, test_ds=test_ds,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            patience=patience,
            seed=seed,
            max_train_samples=max_train_samples,
            glue_metrics_path=glue_metrics_path,
        )
        p_min, p_raw = _extract_score(run["test_metrics"], is_causal_lm, task_name)
        run["P_i_minimization"] = p_min
        run["P_i_raw"] = p_raw
        per_layer_runs.append(run)
        scores_min.append(p_min)
        logger.info(f"layer {layer_idx} — P_i (min-conv) = {p_min:.4f}, raw = {p_raw:.4f}")

    # =========================================================================
    # STAGE B: one union-adapter fine-tune per candidate percentile
    # (tau_100 == leave_out=[], which serves as the baseline)
    # =========================================================================
    logger.info("=" * 60)
    logger.info(
        f"STAGE B — Candidate runs ({len(candidate_percentiles)} percentiles, "
        f"{epochs} epochs each — tau_100 is the all-layers baseline)"
    )
    logger.info("=" * 60)

    finite_scores = np.array([s for s in scores_min if math.isfinite(s)], dtype=float)
    if len(finite_scores) == 0:
        raise RuntimeError("All per-layer probe scores were non-finite — cannot threshold.")

    # We run each unique G1 once. tau_100 always corresponds to the all-layers
    # mask; smaller percentiles may collide with each other (or even with tau_100
    # on small models), in which case the second hit reuses the first run.
    sorted_pcts = sorted(set(candidate_percentiles))
    candidates: List[Dict[str, Any]] = []
    seen_layer_sets: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    for pct in sorted_pcts:
        tau = float(np.percentile(finite_scores, pct))
        searchable = [i for i, s in enumerate(scores_min) if math.isfinite(s) and s <= tau]
        if len(searchable) == 0:
            logger.warning(f"tau_{pct:g} yielded |G1|=0; skipping this candidate.")
            candidates.append({
                "percentile": pct,
                "threshold": tau,
                "searchable_layers": [],
                "frozen_layers": list(range(num_layers)),
                "num_active_layers": 0,
                "raw_metric": float("nan"),
                "param_pct": 0.0,
                "perf_gain": float("nan"),
                "param_gain": float("nan"),
                "selection_score": float("-inf"),
                "skipped": True,
                "is_baseline": (pct == 100.0),
            })
            continue
        frozen = [i for i in range(num_layers) if i not in searchable]

        layer_key = tuple(searchable)
        if layer_key in seen_layer_sets:
            prev = seen_layer_sets[layer_key]
            dup = dict(prev)
            dup["percentile"] = pct
            dup["threshold"] = tau
            dup["duplicate_of"] = prev["percentile"]
            dup["is_baseline"] = (pct == 100.0)
            candidates.append(dup)
            logger.info(
                f"tau_{pct:g} = {tau:.4f} produces same G1 as tau_{prev['percentile']:g} — reusing result."
            )
            continue

        is_baseline = (pct == 100.0)
        tag_role = "BASELINE" if is_baseline else "candidate"
        logger.info(
            f"--- {tag_role} tau_{pct:g} = {tau:.4f} | |G1|={len(searchable)} G1={searchable} ---"
        )
        run = _fine_tune_run(
            run_tag=f"{'baseline' if is_baseline else 'candidate'}/tau_{int(pct)}",
            leave_out=frozen,
            num_layers=num_layers,
            model_name_or_path=model_name_or_path,
            is_causal_lm=is_causal_lm,
            num_labels=num_labels,
            is_regression=is_regression,
            task_name=task_name,
            tokenizer=tokenizer,
            train_ds=train_ds, eval_ds=eval_ds, test_ds=test_ds,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            patience=patience,
            seed=seed,
            max_train_samples=max_train_samples,
            glue_metrics_path=glue_metrics_path,
        )
        _, cand_raw = _extract_score(run["test_metrics"], is_causal_lm, task_name)
        cand_param = run["param_pct"]

        entry = {
            "percentile": pct,
            "threshold": tau,
            "searchable_layers": searchable,
            "frozen_layers": frozen,
            "num_active_layers": len(searchable),
            "raw_metric": cand_raw,
            "param_pct": cand_param,
            "perf_gain": float("nan"),       
            "param_gain": float("nan"),
            "selection_score": float("-inf"),
            "skipped": False,
            "is_baseline": is_baseline,
            "run": run,
        }
        seen_layer_sets[layer_key] = entry
        candidates.append(entry)
        logger.info(
            f"tau_{pct:g}: raw={cand_raw:.4f}, param%={cand_param:.4f}"
            + (" (baseline)" if is_baseline else "")
        )

    # ---- Identify the baseline (tau_100) ----
    baseline = next(
        (c for c in candidates if c.get("is_baseline") and not c.get("skipped")),
        None,
    )
    if baseline is None:
        raise RuntimeError(
            "tau_100 baseline run did not produce valid metrics — cannot compute gains."
        )
    baseline_raw = baseline["raw_metric"]
    baseline_param = baseline["param_pct"]
    logger.info(
        f"BASELINE (tau_100): raw_metric={baseline_raw:.4f}, param%={baseline_param:.4f}"
    )

    # ---- Compute gains for every candidate ----
    # selection_score uses perf_gain ONLY.
    #
    # WHY NOT perf_gain + param_gain (or any other fixed perf-vs-param weighting)?
    #
    # Two reasons, one structural and one design-coherence:
    #
    # 1. STRUCTURAL: |G1| is monotone in the percentile (nested-set property of
    #    the threshold rule G1 = {i : P_i <= tau_p}), and the union adapter
    #    contributes a uniform per-layer parameter cost. Therefore param_gain
    #    is fully determined by the percentile choice — it carries no
    #    information beyond "lower percentile = fewer params" and behaves like
    #    a fixed prior, not a measurement.
    #
    # 2. DESIGN COHERENCE WITH THE MAIN LOOP: the framework's main acquisition
    #    function (eq. 6 in the paper) ALREADY traverses the Pareto front of
    #    (perf, params) inside the search space we are about to fix. It does
    #    so via a randomized scalarization
    #        alpha_t(a) = lambda(t) * [p_tilde(a) - kappa*sigma_tilde(a)]
    #                   + (1 - lambda(t)) * c_tilde(a)
    #    where lambda(t) is drawn uniformly each iteration. This sweeps the
    #    front rather than committing to any one weighting.
    #
    #    If layer selection committed to a fixed weighting (e.g.
    #    perf_gain + param_gain), it would pre-bias the search space toward
    #    one corner of the front BEFORE the main loop's lambda(t) sweep
    #    starts. The main loop would then traverse a front whose shape was
    #    already distorted by an arbitrary pre-commitment. The cleaner
    #    division of labour is:
    #
    #      - layer selection: choose the search-space SHAPE (|G1|) by
    #        picking the most aggressive reduction that doesn't hurt
    #        performance. perf_gain captures exactly this.
    #      - main loop: traverse the perf-vs-param Pareto front WITHIN
    #        the chosen |G1| via lambda(t). c(a) participates here.
    #
    # The candidate set {25, 50, 75} itself encodes the param preference at
    # this stage (we wouldn't propose 90 if we cared little about reduction).
    # param_gain remains in the per-candidate report as a descriptive figure
    # — useful for the paper's analogue of Table 3-2 — but not a selection
    # driver.
    for c in candidates:
        if c.get("skipped"):
            continue
        if c.get("is_baseline"):
            c["perf_gain"] = 0.0
            c["param_gain"] = 0.0
            c["selection_score"] = 0.0
            continue
        c["perf_gain"]  = _perf_gain(baseline_raw, c["raw_metric"], lower_is_better=lower_is_better)
        c["param_gain"] = _param_gain(baseline_param, c["param_pct"])
        if math.isfinite(c["perf_gain"]):
            c["selection_score"] = c["perf_gain"]
        logger.info(
            f"tau_{c['percentile']:g}: perf_gain={c['perf_gain']:+.2f}% "
            f"(param_gain={c['param_gain']:+.2f}%, descriptive only)"
        )

    # =========================================================================
    # STAGE C: auto-select among non-baseline candidates
    #
    #   tau_100 is the baseline and is NEVER returned as the selection — the
    #   goal of layer selection is to find a useful subset, and "all layers"
    #   is the thing we are trying to reduce from. Even if no smaller subset
    #   beats baseline performance, we still return the best subset (i.e. the
    #   one whose perf_gain is closest to zero or most positive). The caller
    #   can read selected_perf_gain to see whether reduction helped.
    #
    #   Tie-break: when two candidates have effectively identical perf_gain
    #   (within PERF_GAIN_TIE_EPS), prefer the one with the smaller |G1|.
    #   "All else equal, fewer trainable layers is better" — this is the
    #   single place in selection where the smaller-is-better preference for
    #   parameters enters, and it only fires when the perf signal cannot
    #   distinguish the candidates.
    # =========================================================================
    PERF_GAIN_TIE_EPS = 0.5  # percent; perf_gains within this band are "tied"
    selectable = [
        c for c in candidates
        if not c.get("skipped") and not c.get("is_baseline")
           and math.isfinite(c["selection_score"])
    ]
    if not selectable:
        raise RuntimeError("No non-baseline candidate produced a finite gain score.")
    # Primary key: perf_gain (higher = better). Tie-break: prefer fewer active layers.
    def _selection_key(c):
        bucket = round(c["selection_score"] / PERF_GAIN_TIE_EPS)
        return (bucket, -c["num_active_layers"])
    best = max(selectable, key=_selection_key)

    result = {
        "model_name_or_path": model_name_or_path,
        "dataset_name": dataset_name,
        "task_name": task_name,
        "is_causal_lm": is_causal_lm,
        "metric_direction": "lower_is_better" if lower_is_better else "higher_is_better",
        "num_layers": num_layers,
        "epochs": epochs,
        "per_layer_scores_minimization": scores_min,
        "per_layer_raw_metric": [r["P_i_raw"] for r in per_layer_runs],

        # Baseline (tau_100, all layers, union adapter only)
        "baseline_percentile": 100.0,
        "baseline_raw_metric": baseline_raw,
        "baseline_param_pct": baseline_param,

        "candidates": candidates,

        "selected_percentile": best["percentile"],
        "selected_threshold": best["threshold"],
        "searchable_layers": best["searchable_layers"],   # G1
        "frozen_layers": best["frozen_layers"],           # G2 (leave_out)
        "num_searchable": best["num_active_layers"],
        "selected_perf_gain": best["perf_gain"],
        "selected_param_gain": best["param_gain"],
        "selected_score": best["selection_score"],

        "output_dir": output_dir,
    }

    summary_path = os.path.join(output_dir, "layer_selection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Summary written to {summary_path}")

    logger.info("=" * 60)
    logger.info(
        f"SELECTED — tau_{best['percentile']:g} | "
        f"|G1| = {best['num_active_layers']} | "
        f"perf_gain = {best['perf_gain']:+.2f}% | "
        f"param_gain = {best['param_gain']:+.2f}%"
    )
    logger.info(f"G1 = {best['searchable_layers']}")
    logger.info("=" * 60)

    return result


# =============================================================================
# CLI
# =============================================================================
def _parse_cli_args():
    import argparse
    p = argparse.ArgumentParser(description="NASPEFT layer-sensitivity + percentile selection")
    p.add_argument("--model", default="roberta-base",
                   help="HF model id or local path (e.g. roberta-base, meta-llama/Llama-3.2-1B)")
    p.add_argument("--dataset", default="sst2",
                   help="GLUE task name (sst2, mnli, ...) or 'wikitext'")
    p.add_argument("--dataset_config", default=None,
                   help="Optional dataset config name (e.g. wikitext-103-raw-v1)")
    p.add_argument("--output_root", default="output/layer_selection")
    p.add_argument("--glue_local_root", default=None,
                   help="Path to local GLUE folder (e.g. ../datasets/glue)")
    p.add_argument("--glue_metrics_path", default=None,
                   help="Optional path to a custom glue_metrics.py")
    p.add_argument("--epochs", type=int, default=5,
                   help="Epochs for every run — probe and candidates alike (default 10)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Optional cap on train examples (applies to all runs)")
    p.add_argument("--percentiles", type=float, nargs="+",
                   default=[25.0, 50.0, 75.0, 100.0],
                   help="Candidate percentiles. tau_100 is the baseline and is "
                        "always included automatically. Default: 25 50 75 100")
    return p.parse_args()


def _fmt(x, prec=4):
    try:
        return f"{x:.{prec}f}"
    except (TypeError, ValueError):
        return str(x)


def _print_report(args, result):
    direction = result["metric_direction"]
    metric_label = "Perplexity" if result["is_causal_lm"] else f"Metric ({result['task_name']})"

    # ---------------- Per-layer probe table ----------------
    print("\n" + "=" * 78)
    print("  Stage A — Per-layer probe scores")
    print(f"  Direction: {direction}")
    print("=" * 78)
    print(f"  {'layer':>5} | {'raw P_i':>12} | {'min-conv':>12}")
    print("  " + "-" * 38)
    for i, (raw, mn) in enumerate(zip(result["per_layer_raw_metric"],
                                      result["per_layer_scores_minimization"])):
        print(f"  {i:>5} | {_fmt(raw):>12} | {_fmt(mn):>12}")

    # ---------------- Threshold table (paper Table 3-2 style) ----------------
    print("\n" + "=" * 78)
    print("  Stage B/C — Threshold partitioning (baseline = tau_100, union adapter all layers)")
    print("=" * 78)
    header = (
        f"  {'Method':<14} {'tau':>10} {'#Active':>9} "
        f"{'Param%':>9} {metric_label[:10]:>10} "
        f"{'perf_gain':>11} {'param_gain':>12} {'sel_score':>11}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    sorted_cands = sorted(
        result["candidates"],
        key=lambda c: (c.get("is_baseline", False), c["percentile"]),
    )
    for c in sorted_cands:
        tag = f"tau_{int(c['percentile'])}" + ("*" if c.get("is_baseline") else "")
        dup_note = f"  (=tau_{int(c['duplicate_of'])})" if "duplicate_of" in c else ""
        if c.get("skipped"):
            print(f"  {tag:<14} {_fmt(c['threshold'], 4):>10}  -- empty G1 --")
            continue
        print(
            f"  {tag:<14} {_fmt(c['threshold'], 4):>10} {c['num_active_layers']:>9} "
            f"{_fmt(c['param_pct'], 2):>9} "
            f"{_fmt(c['raw_metric'], 4):>10} "
            f"{_fmt(c['perf_gain'], 2):>10}% "
            f"{_fmt(c['param_gain'], 2):>11}% "
            f"{_fmt(c['selection_score'], 2):>11}{dup_note}"
        )
    print("  (* = baseline, never selected; param_gain is descriptive only —")
    print("   selection uses perf_gain alone, with smaller-G1 tie-break)")

    # ---------------- Final selection ----------------
    print("\n" + "=" * 78)
    print("  SELECTED")
    print("=" * 78)
    print(f"  percentile        : tau_{int(result['selected_percentile'])}")
    print(f"  threshold         : {result['selected_threshold']:.4f}")
    print(f"  perf_gain         : {result['selected_perf_gain']:+.2f}%  (vs. tau_100 baseline)")
    print(f"  param_gain        : {result['selected_param_gain']:+.2f}%  (descriptive, not used in selection)")
    print(f"  selection_score   : {result['selected_score']:+.2f}  (= perf_gain)")
    print(f"  |G1| searchable   : {result['num_searchable']}")
    print(f"  G1 (search over)  : {result['searchable_layers']}")
    print(f"  G2 (leave_out=1)  : {result['frozen_layers']}")
    print(f"  Full summary file : {os.path.join(result['output_dir'], 'layer_selection_summary.json')}")
    print("=" * 78)


def main():
    args = _parse_cli_args()

    print("=" * 78)
    print(f"  Layer-Sensitivity Probe + Percentile Auto-Selection")
    print(f"  model       : {args.model}")
    print(f"  dataset     : {args.dataset}" + (f" ({args.dataset_config})" if args.dataset_config else ""))
    print(f"  percentiles : {args.percentiles}  (tau_100 = baseline)")
    print(f"  epochs/run  : {args.epochs}")
    print("=" * 78)

    result = run_layer_selection(
        model_name_or_path=args.model,
        dataset_name=args.dataset,
        output_root=args.output_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        glue_local_root=args.glue_local_root,
        glue_metrics_path=args.glue_metrics_path,
        dataset_config_name=args.dataset_config,
        candidate_percentiles=tuple(args.percentiles),
    )

    _print_report(args, result)
    return result["num_searchable"], result["searchable_layers"]


if __name__ == "__main__":
    main()
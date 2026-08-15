#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning models for language modeling or GLUE classification with NAS adapters."""

from enum import auto
import json
import sys
sys.path.append('/root/Martin/NasPEFT/naspeft')
from definition import ROOT_DIR
from logger import setup_logging
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional
import re
import shutil
from collections import Counter
import datasets
import numpy as np
import torch
import math
from datasets import load_dataset, load_from_disk
from evaluate import load as load_metric
import transformers
import adapters
from adapters import (
    AutoAdapterModel,
    AdapterTrainer,
    SeqBnConfig,
    PrefixTuningConfig,
    ParBnConfig,
    LoRAConfig,
    ConfigUnion,
    Seq2SeqAdapterTrainer
)
from transformers import (
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    default_data_collator,
    EvalPrediction,
    EarlyStoppingCallback,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback
)
from transformers.trainer_utils import get_last_checkpoint

#--------------------------------------------------------------------------------
# Silence internal Hugging Face and Datasets loggers
#--------------------------------------------------------------------------------
from transformers.utils import logging as hf_logging

#--------------------------------------------------------------------------------
# Logging setup
#--------------------------------------------------------------------------------
logger = logging.getLogger(__name__)

#--------------------------------------------------------------------------------
# GLUE task keys
#--------------------------------------------------------------------------------
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

#--------------------------------------------------------------------------------
# Function to split datasets
#--------------------------------------------------------------------------------
def split_datasets(train_ds, n: int = None):
    logger.info(
        "Splitting the train/eval datasets into train/eval by "
        "using 90% and 10% of train as train and eval."
    )
    if n is None:
        n = len(train_ds)
        logger.info(f"Using the whole train dataset of {n} samples.")
    else:
        logger.info(f"Reducing the train dataset to only {n} samples.")
    split_at = int(n * 0.90)
    train_ds = train_ds.shuffle()
    new_train_ds = train_ds.select(range(split_at))
    new_eval_ds = train_ds.select(range(split_at, n))
    return new_train_ds, new_eval_ds

#---------------------------------------------------------------------------------
# Data Classes
#---------------------------------------------------------------------------------
@dataclass
class DataTrainingArguments:
    local_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to local copy of dataset"}
    )
    task_name: str = field(
        default=None,
        metadata={"help": "Name of the task (e.g. wikitext, sst2, mnli, ...)"}
    )
    patience: int = field(
        default=10,
        metadata={"help": "Number of epochs to wait before early stopping"}
    )
    resplit_dataset: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to resplit the dataset."}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset (wikitext or glue)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "Maximum total input sequence length after tokenization."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={"help": "Whether to pad all samples to `max_seq_length`."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of training examples."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of evaluation examples."}
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of prediction examples."}
    )
    custom_split: bool = field(
        default=False,
        metadata={"help": "Whether to use a custom split for the dataset."}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.dataset_name is None or self.local_dataset_path is None:
            raise ValueError(
                "Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.local_dataset_path.split(".")[-1]
            assert train_extension in [
                "csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.local_dataset_path.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."

#---------------------------------------------------------------------------------
# Training Arguments
#---------------------------------------------------------------------------------
@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to do training."}
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to do evaluation."}
    )
    output_dir: str = field(
        default="../output",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Overwrite the output directory if it exists."}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "The batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=32,
        metadata={"help": "The batch size per GPU/TPU core/CPU for evaluation."}
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."}
    )
    eval_steps: int = field(
        default=150,
        metadata={"help": "Evaluate every X steps."}
    )
    logging_strategy: str = field(
        default="steps",
        metadata={"help": "The logging strategy to use."}
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "Log every X steps."}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "The save strategy to use."}
    )
    save_steps: int = field(
        default=1000,
        metadata={"help": "Save checkpoint every X steps."}
    )
    save_total_limit: int = field(
        default=1,
        metadata={"help": "Limit the total number of checkpoints."}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "The maximum number of steps to train."}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "The initial learning rate for AdamW."}
    )
    warmup_ratio: float = field(
        default=0.00,
        metadata={"help": "The warmup ratio for learning rate."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "The maximum gradient norm."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "The seed for random number generation."}
    )
    weight_decay: float = field(
        default=0.00,
        metadata={"help": "The weight decay for AdamW."}
    )
    predict_with_generate: bool = field(
        default=True,
        metadata={"help": "Use generate() during eval/predict (required for seq2seq)."}
    )
    generation_max_length: Optional[int] = field(
        default=8,
        metadata={"help": "Max tokens to generate during eval/predict."}
    )
    generation_num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Beam width during eval/predict."}
    )

#---------------------------------------------------------------------------------
# Model Arguments
#---------------------------------------------------------------------------------
@dataclass
class ModelArguments:
    nas_adapter_config_path: str = field(
        metadata={"help": "NAS adapter config path"}
    )
    model_name_or_path: str = field(
        default="t5-large",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    adapter_name: str = field(
        default="naspeft",
        metadata={"help": "The name of the adapter to use."}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "The tokenizer to use."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
            }
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use a fast tokenizer."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Whether to use auth token."}
    )

#---------------------------------------------------------------------------------
# Adapter Arguments
#---------------------------------------------------------------------------------
@dataclass
class MultiLingAdapterArguments:
    train_adapter: bool = field(
        default=True,
        metadata={"help": "Whether to train the adapter."}
    )
    load_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the adapter to load."}
    )
    load_lang_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the language adapter to load."}
    )

#---------------------------------------------------------------------------------
# Get all checkpoints
#---------------------------------------------------------------------------------
def get_all_checkpoint(folder):
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    content = os.listdir(folder)
    checkpoints = [
        path for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    return checkpoints if checkpoints else None

#---------------------------------------------------------------------------------
# Default argument dictionaries
#---------------------------------------------------------------------------------
default_arg = {
    "non_linearity": "relu",
    "residual_before_ln": True,
    "adapter_residual_before_ln": False,
    "ln_after": False,
    "ln_before": False,
    "reduction_factor": 64,
    "leave_out": [],
    "mh_adapter": False,
    "output_adapter": True,
    "original_ln_before": True,
    "original_ln_after": True,
    "is_parallel": False,
}

default_prefix_arg = {
    "prefix_length": 1,
    "leave_out": [],
}

default_pfeiffer_arg = {
    "reduction_factor": 16,
    "leave_out": [],
    "non_linearity": "relu",
}

default_mam_arg = {
    "non_linearity": "relu",
    "reduction_factor": 64,
    "prefix_length": 1,
    "leave_out": [],
    "mh_adapter": False,
    "output_adapter": True,
}

#---------------------------------------------------------------------------------
# Main function
#---------------------------------------------------------------------------------
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments, MultiLingAdapterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
        training_args.dataloader_num_workers = 0
        training_args.dataloader_prefetch_factor = None
        training_args.load_best_model_at_end = True

    is_seq2seq = True

    # Pick the right metric for the task
    if data_args.task_name == "cola":
        training_args.metric_for_best_model = "eval_matthews_correlation"
    elif data_args.task_name in ("mrpc", "qqp"):
        training_args.metric_for_best_model = "eval_f1"
    elif data_args.task_name == "stsb":
        training_args.metric_for_best_model = "eval_spearmanr"
    else:
        training_args.metric_for_best_model = "eval_accuracy"
    training_args.greater_is_better = True

    os.makedirs(training_args.output_dir, exist_ok=True)
    setup_logging(os.path.join(training_args.output_dir, "training.log"))
    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory {training_args.output_dir} already exists and is not empty.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming at {last_checkpoint}.")

    set_seed(training_args.seed)

    # ──────────────────────────────────────────────────────────────────────────────
    #  Dataset loading
    # ──────────────────────────────────────────────────────────────────────────────
    if data_args.task_name is None:
        raise ValueError("task_name is required for GLUE classification tasks.")
    raw_datasets = load_from_disk(data_args.local_dataset_path)

    is_regression = data_args.task_name == "stsb"
    label_list = []
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ──────────────────────────────────────────────────────────────────────────────
    #  Model loading + classification head (if needed)
    # ──────────────────────────────────────────────────────────────────────────────
    model = AutoAdapterModel.from_pretrained(
        model_args.model_name_or_path,
    )

    model.config.use_cache = False
    model_param_dict = {"model": model.num_parameters()}

    # ──────────────────────────────────────────────────────────────────────────────
    #  Adapter setup
    # ──────────────────────────────────────────────────────────────────────────────
    adapters.init(model)
    if model.has_adapters():
        for adapter_name in model.get_configured_adapters():
            logger.info(f"Deleting pre-loaded adapter: {adapter_name}")
            model.delete_adapter(adapter_name)

    if adapter_args.train_adapter:
        logger.info(f"Random seed for adapter training: {training_args.seed}")
        with open(model_args.nas_adapter_config_path, "r") as f:
            config_args = json.load(f)

        leave_out_list = []
        if hasattr(model.config, "num_hidden_layers"):
            number_layer = model.config.num_hidden_layers
        elif hasattr(model.config, "num_layers"):
            number_layer = model.config.num_layers  # T5
        else:
            raise ValueError("Cannot determine number of layers for model")

        for i in range(number_layer):
            if config_args.get(f"leave_out_{i}", False):
                leave_out_list.append(int(i))
            if f"leave_out_{i}" in config_args:
                del config_args[f"leave_out_{i}"]
        config_args["leave_out"] = leave_out_list

        hidden_size = model.config.hidden_size

        if model_args.adapter_name == "prefix":
            config_args["prefix_length"] = config_args.get("prefix_length", 1)
            if "reduction_factor" in config_args:
                del config_args["reduction_factor"]
            default_arg = default_prefix_arg.copy()
            default_arg.update(config_args)
            adapter_config = PrefixTuningConfig(
                prefix_length= int(default_arg["prefix_length"]),
                bottleneck_size= int(default_arg["bottleneck_size"]),
                flat=False,
                leave_out=[]
            )

        elif model_args.adapter_name == "pfeiffers":
            config_args["reduction_factor"] = config_args.get("reduction_factor", 16)
            default_arg = default_pfeiffer_arg.copy()
            default_arg.update(config_args)
            adapter_config = SeqBnConfig(
                reduction_factor=default_arg["reduction_factor"],
                leave_out=default_arg["leave_out"],
                non_linearity=default_arg["non_linearity"]
            )

        elif model_args.adapter_name == "parallel":
            config_args["reduction_factor"] = config_args.get("reduction_factor", 64)
            default_arg = default_arg.copy()
            default_arg.update(config_args)
            adapter_config = ParBnConfig(
                reduction_factor=default_arg["reduction_factor"],
                leave_out=default_arg["leave_out"],
                non_linearity="relu", #default_arg["non_linearity"],
                # output_adapter=default_arg["output_adapter"],
                # mh_adapter=default_arg["mh_adapter"],
            )

        elif model_args.adapter_name == "lora":
            adapter_config = LoRAConfig(
                r=8,
                alpha=16,
                attn_matrices=["q", "v"],  
                dropout=0.1,
                leave_out=config_args.get("leave_out", []),                
            )
            logger.info("Using HARDCODED LoRA config: r=8, alpha=16, q+v only, all layers")

        elif model_args.adapter_name == "mam":
            prefix_flag = config_args.get("reduction_prefix", 512) != 512
            parallel_flag = config_args.get("reduction_factor", 512) <= hidden_size
            if not prefix_flag:
                config_args["prefix_length"] = 1
            else:
                config_args["prefix_length"] = hidden_size / config_args["reduction_prefix"]
            if config_args["reduction_factor"] == 512:
                config_args["reduction_factor"] = hidden_size
            if "reduction_prefix" in config_args:
                del config_args["reduction_prefix"]
            default_arg = default_mam_arg.copy()
            default_arg.update(config_args)
            config_list = []
            if prefix_flag:
                config_list.append(PrefixTuningConfig(
                    prefix_length=int(default_arg["prefix_length"]),
                    bottleneck_size=hidden_size,
                    leave_out=default_arg["leave_out"]
                ))
            if parallel_flag:
                config_list.append(ParBnConfig(
                    reduction_factor=default_arg["reduction_factor"],
                    leave_out=default_arg["leave_out"],
                    non_linearity="relu",
                    mh_adapter=False,
                    output_adapter=True
                ))
            adapter_config = ConfigUnion(*config_list) if config_list else None

        elif model_args.adapter_name == "naspeft":
            exclude_prefix = False
            exclude_lora = False
            exclude_parallel = False

            if config_args.get("reduction_prefix", hidden_size + 1) > hidden_size:
                exclude_prefix = True
            if config_args.get("lora_r", 65) > 64:
                exclude_lora = True
            if config_args.get("reduction_parallel", hidden_size + 1) > hidden_size:
                exclude_parallel = True

            config_args["reduction_prefix"] = max(1, min(config_args.get("reduction_prefix", 1), hidden_size))
            config_args["lora_r"] = max(1, min(config_args.get("lora_r", 64), 64))
            config_args["reduction_parallel"] = max(1, min(config_args.get("reduction_parallel", 1), hidden_size))

            config_flag_list = [not exclude_prefix, not exclude_lora, not exclude_parallel]
            config_list = [
                LoRAConfig(
                    r=config_args["lora_r"],
                    alpha=config_args.get("lora_alpha", 16),
                    attn_matrices=["q", "v",],
                    leave_out=config_args["leave_out"],
                    dropout=0.1,
                ),
                ParBnConfig(
                    reduction_factor=config_args["reduction_parallel"],
                    leave_out=config_args["leave_out"],
                    non_linearity="relu",
                    output_adapter=True,
                    dropout=0.1,
                ),
                PrefixTuningConfig(
                    prefix_length=config_args["reduction_prefix"],     
                    flat=True,  
                    bottleneck_size=hidden_size,
                    leave_out=config_args["leave_out"],
                    non_linearity="tanh",
                    # dropout=0.1,
                )
            ]
            adapter_config = ConfigUnion(*[config_list[i] for i in range(len(config_list)) if config_flag_list[i]])

        if adapter_config is not None:
            model.add_adapter(model_args.adapter_name, config=adapter_config)
            model.train_adapter(model_args.adapter_name)
            logger.info(f"Adapter summary after adding:\n{model.adapter_summary()}")

    model_param_dict["w. adapters"] = model.num_parameters()
    model_param_dict["adapters"] = model_param_dict["w. adapters"] - model_param_dict["model"]
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "model_param_dict.json"), "w", encoding="utf8") as f:
        json.dump(model_param_dict, f, indent=2, ensure_ascii=False)

    # ──────────────────────────────────────────────────────────────────────────────
    #  Preprocessing
    # ──────────────────────────────────────────────────────────────────────────────
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    def preprocess_function(examples):
        inputs = []
        targets = []

        for i in range(len(examples[sentence1_key])):
            if sentence2_key is None:
                input_text = f"{data_args.task_name}: {examples[sentence1_key][i]}"
            else:
                input_text = (
                    f"{data_args.task_name}: "
                    f"{examples[sentence1_key][i]} "
                    f"{examples[sentence2_key][i]}"
                )

            label = examples["label"][i]
            # Use a placeholder for unlabeled test examples instead of dropping them.
            # Dropping inside batched=True can leave the batch empty and crash the tokenizer.
            if is_regression:
                target_text = str(float(label)) if label != -1 else "0.0"
            elif label == -1:
                target_text = label_list[0]   # any valid label; won't be evaluated
            else:
                target_text = label_list[label]

            inputs.append(input_text)
            targets.append(target_text)

        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding=padding,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=8,
                truncation=True,
                padding=padding,
            )

        processed_labels = []
        for label_ids in labels["input_ids"]:
            if isinstance(label_ids, (int, np.integer)):
                label_ids = [int(label_ids)]
            elif not isinstance(label_ids, list):
                label_ids = list(label_ids)
            processed_label = [tok if tok != tokenizer.pad_token_id else -100 for tok in label_ids]
            processed_labels.append(processed_label)

        model_inputs["labels"] = processed_labels
        return model_inputs

    with training_args.main_process_first(desc="dataset map pre-processing"):
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
            remove_columns=raw_datasets["train"].column_names,
        )

    # ──────────────────────────────────────────────────────────────────────────────
    #  Train / eval / predict datasets
    # ──────────────────────────────────────────────────────────────────────────────
    LARGE_DATASETS = {
        "qqp",
        "qnli",
        "sst2",
        # "mnli",
    }

    def create_custom_split(tokenized_datasets, task_name, seed):

        rng = np.random.RandomState(seed)

        # ------------------------------------------------------------------
        # MNLI special handling
        # ------------------------------------------------------------------
        if task_name == "mnli":

            train_ds = tokenized_datasets["train"]
            original_validation = tokenized_datasets["validation_matched"]

            indices = np.arange(len(train_ds))
            rng.shuffle(indices)

            val_indices = indices[:2000]
            train_indices = indices[2000:]

            return {
                "train": train_ds.select(train_indices),
                "validation": train_ds.select(val_indices),
                "test": original_validation,
            }

        # ------------------------------------------------------------------
        # Large datasets
        # ------------------------------------------------------------------
        if task_name in LARGE_DATASETS:

            train_ds = tokenized_datasets["train"]
            original_validation = tokenized_datasets["validation"]

            if len(train_ds) <= 2000:
                raise ValueError(
                    f"{task_name} has only {len(train_ds)} training examples; "
                    "cannot reserve 2000 for validation."
                )

            indices = np.arange(len(train_ds))
            rng.shuffle(indices)

            val_indices = indices[:2000]
            train_indices = indices[2000:]

            return {
                "train": train_ds.select(train_indices),
                "validation": train_ds.select(val_indices),
                "test": original_validation,
            }

        # ------------------------------------------------------------------
        # Small datasets
        # ------------------------------------------------------------------
        train_ds = tokenized_datasets["train"]
        original_validation = tokenized_datasets["validation"]

        indices = np.arange(len(original_validation))
        rng.shuffle(indices)

        midpoint = len(indices) // 2

        validation_indices = indices[:midpoint]
        test_indices = indices[midpoint:]

        return {
            "train": train_ds,
            "validation": original_validation.select(validation_indices),
            "test": original_validation.select(test_indices),
        }

    if data_args.custom_split:

        split_data = create_custom_split(
            tokenized_datasets,
            data_args.task_name,
            training_args.seed,
        )

        train_dataset = split_data["train"]
        eval_dataset = split_data["validation"]
        predict_dataset = split_data["test"]

        logger.info(
            f"Custom split enabled:"
            f" train={len(train_dataset)}"
            f" val={len(eval_dataset)}"
            f" test={len(predict_dataset)}"
        )

    else:

        train_dataset = tokenized_datasets["train"]

        eval_dataset = tokenized_datasets[
            "validation_matched" if data_args.task_name == "mnli"
            else "validation"
        ]

        predict_dataset = tokenized_datasets[
            "test_matched" if data_args.task_name == "mnli"
            else "test"
        ]

        
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))
    if data_args.max_predict_samples is not None:
        predict_dataset = predict_dataset.select(range(min(len(predict_dataset), data_args.max_predict_samples)))

    if data_args.resplit_dataset:
        logger.info(f"Original train length: {len(tokenized_datasets['train'])}")
        train_dataset, eval_dataset = split_datasets(tokenized_datasets["train"], n=data_args.max_train_samples)
        logger.info(f"After resplit - train: {len(train_dataset)}, eval: {len(eval_dataset)}")
        predict_dataset = tokenized_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]

    # ──────────────────────────────────────────────────────────────────────────────
    #  Data collator
    # ──────────────────────────────────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # ──────────────────────────────────────────────────────────────────────────────
    #  Metrics & callbacks
    # ──────────────────────────────────────────────────────────────────────────────
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        # Defensive: replace any -100 in preds (rare but possible)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        if data_args.task_name == "stsb":

            pred_values = []
            label_values = []
            for pred_text, label_text in zip(decoded_preds, decoded_labels):
                try:
                    pred_values.append(float(pred_text))
                    label_values.append(float(label_text))
                except (TypeError, ValueError):
                    continue

            if pred_values:
                from scipy.stats import pearsonr, spearmanr
                pred_array = np.asarray(pred_values, dtype=np.float64)
                label_array = np.asarray(label_values, dtype=np.float64)
                results = {
                    "pearson": float(pearsonr(pred_array, label_array)[0]),
                    "spearmanr": float(spearmanr(pred_array, label_array)[0]),
                }
            else:
                results = {"pearson": 0.0, "spearmanr": 0.0}
        else:
            correct = sum(p == l for p, l in zip(decoded_preds, decoded_labels))
            acc = correct / max(len(decoded_preds), 1)
            results = {"accuracy": acc}

        # Binary F1 for MRPC (positive class = "equivalent")
        if data_args.task_name == "mrpc" or data_args.task_name == "qqp":
            # label_list comes from raw_datasets["train"].features["label"].names
            # Typically ["not_equivalent", "equivalent"] — positive is "equivalent"
            pos = "equivalent" if "equivalent" in label_list else label_list[1]
            tp = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p == pos and l == pos)
            fp = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p == pos and l != pos)
            fn = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p != pos and l == pos)
            if tp == 0:
                f1 = 0.0
            else:
                precision = tp / (tp + fp)
                recall    = tp / (tp + fn)
                f1 = 2 * precision * recall / (precision + recall)
            results["f1"] = f1

        # Matthews correlation for CoLA
        if data_args.task_name == "cola":
            label_set = set(label_list)
            valid = [(p, l) for p, l in zip(decoded_preds, decoded_labels)
                     if p in label_set and l in label_set]
            if valid:
                from sklearn.metrics import matthews_corrcoef
                ps = [label_list.index(p) for p, _ in valid]
                ls = [label_list.index(l) for _, l in valid]
                results["matthews_correlation"] = float(matthews_corrcoef(ls, ps))
            else:
                results["matthews_correlation"] = 0.0

        # Diagnostic: how diverse are the predictions?
        print(f"[{data_args.task_name}] preds:", Counter(decoded_preds).most_common(3))
        print(f"[{data_args.task_name}] labels:", Counter(decoded_labels).most_common(3))

        return results

    callbacks = [EarlyStoppingCallback(early_stopping_patience=data_args.patience)]

    # ──────────────────────────────────────────────────────────────────────────────
    #  Trainer
    # ──────────────────────────────────────────────────────────────────────────────
    trainer = Seq2SeqAdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # ──────────────────────────────────────────────────────────────────────────────
    #  Training
    # ──────────────────────────────────────────────────────────────────────────────
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        model.save_adapter(training_args.output_dir, model_args.adapter_name)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # ──────────────────────────────────────────────────────────────────────────────
    #  Evaluation
    # ──────────────────────────────────────────────────────────────────────────────
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples or len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        if data_args.task_name == "mnli":
            metrics_mm = trainer.evaluate(eval_dataset=tokenized_datasets["validation_mismatched"])
            metrics_mm = {k + "_mm": v for k, v in metrics_mm.items()}
            metrics.update(metrics_mm)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ──────────────────────────────────────────────────────────────────────────────
    #  Predict / Test
    # ──────────────────────────────────────────────────────────────────────────────
    if training_args.do_predict:
        logger.info("*** Predict ***")
        if data_args.resplit_dataset or data_args.custom_split:
            metrics = trainer.evaluate(eval_dataset=predict_dataset, metric_key_prefix="test")
            max_predict_samples = data_args.max_predict_samples or len(predict_dataset)
            metrics["test_samples"] = min(max_predict_samples, len(predict_dataset))
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)
        else:
            predictions = trainer.predict(
                predict_dataset,
                metric_key_prefix="predict"
            ).predictions

            # Convert predictions to tensor if needed
            if isinstance(predictions, tuple):
                predictions = predictions[0]

            # Decode generated sequences
            decoded_preds = tokenizer.batch_decode(
                predictions,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            output_predict_file = os.path.join(
                training_args.output_dir,
                f"predict_results_{data_args.task_name}.txt"
            )
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results {data_args.task_name} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(decoded_preds):
                    writer.write(f"{index}\t{item}\n")

    all_checkpoints = get_all_checkpoint(training_args.output_dir)
    if all_checkpoints:
        for checkpoint in all_checkpoints:
            shutil.rmtree(os.path.join(training_args.output_dir, checkpoint), ignore_errors=True)


def _mp_fn(index):
    main()

if __name__ == "__main__":
    main()
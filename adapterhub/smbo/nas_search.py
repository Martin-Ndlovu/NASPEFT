#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning Llama models for language modeling on WikiText with NAS adapters."""

from enum import auto
import json
import sys
from unittest import result
sys.path.append('/root/Martin/NasPEFT/naspeft')
from logger import setup_logging
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import re
import shutil
import datasets
import numpy as np
import torch
import math
from datasets import load_dataset, load_from_disk
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
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    TrainerCallback
)
from transformers.trainer_utils import get_last_checkpoint
from accelerate import Accelerator
import torch.distributed as dist

#--------------------------------------------------------------------------------
# Silence internal Hugging Face and Datasets loggers
#--------------------------------------------------------------------------------
from transformers.utils import logging as hf_logging

#--------------------------------------------------------------------------------
# Logging setup
#--------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
accelerator = Accelerator()

#--------------------------------------------------------------------------------
# Function to split datasets
#--------------------------------------------------------------------------------
def split_datasets(train_ds, n: int = None):
    if accelerator.is_main_process:
        logger.info(
            "Splitting the train/eval datasets into train/eval by "
            "using 90% and 10% of train as train and eval."
        )
    if n is None:
        n = len(train_ds)
        if accelerator.is_main_process:
            logger.info(f"Using the whole train dataset of {n} samples.")
    else:
        if accelerator.is_main_process:
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
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    local_dataset_path: str = field(
        default="../datasets/wikitext/wikitext-2-v1",
        metadata={"help": "Path to local copy of dataset"}
    )
   
    task_name: str = field(
        default="wikitext",
        metadata={"help": "Name of the taske.g., wikitext)"}
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
        default="wikitext",
        metadata={"help": "The name of the dataset (wikitext)."}
    )
    dataset_config_name: Optional[str] = field(
        default="wikitext-103-v1",
        metadata={"help": "The configuration name of the dataset (wikitext-103-v1)."}
    )
   
    max_seq_length: int = field(
        default=512,
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
    def __post_init__(self):
        if self.dataset_name is not None:
            self.dataset_name = self.dataset_name.lower()

#---------------------------------------------------------------------------------
# Training Arguments
#---------------------------------------------------------------------------------
@dataclass
class MyTrainingArguments(TrainingArguments):
    """
    Arguments pertaining to the training process.
    """
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."}
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation on the validation set."}
    )
    output_dir: str = field(
        default="../output",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Overwrite the content of the output directory."}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device during training."}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device during evaluation."}
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy to adopt during training."}
    )
    logging_strategy: str = field(
        default="steps",
        metadata={"help": "Logging strategy to adopt during training."}
    )
    logging_steps: int = field(
        default=500,
        metadata={"help": "Log every X updates steps."}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "Save strategy to adopt during training."}
    )
    save_steps: int = field(
        default=2000,
        metadata={"help": "Save every X updates steps."}
    )
    save_total_limit: int = field(
        default=1,
        metadata={"help": "Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir."}
    )
    max_steps: int = field(
        default=5000,
        metadata={"help": "Total number of training steps to perform. Override num_train_epochs."}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Initial learning rate (after the potential warmup periodto use."}
    )
    warmup_ratio: float = field(
        default=0.05,
        metadata={"help": "Ratio of total steps for the warmup phase."}
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={"help": "Max gradient norm for gradient clipping."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for initialization."}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay to use."}
    )

#---------------------------------------------------------------------------------
# Data Classes for Model Arguments
#---------------------------------------------------------------------------------
@dataclass
class ModelArguments:
    nas_adapter_config_path: str = field(
        metadata={"help": "NAS adapter config path"}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    adapter_name: str = field(
        default="naspeft",
        metadata={"help": "The name of the adapter to use."}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to usecan be a branch name, tag name or commit id)."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Will use the token for private models."}
    )
   
#---------------------------------------------------------------------------------
# Data Classes for Adapter Arguments
#---------------------------------------------------------------------------------
@dataclass
class MultiLingAdapterArguments:
    train_adapter: bool = field(
        default=True,
        metadata={"help": "Whether to train adapters."}
    )
    load_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-trained adapter to load."}
    )
   
    load_lang_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-trained language adapter to load."}
    )

#---------------------------------------------------------------------------------
# Get all checkpoints in a folder
#---------------------------------------------------------------------------------
def get_all_checkpoint(folder):
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
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

#---------------------------------------------------------------------------------
# Default argument dictionaries for different adapter types
#---------------------------------------------------------------------------------
default_prefix_arg = {
    "prefix_length": 1,
    "bottleneck_size": 2048,
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

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main():
    #--------------------------------------------------------------------------
    # Set environment variables for distributed training
    #--------------------------------------------------------------------------
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    #-------------------------------------------------------------------------
    # Parse arguments
    #-------------------------------------------------------------------------
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments, MultiLingAdapterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
        training_args.dataloader_num_workers = 8
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "eval_perplexity"
        training_args.greater_is_better = False
        training_args.ddp_find_unused_parameters = False

    #-------------------------------------------------------------------------
    #  Setup logging
    #-------------------------------------------------------------------------
    setup_logging(os.path.join(training_args.output_dir, "training.log"))
    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    training_args.evaluation_strategy = "steps"
    training_args.eval_steps = 500
    
    #-------------------------------------------------------------------------
    # Detecting last checkpoint
    #-------------------------------------------------------------------------
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory{training_args.output_dir}already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            if accelerator.is_main_process:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
           
    #-------------------------------------------------------------------------
    # Set seed before initialing the model
    #-------------------------------------------------------------------------
    set_seed(training_args.seed)

    dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name)

    #-------------------------------------------------------------------------
    # A filtering function to remove empty or very short texts
    #-------------------------------------------------------------------------
    def filter_texts(example):
        text = example.get("text")
        if not text or len(text) < 5:
            return False
        return True

    #-------------------------------------------------------------------------
    # Add text cleaning to filter out special tokens
    #-------------------------------------------------------------------------
    def clean_special_tokens(example):
        text = example["text"]
        text = re.sub(r'@-@', '-', text)
        text = re.sub(r'@,@', ',', text)
        text = re.sub(r'@.@', '.', text)
        return {"text": text}
    
    for split in ["train", "validation", "test"]:
        dataset[split] = dataset[split].filter(filter_texts, batched=False)
        dataset[split] = dataset[split].map(clean_special_tokens, batched=False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #-----------------------------------------------------------------------
    # Loading model, quantaisation can be used to reduce memory usage
    #----------------------------------------------------------------------
    model = AutoAdapterModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        # device_map=None, 
        # use_gradient_checkpointing=True,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # ),
        # torch_dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.use_cache = False  # Disable cache for training
    model_param_dict = {"model": model.num_parameters()}

    #--------------------------------------------------------------------
    # Initialize adapters and check for pre-loaded adapters
    #--------------------------------------------------------------------
    adapters.init(model)
    if model.has_adapters():
        for adapter_name in model.get_configured_adapters():
            if accelerator.is_main_process:
                logger.info(f"Deleting pre-loaded adapter: {adapter_name}")
            model.delete_adapter(adapter_name)
        if accelerator.is_main_process:
            logger.info("Adapter summary after deletion:", model.adapter_summary())
        if model.has_adapters():
            if accelerator.is_main_process:
                logger.info("WARNING: Pre-loaded adapters still present after deletion:", model.get_configured_adapters())

    if adapter_args.train_adapter:
        if accelerator.is_main_process:
            logger.info(f"Random seed for adapter training: {training_args.seed}")
        with open(model_args.nas_adapter_config_path, "r")as f:
            config_args = json.load(f)

        #--------------------------------------------------------------------
        # Handle leave_out layers
        #--------------------------------------------------------------------
        leave_out_list = []
        number_layer = 16  # LLaMA-3.2-1B has 16 layers
        for i in range(number_layer):
            if config_args.get(f"leave_out_{i}", False):
                leave_out_list.append(int(i))
            if f"leave_out_{i}" in config_args:
                del config_args[f"leave_out_{i}"]
        config_args["leave_out"] = leave_out_list

        hidden_size = 2048  # LLaMA-3.2-1B has hidden size of 2048

        #-------------------------------------------------------------------
        # Configure adapters
        #-------------------------------------------------------------------
        if model_args.adapter_name == "prefix":
            config_args["prefix_length"] = config_args.get("prefix_length", 1)
            del config_args["reduction_factor"]
            default_arg = default_prefix_arg.copy()
            default_arg.update(config_args)
            adapter_config = PrefixTuningConfig(default_arg["prefix_length"], default_arg["bottleneck_size"], default_arg["leave_out"])
        elif model_args.adapter_name == "pfeiffers":
            config_args["reduction_factor"] = config_args.get("reduction_factor", 16)
            del config_args["reduction_factor"]
            default_arg = default_pfeiffer_arg.copy()
            default_arg.update(config_args)
            adapter_config = SeqBnConfig(16, default_arg["leave_out"], default_arg["non_linearity"])
        elif model_args.adapter_name == "lora":
            config_args["r"] = config_args.get("reduction_rank", 64)
            # del config_args["reduction_rank"]
            default_arg = default_mam_arg.copy()
            default_arg.update(config_args)
            if accelerator.is_main_process:
                logger.info("Config arguments:", config_args)
            adapter_config = LoRAConfig(
                selfattn_lora=True, intermediate_lora=True, output_lora=True,
                attn_matrices=["q", "k", "v"],
                alpha=config_args['lora_alpha'], r=config_args['lora_r'], dropout=0.1,
                leave_out=config_args.get("leave_out", []),)
    
        elif model_args.adapter_name == "mam":
            prefix_flag = config_args.get("reduction_prefix", 512) != 512
            parallel_flag = config_args.get("reduction_factor", 512) <= hidden_size
            if not prefix_flag:
                config_args["prefix_length"] = 1
            else:
                config_args["prefix_length"] = hidden_size / config_args["reduction_prefix"]
            if config_args["reduction_factor"] == 512:
                config_args["reduction_factor"] = hidden_size
            del config_args["reduction_prefix"]
            default_arg = default_mam_arg.copy()
            default_arg.update(config_args)
            config_flag_list = [prefix_flag, parallel_flag]
            config_list = [
                PrefixTuningConfig(
                    prefix_length=int(default_arg["prefix_length"]),
                    bottleneck_size=hidden_size,
                    leave_out=default_arg["leave_out"])
               ,
                ParBnConfig(
                    reduction_factor=default_arg["reduction_factor"],
                    leave_out=default_arg["leave_out"],
                    non_linearity="relu",
                    mh_adapter=False,
                    output_adapter=True
                )
            ]
            adapter_config = ConfigUnion(*[config_list[i] for i in range(len(config_list)) if config_flag_list[i]])
        elif model_args.adapter_name == "unipelt":
            prefix_flag = config_args.get("reduction_prefix", 512) != 512
            parallel_flag = config_args.get("reduction_parallel", 512) <= hidden_size
            lora_flag = config_args.get("reduction_rank", 512) != 512
            if not prefix_flag:
                config_args["prefix_length"] = 1
            else:
                config_args["prefix_length"] = hidden_size / config_args["reduction_prefix"]
            if config_args["reduction_factor"] == 512:
                config_args["reduction_factor"] = hidden_size
            if not lora_flag:
                config_args["r"] = 1
            else:
                config_args["r"] = 64 / config_args["reduction_rank"]
            del config_args["reduction_prefix"]
            del config_args["reduction_rank"]
            default_arg = default_mam_arg.copy()
            default_arg.update(config_args)
            config_flag_list = [prefix_flag, parallel_flag, lora_flag]
            config_list = [
                PrefixTuningConfig(
                    prefix_length=int(default_arg["prefix_length"]),
                    bottleneck_size=hidden_size,
                    leave_out=default_arg["leave_out"]
                ),
                SeqBnConfig(
                    reduction_factor=default_arg["reduction_factor"],
                    leave_out=default_arg["leave_out"],
                    non_linearity="relu",
                    mh_adapter=False,
                    output_adapter=True
                ),
                LoRAConfig(
                    r=int(default_arg["r"]),
                    alpha=16,
                    attn_matrices=["q", "k", "v", "o", "g", "u", "d"],
                    dropout=0.0
                )
            ]
            adapter_config = ConfigUnion(*[config_list[i] for i in range(len(config_list)) if config_flag_list[i]])
        elif model_args.adapter_name == "naspeft":
            exclude_prefix = False
            exclude_lora = False
            exclude_parallel = False

            #--------------------------------------------------------------------
            # prefix tuning configuration
            #--------------------------------------------------------------------
            if config_args.get("reduction_prefix") > hidden_size:
                config_args["reduction_prefix"] = hidden_size
                exclude_prefix = True

            #--------------------------------------------------------------------
            # LoRA configuration
            #--------------------------------------------------------------------
            if config_args.get("lora_r", 16) > 64:
                config_args["lora_r"] = 1
                exclude_lora = True

            #--------------------------------------------------------------------
            # Parallel (ParBnAdapter) configuration
            #--------------------------------------------------------------------
            if config_args.get("reduction_parallel") > hidden_size:
                config_args["reduction_parallel"] = hidden_size
                exclude_parallel = True
            
            if config_args["reduction_prefix"] < 1:
                config_args["reduction_prefix"] = 1
                exclude_prefix = True

            if config_args["lora_r"] < 1:
                config_args["lora_r"] = 1
                exclude_lora = True

            if config_args["reduction_parallel"] < 1:
                config_args["reduction_parallel"] = 1
                exclude_parallel = True

            if accelerator.is_main_process:
                logger.info(f"============= Config Args {config_args} =======")

            #--------------------------------------------------------------------
            # Configure adapter based on exclusion flags
            #--------------------------------------------------------------------
            config_flag_list = [not exclude_prefix, not exclude_lora, not exclude_parallel]
            config_list = [
                LoRAConfig(
                    r=config_args["lora_r"],
                    alpha=config_args["lora_alpha"],
                    attn_matrices=["q", "k", "v", "o",],
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
                    leave_out=config_args["leave_out"],
                    non_linearity="tanh",
                    dropout=0.1,
                )
            ]
            adapter_config = ConfigUnion(*[config_list[i] for i in range(len(config_list)) if config_flag_list[i]])

        #--------------------------------------------------------------------
        # Add and train adapter
        #--------------------------------------------------------------------
        model.add_adapter(model_args.adapter_name, config=adapter_config)
        model.train_adapter(model_args.adapter_name)
        if accelerator.is_main_process:
            logger.info(f"Adapter summary after adding:\n{model.adapter_summary()}")
    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode. "
                "Use --train_adapter to enable adapter training")
           
    #--------------------------------------------------------------------
    # Move model to device
    #--------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #--------------------------------------------------------------------
    # Save model parameter counts
    #--------------------------------------------------------------------
    model_param_dict["w. adapters"] = model.num_parameters()
    model_param_dict["adapters"] = model_param_dict["w. adapters"] - model_param_dict["model"]
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "model_param_dict.json"), "w", encoding="utf8") as f:
        json.dump(model_param_dict, f, indent=2, ensure_ascii=False)

    #--------------------------------------------------------------------
    # Preprocessing the datasets
    #--------------------------------------------------------------------
    def preprocess_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=data_args.max_seq_length,
            # padding="max_length",
            # add_special_tokens=True,
            # return_tensors="pt",
        )
        if isinstance(result["input_ids"], torch.Tensor):
            result["labels"] = result["input_ids"].clone()
        else:
            result["labels"] = result["input_ids"].copy()

        return result
    
    #--------------------------------------------------------------------
    # Apply size limits before tokenization
    #--------------------------------------------------------------------
    if training_args.do_train and "train" in dataset:
        max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(dataset["train"])
        max_train_samples = min(len(dataset["train"]), max_train_samples)
        dataset["train"] = dataset["train"].select(range(max_train_samples))

    #--------------------------------------------------------------------
    # Apply size limits before tokenization
    #--------------------------------------------------------------------
    if training_args.do_eval and "validation" in dataset:
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(dataset["validation"])
        max_eval_samples = min(len(dataset["validation"]), max_eval_samples)
        dataset["validation"] = dataset["validation"].select(range(max_eval_samples))

    #--------------------------------------------------------------------
    # Apply size limits before tokenization
    #--------------------------------------------------------------------
    if training_args.do_predict and "test" in dataset:
        max_predict_samples = data_args.max_predict_samples if data_args.max_predict_samples is not None else len(dataset["test"])
        max_predict_samples = min(len(dataset["test"]), max_predict_samples)
        dataset["test"] = dataset["test"].select(range(max_predict_samples))

    #--------------------------------------------------------------------
    # Tokenize each split separately
    #--------------------------------------------------------------------
    num_proc = min(4, len(dataset["train"]) // 250) if training_args.do_train else 1
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=max(1, num_proc),
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )

    #--------------------------------------------------------------------
    # Grouping/chunking
    #--------------------------------------------------------------------
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // data_args.max_seq_length) * data_args.max_seq_length
        result = {
            k: [t[i : i + data_args.max_seq_length] for i in range(0, total_length, data_args.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result
    
    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts into chunks of {data_args.max_seq_length}",
    )

    #--------------------------------------------------------------------
    # Data collator and datasets for training/evaluation
    #--------------------------------------------------------------------
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    #--------------------------------------------------------------------
    # Datasets for evaluation
    #--------------------------------------------------------------------
    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    #--------------------------------------------------------------------
    # Datasets for prediction
    #--------------------------------------------------------------------
    if training_args.do_predict:
        if "test" not in tokenized_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = tokenized_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    #--------------------------------------------------------------------
    # Resplit dataset if required
    #--------------------------------------------------------------------
    if data_args.resplit_dataset:
        logger.info(f"Original length of training dataset: {len(tokenized_datasets['train'])}")
        train_dataset, eval_dataset = split_datasets(tokenized_datasets["train"], n=data_args.max_train_samples)
        logger.info(f"Length of training dataset after resplit: {len(train_dataset)}")
        logger.info(f"Length of eval dataset after resplit: {len(eval_dataset)}")
        predict_dataset = tokenized_datasets["validation"]
    else:
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]
        predict_dataset = tokenized_datasets["test"]

    #--------------------------------------------------------------------
    # Data collator for language modeling
    #--------------------------------------------------------------------
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    #--------------------------------------------------------------------
    # Initialize AdapterTrainer
    #--------------------------------------------------------------------
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
        perplexity = torch.exp(loss)
        return {"perplexity": perplexity.item()}

    #--------------------------------------------------------------------
    # Adding a callback for evaluation without requiring extra space 
    #--------------------------------------------------------------------
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

                #--------------------------------------------------------------------
                # Update the trainer's state log_history for consistancy logging and history storage 
                # to include perplexity.
                #--------------------------------------------------------------------
                state.log_history.append({
                    "epoch": state.epoch,
                    "step": state.global_step,
                    "eval_loss": eval_loss,
                    "eval_perplexity": perplexity,
                    "eval_runtime": metrics.get("eval_runtime"),
                    "eval_samples_per_second": metrics.get("eval_samples_per_second"),
                    "eval_steps_per_second": metrics.get("eval_steps_per_second")
                })

            #--------------------------------------------------------------------
            # Store for later logging
            #--------------------------------------------------------------------
            self.last_eval_metrics = metrics.copy()
            
            return control
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return control
            # If this log is from eval (check for eval keys), merge perplexity
            if "eval_loss" in logs and "eval_perplexity" not in logs:
                # Pull from stored metrics or recompute
                eval_loss = logs.get("eval_loss")
                if eval_loss is not None:
                    logs["eval_perplexity"] = math.exp(eval_loss)
            
            return control

    #--------------------------------------------------------------------
    # Initialize AdapterTrainer
    #--------------------------------------------------------------------
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        # compute_metrics=compute_metrics, 
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[PerplexityCallback(), EarlyStoppingCallback(early_stopping_patience=data_args.patience)],)
   
    #--------------------------------------------------------------------
    # Train the model
    #--------------------------------------------------------------------
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model() # Replaced with save_adapter below
        if accelerator.is_main_process:
            model.save_adapter(training_args.output_dir, model_args.adapter_name)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    #--------------------------------------------------------------------
    # Evaluate the model
    #--------------------------------------------------------------------
    if training_args.do_eval:
        if accelerator.is_main_process:
            logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples or len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #--------------------------------------------------------------------
    # Do prediction
    #--------------------------------------------------------------------
    if training_args.do_predict and data_args.resplit_dataset:
        if accelerator.is_main_process:
            logger.info("*** Test ***")
        metrics = trainer.evaluate(eval_dataset=predict_dataset)
        max_predict_samples = data_args.max_predict_samples or len(predict_dataset)
        metrics["test_samples"] = min(max_predict_samples, len(predict_dataset))
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    if training_args.do_predict and not data_args.resplit_dataset:
        if accelerator.is_main_process:
            logger.info("*** Predict ***")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.squeeze(predictions if isinstance(predictions, tuple) else predictions)
        output_predict_file = os.path.join(training_args.output_dir, "predict_results_wikitext.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                if accelerator.is_main_process:
                    logger.info("***** Predict results wikitext *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "language-modeling"}
    if data_args.dataset_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "wikitext" if "wikitext" in data_args.dataset_name else data_args.dataset_name
        kwargs["dataset_args"] = data_args.dataset_config_name
        kwargs["dataset"] = "WikiText-2-v1" if "wikitext" in data_args.dataset_name else data_args.dataset_name

    if accelerator.is_main_process:
        all_checkpoints = get_all_checkpoint(training_args.output_dir)
        if all_checkpoints:
            for checkpoint in all_checkpoints:
                last_checkpoint = os.path.join(training_args.output_dir, checkpoint)
                shutil.rmtree(last_checkpoint, ignore_errors=True)


def _mp_fn(index):
    main()

if __name__ == "__main__":
    main()
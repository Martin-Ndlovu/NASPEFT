#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
""" Fine-tuning Llama 3.1 8B Instruct on Wikitext-103-v1 with adapters. """

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import numpy as np
import transformers
from datasets import load_dataset
import adapters
from adapters import (
    AdapterTrainer,
    SeqBnConfig,
    PrefixTuningConfig,
    ParBnConfig,
    LoRAConfig,
    ConfigUnion,
)
from adapters.composition import Fuse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="../models/Meta-Llama-3.1-8B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    adapter_name: str = field(
        default="naspeft",
        metadata={"help": "Adapter type: pfeiffer, prefix, parallel, lora, mam, unipelt, sappa"}
    )
    prefix_length: int = field(
        default=10,
        metadata={"help": "Prefix length for prefix tuning adapters"}
    )
    reduction_factor: int = field(
        default=16,
        metadata={"help": "Reduction factor for adapters"}
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "Rank for LoRA adapters"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store pretrained models from huggingface.co"}
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "Token for accessing private models from huggingface.co"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    local_dataset_path: str = field(
        default="../datasets/wikitext/wikitext-103-v1",
        metadata={"help": "Path to local copy of the dataset"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length after tokenization"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite cached preprocessed datasets"}
    )
    max_train_samples: Optional[int] = field(
        default=10000,
        metadata={"help": "Truncate training examples for debugging"}
    )
    max_eval_samples: Optional[int] = field(
        default=100,
        metadata={"help": "Truncate evaluation examples for debugging"}
    )

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
        default=True,
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
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Total number of training epochs to perform."}
    )
    per_device_train_batch_size: int = field(
        default=1,
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
        default=100,
        metadata={"help": "Log every X updates steps."}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "Save strategy to adopt during training."}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir."}
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    max_steps: int = field(
        default=2000,
        metadata={"help": "Total number of training steps to perform. Override num_train_epochs."}
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "The scheduler type to use."}
    )
    optim: str = field(
        default="paged_adamw_32bit",
        metadata={"help": "Optimizer to use during training."}
    )
    learning_rate: float = field(
        default=0.0001,
        metadata={"help": "Initial learning rate (after the potential warmup period) to use."}
    )
    group_by_length: bool = field(
        default=True,
        metadata={"help": "Whether to group sequences of similar lengths together during training."}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bfloat16 (requires PyTorch 1.10 or later)."}
    )
    warmup_ratio: float = field(
        default=0.03,
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



def main():
    # Set the environment variable to use only the first GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Ensure eval_strategy and early stopping compatibility
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "perplexity"
    training_args.greater_is_better = False
    # Set gradient accumulation to reduce memory usage
    if not hasattr(training_args, "gradient_accumulation_steps"):
        training_args.gradient_accumulation_steps = 4

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(os.path.join(training_args.output_dir, model_args.adapter_name)) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory {training_args.output_dir} already exists and is not empty.")
        elif last_checkpoint is not None:
            logger.info(f"Resuming training from {last_checkpoint}")

    # Set seed
    set_seed(training_args.seed)

    # Load dataset
    dataset = load_dataset(data_args.local_dataset_path)
    print(f"Loaded dataset from: {data_args.local_dataset_path}")

    def preprocess_dataset(dataset):
        def clean_example(example):
            if example["text"] is None or not isinstance(example["text"], str):
                example["text"] = ""  # Replace invalid entries with empty string
            return example
        return dataset.map(clean_example, num_proc=max(1, min(4, len(dataset) // 250)))

    dataset = {
        'train': preprocess_dataset(dataset['train']).select(range(data_args.max_train_samples)),
        'validation': preprocess_dataset(dataset['validation']).select(range(data_args.max_eval_samples)),
        'test': preprocess_dataset(dataset['test']).select(range(100))
    }
    print("Preprocessed dataset:", dataset)


    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=torch.bfloat16,
    )
    model.use_cache = False  # Disable cache for training

    # print(f"GPU memory after model load: {torch.cuda.memory_allocated()/1e9:.2f} GiB")
    # print(f"Memory summary after model load:\n{torch.cuda.memory_summary()}")

    print(f"Model head count: {model.config.num_attention_heads}")
    print(f"Model config: {model.config}")

    
    # Initialize adapters
    adapters.init(model)
    print("Adapters initialized for model")

    # Check for pre-loaded adapters
    print("Adapter summary before setup:", model.adapter_summary())
    if model.has_adapters():
        for adapter_name in model.get_configured_adapters():
            print(f"Deleting pre-loaded adapter: {adapter_name}")
            model.delete_adapter(adapter_name)
        print("Adapter summary after deletion:", model.adapter_summary())
        if model.has_adapters():
            print("WARNING: Pre-loaded adapters still present after deletion:", model.get_configured_adapters())

    model_param_dict = {'model': model.num_parameters()}
    print(f"Total model parameters before adapters: {model.num_parameters()}")


    # Setup adapters
    task_name = "wikitext"
    if model_args.adapter_name == "pfeiffer":
        config = SeqBnConfig(reduction_factor=model_args.reduction_factor)
        model.add_adapter(model_args.adapter_name, config=config)

    elif model_args.adapter_name == "prefix":
        config = PrefixTuningConfig(prefix_length=model_args.prefix_length, flat=False)
        model.add_adapter(model_args.adapter_name, config=config)

    elif model_args.adapter_name == "parallel":
        config = ParBnConfig(reduction_factor=model_args.reduction_factor)
        model.add_adapter(model_args.adapter_name, config=config)

    elif model_args.adapter_name == "lora":
        config = LoRAConfig(
            selfattn_lora=True, intermediate_lora=True, output_lora=True,
            attn_matrices=["q", "k", "v"],
            alpha=16, r=64, dropout=0.1,
        )
        model.add_adapter(model_args.adapter_name, config=config)

    elif model_args.adapter_name in ["mam", "unipelt", "sappa"]:
        if model_args.adapter_name == "mam":
            adapters_to_fuse = [
                ("prefix", PrefixTuningConfig(prefix_length=model_args.prefix_length, flat=False)),
                ("pfeiffer", SeqBnConfig(reduction_factor=model_args.reduction_factor)),
            ]
        elif model_args.adapter_name == "unipelt":
            adapters_to_fuse = [
                ("lora", LoRAConfig(
                    r=model_args.lora_rank,
                    alpha=16,
                    attn_matrices=["q", "k", "v", "o"],
                    init_weights="lora",
                    dropout=0.1,
                    use_gating=False,
                )),
                ("prefix", PrefixTuningConfig(prefix_length=model_args.prefix_length, flat=False)),
                ("pfeiffer", SeqBnConfig(reduction_factor=model_args.reduction_factor)),
            ]
        elif model_args.adapter_name == "sappa":
            adapters_to_fuse = [
                ("prefix", PrefixTuningConfig(prefix_length=model_args.prefix_length, flat=False)),
                ("pfeiffer", SeqBnConfig(reduction_factor=model_args.reduction_factor * 2)),
            ]
        
        for adapter_id, config in adapters_to_fuse:
            model.add_adapter(f"{task_name}_{adapter_id}", config=config)
            print(f"Added adapter: {task_name}_{adapter_id}")
        
        fusion_adapters = [f"{task_name}_{adapter_id}" for adapter_id, _ in adapters_to_fuse]
        model.add_adapter_fusion(Fuse(*fusion_adapters))
        model.train_adapter_fusion(fusion_adapters)
        print(f"Added and activated fusion adapters: {fusion_adapters}")
        print(f"Active adapters: {model.active_adapters}")
        print("Adapter summary after fusion:", model.adapter_summary())

    elif model_args.adapter_name == "naspeft":
        config = ConfigUnion(
        LoRAConfig(
            selfattn_lora=True, intermediate_lora=True, output_lora=True,
            attn_matrices=["q", "k", "v"],
            alpha=16, r=64, dropout=0.1,
        ),
        # PrefixTuningConfig(flat=True, prefix_length=30),
        ParBnConfig(reduction_factor=32)
        )
        model.add_adapter(model_args.adapter_name, config=config)
        
    if model_args.adapter_name not in ["mam", "unipelt", "sappa"]:
        model.train_adapter(model_args.adapter_name)
        print("Activated adapter:", model_args.adapter_name)
        print("Adapter summary after setup:", model.adapter_summary())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for _, module in model.named_modules():
        if hasattr(module, 'prefix_tuning'):
            module.prefix_tuning.to(device)

    for param in model.parameters():
        if param.dim() == 1:
            param.data = param.data.to(torch.float32)  # Cast small parameters to fp32 for stability


    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params}")
    if trainable_params == 0:
        print("ERROR: No trainable parameters found. Check adapter configuration.")
        sys.exit(1)

    print(f"Total model parameters after adapters: {model.num_parameters()}")
    model_param_dict['total_with_adapters'] = model.num_parameters()
    model_param_dict['adapters_only'] = model_param_dict['total_with_adapters'] - model_param_dict['model']
    
    # class CastOutputToFloat(torch.nn.Sequential):
    #     """
    #     Custom module to cast model outputs to float32.
    #     """
    #     def forward(self, x):
    #         return super().forward(x.to(torch.float32))
    # model.lm_head = CastOutputToFloat(model.lm_head)

    print("Model: ", model)

    # Verify the datatypes of model parameters
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(f"Parameter type: {k}, Count: {v}, Percentage: {v/total:.2%}")

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    with open(os.path.join(training_args.output_dir, "model_info.json"), "w", encoding='utf8') as f:
        json.dump(model_param_dict, f, indent=2)

    # # Tokenization 
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
        )
    tokenizer.pad_token = tokenizer.eos_token 

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], 
            truncation=False, 
            max_length=data_args.max_seq_length, 
            add_special_tokens=True)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

  # Tokenize each split separately
    num_proc = min(4, len(dataset["train"]) // 250) # Adjust number of processes based on dataset size
    tokenized_dataset = {
        "train": dataset["train"].map(tokenize_function, batched=True, num_proc=max(1, num_proc),   remove_columns=["text"]),
        "validation": dataset["validation"].map(tokenize_function, batched=True, num_proc=max(1, num_proc), remove_columns=["text"]),
        "test": dataset["test"].map(tokenize_function, batched=True, num_proc=max(1, num_proc), remove_columns=["text"])
    }

    print("Tokenized dataset:", tokenized_dataset)

    # Compute metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Convert to torch tensors if needed
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
    
    class MyDataCollator(DataCollatorForLanguageModeling):
        def __call__(self, features):
            batch = super().__call__(features)
            if "attention_mask" in batch:
                batch["attention_mask"] = batch["attention_mask"].to(torch.bfloat16)
                # print("Collator attention_mask dtype:", batch["attention_mask"].dtype)
            return batch

    data_collator = MyDataCollator(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    # Initialize Trainer
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    trainer.create_model_card(
        finetuned_from=model_args.model_name_or_path,
        tasks="language-modeling",
        dataset="Wikitext-103-v1",
    )

if __name__ == "__main__":
    main()
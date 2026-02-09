import argparse
import numpy as np
import os
import re
import random
import torch
import math
from transformers import set_seed
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    set_seed,
)
from adapters import (
    AutoAdapterModel,
    AdapterConfig, 
    AdapterTrainer, 
    ConfigUnion, 
    LoRAConfig, 
    PrefixTuningConfig, 
    ParBnConfig
)
import logging

# Ensure INFO-level logs are visible (add this once at top of main() or after imports)
logging.getLogger("transformers").setLevel(logging.INFO)

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a language adapter for LLaMA-3-8B using CLM on wikitext-103-v1.")
    parser.add_argument("--language", type=str, default="en", help="Target language (e.g., cy for Cornish)")
    parser.add_argument("--output_dir", type=str, default="./output/union_adapter/", help="Output directory for adapter checkpoints")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--max_steps", type=int, default=200, help="Number of training steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size per device during evaluation")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimization")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    return parser.parse_args()

args = parse_arguments()

def main():
    # Set seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    # Load tokenizer and model
    model_name = "/root/Martin/NasPEFT/naspeft/models/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoAdapterModel.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Initialize adapters
    import adapters
    adapters.init(model)

    # Define and add configurations for multiple adapter types
    lora_config = LoRAConfig(r=8, alpha=16, dropout=0.1)
    prefix_config = PrefixTuningConfig(prefix_length=30, dropout=0.1)
    parbn_config = ParBnConfig(reduction_factor=2, dropout=0.1)

    # Combine configurations using ConfigUnion
    adapter_config = ConfigUnion(*[lora_config, prefix_config, parbn_config])

    # Add Seq_bn_inv adapter for CLM
    # adapter_config = AdapterConfig.load("seq_bn_inv")  # Initialize Sequential Bottleneck with Invertible Layers
    model.add_adapter("clm", config=adapter_config)
    model.train_adapter("clm")  # Train only the adapter

    model.to(device)

    # Load and preprocess dataset
    def filter_texts(example):
        text = example.get("text")

        if not text or len(text) < 5:
            return False
        
        return True
    
    def clean_special_tokens(example):
        """Clean the special @ tokens from text"""
        text = example["text"]
        
        # Replace special tokens
        text = re.sub(r'@-@', '-', text)
        text = re.sub(r'@,@', ',', text) 
        text = re.sub(r'@.@', '.', text)
        
        return {"text": text}

    dataset = load_dataset("wikitext", "wikitext-2-v1")
    for split in ("train", "validation", "test"):
        dataset[split] = dataset[split].filter(filter_texts, batched=False)
        dataset[split] = dataset[split].map(clean_special_tokens, batched=False)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_seq_length)

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )

    # Group texts into chunks
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // args.max_seq_length) * args.max_seq_length
        result = {
            k: [t[i : i + args.max_seq_length] for i in range(0, total_length, args.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts into chunks of {args.max_seq_length}",
    )

    # Data collator for CLM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    output_dir = os.path.join(args.output_dir, args.language, "clm")
    training_args = TrainingArguments(
        output_dir=output_dir,#
        max_steps=args.max_steps,#
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,#
        weight_decay=args.weight_decay, #
        eval_strategy="steps",#
        eval_steps=100,#
        save_strategy="steps",#
        save_steps=2000,#
        save_total_limit=10,#
        load_best_model_at_end=True,#
        logging_steps=100,#
        warmup_ratio=0.05,#
        overwrite_output_dir=True,#
        seed=args.seed,#
        report_to="all",#
        metric_for_best_model="eval_perplexity", #
        greater_is_better=False,#

    )

    class PerplexityCallback(TrainerCallback):
        def __init__(self):
            self.last_eval_metrics = {}  # Store for on_log
        
        def on_evaluate(self, args, state, control, **kwargs):
            metrics = kwargs.get("metrics")
            if metrics is None:
                return control
            
            eval_loss = metrics.get("eval_loss")
            if eval_loss is not None:
                perplexity = math.exp(eval_loss)
                metrics["eval_perplexity"] = perplexity 
            
            # Store for later logging
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

    # Initialize trainer
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        callbacks=[PerplexityCallback()],
    )

    # Function to find the latest checkpoint
    def find_latest_checkpoint(adapter_base_path: str, language_code: str) -> str:
        """
        Finds the latest checkpoint directory for the given language.
        """
        language_code = language_code.replace("_", "-")
        lang_adapter_path = os.path.join(adapter_base_path, language_code, "clm")
        checkpoints = [d for d in os.listdir(lang_adapter_path) if d.startswith("checkpoint")]
        # check if the last checkpoint is 'checkpoint-final'
        if "checkpoint-final" in checkpoints:
            return os.path.join(lang_adapter_path, "checkpoint-final")
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))  # Sort by checkpoint number
        latest_checkpoint = checkpoints[-1] if checkpoints else None  # Get the latest checkpoint
        return os.path.join(lang_adapter_path, latest_checkpoint) if latest_checkpoint else None
    
    # Train the adapter
    # Resume from checkpoint if exists
    checkpoint = find_latest_checkpoint(args.output_dir, args.language) if os.path.exists(args.output_dir) else None

    resume_flag = False
    
    if checkpoint and os.path.isdir(checkpoint) and resume_flag:
        print(f"Resuming training from checkpoint: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save the adapter
    model.save_adapter(os.path.join(output_dir, "checkpoint-final"), "clm")

if __name__ == "__main__":
    main()
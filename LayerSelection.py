# This file is aimed to figure out what layers of a llama 3.2 model are best to fine-tune 
# It uses LoRA to fine-tune one layer at a time with the same configuration.
# The results are saved in JSON files for later analysis.

import torch
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import logging
import os
import json
import math

# -----------------------------
# Distributed setup
# -----------------------------
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}")

def is_main_process():
    return int(os.environ.get("LOCAL_RANK", 0)) == 0

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Paths
# -----------------------------
model_name = "models/Llama-3.2-1B"           
output_dir = "output/layer_selection"  
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Dataset Loading
# -----------------------------
def load_and_split_alpaca_dataset():
    logger.info("Loading Alpaca dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned")
    logger.info(f"Alpaca dataset loaded with splits: {list(dataset.keys())}")

    logger.info("Splitting Alpaca dataset into 80/10/10 train/validation/test...")
    splits = dataset['train'].train_test_split(test_size=0.1, seed=42)
    train_val = splits['train'].train_test_split(test_size=0.1111, seed=42)
    
    alpaca_splits = DatasetDict({
        'train': train_val['train'],
        'validation': train_val['test'],
        'test': splits['test']
    })
    
    logger.info(f"Final splits: train={len(alpaca_splits['train'])}, "
                f"validation={len(alpaca_splits['validation'])}, "
                f"test={len(alpaca_splits['test'])}")
    return alpaca_splits

# -----------------------------
# Tokenization with masking (only Response contributes to loss)
# -----------------------------
def tokenize_with_labels(example, tokenizer, max_length=512):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    response = example.get("output", "")

    # Build prompt
    prompt = f"### Instruction:\n{instruction}"
    if input_text:
        prompt += f"\n### Input:\n{input_text}"
    prompt += "\n### Response:\n"

    # Full text = prompt + response
    full_text = prompt + response

    # Tokenize full string
    tokenized = tokenizer(full_text, truncation=True, max_length=max_length)

    # Tokenize prompt only
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]

    # Labels = copy of input_ids
    labels = tokenized["input_ids"].copy()
    # Mask out prompt tokens
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

    tokenized["labels"] = labels
    return tokenized

# -----------------------------
# Main loop
# -----------------------------
def main():
    dataset = load_and_split_alpaca_dataset()

    # Quantization for efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare tokenized datasets (train + val)
    train_dataset = dataset["train"].map(lambda ex: tokenize_with_labels(ex, tokenizer), batched=False)
    val_dataset = dataset["validation"].map(lambda ex: tokenize_with_labels(ex, tokenizer), batched=False)

    # Number of layers in Llama 3.2
    num_layers = 16

    # Loop over all layers
    for layer_idx in range(num_layers):
        logger.info(f"=== Fine-tuning only layer {layer_idx} ===")

        # Reload fresh model each time
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": device},
        )
        model.config.use_cache = False

        # LoRA config (layer-specific)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            layers_to_transform=[layer_idx],
            layers_pattern="layers",  # matches model.layers
        )

        model = get_peft_model(model, peft_config)

        # Output directory per layer
        layer_output_dir = os.path.join(output_dir, f"layer_{layer_idx}")
        os.makedirs(layer_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=layer_output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            warmup_steps=100,
            logging_steps=100,
            fp16=False,
            bf16=True,
            report_to="none",
            dataloader_num_workers=4,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
            args=training_args,
        )

        # -----------------------------
        # Training
        # -----------------------------
        logger.info("Starting training...")
        train_result = trainer.train()

        # -----------------------------
        # Evaluation
        # -----------------------------
        logger.info("Evaluating...")
        eval_results = trainer.evaluate()
        if "eval_loss" in eval_results:
            eval_results["eval_perplexity"] = math.exp(eval_results["eval_loss"])
        else:
            eval_results["eval_perplexity"] = float("nan")

        # Save eval_results.json
        with open(os.path.join(layer_output_dir, "eval_results.json"), "w") as f:
            json.dump(eval_results, f, indent=4)

        # Save all_results.json
        all_results = {**train_result.metrics, **eval_results}
        with open(os.path.join(layer_output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=4)

        # Save model checkpoint
        model.save_pretrained(os.path.join(layer_output_dir, f"finetuned_layer_{layer_idx}"))
        tokenizer.save_pretrained(os.path.join(layer_output_dir, f"finetuned_layer_{layer_idx}"))

        logger.info(f"Finished layer {layer_idx}, results saved.")

if __name__ == "__main__":
    main()

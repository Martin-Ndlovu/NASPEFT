import torch
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import logging

import torch.distributed as dist
import os

# detect local rank (accelerate / torch.distributed sets this)
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}")

def is_main_process():
    # Accelerate sets LOCAL_RANK or RANK
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model setup
model_name = "models/Llama-3.2-1B"

# Load and split Alpaca dataset
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

dataset = load_and_split_alpaca_dataset()
train_dataset = dataset["train"]

# Quantization for efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": device},
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, peft_config)


# Formatting function for Alpaca
def formatting_prompts_func(example):
    outputs = []
    for instruction, input_text, output in zip(example["instruction"], example["input"], example["output"]):
        text = f"### Instruction:\n{instruction}"
        if input_text:
            text += f"\n### Input:\n{input_text}"
        text += f"\n### Response:\n{output}"
        outputs.append(text)
    return {"text": outputs}

def formatting_single(example):
    # example is a mapping of single fields (not batched)
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    text = f"### Instruction:\n{instruction}"
    if input_text:
        text += f"\n### Input:\n{input_text}"
    text += f"\n### Response:\n{output}"
    return text  # SFTTrainer expects a string

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

# Training config
training_args = TrainingArguments(
    output_dir="output/llama-3.2-alpaca_sft_peft",
    num_train_epochs=2,
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
    save_total_limit=20,
    # Optional: Add evaluation on validation set
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

# SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=dataset["validation"],  # Optional: Added for evaluation
    peft_config=peft_config,
    # dataset_text_field="text",
    processing_class=tokenizer,
    formatting_func=formatting_single,
    args=training_args,
)

logger.info("Starting training...")
trainer.train()
trainer.save_model()
model.save_pretrained("output/llama-3.2-alpaca-sft-lora_peft")
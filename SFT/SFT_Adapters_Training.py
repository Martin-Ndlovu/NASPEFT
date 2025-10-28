import os
import torch
import logging
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
# Adapter-enabled model loader from adapter-transformers
from adapters import AutoAdapterModel, LoRAConfig
import adapters
from trl import SFTTrainer

# ----------------------------------------------------------------------
# Distributed / device setup
# ----------------------------------------------------------------------
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}")

def is_main_process():
    return int(os.environ.get("LOCAL_RANK", 0)) == 0

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Model + Tokenizer
# ----------------------------------------------------------------------
model_name = "models/Llama-3.2-1B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

logger.info("Loading base adapter-capable model (AutoAdapterModel)...")
# Use AutoAdapterModel so adapter-transformers methods are available
model = AutoAdapterModel.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    device_map={"": device},
    trust_remote_code=True,
)
model.config.use_cache = False

# initialize model with adapters
adapters.init(model)

logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

# ----------------------------------------------------------------------
# Load Alpaca dataset and split
# ----------------------------------------------------------------------
def load_and_split_alpaca_dataset():
    logger.info("Loading Alpaca dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned")
    logger.info(f"Dataset splits: {list(dataset.keys())}")

    logger.info("Splitting into train/val/test...")
    splits = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_val = splits["train"].train_test_split(test_size=0.1111, seed=42)

    return DatasetDict(
        {
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": splits["test"],
        }
    )

dataset = load_and_split_alpaca_dataset()
train_dataset = dataset["train"]

# ----------------------------------------------------------------------
# Adapters configuration (adapter-transformers)
# ----------------------------------------------------------------------
adapter_name = "alpaca_adapter"
lora_config = LoRAConfig(
    r=8,
    alpha=32,
    dropout=0.1,
    attn_matrices=["q", "k", "v", "o", "g", "u", "d"],  # q,k,v,o,gate,up,down projections
    selfattn_lora=True,
)
model.add_adapter(adapter_name, config=lora_config)
# model.set_active_adapters(adapter_name)
model.add_causal_lm_head(adapter_name, overwrite_ok=True)
# model.train_adapter(adapter_name)


try:
    # Add a causal-lm head if one is not present
    # (vocab_size uses tokenizer; layers/other args optional)
    if "default" not in getattr(model, "adapters", {}).get("heads", {}):
        model.add_head(
            name="default",
            head_type="causal_lm",
            layers=1,
            vocab_size=getattr(tokenizer, "vocab_size", None) or getattr(tokenizer, "len", None)
        )
except Exception:
    # Fallback: call the more generic add_head interface (some versions expose add_head directly)
    try:
        model.add_head("default", head_type="causal_lm", layers=1, vocab_size=tokenizer.vocab_size)
    except Exception:
        pass


# Activate and mark adapter for training
model.set_active_adapters(adapter_name)
model.train_adapter(adapter_name)

# Optional: print active adapters / heads to verify
print("Active adapters:", model.active_adapters)
print("Active head:", getattr(model, "active_head", None))
print("Available heads:", getattr(model, "heads", {}).keys())

# ----------------------------------------------------------------------
# Prompt formatting
# ----------------------------------------------------------------------
def formatting_prompts_func(example):
    outputs = []
    for instruction, input_text, output in zip(
        example["instruction"], example["input"], example["output"]
    ):
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
    output_dir="output/llama-3.2-alpaca-sft",
    num_train_epochs=4,
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
    load_best_model_at_end=False,
    # metric_for_best_model="epoch",
)

# SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=dataset["validation"],  # Optional: Added for evaluation
    # dataset_text_field="text",
    processing_class=tokenizer,
    formatting_func=formatting_single,
    args=training_args,
)

logger.info("Starting training...")
trainer.train()
trainer.save_model()
model.save_adapter("output/llama-3.2-alpaca-sft", "alpaca_adapter")
model.save_pretrained("output./llama-3.2-alpaca100")

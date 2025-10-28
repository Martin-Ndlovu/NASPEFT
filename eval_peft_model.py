# import os
# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
# from datasets import load_dataset
# import logging
# import adapters

# BASE_MODEL_PATH = "/root/Martin/NasPEFT/naspeft/models/Llama-3.2-1B/"
# DATA_PATH = "/root/Martin/NasPEFT/naspeft/datasets/wikitext/wikitext-103-v1/"
# OUTPUT_DIR = "/root/Martin/NasPEFT/naspeft/output/llama1B_lora2/"
# BATCH_SIZE = 16

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("eval_peft_model")

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # Load and tokenize dataset
# dataset = load_dataset("wikitext", "wikitext-103-v1", split="test", cache_dir=DATA_PATH)
# logger.info(f"Loaded test dataset with {len(dataset)} samples.")

# def tokenize_function(examples):
#     return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

# logger.info("Tokenizing dataset...")
# tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
# logger.info("Tokenization complete.")

# collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, collate_fn=collator)

# # Find all PEFT model directories
# peft_dirs = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR)
#              if os.path.isdir(os.path.join(OUTPUT_DIR, d))]

# logger.info(f"Found {len(peft_dirs)} PEFT model directories.")

# for peft_dir in peft_dirs:
#     logger.info(f"Evaluating model in {peft_dir}")

#     # Load base model and move to device
#     model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16)
#     adapters.init(model)
#     model.to(device)
#     model.eval()

#     # Try to load PEFT adapter if exists
#     adapter_path = os.path.join(peft_dir, "lora")
#     if os.path.isdir(adapter_path):
#         try:
#             adapter_name = "lora"
#             model.load_adapter(adapter_path, load_as=adapter_name)
#             model.set_active_adapters(adapter_name)
#             logger.info(f"Loaded adapter from {adapter_path}")

#             # Ensure all parameters (including adapter weights) match model dtype and device
#             for name, param in model.named_parameters():
#                 if param.dtype != model.dtype or param.device != device:
#                     param.data = param.data.to(dtype=model.dtype, device=device)
#         except Exception as e:
#             logger.warning(f"Could not load adapter from {adapter_path}: {e}")
#             continue
#     else:
#         logger.warning(f"No adapter found in {adapter_path}, skipping.")
#         continue
    

#     # Evaluation loop
#     total_loss = 0.0
#     total_tokens = 0

#     with torch.no_grad():
#         for i, batch in enumerate(dataloader):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             shift_logits = outputs.logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#             total_loss += loss.item()
#             total_tokens += (shift_labels != -100).sum().item()
#             if i % 50 == 0:
#                 logger.info(f"[{peft_dir}] Processed batch {i}, cumulative loss: {total_loss:.2f}, tokens: {total_tokens}")

#     avg_loss = total_loss / total_tokens
#     perplexity = torch.exp(torch.tensor(avg_loss))
#     print(f"\nPEFT model: {peft_dir}")
#     print(f"Perplexity on wikitext-103-v1: {perplexity.item():.2f}")

#     # Example inference
#    # 5 diverse prompts for generation
#     prompts = [
#         "The theory of relativity was developed by",
#         "The capital city of France is",
#         "Python is a programming language that",
#         "The process of photosynthesis in plants involves",
#         "The Great Wall of China was built to",
#     ]
#     print("\n=== Generation Examples ===")
#     for prompt in prompts:
#         inputs = tokenizer(prompt, return_tensors="pt").to(device)
#         with torch.no_grad():
#             generated_ids = model.generate(
#                 **inputs,
#                 max_new_tokens=100,
#                 do_sample=True,
#                 top_p=0.90,
#                 temperature=0.9,
#                 pad_token_id=tokenizer.eos_token_id
#             )
#         output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#         print(f"Prompt: {prompt}")
#         print(f"Generated: {output_text}\n")



import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict
import logging
import adapters

BASE_MODEL_PATH = "/root/Martin/NasPEFT/naspeft/models/Llama-3.2-1B/"
OUTPUT_DIR = "/root/Martin/NasPEFT/naspeft//llama-3.2-alpaca-sft/"
BATCH_SIZE = 8

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_peft_model")

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Alpaca dataset
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
dataset = dataset["test"]  # Use test split for evaluation
logger.info(f"Loaded test dataset with {len(dataset)} samples.")

# Tokenize dataset
def tokenize_function(examples):
    # Combine instruction, input, and output into a single text
    texts = []
    for ex in zip(examples["instruction"], examples["input"], examples["output"]):
        instruction, input_text, output = ex
        text = f"{instruction}\n{input_text}\n{output}" if input_text.strip() else f"{instruction}\n{output}"
        texts.append(text)
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

logger.info("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["instruction", "input", "output"])
logger.info("Tokenization complete.")

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, collate_fn=collator)

# Find all PEFT model directories
peft_dirs = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR)
             if os.path.isdir(os.path.join(OUTPUT_DIR, d))]

logger.info(f"Found {len(peft_dirs)} PEFT model directories.")

for peft_dir in peft_dirs:
    logger.info(f"Evaluating model in {peft_dir}")

    # Load base model and move to device
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16)
    adapters.init(model)
    model.to(device)
    model.eval()

    # Try to load PEFT adapter if exists
    adapter_path = os.path.join(peft_dir, "lora")
    if os.path.isdir(adapter_path):
        try:
            adapter_name = "lora"
            model.load_adapter(adapter_path, load_as=adapter_name)
            model.set_active_adapters(adapter_name)
            logger.info(f"Loaded adapter from {adapter_path}")

            # Ensure all parameters (including adapter weights) match model dtype and device
            for name, param in model.named_parameters():
                if param.dtype != model.dtype or param.device != device:
                    param.data = param.data.to(dtype=model.dtype, device=device)
        except Exception as e:
            logger.warning(f"Could not load adapter from {adapter_path}: {e}")
            continue
    else:
        logger.warning(f"No adapter found in {adapter_path}, skipping.")
        continue
    

    # Evaluation loop
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()
            total_tokens += (shift_labels != -100).sum().item()
            if i % 50 == 0:
                logger.info(f"[{peft_dir}] Processed batch {i}, cumulative loss: {total_loss:.2f}, tokens: {total_tokens}")

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"\nPEFT model: {peft_dir}")
    print(f"Perplexity on Alpaca test split: {perplexity.item():.2f}")

    # Example inference
    # 5 diverse prompts for generation, tailored for Alpaca's instruction-based format
    prompts = [
        "Write a short poem about the moon.",
        "Explain the concept of gravity in simple terms.",
        "Generate a tweet about artificial intelligence.",
        "Describe how to make a sandwich step-by-step.",
        "Summarize the benefits of recycling in one sentence."
    ]
    print("\n=== Generation Examples ===")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.90,
                temperature=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Generated: {output_text}\n")
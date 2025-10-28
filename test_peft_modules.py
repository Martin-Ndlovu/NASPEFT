
import os
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.INFO)
logging.getLogger("adapters").setLevel(logging.INFO)

import adapters
import torch
from datasets import load_dataset
import re
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    EarlyStoppingCallback,
    Trainer
)
# from adapters import AdapterTrainer, LoRAConfig
from peft import LoraConfig, TaskType, get_peft_model
from adapters import LoRAConfig, PrefixTuningConfig
import torch.distributed as dist

def is_main_process():
    # Accelerate sets LOCAL_RANK or RANK
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/Llama-3.2-1B"
OUTPUT_DIR = "output/llama1B_lora_wikitext103_adapters_epoch3"
MAX_LEN = 512
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 8
EPOCHS = 3
LEARNING_RATE = 5e-4


def filter_texts(ex):
    text = ex["text"].strip()
    # Remove markup, special tokens, and headers
    text = re.sub(r'@-@', '-', text)  # Replace hyphen markup
    text = re.sub(r'<unk>', '', text)  # Remove unknown tokens
    text = re.sub(r'=+ .*? =+', '', text)  # Remove wiki section headers (e.g., "= = Gameplay = =")
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    # Check for non-empty, minimum length, and meaningful content
    words = text.split()
    unique_words = len(set(words))
    return (
        bool(text) and
        len(words) >= 8 and  # Slightly lower threshold for flexibility
        len(text) >= 50 and  # Ensure sufficient character length
        unique_words >= 5 and  # Ensure some lexical diversity
        not re.match(r'^\W+$', text)  # Exclude texts with only punctuation/symbols
    )

def tokenize_batch(examples, tokenizer):
    texts = examples["text"]
    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc["labels"] = enc["input_ids"].clone()
    return enc

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
        return {"eval_perplexity": perplexity.item()}

if __name__ == "__main__":
    set_seed(42)

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    for split in ("train", "validation", "test"):
        dataset[split] = dataset[split].filter(filter_texts, batched=False)

    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(100000))  # Limit for quicker runs
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(1000))
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(1000))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Tokenize datasets
    tokenized = {}
    for split in ("train", "validation", "test"):
        tokenized[split] = dataset[split].map(
            lambda x: tokenize_batch(x, tokenizer),
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing {split}",
        )
        tokenized[split].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    num_layers = 1 # For 1B model

    for layer in range(num_layers):
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        # model.to(device)/
        model.config.use_cache = False
        # model.train()

        # Initialize adapters
        adapters.init(model)

        # Define LoRA configuration for peft library
        # lora_config = LoraConfig(
        #     r=16,
        #     lora_alpha=64,
        #     lora_dropout=0.1,
        #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        #     task_type=TaskType.CAUSAL_LM,
        #     inference_mode=False,
        #     # layers_to_transform=[layer],
        #     # layers_pattern="layers",  
        # )

        # # Define LoRA configuration for adapters library
        # lora_config = LoRAConfig(
        #     r=16,
        #     alpha=64,
        #     dropout=0.1,
        #     attn_matrices=["q", "k", "v", "o", "g", "u", "d"],
        #     init_weights_seed=42,
        #     # dtype=torch.bfloat16, 
        # )

        # Define Prefix Tuning configuration for adapters library
        prefix_config = PrefixTuningConfig(
            prefix_length=512,
            dropout=0.1,
        )

        model.add_adapter("prefix", config=prefix_config)
        model.train_adapter("prefix")
        model.set_active_adapters("prefix")
    
        # model = get_peft_model(model, lora_config)


        if is_main_process():
            print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            # model.print_trainable_parameters()

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

        LAYER_OUTPUT_DIR = os.path.join(OUTPUT_DIR, f"layer_{layer}")
        os.makedirs(LAYER_OUTPUT_DIR, exist_ok=True)

        train_args = TrainingArguments(
            output_dir=LAYER_OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            warmup_steps=100,
            bf16=True,
            fp16=False,
            logging_steps=50,
            save_steps=100,
            dataloader_num_workers=4,
            save_total_limit=3,
            eval_strategy="no",
            metric_for_best_model="eval_perplexity",
            eval_steps=100,
            # load_best_model_at_end=True,
            report_to="none",
            ddp_find_unused_parameters=False,
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=tokenized["train"],
            # eval_dataset=tokenized["validation"],
            # compute_metrics=compute_metrics,
            data_collator=collator,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)],
        )

        if is_main_process():
            print(f"Starting training layer {layer}...")
        checkpoint_dir = os.path.join(LAYER_OUTPUT_DIR, "checkpoint-3001")
        if os.path.exists(checkpoint_dir):
            trainer.train(resume_from_checkpoint=checkpoint_dir)
        else:
            trainer.train()
        trainer.save_model(os.path.join(LAYER_OUTPUT_DIR, "final"))
import argparse
import math
import numpy as np
import os
import random
import torch
from transformers import set_seed
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from adapters import AutoAdapterModel, AdapterConfig, AdapterTrainer

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_main_process():
    # Accelerate sets LOCAL_RANK or RANK
    return int(os.environ.get("LOCAL_RANK", 0)) == 0

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate the trained language adapter for LLaMA-3-8B using CLM on wikitext-103-v1 test set.")
    parser.add_argument("--language", type=str, default="en", help="Target language (e.g., cy for Cornish)")
    parser.add_argument("--adapter_dir", type=str, default="./output/union_adapter", help="Base directory for adapter checkpoints")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--num_generation_examples", type=int, default=5, help="Number of examples for text generation")
    parser.add_argument("--generation_max_length", type=int, default=50, help="Maximum length for generated text")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Number of new tokens to generate")
    parser.add_argument("--prompt_length", type=int, default=50, help="Length of the prompt for generation")
    return parser.parse_args()

args = parse_arguments()

def find_latest_checkpoint(adapter_base_path: str, language_code: str) -> str:
    """
    Finds the latest checkpoint directory for the given language.
    """
    language_code = language_code.replace("_", "-")
    lang_adapter_path = os.path.join(adapter_base_path, language_code, "clm")
    checkpoints = [d for d in os.listdir(lang_adapter_path) if d.startswith("checkpoint")]
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))  # Sort by checkpoint number
    latest_checkpoint = checkpoints[-1]  # Get the latest checkpoint
    return os.path.join(lang_adapter_path, latest_checkpoint)

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

    # Load the trained Seq_bn_inv adapter
    adapter_dir =  args.adapter_dir + "/en/clm/checkpoint-16000/clm" #find_latest_checkpoint(args.adapter_dir, args.language) + "/clm"
    lang_adapter_config = AdapterConfig.load(os.path.join(adapter_dir, "adapter_config.json"))
    model.load_adapter(adapter_dir, config=lang_adapter_config, load_as="clm", with_head=False)
    model.set_active_adapters("clm")

    # Load and preprocess test dataset (consistent with training)
    def filter_texts(example):
        return example["text"] is not None and len(example["text"].strip()) > 0

    dataset = load_dataset("wikitext", "wikitext-2-v1")
    dataset["test"] = dataset["test"].filter(filter_texts, batched=False)
    dataset["test"] = dataset["test"].shuffle(seed=42)  # Consistent with training's test selection

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_seq_length)

    tokenized_test = dataset["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing test dataset",
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

    tokenized_test = tokenized_test.map(
        group_texts,
        batched=True,
        desc=f"Grouping test texts into chunks of {args.max_seq_length}",
    )

    # Data collator for CLM (consistent with training)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments for evaluation
    eval_args = TrainingArguments(
        output_dir="./eval_temp",  # Temporary directory for evaluation
        per_device_eval_batch_size=1,
        overwrite_output_dir=True,
    )

    # Initialize trainer for evaluation
    trainer = AdapterTrainer(
        model=model,
        args=eval_args,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
    )

    # Calculate perplexity
    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results["eval_loss"])
    if is_main_process():
        print(f"Perplexity on test set: {perplexity}")

    # Test generation
    if is_main_process():
        print("\nTest Generation Examples:")
    model.eval()
    for i in range(args.num_generation_examples):
        # Take a prompt from the test set (first tokens of a chunk)
        example = tokenized_test[i]
        input_ids = torch.tensor([example["input_ids"][:args.prompt_length]])  # Use first 50 tokens as prompt
        attention_mask = torch.tensor([example["attention_mask"][:args.prompt_length]])

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

    # Generate text with better parameters
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=min(args.max_new_tokens, 200),  
        num_return_sequences=1,
        do_sample=True,  # Add sampling for more diverse output
        temperature=0.7,  # Control randomness
        top_p=0.9,       # Nucleus sampling
        repetition_penalty=1.3,  # Reduce repetition
        no_repeat_ngram_size=3,  # Slightly larger n-gram prevention
        pad_token_id=tokenizer.eos_token_id,
        penalty_alpha=0.6,  # Mirostat parameter
        top_k=4,           # Mirostat parameter
        # early_stopping=True,
    )

    def clean_generated_text(text):
        """Clean special tokens from generated text"""
        import re
        # Replace special tokens
        text = re.sub(r'@-@', '-', text)
        text = re.sub(r'@,@', ',', text)
        text = re.sub(r'@\.@', '.', text)
        # Remove <unk> tokens
        text = re.sub(r'<unk>', '', text)
        # Clean up any extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Decode and clean the generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    cleaned_text = clean_generated_text(generated_text)
    
    # Also clean the prompt for display
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    cleaned_prompt = clean_generated_text(prompt_text)

    if is_main_process():
        print(f"Example {i+1}:")
        print(f"Prompt: {cleaned_prompt}")
        print(f"Generated: {cleaned_text}")
        print("-" * 80)

if __name__ == "__main__":
    main()


# # dist_test.py
# # allreduce_safe_diag.py
# import os
# import time
# import datetime
# import traceback
# import torch
# import torch.distributed as dist
# from torch.multiprocessing import spawn

# # --------- Set early env vars (tweakable) ----------
# os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
# os.environ.setdefault("MASTER_PORT", "12357")

# # Diagnostic NCCL/CUDA envs (you can change recommended toggles below)
# os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
# os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
# os.environ.setdefault("NCCL_DEBUG", "INFO")
# os.environ.setdefault("NCCL_P2P_DISABLE", "1")   # try toggling 1 if hang persists
# os.environ.setdefault("NCCL_IB_DISABLE", "1")
# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
# os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

# TIMEOUT_SECONDS = 40  # how long to wait on a collective before we declare a problem

# def now():
#     return time.strftime("%H:%M:%S")

# def wait_with_timeout(work, timeout_secs, desc="work"):
#     """Wait for a distributed Work, but timeout and raise if exceeded."""
#     start = time.time()
#     if work is None:
#         return
#     # Work.wait() will block; wrap in small sleeps to allow timeout detection.
#     while True:
#         try:
#             work.wait(timeout=1.0)  # some torch versions accept timeout on wait()
#             return
#         except TypeError:
#             # If wait(timeout=..) not supported, fallback to calling wait() once and rely on timeout externally
#             work.wait()
#             return
#         except Exception as e:
#             # If it's a timeout from underlying wait, check total elapsed
#             elapsed = time.time() - start
#             if elapsed >= timeout_secs:
#                 raise TimeoutError(f"Timeout waiting for {desc} after {elapsed:.1f}s") from e
#             # otherwise loop and try again
#             time.sleep(0.1)

# def run(rank, world_size):
#     print(f"[{now()}] Rank {rank}: starting. Env NCCL_DEBUG={os.environ.get('NCCL_DEBUG')} TORCH_NCCL_BLOCKING_WAIT={os.environ.get('TORCH_NCCL_BLOCKING_WAIT')}")
#     try:
#         dist.init_process_group(
#             backend="nccl",
#             init_method=f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}",
#             rank=rank,
#             world_size=world_size,
#             timeout=datetime.timedelta(seconds=60),
#         )
#         torch.cuda.set_device(rank)
#         dev = torch.device(f"cuda:{rank}")

#         # Use a larger tensor (avoid tiny-size edge cases)
#         tensor = torch.ones(256, device=dev, dtype=torch.float32) * (rank + 1.0)
#         if rank == 0:
#             tensor.fill_(999.0)

#         print(f"[{now()}] Rank {rank}: before broadcast")
#         dist.broadcast(tensor, src=0)
#         torch.cuda.synchronize()
#         print(f"[{now()}] Rank {rank}: after broadcast sample={tensor.flatten()[0].item()}")

#         # barrier before all_reduce to ensure ordering
#         print(f"[{now()}] Rank {rank}: issuing async barrier")
#         barrier_work = dist.barrier(async_op=True)
#         wait_with_timeout(barrier_work, TIMEOUT_SECONDS, desc="barrier")
#         print(f"[{now()}] Rank {rank}: barrier complete")

#         # short pause to let GPU kernels settle
#         time.sleep(0.2)

#         # async all_reduce + wait with timeout
#         print(f"[{now()}] Rank {rank}: issuing async all_reduce")
#         ar_work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
#         wait_with_timeout(ar_work, TIMEOUT_SECONDS, desc="all_reduce")
#         torch.cuda.synchronize()
#         print(f"[{now()}] Rank {rank}: after all_reduce sample={tensor.flatten()[0].item()}")

#         # try graceful destroy
#         try:
#             dist.destroy_process_group()
#             print(f"[{now()}] Rank {rank}: destroy_process_group OK")
#         except Exception as e:
#             print(f"[{now()}] Rank {rank}: destroy_process_group error: {e}")

#         print(f"[{now()}] Rank {rank}: finished successfully.")

#     except Exception as e:
#         print(f"[{now()}] Rank {rank}: Exception:\n{traceback.format_exc()}")
#         # best-effort cleanup
#         try:
#             dist.destroy_process_group()
#         except Exception:
#             pass
#         raise

# if __name__ == "__main__":
#     world_size = torch.cuda.device_count() or 2
#     print(f"[{now()}] Launching {world_size} procs")
#     spawn(run, args=(world_size,), nprocs=world_size, join=True)

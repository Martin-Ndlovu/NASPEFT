import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset, DatasetDict
import math
from bert_score import score

# -----------------------------
# Paths
# -----------------------------
base_model_name = "models/Llama-3.2-1B"           # same base model you used
adapter_path    = "output/layer_selection/layer_0/finetuned_layer_0"  # path where LoRA adapter was saved

# -----------------------------
# Load base model in 4-bit
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = True  # for inference

# -----------------------------
# Load LoRA adapter
# -----------------------------
print("Loading LoRA adapter...")
# model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()  # set to evaluation mode

# Optional: merge LoRA into base for faster inference
# model = model.merge_and_unload()

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------------
# Function to generate text
# -----------------------------
def generate(prompt, max_tokens=120, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.2, 
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# -----------------------------
# Function to compute perplexity
# -----------------------------
# def compute_perplexity(texts):
#     model.eval()
#     total_loss = 0.0
#     total_tokens = 0
#     for text in texts:
#         inputs = tokenizer(text, return_tensors="pt").to(model.device)
#         with torch.no_grad():
#             outputs = model(**inputs, labels=inputs["input_ids"])
#         total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
#         total_tokens += inputs["input_ids"].size(1)
#     ppl = math.exp(total_loss / total_tokens)
#     return ppl

def compute_perplexity(texts, batch_size=8, max_length=512, device=None):
    """
    Robust perplexity computation:
      - Tokenizes in batches (padding/truncation)
      - Sets labels = input_ids with pad tokens -> -100 (ignored)
      - Counts actual label tokens used by the model (labels[:,1:] non -100)
      - Accumulates sum negative log-likelihood, then exponentiates average.
    """
    device = device or model.device
    model.eval()
    total_nll = 0.0        # sum of negative log-likelihood across all active tokens
    total_active_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        # prepare labels: copy input_ids and convert pad tokens to -100
        labels = enc["input_ids"].clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100
        # If you explicitly want to ignore the first token (causal LM shift), model does that internally.
        # But to compute the number of tokens used in loss, consider labels[:,1:].
        with torch.no_grad():
            outputs = model(**enc, labels=labels)

        # outputs.loss is mean NLL per active (non -100) token in shift_labels (i.e., labels[:,1:])
        # So compute number of *active* label tokens that loss averaged over:
        # active_tokens = count of labels[:,1:] that are != -100
        active_tokens = (labels[:, 1:] != -100).sum().item()  # integer
        if active_tokens == 0:
            continue  # skip examples that provide no label tokens

        # outputs.loss is a scalar averaged over those `active_tokens`
        nll_sum = outputs.loss.item() * active_tokens

        total_nll += nll_sum
        total_active_tokens += active_tokens

    if total_active_tokens == 0:
        return float("inf")  # no tokens => undefined perplexity

    avg_nll = total_nll / total_active_tokens
    ppl = math.exp(avg_nll)
    return ppl


# -----------------------------
# Function to compute BERTScore
# -----------------------------
def compute_bertscore(preds, refs, lang="en"):
    P, R, F1 = score(preds, refs, lang=lang, rescale_with_baseline=True)
    return P.mean().item(), R.mean().item(), F1.mean().item()


# Load and split Alpaca dataset
def load_and_split_alpaca_dataset():
    print("Loading Alpaca dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned")
    # print(f"Alpaca dataset loaded with splits: {list(dataset.keys())}")

    print("Splitting Alpaca dataset into 80/10/10 train/validation/test...")
    splits = dataset['train'].train_test_split(test_size=0.1, seed=42)
    train_val = splits['train'].train_test_split(test_size=0.1111, seed=42)
    
    alpaca_splits = DatasetDict({
        'train': train_val['train'],
        'validation': train_val['test'],
        'test': splits['test']
    })
    
    # print(f"Final splits: train={len(alpaca_splits['train'])}, "
    #             f"validation={len(alpaca_splits['validation'])}, "
    #             f"test={len(alpaca_splits['test'])}")
    return alpaca_splits

dataset = load_and_split_alpaca_dataset()
train_dataset = dataset["train"]

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    prompt = """### Instruction:
Write a short poem about AI and the ocean.

### Response:
"""
    output = generate(prompt)
    print("\n===== Generated Output =====\n")
    print(output)

    # -----------------------------
    # Load Alpaca validation set for evaluation
    # -----------------------------
    print("\nLoading Alpaca validation dataset for metrics...")
    dataset = load_and_split_alpaca_dataset()
    val_texts = dataset["test"]

    # print("The evaluation dataset has", len(val_texts), "examples.")
    # print("Val_texts example:", val_texts)

    # Generate outputs for BERTScore
    print("Generating model outputs for BERTScore...")

    new_prompts = [f"### Instruction: {ex['instruction']}"
        + (f"### Input:{ex['input']}" if ex['input'] else "")
        + f"### Response: {ex['output']}" for ex in val_texts]
    
    generated_texts = [generate(new_prompt, max_tokens=100) for new_prompt in new_prompts[:100]]  # 100 examples for demo

    reference_texts = val_texts["output"][:100]

    # Compute perplexity
    print("Computing perplexity on validation set...")
    ppl = compute_perplexity(new_prompts)  # limit to 100 examples for speed
    print(f"Perplexity: {ppl:.2f}")

    # print("Here is generated texts and reference texts lengths: ")
    # print(len(generated_texts), len(reference_texts))

    print(f"Generated example text: {generated_texts[2].split('### Response:')[-1]}")
    print(f"Reference example text: {reference_texts[2]}")


    # Compute BERTScore
    P, R, F1 = compute_bertscore(generated_texts, reference_texts)
    print(f"BERTScore -> Precision: {P:.4f}, Recall: {R:.4f}, F1: {F1:.4f}")

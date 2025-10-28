import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from datasets import load_dataset
import math
from bert_score import score
from adapters import AutoAdapterModel
import os

# -----------------------------
# Paths
# -----------------------------
base_model_name = "models/Llama-3.2-1B"           # same base model you used
adapter_path    = "output/llama-3.2-alpaca-sft"  # path where LoRA adapter was saved
# adapter_name = "alpaca_adapter"

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
base_model = AutoAdapterModel.from_pretrained(
    base_model_name,
    # quantization_config=bnb_config,
    # device_map="auto",
    trust_remote_code=True,
)
base_model.config.use_cache = True  # for inference
# base_model.to(device)

# Load LoRA adapter with adapters library
# print("Loading LoRA adapter...")
# adapter_name = base_model.load_adapter(
#     adapter_path,
#     config = "lora",
#     set_active=True,
#     load_as="alpaca_adapter")
# base_model.set_active_adapters(adapter_name)
base_model.eval()
model = base_model
model.to(device)


# Optional: merge LoRA into base for faster inference
# model = model.merge_and_unload()

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# -----------------------------
# Function to generate text
# -----------------------------
def generate(prompt, max_tokens=120, temperature=0.7, top_p=0.9):
    # device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
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
def compute_perplexity(texts):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
        total_tokens += inputs["input_ids"].size(1)
    ppl = math.exp(total_loss / total_tokens)
    return ppl

# -----------------------------
# Function to compute BERTScore
# -----------------------------
def compute_bertscore(preds, refs, lang="en"):
    P, R, F1 = score(preds, refs, lang=lang, rescale_with_baseline=True)
    return P.mean().item(), R.mean().item(), F1.mean().item()

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    prompt = """### Instruction:
What do you know about Artificial intelligence.

### Response:
"""
    output = generate(prompt)
    print("\n===== Generated Output =====\n")
    print(output.split("### Response:")[-1].strip())

    # -----------------------------
    # Load Alpaca validation set for evaluation
    # -----------------------------
    print("\nLoading Alpaca validation dataset for metrics...")
    dataset = load_dataset("yahma/alpaca-cleaned")
    val_texts = [
        f"### Instruction:\n{ex['instruction']}"
        + (f"\n### Input:\n{ex['input']}" if ex['input'] else "")
        + f"\n### Response: "
        for ex in dataset['train'].train_test_split(test_size=0.1, seed=42)['test']
    ]

    print("The evaluation dataset has", len(val_texts), "examples.")

    # Compute perplexity
    # print("Computing perplexity on validation set...")
    # ppl = compute_perplexity(val_texts)  # limit to 100 examples for speed
    # print(f"Perplexity: {ppl:.2f}")

    # Generate outputs for BERTScore
    print("Generating model outputs for BERTScore...")
    generated_texts = [generate(text, max_tokens=100) for text in val_texts[:100]]  # 100 examples for demo
    reference_texts = [text.split("### Response:")[-1].strip() for text in val_texts[:100]]

    print("Generated text: \n", generated_texts[2])
    print(f"\n\nReference texts: \n{reference_texts[2]}")


    # Compute BERTScore
    # P, R, F1 = compute_bertscore(generated_texts, reference_texts)
    # print(f"BERTScore -> Precision: {P:.4f}, Recall: {R:.4f}, F1: {F1:.4f}")

# import os
# import json
# import torch
# from peft import PeftModel
# from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
# from datasets import load_dataset
# from torch.utils.data import DataLoader

# MODEL_PATH = "/root/Martin/NasPEFT/naspeft/models/Llama-3.2-1B/"
# DATA_PATH = "/root/Martin/NasPEFT/naspeft/datasets/wikitext/wikitext-103-v1/"
# ADAPTERS_DIR = "output/llama1B_lora_wikitext103_layers_epoch3"  # Directory containing layer subdirectories
# RESULTS_FILE = "layer_selection_3epoch_eval_results.json"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load tokenizer once
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # Load dataset once
# dataset = load_dataset("wikitext", "wikitext-103-v1", split="test", cache_dir=DATA_PATH)

# def tokenize_function(examples):
#     return tokenizer(examples["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=512)

# tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# dataloader = DataLoader(
#     tokenized_dataset,
#     batch_size=8,
#     collate_fn=collator
# )

# loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

# # Load previous results if file exists
# if os.path.exists(RESULTS_FILE):
#     with open(RESULTS_FILE, "r") as f:
#         all_results = json.load(f)
# else:
#     all_results = []

# # Loop over all layer folders
# for layer_dir in sorted(os.listdir(ADAPTERS_DIR)):
#     adapter_path = os.path.join(ADAPTERS_DIR, layer_dir, "final")
#     if not os.path.isdir(adapter_path):
#         continue  # skip if not a directory

#     print(f"Evaluating {layer_dir}...")

#     # Fresh base model each time
#     base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
#     model = PeftModel.from_pretrained(base_model, adapter_path)
#     model.to(device)
#     model.eval()

#     # Evaluation loop
#     total_loss = 0.0
#     total_tokens = 0

#     with torch.no_grad():
#         for batch in dataloader:
#             inputs = {k: v.to(device) for k, v in batch.items()}
#             labels = inputs["labels"]
#             outputs = model(**inputs)

#             shift_logits = outputs.logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()

#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
#                             shift_labels.view(-1))

#             total_loss += loss.item()
#             total_tokens += (shift_labels != -100).sum().item()

#     avg_loss = total_loss / total_tokens
#     perplexity = torch.exp(torch.tensor(avg_loss)).item()

#     print(f"{layer_dir} perplexity: {perplexity:.2f}")

#     # Append results
#     all_results.append({
#         "layer": layer_dir,
#         "perplexity": perplexity
#     })

#     # Save after each run (so progress is not lost if interrupted)
#     with open(RESULTS_FILE, "w") as f:
#         json.dump(all_results, f, indent=4)

# print("Evaluation complete. Results saved to", RESULTS_FILE)













# This script evaluates a base language model and a single adapter with a LoRA adapter on the wikitext-103
import torch
import re
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
from adapters import AutoAdapterModel

# Configuration
MODEL_PATH = "/root/Martin/NasPEFT/naspeft/models/Llama-3.2-1B/"
DATA_PATH = "/root/Martin/NasPEFT/naspeft/datasets/wikitext/wikitext-2/"
ADAPTER_PATH = "output/llama1B_lora1/final/lora/"
MAX_LEN = 512
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model
model = AutoAdapterModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
try:
    model.load_adapter(ADAPTER_PATH, load_as="lora", set_active=True)
    model.active_adapters = "lora"  # Explicitly set active adapter
except OSError as e:
    print(f"Error loading adapter: {e}")
    print(f"Please verify that {ADAPTER_PATH} contains pytorch_adapter.bin and adapter_config.json")
    exit(1)

# Ensure all parameters are in bfloat16
model = model.to(torch.bfloat16)
for name, param in model.named_parameters():
    if param.dtype != torch.bfloat16:
        param.data = param.data.to(torch.bfloat16)

model.to(device)
model.eval()

# Verify active adapters
print(f"Active adapters: {model.active_adapters}")

# Load and filter dataset
def filter_texts(ex):
    text = ex["text"].strip()
    text = re.sub(r'@-@', '-', text)
    text = re.sub(r'<unk>', '', text)
    text = re.sub(r'=+ .*? =+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    unique_words = len(set(words))
    return (
        bool(text) and
        len(words) >= 8 and
        len(text) >= 50 and
        unique_words >= 5 and
        not re.match(r'^\W+$', text)
    )

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=DATA_PATH)
dataset = dataset.filter(filter_texts, batched=False)
dataset = dataset.shuffle(seed=42).select(range(1000))

# Tokenize dataset
def tokenize_batch(examples):
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

tokenized_dataset = dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing test",
)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Data collator
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

# Create DataLoader
dataloader = DataLoader(
    tokenized_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collator,
)

# Evaluation loop
total_loss = 0.0
total_tokens = 0
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

with torch.no_grad():
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        labels = inputs["labels"]
        outputs = model(**inputs)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        total_loss += loss.item()
        total_tokens += (shift_labels != -100).sum().item()

# Calculate perplexity
avg_loss = total_loss / total_tokens
perplexity = torch.exp(torch.tensor(avg_loss))
print(f"Total valid tokens: {total_tokens}")
print(f"Average loss per token: {avg_loss:.4f}")
print(f"Model perplexity on wikitext-103-v1 test set: {perplexity.item():.2f}")





# # # This script evaluates a base language model and a single adapter with a LoRA adapter on the wikitext-103
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import adapters
# import torch

# MODEL_PATH = "models/Llama-3.2-1B"
# ADAPTER_PATH = "output/llama1B_lora_wikitext103_adapters_epoch3/layer_0/adapter"

# # 1️⃣ Load the full base model including the LM head
# base_model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=torch.float32,
#     trust_remote_code=True,
# )

# # 2️⃣ Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

# # 3️⃣ Resize embeddings to match tokenizer
# base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

# # 4️⃣ Initialize the adapters system
# adapters.init(base_model)

# import os
# import json

# # 5️⃣ Check adapter_config.json to confirm adapter name
# try:
#     with open(os.path.join(ADAPTER_PATH, "adapter_config.json"), "r") as f:
#         adapter_config = json.load(f)
#     # print(f"Adapter config: {adapter_config}")
#     adapter_name_in_config = adapter_config.get("name", "prefix")  # Default to 'prefix' if name not specified
# except FileNotFoundError:
#     # print(f"Error: adapter_config.json not found in {ADAPTER_PATH}")
#     raise

# # 6️⃣ Load the adapter and activate it
# try:
#     adapter_name = base_model.load_adapter(
#         ADAPTER_PATH,
#         load_as=adapter_name_in_config,  # Use name from config
#         source="local",
#     )
#     # print(f"Loaded adapter: {adapter_name}")
#     base_model.set_active_adapters(adapter_name)
#     # print(f"Active adapters: {base_model.active_adapters}")
# except OSError as e:
#     print(f"Error loading adapter: {e}")
#     raise

# # 7️⃣ Print adapter summary
# print("Adapter Summary:")
# print(base_model.adapter_summary())

# # 8️⃣ Inspect adapter module
# adapter_module = base_model.get_adapter(adapter_name)
# if adapter_module:
#     print("Adapter Module Structure:")
#     # print(adapter_module)
#     print("Adapter Trainable Parameters:")
#     # for name, param in adapter_module.named_parameters():
#     #     if param.requires_grad:
#     #         print(f"{name}: {param.numel()} parameters")
# else:
#     print(f"Error: Could not retrieve adapter module for {adapter_name}")

# # 9️⃣ Optional: Summarize adapter with torchsummary (if installed)
# try:
#     from torch import torchsummary as summary
#     print("Adapter Parameter Summary:")
#     summary(adapter_module, input_size=(512, base_model.config.hidden_size))
# except ImportError:
#     print("torchsummary not installed. Skipping parameter summary.")
# except Exception as e:
#     print(f"Error summarizing adapter: {e}")

# # 10️⃣ Optional: Visualize adapter computation graph with torchviz (if installed)
# try:
#     from torch import torchviz as make_dot
#     dummy_input = torch.randn(1, 512, base_model.config.hidden_size).to("cuda")
#     base_model.to("cuda")
#     base_model.eval()
#     with torch.no_grad():
#         output = base_model(dummy_input)
#     dot = make_dot(output, params=dict(base_model.named_parameters()))
#     dot.render("adapter_graph", format="png")
#     print("Saved adapter computation graph as adapter_graph.png")
# except ImportError:
#     print("torchviz not installed. Skipping computation graph visualization.")
# except Exception as e:
#     print(f"Error visualizing adapter: {e}")

# # 11️⃣ Convert to bfloat16 and move to device
# base_model.to(torch.bfloat16)
# base_model.to("cuda")
# base_model.eval()

# 12️⃣ Test inference
prompt = "Explain the importance of neural architecture search (NAS) in AI."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")

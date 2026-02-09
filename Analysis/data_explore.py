
'''

# This file will be used to explore the dataset. The dataset name is wikitext-103-v1 and the aim is to know its length,
# number of samples and other basic statistics. This information will be critical for determining max_seq_length
# and other hyperparameters when fine-tuning.


import logging
import numpy as np
from datasets import load_dataset, DatasetDict
import re
import unicodedata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-v1")
logger.info(f"Original dataset structure: {dataset}")

def is_coherent(text):
    """Check if text has at least 3 sentences for coherence"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences) >= 3

def comprehensive_clean_text(text):
    """Comprehensive cleaning function for Wikipedia-specific artifacts"""
    if not text or not text.strip():
        logger.debug("Skipping empty text")
        return ""
    
    text = text.strip()
    
    # Step 1: Remove Wikipedia markup
    logger.debug("Removing Wikipedia markup")
    text = re.sub(r'=+\s*[^=]+\s*=+', '', text)  # Section headers
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)  # Wiki links
    text = re.sub(r'\[\[(File|Image):[^\]]+\]\]', '', text)  # File/image links
    text = re.sub(r'\{\{[^}]+\}\}', '', text)  # Templates
    text = re.sub(r'@-@', '-', text)  # Special characters
    text = re.sub(r'@,@', ',', text)
    text = re.sub(r'@.@', '.', text)
    text = re.sub(r'^\s*[\*\+\-]\s*.*$', '', text, flags=re.MULTILINE)  # Lists
    text = re.sub(r'\(\s*Japanese\s*:.*?\)', '', text)  # Metadata
    text = re.sub(r'\|\s*[^|]+\s*\|', '', text)  # Tables
    
    # Step 2: Normalize Unicode (including apostrophes)
    logger.debug("Normalizing Unicode")
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    # Explicitly replace any remaining curly apostrophes with straight ones
    text = text.replace("'", "'").replace("'", "'")
    
    # Step 3: Remove citations and URLs
    logger.debug("Removing citations and URLs")
    text = re.sub(r'\[\d+(?:\s*-\s*\d+)?\]', '', text)
    text = re.sub(r'\[citation needed\]', '', text)
    text = re.sub(r'\[[^\]]+\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\b(according to|as stated in|cited in)\b.*?(?=\.\s|$)', '', text, flags=re.IGNORECASE)
    
    # Step 4: Clean punctuation
    logger.debug("Cleaning punctuation")
    text = re.sub(r'\.{3,}', '...', text)
    text = re.sub(r'\!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'[\'"]+', '"', text)
    text = re.sub(r'[:;]\s*$', '', text)
    
    # Step 5: Fix spacing around punctuation and decimals
    logger.debug("Fixing punctuation and decimal spacing")
    # Remove spaces before punctuation, including apostrophes
    text = re.sub(r'\s+([.,!?;:")\'-])', r'\1', text)
    # Ensure space after punctuation (except at end or before another punctuation)
    text = re.sub(r'([.,!?;:])(?=[^\s.,!?;:])', r'\1 ', text)
    # Fix decimal numbers (e.g., "11. 14" -> "11.14")
    text = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', text)
    # Fix possessives and contractions (e.g., "men"s" -> "men's")
    text = re.sub(r'(\w+)\s+\'s', r"\1's", text)
    text = re.sub(r'(\w+)\s+\'(\w+)', r"\1'\2", text)
    
    # Step 6: Filter short or incoherent texts
    logger.debug("Checking length and coherence")
    if len(text) < 50 or len(text.split()) < 15 or not is_coherent(text):
        logger.debug("Text filtered out: too short or not coherent")
        return ""
    
    # Step 7: Normalize capitalization and punctuation
    logger.debug("Normalizing capitalization and punctuation")
    if text and text[0].isalpha() and not text[0].isupper():
        text = text[0].upper() + text[1:]
    if text and len(text.split()) > 5 and text[-1] not in ['.', '!', '?']:
        text += '.'
    
    # Step 8: Normalize whitespace
    logger.debug("Normalizing whitespace")
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def filter_and_clean_dataset(example):
    """Apply comprehensive cleaning and filtering"""
    original_text = example['text']
    cleaned_text = comprehensive_clean_text(original_text)
    
    if cleaned_text:
        return {'text': cleaned_text, 'valid': True}
    else:
        return {'text': '', 'valid': False}

# Apply cleaning to all splits
cleaned_dataset = {}
for split in ['train', 'validation', 'test']:
    if split in dataset:
        logger.info(f"Processing {split} split")
        # Apply cleaning
        temp_dataset = dataset[split].map(
            filter_and_clean_dataset,
            batched=False,
            desc=f"Cleaning {split} set"
        )
        # Filter out invalid entries
        cleaned_dataset[split] = temp_dataset.filter(
            lambda example: example['valid'],
            batched=False,
            desc=f"Filtering {split} set"
        )
        # Remove the 'valid' column
        cleaned_dataset[split] = cleaned_dataset[split].remove_columns(['valid'])

# Convert to DatasetDict
cleaned_dataset_dict = DatasetDict(cleaned_dataset)

logger.info(f"\nCleaned dataset structure:")
for split in ['train', 'validation', 'test']:
    if split in cleaned_dataset_dict:
        logger.info(f"{split.capitalize()} set: {len(cleaned_dataset_dict[split])} samples")

# Save the cleaned dataset
output_path = "./datasets/wikitext/wikitext-2-cleaned"
cleaned_dataset_dict.save_to_disk(output_path)
logger.info(f"\nCleaned dataset saved to: {output_path}")



import logging
from datasets import load_from_disk, DatasetDict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the cleaned dataset
dataset_path = "./datasets/wikitext/wikitext-2-cleaned"
logger.info(f"Loading cleaned dataset from: {dataset_path}")
cleaned_dataset = load_from_disk(dataset_path)

# Verify dataset is a DatasetDict
if not isinstance(cleaned_dataset, DatasetDict):
    logger.error("Loaded dataset is not a DatasetDict")
    raise ValueError("Expected a DatasetDict, but got something else")

# Print 5 diverse examples from each split
for split in ['train', 'validation', 'test']:
    if split in cleaned_dataset:
        logger.info(f"\n=== {split.upper()} Split Examples ===")
        texts = cleaned_dataset[split]['text']
        total_samples = len(texts)
        logger.info(f"Total samples in {split}: {total_samples}")

        # Select 5 diverse indices (evenly spaced)
        if total_samples >= 5:
            step = max(1, total_samples // 5)
            indices = range(0, total_samples, step)[:5]
        else:
            # If fewer than 5 samples, use all available
            indices = range(total_samples)

        # Print examples
        for i, idx in enumerate(indices, 1):
            sample = texts[idx]
            word_count = len(sample.split())
            logger.info(f"\nExample {i} ({word_count} words):")
            logger.info(f"{sample}")
    else:
        logger.info(f"\n=== {split.upper()} Split Examples ===")
        logger.info(f"No data found for {split} split")
'''


'''
# ============================================
# WIKITEXT DATASET ANALYSIS FOR THESIS
# ============================================

from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import os

# ============================================
# GLOBAL FONT SETTINGS (Times New Roman)
# ============================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 32
plt.rcParams["axes.titlesize"] = 28
plt.rcParams["axes.labelsize"] = 28
plt.rcParams["xtick.labelsize"] = 28
plt.rcParams["ytick.labelsize"] = 28
plt.rcParams["legend.fontsize"] = 28

# --------------------------------------------
# 1. Load datasets
# --------------------------------------------
wikitext2 = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
wikitext103 = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

# --------------------------------------------
# 2. Initialize tokenizer (LLaMA 3.2)
# --------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("models/Llama-3.2-1B")

# --------------------------------------------
# 3. Function to analyze a dataset split
# --------------------------------------------
def analyze_split(dataset, split_name="train"):
    texts = dataset[split_name]["text"]
    
    token_lengths = []
    all_tokens = []

    for text in tqdm(texts, desc=f"Tokenizing {split_name}"):
        tokens = tokenizer.tokenize(text)
        token_lengths.append(len(tokens))
        all_tokens.extend(tokens)
    
    return {
        "split": split_name,
        "total_tokens": len(all_tokens),
        "vocab_size": len(set(all_tokens)),
        "avg_seq_len": np.mean(token_lengths),
        "median_seq_len": np.median(token_lengths),
        "std_seq_len": np.std(token_lengths),
        "min_seq_len": np.min(token_lengths),
        "max_seq_len": np.max(token_lengths),
        "token_lengths": token_lengths,
        "all_tokens": all_tokens
    }

# --------------------------------------------
# 4. Analyze all splits for both datasets
# --------------------------------------------
def analyze_dataset(dataset, name):
    results = {}
    for split in dataset.keys():
        results[split] = analyze_split(dataset, split)

    print(f"\n===== {name.upper()} STATS =====")
    for split, res in results.items():
        print(f"\n--- Split: {split.upper()} ---")
        print(f"Total Tokens: {res['total_tokens']:,}")
        print(f"Vocab Size: {res['vocab_size']:,}")
        print(f"Avg Seq Len: {res['avg_seq_len']:.2f}")
        print(f"Median Seq Len: {res['median_seq_len']:.2f}")
        print(f"Std Dev Seq Len: {res['std_seq_len']:.2f}")
        print(f"Min Seq Len: {res['min_seq_len']}")
        print(f"Max Seq Len: {res['max_seq_len']}")
    return results

results_wt2 = analyze_dataset(wikitext2, "WikiText-2")
results_wt103 = analyze_dataset(wikitext103, "WikiText-103")

# --------------------------------------------
# 5. Create summary table (train splits only)
# --------------------------------------------
summary = pd.DataFrame([
    {
        "Dataset": "WikiText-2",
        "Total Tokens": results_wt2["train"]["total_tokens"],
        "Vocab Size": results_wt2["train"]["vocab_size"],
        "Avg Seq Len": round(results_wt2["train"]["avg_seq_len"], 2),
        "Median Seq Len": round(results_wt2["train"]["median_seq_len"], 2)
    },
    {
        "Dataset": "WikiText-103",
        "Total Tokens": results_wt103["train"]["total_tokens"],
        "Vocab Size": results_wt103["train"]["vocab_size"],
        "Avg Seq Len": round(results_wt103["train"]["avg_seq_len"], 2),
        "Median Seq Len": round(results_wt103["train"]["median_seq_len"], 2)
    }
])

print("\n===== DATASET SUMMARY =====")
print(summary.to_string(index=False))

# ============================================
# VISUALIZATIONS
# ============================================

from matplotlib.ticker import ScalarFormatter
os.makedirs("figures", exist_ok=True)

# --- 6. Token Length Distribution ---
def plot_split_distributions(dataset_results, dataset_name):
    splits = list(dataset_results.keys())
    fig, axs = plt.subplots(1, len(splits), figsize=(18, 5), sharey=False)

    for i, split in enumerate(splits):
        lengths = np.array(dataset_results[split]["token_lengths"])
        cutoff = np.percentile(lengths, 99)

        sns.histplot(
            lengths,
            bins=100,
            kde=True,
            ax=axs[i],
            color=sns.color_palette("Set2")[i],
            stat="density"
        )

        axs[i].set_xlim(0, cutoff)
        axs[i].set_ylim(bottom=0.0001)

        # axs[i].set_title(f"{dataset_name} - {split.capitalize()} Split", fontsize=24)   # COMMENTED OUT
        # axs[i].set_xlabel(f"Tokens per Text") #(≤ {int(cutoff)} tokens)
        axs[i].set_ylabel("")

        axs[i].tick_params(axis='both', labelsize=28)
        axs[i].grid(False)

    # fig.suptitle(f"{dataset_name} Token Length Distribution Across Splits", fontsize=24)  # COMMENTED OUT
    plt.tight_layout(rect=[0.08, 0, 1, 0.96])
    fig.text(0.02, 0.5, "Density (normalized)", va='center', rotation='vertical')

    plt.savefig(f"figures/{dataset_name}_split_distributions_scaled.png")


# --- 7. Combined Token Length Distribution ---
plt.figure(figsize=(10, 6))

wt2_cutoff = np.percentile(results_wt2["train"]["token_lengths"], 99)
wt103_cutoff = np.percentile(results_wt103["train"]["token_lengths"], 99)
x_limit = max(wt2_cutoff, wt103_cutoff)

sns.histplot(results_wt2["train"]["token_lengths"], bins=100, label="WikiText-2", kde=True, stat="density")
sns.histplot(results_wt103["train"]["token_lengths"], bins=100, label="WikiText-103", kde=True, stat="density")

plt.xlim(0, x_limit)
plt.ylim(bottom=0)
# plt.title("Token Length Distribution (Train Splits…)", fontsize=16)   # COMMENTED OUT
plt.xlabel(f"Number of Tokens per Text Sample (≤ {int(x_limit)} tokens)")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/train_token_length_comparison_scaled.png")
plt.close()

# --- 8. Vocabulary Growth Curve ---
def vocab_growth(tokens):
    seen = set()
    growth = []
    for i, t in enumerate(tokens):
        seen.add(t)
        if (i + 1) % 1000 == 0:
            growth.append(len(seen))
    return growth

growth_wt2 = vocab_growth(results_wt2["train"]["all_tokens"])
growth_wt103 = vocab_growth(results_wt103["train"]["all_tokens"])

plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, len(growth_wt2)) * 1000, growth_wt2, label="WikiText-2")
plt.plot(np.arange(0, len(growth_wt103)) * 1000, growth_wt103, label="WikiText-103")
# plt.title("Vocabulary Growth Curve (Train)")   # COMMENTED OUT
plt.xlabel("Number of Tokens (log scale)")
plt.ylabel("Unique Tokens")
plt.xscale("log")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/vocab_growth_curve_scaled.png")
plt.close()

# --- 9. Word Frequency Distribution ---
def top_k_freq(tokens, k=20):
    counter = Counter(tokens)
    return counter.most_common(k)

top20_wt2 = top_k_freq(results_wt2["train"]["all_tokens"])
top20_wt103 = top_k_freq(results_wt103["train"]["all_tokens"])

df_top20_wt2 = pd.DataFrame(top20_wt2, columns=["Token", "Frequency"])
df_top20_wt103 = pd.DataFrame(top20_wt103, columns=["Token", "Frequency"])

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(y="Token", x="Frequency", data=df_top20_wt2, ax=axs[0])
# axs[0].set_title("Top 20 Tokens - WikiText-2")   # COMMENTED OUT
axs[0].set_xscale("log")
axs[0].set_xlabel("Frequency (log scale)")
axs[0].set_ylabel("Token")
axs[0].grid(True, alpha=0.3)

sns.barplot(y="Token", x="Frequency", data=df_top20_wt103, ax=axs[1])
# axs[1].set_title("Top 20 Tokens - WikiText-103") # COMMENTED OUT
axs[1].set_xscale("log")
axs[1].set_xlabel("Frequency (log scale)")
axs[1].set_ylabel("")
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/top20_token_freq_comparison_scaled.png")
plt.close()

plot_split_distributions(results_wt2, "WikiText-2")
plot_split_distributions(results_wt103, "WikiText-103")

print("\nScaled analysis complete! Improved plots saved in 'figures/' folder.")

'''




# ============================================
# WIKITEXT DATASET ANALYSIS FOR THESIS
# ============================================

from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import scipy.stats as stats
import os

# ============================================
# GLOBAL FONT SETTINGS (Times New Roman)
# ============================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 22
plt.rcParams["axes.labelsize"] = 22
plt.rcParams["xtick.labelsize"] = 22
plt.rcParams["ytick.labelsize"] = 22
plt.rcParams["legend.fontsize"] = 22

# ============================================
# PPT-THEME COLORS
# ============================================
COLOR_GOLD = "#F7E6C7"   # Gold, Accent 4, Lighter 80%
COLOR_GREEN = "#C5E0B3"  # Green, Accent 6, Lighter 60%
COLOR_BLUE = "#A9D0F5"   # Blue, Accent 5, Lighter 40%
COLOR_ORANGE = "#F8CBAD"  # Orange, Accent 2, Lighter 60% 


# ============================================
# LOAD DATASETS
# ============================================
wikitext2 = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
wikitext103 = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

# ============================================
# TOKENIZER
# ============================================
tokenizer = AutoTokenizer.from_pretrained("models/Llama-3.2-1B")

# ============================================
# CATEGORY FUNCTIONS FOR PIE CHART
# ============================================
def categorize_token(tok):
    if tok.isalpha():
        return "Alphabet"
    if tok.isdigit():
        return "Number"
    if tok in {".", ",", "!", "?", ";", ":"}:
        return "Punctuation"
    return "Special"

# ============================================
# SPLIT ANALYSIS
# ============================================
def analyze_split(dataset, split_name="train"):

    texts = dataset[split_name]["text"]

    token_lengths = []
    all_tokens = []
    category_counter = Counter()

    for text in tqdm(texts, desc=f"Tokenizing {split_name}"):
        tokens = tokenizer.tokenize(text)
        token_lengths.append(len(tokens))
        all_tokens.extend(tokens)

        for t in tokens:
            category_counter[categorize_token(t)] += 1

    lengths = np.array(token_lengths)

    # descriptive stats
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    std_len = np.std(lengths)
    min_len = np.min(lengths)
    max_len = np.max(lengths)
    p95 = np.percentile(lengths, 95)
    p99 = np.percentile(lengths, 99)
    skew = stats.skew(lengths)
    kurtosis = stats.kurtosis(lengths)

    # Logging
    print(f"\n===== SPLIT: {split_name.upper()} =====")
    print(f"Total Tokens: {len(all_tokens):,}")
    print(f"Vocab Size: {len(set(all_tokens)):,}")
    print(f"Avg Seq Len: {mean_len:.2f}")
    print(f"Median Seq Len: {median_len:.2f}")
    print(f"Std Dev Seq Len: {std_len:.2f}")
    print(f"Min / Max: {min_len} / {max_len}")
    print(f"95th / 99th Percentile: {p95:.2f} / {p99:.2f}")
    print(f"Skewness: {skew:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
    print("Token Category Distribution:", dict(category_counter))

    return {
        "split": split_name,
        "total_tokens": len(all_tokens),
        "vocab_size": len(set(all_tokens)),
        "avg_seq_len": mean_len,
        "median_seq_len": median_len,
        "std_seq_len": std_len,
        "min_seq_len": min_len,
        "max_seq_len": max_len,
        "p95_seq_len": p95,
        "p99_seq_len": p99,
        "skewness": skew,
        "kurtosis": kurtosis,
        "token_lengths": token_lengths,
        "all_tokens": all_tokens,
        "token_categories": category_counter,
    }

# ============================================
# DATASET ANALYSIS
# ============================================
def analyze_dataset(dataset, name):
    results = {}
    print(f"\n========== DATASET: {name.upper()} ==========\n")
    for split in dataset.keys():
        results[split] = analyze_split(dataset, split)
    return results

results_wt2 = analyze_dataset(wikitext2, "WikiText-2-v1")
results_wt103 = analyze_dataset(wikitext103, "WikiText-103-v1")

os.makedirs("figures", exist_ok=True)

# ==========================================================
# FIGURE 1 — TOKEN LENGTH HISTOGRAM (WT2 & WT103, TRAIN ONLY)
# ==========================================================
plt.figure(figsize=(10, 4))

wt2_lengths = np.array(results_wt2["train"]["token_lengths"])
wt103_lengths = np.array(results_wt103["train"]["token_lengths"])

max_x = max(
    np.percentile(wt2_lengths, 99),
    np.percentile(wt103_lengths, 99)
)

sns.histplot(wt2_lengths, bins=120, stat="density", label="WikiText-2-v1",
             kde=True, color=COLOR_GOLD)
sns.histplot(wt103_lengths, bins=120, stat="density", label="WikiText-103-v1",
             kde=True, color=COLOR_BLUE)

plt.xlim(0, max_x)
plt.xlabel("Token Length")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("figures/hist_token_length_comparison.png", dpi=300)
plt.close()

# ==========================================================
# FIGURE 2 — VOCABULARY DIVERSITY BAR CHART (TRAIN/VAL/TEST)
# ==========================================================
splits = ["train", "validation", "test"]
datasets = ["WikiText-2-V1", "WikiText-103-V1"]

vocab_values = [
    [results_wt2[s]["vocab_size"] for s in splits],
    [results_wt103[s]["vocab_size"] for s in splits],
]

df_vocab = pd.DataFrame(vocab_values, index=datasets, columns=splits).T

plt.figure(figsize=(10, 4))
df_vocab.plot(
    kind="bar",
    figsize=(10, 4),
    color=[COLOR_GOLD, COLOR_GREEN, COLOR_BLUE]
)

plt.ylabel("Vocabulary Size")
plt.legend()
plt.tight_layout()
plt.savefig("figures/vocab_diversity_bar.png", dpi=300)
plt.close()

# ==========================================================
# FIGURE 3 — TOKEN CATEGORY PIE CHART (TRAIN WT103)
# ==========================================================
cat_counts = results_wt103["train"]["token_categories"]
labels = list(cat_counts.keys())
sizes = [cat_counts[k] for k in labels]

# Improved percentage formatter for tiny slices
def autopct_force_small(pct):
    if pct < 1:
        return f"{pct:.2f}%"
    return f"{pct:.1f}%"

explode = [0, 0, 0.25, 0]

plt.figure(figsize=(7, 7))
plt.pie(
    sizes,
    labels=labels,
    autopct=autopct_force_small,
    explode=explode,
    pctdistance=0.75,
    labeldistance=1.12,       # push labels slightly outward
    startangle=140,
    colors=[COLOR_GOLD, COLOR_GREEN, COLOR_BLUE, COLOR_ORANGE]  # ALL FOUR COLORS
)

plt.tight_layout()
plt.savefig("figures/token_category_pie.png", dpi=300)
plt.close()



print("\nAll figures generated in 'figures/' and full statistics logged.")


'''
#------------------------------------------------------------------------------------------------
#------------------------------------- MIXED DATASET CREATION -----------------------------------
#------------------------------------------------------------------------------------------------

#------------------------- Import required Packages ---------------------------------------------
import logging
# import os
# import random
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
# from transformers import AutoTokenizer
# import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#------------------------- Config and Logger Setup ---------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("")

#-------------------------- Config Class -----------------------------------------------------
class Config:
    SEED = 42
    DATASET_CONFIGS = {
        "wikitext": {"path":"wikitext", "config": "wikitext-103-v1", "train_split": "train", "val_split": "validation", "test_split": "test"},
        "gsm8k": {"path":"gsm8k", "config": "main", "train_split": "train", "val_split": "test",},
        "alpaca": {"path":"yahma/alpaca-cleaned", "config": None, "train_split": "train", "val_split": None,},
    }

    BLOCK_SIZE = 512

    DATA_PROPORTIONS = {
        "wikitext": 0.6,
        "gsm8k": 0.25,
        "alpaca": 0.15,
    }

    TOTAL_TRAIN_SAMPLES = 100000
    TOTAL_VAL_SAMPLES = 10000
    TOTAL_TEST_SAMPLES = 10000

#----------------------- Functions to format datasets -----------------------------------
def format_gsm8k_example(example, tokenizer):
    """Format GSM8K examples with chain-of-thought"""
    question = example["question"]
    answer = example["answer"]
    
    # Extract the final answer if it's in the reasoning
    if "####" in answer:
        reasoning, final_answer = answer.split("####", 1)
        final_answer = final_answer.strip()
    else:
        reasoning = answer
        final_answer = ""
    
    text = f"Question: {question}\nReasoning: Let's think step by step. {reasoning.strip()}"
    if final_answer:
        text += f"\nFinal Answer: {final_answer}"
    
    return text + tokenizer.eos_token

def format_alpaca_example(example, tokenizer):
    """Format Alpaca instruction examples"""
    if example.get("input"):
        text = f"Instruction: {example['instruction']}\nInput: {example['input']}\nResponse: {example['output']}"
    else:
        text = f"Instruction: {example['instruction']}\nResponse: {example['output']}"
    
    return text + tokenizer.eos_token

def format_wikitext_example(example, tokenizer):
    """Format WikiText examples"""
    text = example["text"].strip()
    if text and not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    return text

#----------------------- Function to load datasets --------------------------
def load_datasets():
    datasets = {}

    #load each dataset
    for name, config in Config.DATASET_CONFIGS.items():
        logger.info(f"Loading {name} dataset...")
        if config["config"]:
            dataset = load_dataset(config["path"], config["config"])
        else:
            dataset = load_dataset(config["path"])
        datasets[name] = dataset
        logger.info(f"{name} dataset loaded with splits: {list(dataset.keys())}")

    return datasets

# ------------------------------- Load alphaca dataset ----------------------------------

def load_and_split_alpaca_dataset():
    logger.info("Loading Alpaca dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned")
    logger.info(f"Alpaca dataset loaded with splits: {list(dataset.keys())}")

    # Split the dataset into train, val, test (80%, 10%, 10%)
    logger.info("Splitting Alpaca dataset into 80/10/10 train/validation/test...")
    # First split: 90% (train+val) and 10% test
    splits = dataset['train'].train_test_split(test_size=0.1, seed=42)
    # Second split: 88.89% of 90% (train, 80% of total) and 11.11% of 90% (val, 10% of total)
    train_val = splits['train'].train_test_split(test_size=0.1111, seed=42)
    
    # Create DatasetDict with final splits
    alpaca_splits = DatasetDict({
        'train': train_val['train'],      # 80% of total
        'validation': train_val['test'],  # 10% of total
        'test': splits['test']            # 10% of total
    })
    
    logger.info(f"Final splits: train={len(alpaca_splits['train'])}, "
                f"validation={len(alpaca_splits['validation'])}, "
                f"test={len(alpaca_splits['test'])}")
    return alpaca_splits

# Analyzing alpaca dataset
plt.style.use('ggplot')

def analyze_alpaca_stats(dataset: DatasetDict, output_dir: str = "alpaca_stats") -> dict:
    """
    Analyze the Alpaca dataset to compute statistics and generate histograms for each split.
    Computes example counts and character length distributions for instruction, input, output,
    and combined fields. Applies filtering to remove outliers and saves histograms.

    Args:
        dataset (DatasetDict): Alpaca dataset with train, validation, and test splits.
        output_dir (str): Directory to save histogram plots.

    Returns:
        dict: Statistics for each split (example counts, length stats before/after filtering).
    """
    logger.info("Starting analysis of Alpaca dataset...")
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {}
    for split in dataset.keys():
        logger.info(f"Analyzing {split} split...")
        data = pd.DataFrame(dataset[split])
        
        # Compute character lengths
        data['instruction_length'] = data['instruction'].apply(len)
        data['input_length'] = data['input'].apply(len)
        data['output_length'] = data['output'].apply(len)
        data['combined_length'] = data['instruction_length'] + data['input_length'] + data['output_length']
        
        # Descriptive statistics before filtering
        desc_stats = {
            'num_examples': len(data),
            'instruction': data['instruction_length'].describe().to_dict(),
            'input': data['input_length'].describe().to_dict(),
            'output': data['output_length'].describe().to_dict(),
            'combined': data['combined_length'].describe().to_dict()
        }
        stats[split] = desc_stats
        
        logger.info(f"{split} split stats (before filtering):")
        logger.info(f"  Examples: {desc_stats['num_examples']}")
        for field in ['instruction', 'input', 'output', 'combined']:
            logger.info(f"  {field.capitalize()} length: "
                        f"mean={desc_stats[field]['mean']:.2f}, "
                        f"std={desc_stats[field]['std']:.2f}, "
                        f"min={desc_stats[field]['min']:.0f}, "
                        f"max={desc_stats[field]['max']:.0f}")
        
        # Plot histograms before filtering
        for field, color in [('instruction_length', 'blue'), ('input_length', 'green'), 
                            ('output_length', 'teal'), ('combined_length', 'purple')]:
            plt.figure(figsize=(10, 5))
            sns.histplot(data[field], bins=50, kde=True, color=color)
            plt.title(f'{field.replace("_", " ").title()} Distribution in {split} Split')
            plt.xlabel('Length (characters)')
            plt.ylabel('Frequency')
            plot_path = os.path.join(output_dir, f'{split}_{field}_before.png')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved histogram: {plot_path}")
        
        # Filter outliers (based on blog: instruction/input <=1500, output <=4000, combined <=6000)
        filtered_data = data[
            (data['instruction_length'] <= 1500) &
            (data['input_length'] <= 1500) &
            (data['output_length'] <= 4000) &
            (data['combined_length'] <= 6000)
        ]
        
        logger.info(f"Filtered {split}: original={len(data)}, filtered={len(filtered_data)}")
        
        # Descriptive statistics after filtering
        filtered_desc_stats = {
            'num_examples': len(filtered_data),
            'instruction': filtered_data['instruction_length'].describe().to_dict(),
            'input': filtered_data['input_length'].describe().to_dict(),
            'output': filtered_data['output_length'].describe().to_dict(),
            'combined': filtered_data['combined_length'].describe().to_dict()
        }
        stats[f'{split}_filtered'] = filtered_desc_stats
        
        logger.info(f"{split} split stats (after filtering):")
        logger.info(f"  Examples: {filtered_desc_stats['num_examples']}")
        for field in ['instruction', 'input', 'output', 'combined']:
            logger.info(f"  {field.capitalize()} length: "
                        f"mean={filtered_desc_stats[field]['mean']:.2f}, "
                        f"std={filtered_desc_stats[field]['std']:.2f}, "
                        f"min={filtered_desc_stats[field]['min']:.0f}, "
                        f"max={filtered_desc_stats[field]['max']:.0f}")
        
        # Plot histograms after filtering
        for field, color in [('instruction_length', 'blue'), ('input_length', 'green'), 
                            ('output_length', 'teal'), ('combined_length', 'purple')]:
            plt.figure(figsize=(10, 5))
            sns.histplot(filtered_data[field], bins=50, kde=True, color=color)
            plt.title(f'{field.replace("_", " ").title()} Distribution in {split} Split (Filtered)')
            plt.xlabel('Length (characters)')
            plt.ylabel('Frequency')
            plot_path = os.path.join(output_dir, f'{split}_{field}_after.png')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved filtered histogram: {plot_path}")
    
    logger.info("Alpaca dataset analysis complete.")
    return stats

#------------------------ Main Function -----------------------------------------------------
def main():
    # Load datasets
    # logger.info("Loading datasets...")
    # datasets = load_datasets()
    # logger.info("Datasets loaded.")

    dataset = load_and_split_alpaca_dataset()
    logger.info(f"Dataset details: {dataset}")

    # Analyze statistics
    # stats = analyze_alpaca_stats(dataset, output_dir="output/alpaca_stats")
    logger.info("Statistics computed and plots saved.")

    # logger.info(f"Dataset details: { {name: {split: len(ds)} for name, ds in datasets.items() for split in ds} }")

    # logging 10 examples from each dataset
    # for name, dataset in datasets.items():
    #     for split in dataset.keys():
    #         logger.info(f"First 10 examples from {name} - {split} split:")
    #         for i in range(min(10, len(dataset[split]))):
    #             logger.info(f"Example {i+1}: {dataset[split][i]}")

    # Create mixed datasets
    # logger.info("Creating mixed datasets...")

    # Save mixed datasets
    # logger.info("Saving mixed datasets...")

if  __name__ == "__main__":
    main()

'''











#------------------------------------------------------------------------------------------------
#------------------------------------- WIKITEXT DATASET EXPLORATION -------------------------------
#------------------------------------------------------------------------------------------------

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

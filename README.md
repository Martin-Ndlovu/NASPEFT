# NASPEFT: Neural Architecture Search for Parameter-Efficient Fine-Tuning of Pre-trained Large Language Models

**Paper**: [NASPEFT: Neural Architecture Search for Parameter-Efficient Fine-Tuning of Pre-trained Large Language Models](https://openreview.net/forum?id=FVGiqAoma7) — NLPCC 2026

---

## Overview

This repository builds on [AutoPEFT](https://github.com/cambridgeltl/autopeft) (Zhou et al., TACL 2024), adapting its codebase to support seq2seq backbones (T5), an exact parameter oracle, a Random Forest surrogate, and a union adapter search space combining LoRA, Parallel Bottleneck adapters, and Prefix Tuning.

---

## Repository Structure

```
NASPEFT/
├── adapterhub/
│   ├── config/                         # Adapter configuration files
│   ├── smbo/
│   │   ├── acquisition.py              # Acquisition function (LCB, eq. 7)
│   │   ├── base_function.py            # Problem class; config evaluation via subprocess
│   │   ├── nas_search.py               # BERT/RoBERTa search loop
│   │   ├── param_oracle.py             # Exact parameter percentage oracle per backbone
│   │   ├── search_space.py             # Config search space and encoding
│   │   └── utils.py                    # Pareto front utilities
│   ├── surrogate/
│   │   ├── features.py                 # Feature construction for surrogate input
│   │   └── random_forest_surrogate.py  # Random Forest surrogate model
│   ├── eval_model.py                   # Standalone evaluation of a trained adapter
│   ├── layer_selection.py              # Layer-sensitivity probe (§3.3)
│   ├── nas_search_T5.py                # Training script for seq2seq / encoder backbones
│   ├── nas_search_plus.py              # Extended search loop with acquisition
│   └── run_wikitext.py                 # WikiText causal LM training script
├── models/                             # Downloaded backbone models (not tracked by git)
├── output/                             # Search outputs (not tracked by git)
├── tools/                              # Utility scripts
├── definition.py                       # Sets ROOT_DIR to the repo root
├── full_finetuning.py                  # Full fine-tuning baseline (LLaMA / WikiText)
├── glue_metrics.py                     # GLUE metric computation helpers
├── logger.py                           # Logging setup
├── plot_pareto.py                      # Pareto front plotting utility
├── run_naspeft.py                      # Main search entry point
├── settings.py                         # Per-task training hyperparameters
└── requirements.txt
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Martin-Ndlovu/NASPEFT.git
cd NASPEFT
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Download and save the GLUE tasks locally. Run this once from the repo root:

```python
import datasets

tasks = ['cola', 'mrpc', 'sst2', 'qnli', 'qqp', 'rte', 'stsb', 'mnli']
for task in tasks:
    dataset = datasets.load_dataset('glue', task)
    dataset.save_to_disk(f'./datasets/glue/{task}')
```

This creates `datasets/glue/<task>/` under the repo root. Pass the **absolute path** to the `glue/` directory via `--data_path` — the task name is appended automatically by the code.

---

## Model Preparation

Download the backbone locally into the `models/` directory:

```python
from transformers import AutoTokenizer, AutoModel

name     = "bert-base-uncased"
save_dir = f"./models/{name}"
AutoTokenizer.from_pretrained(name).save_pretrained(save_dir)
AutoModel.from_pretrained(name).save_pretrained(save_dir)
```

For T5 models, use `AutoModelForSeq2SeqLM`. For LLaMA, use `AutoModelForCausalLM`.

---

## Running NASPEFT

All commands are run from the repo root. Pass the **absolute path** to `--data_path`.

### Basic search (BERT-base, MRPC)

```bash
python run_naspeft.py \
    --model_path ./models/bert-base-uncased \
    --data_path  /absolute/path/to/NASPEFT/datasets/glue/ \
    --task       mrpc \
    --objectives param acc \
    --seed       42 \
    --max_iter   200 \
    --batch_size 4 \
    --n_init     20 \
    --kappa      0.1 \
    --adapter_name naspeft \
    --overwrite
```

### With a parameter budget

Only configurations with `param% < 1.5` are considered:

```bash
python run_naspeft.py \
    --model_path   ./models/bert-base-uncased \
    --data_path    /absolute/path/to/NASPEFT/datasets/glue/ \
    --task         mrpc \
    --objectives   param acc \
    --param_budget 1.5 \
    --seed         42 \
    --max_iter     200 \
    --batch_size   4 \
    --n_init       20 \
    --overwrite
```

### With a held-out test split

Creates a reproducible train / validation / test split from the GLUE training data:

```bash
python run_naspeft.py \
    --model_path   ./models/bert-base-uncased \
    --data_path    /absolute/path/to/NASPEFT/datasets/glue/ \
    --task         mrpc \
    --objectives   param acc \
    --custom_split true \
    --seed         42 \
    --max_iter     200 \
    --batch_size   4 \
    --n_init       20 \
    --overwrite
```

### Skip the layer-selection probe

```bash
python run_naspeft.py \
    --model_path           ./models/bert-base-uncased \
    --data_path            /absolute/path/to/NASPEFT/datasets/glue/ \
    --task                 mrpc \
    --objectives           param acc \
    --skip_layer_selection \
    --seed                 42 \
    --max_iter             200 \
    --batch_size           4 \
    --n_init               20 \
    --overwrite
```

### Resume an interrupted run

```bash
python run_naspeft.py \
    --model_path ./models/bert-base-uncased \
    --data_path  /absolute/path/to/NASPEFT/datasets/glue/ \
    --task       mrpc \
    --objectives param acc \
    --seed       42 \
    --max_iter   200 \
    --batch_size 4 \
    --n_init     20 \
    --resume
```

### WikiText / LLaMA (causal LM)

```bash
python run_naspeft.py \
    --model_path ./models/Llama-3.2-1B \
    --data_path  /absolute/path/to/NASPEFT/datasets/ \
    --task       wikitext \
    --objectives param perplexity \
    --seed       42 \
    --max_iter   200 \
    --batch_size 2 \
    --n_init     10 \
    --overwrite
```

---

## Key Arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--model_path` | `-mp` | — | Path to local model directory under `models/` |
| `--data_path` | `-dp` | `{ROOT_DIR}/datasets/glue/` | Absolute path to the `glue/` root (task appended automatically) |
| `--task` | `-t` | `mrpc` | GLUE task: `cola`, `mrpc`, `sst2`, `rte`, `stsb`, `qnli`, `qqp`, `mnli`; or `wikitext` for causal LM |
| `--objectives` | `-o` | `param acc` | `param acc` for GLUE; `param perplexity` for WikiText/LLaMA |
| `--max_iter` | `-mi` | `100` | Total configurations to evaluate (including init) |
| `--n_init` | `-ni` | `20` | Random warm-start configurations |
| `--batch_size` | `-bs` | `4` | Configurations evaluated per search iteration |
| `--kappa` | `-k` | `0.1` | Exploration constant in the acquisition function |
| `--param_budget` | `-pb` | `None` | If set, only configs with `param% < param_budget` are considered |
| `--seed` | `-s` | `42` | Random seed |
| `--adapter_name` | `-an` | `naspeft` | Adapter name; use `naspeft` for the union adapter |
| `--save_path` | `-sp` | `{ROOT_DIR}/output/` | Root output directory |
| `--overwrite` | — | `False` | Overwrite existing output directory |
| `--resume` | — | `False` | Resume from a previous run |
| `--custom_split` | — | `False` | Held-out test split carved from training data |
| `--resplit_dataset` | `-rd` | `False` | 90/10 train split; GLUE validation used as test |
| `--skip_layer_selection` | — | `False` | Search all layers without running the sensitivity probe |
| `--force_layer_selection` | — | `False` | Re-run the probe even if cached results exist |
| `--layer_selection_summary` | — | `None` | Path to an existing `layer_selection_summary.json` to reuse |
| `--ls_epochs` | — | `5` | Epochs per layer during the sensitivity probe |
| `--mock_run` | — | `False` | Dry run — no training is performed |

---

## Output Structure

Results are written to `output/{model}_{task}_random_seed_{seed}_bs_{batch_size}/`:

```
output/
└── bert-base-uncased_mrpc_random_seed_42_bs_4/
    ├── train_logs.log
    ├── result_stats.pt
    ├── layer_selection/
    │   └── bert-base-uncased_mrpc/
    │       └── layer_selection_summary.json
    ├── <config_id>/
    │   ├── config.json
    │   ├── model_param_dict.json
    │   ├── trainer_state.json
    │   ├── train_results.json
    │   ├── eval_results.json
    │   └── test_results.json
    └── test/
        └── seed_{40..44}/
```

---

## Citation

```bibtex
@inproceedings{anonymous2026naspeft,
  title     = {{NASPEFT}: Neural Architecture Search for Parameter-Efficient Fine-Tuning of Pre-trained Large Language Models},
  author    = {Anonymous},
  booktitle = {The 15th CCF International Conference on Natural Language Processing and Chinese Computing},
  year      = {2026},
  url       = {https://openreview.net/forum?id=FVGiqAoma7}
}

@article{zhou2024autopeft,
  title   = {{AutoPEFT}: Automatic Configuration Search for Parameter-Efficient Fine-Tuning},
  author  = {Zhou, Han and Wan, Xingchen and Vuli{\'c}, Ivan and Korhonen, Anna},
  journal = {Transactions of the Association for Computational Linguistics},
  year    = {2024}
}
```
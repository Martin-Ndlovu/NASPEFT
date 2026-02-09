import os
import json
import subprocess
import torch
from datetime import datetime

PROJECT_NAME = "NAS-PEFT Framework"
OUTPUT_FILE = "README.md"
LAYER_JSON = "output/naspeft_layerwise/layer_sensitivity_results.json"

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return "N/A"

def gpu_info():
    if not torch.cuda.is_available():
        return "No GPU detected"
    gpus = []
    for i in range(torch.cuda.device_count()):
        gpus.append(torch.cuda.get_device_name(i))
    return ", ".join(gpus)

def layer_info():
    if not os.path.exists(LAYER_JSON):
        return "Layer sensitivity file not found."
    with open(LAYER_JSON) as f:
        data = json.load(f)
    return f"""
- Median Perplexity τₚ: {data['median_perplexity']:.4f}
- G1 Layers (searched): {len(data['G1_layers'])} → {data['G1_layers']}
- G2 Layers (pruned): {len(data['G2_layers'])}
"""

readme = f"""
# {PROJECT_NAME}

> Automatically generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Overview
This repository implements a **NAS-PEFT framework** for LLaMA-3.2-1B that combines
layer sensitivity analysis with parameter-efficient fine-tuning (LoRA, Prefix, Parallel Adapters).

## System Information
- OS: {run("uname -a")}
- Python: {run("python --version")}
- CUDA: {run("nvcc --version | tail -n 1")}
- GPUs: {gpu_info()}

## Layer Sensitivity Analysis
{layer_info()}

## Main Components
- `layerwise_train.py` – single-layer fine-tuning & perplexity profiling
- `search_space.py` – ConfigSpace-based NAS search space
- `nas_runner.py` – random search over PEFT configurations

## Reproducibility
- Seed: 42
- Model: LLaMA-3.2-1B
- Dataset: WikiText-2

## Notes
This README was generated automatically. Do not edit manually.
"""

with open(OUTPUT_FILE, "w") as f:
    f.write(readme)

print("README.md generated successfully.")

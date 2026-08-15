#!/usr/bin/env python3
"""Pareto-front / search-process figure for the NASPEFT ablation.

Reads result_stats.pt directly from each run (NASPEFT, random search, random
layer selection) and plots trainable-parameter %% vs. accuracy for every
sampled architecture, coloured by method, opacity by search iteration, with
each method's Pareto front drawn as a line.

Storage convention (adapterhub/smbo/base_function.py):
    GLUE  : Y = (-param%, +metric)  -> param% = -Y[:,0], acc = Y[:,1]
    LLaMA : Y = (+param%, +ppl)     -> param% =  Y[:,0], acc = Y[:,1]

Usage:
    python plot_pareto.py \
        --naspeft       output/bert-base-uncased_mrpc_random_seed_42_bs_4/result_stats.pt \
        --random_search output/random_search/bert-base-uncased_mrpc_random_seed_42_bs_4/result_stats.pt \
        --random_layers output/random_layer_selection/bert-base-uncased_mrpc_random_seed_42_bs_4/result_stats.pt \
        --n_init 20 --batch 4 --xmax 7 --metric "MRPC accuracy" --out pareto_search
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt


def load_run(path, n_init, batch, is_llama=False):
    d = torch.load(path, map_location="cpu")
    Y = d["Y"].numpy().astype(float)
    param = Y[:, 0] if is_llama else -Y[:, 0]
    acc = Y[:, 1]
    idx = np.arange(len(param))
    it = np.where(idx < n_init, 0, 1 + (idx - n_init) // batch)
    return param, acc, it


def pareto_mask(param, acc, higher_better=True):
    order = np.argsort(param, kind="mergesort")
    mask = np.zeros(len(param), dtype=bool)
    best = -np.inf
    for i in order:
        v = acc[i] if higher_better else -acc[i]
        if v > best:
            mask[i] = True
            best = v
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--naspeft", required=True)
    ap.add_argument("--random_search", required=True)
    ap.add_argument("--random_layers", required=True)
    ap.add_argument("--n_init", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--llama", action="store_true",
                    help="Perplexity run: flips axis sign + Pareto direction.")
    ap.add_argument("--metric", default="MRPC accuracy")
    ap.add_argument("--xmax", type=float, default=None,
                    help="Clip x-axis (param%%) for readability. Points beyond "
                         "are still drawn faintly if within limits.")
    ap.add_argument("--out", default="pareto_search",
                    help="Output basename; writes <out>.png and <out>.pdf")
    args = ap.parse_args()

    higher_better = not args.llama
    runs = {
        "NASPEFT":                (args.naspeft,       "#1f77b4", "o"),
        "Random search":          (args.random_search, "#d62728", "s"),
        "Random layer selection": (args.random_layers, "#2ca02c", "^"),
    }

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for label, (path, color, marker) in runs.items():
        if not os.path.exists(path):
            print(f"[skip] {label}: {path} not found")
            continue
        param, acc, it = load_run(path, args.n_init, args.batch, args.llama)
        alphas = 0.18 + 0.62 * (it / max(it.max(), 1))
        ax.scatter(param, acc, s=24, c=color, marker=marker, alpha=alphas,
                   edgecolors="none", zorder=2)
        m = pareto_mask(param, acc, higher_better)
        o = np.argsort(param[m])
        ax.plot(param[m][o], acc[m][o], color=color, lw=1.8, marker=marker,
                ms=5, label=label, zorder=3)
        print(f"{label}: {len(param)} points, {m.sum()} on Pareto front")

    ax.set_xlabel("Trainable parameters (%)")
    ax.set_ylabel(args.metric + (" (perplexity)" if args.llama else ""))
    if args.xmax is not None:
        ax.set_xlim(0, args.xmax)
    ax.legend(frameon=False, fontsize=9,
              loc="upper right" if args.llama else "lower right")
    ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=200)
    fig.savefig(args.out + ".pdf")
    print(f"wrote {args.out}.png and {args.out}.pdf")


if __name__ == "__main__":
    main()
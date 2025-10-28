import json
import matplotlib.pyplot as plt
import numpy as np

# Load results from JSON
with open("layer_selection_3epoch_eval_results.json", "r") as f:
    results = json.load(f)

# Extract layers and perplexities
layers = [int(r["layer"].split("_")[1]) for r in results]
perplexities = [r["perplexity"] for r in results]

# Sort by layer index so bars are in order
layers, perplexities = zip(*sorted(zip(layers, perplexities)))

# Normalize colors so lower perplexity = darker
norm = plt.Normalize(min(perplexities), max(perplexities))
cmap = plt.cm.viridis_r
colors = cmap(norm(perplexities))

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 2.5))

# Plot heatmap-like bar chart
bars = ax.bar(layers, perplexities, color=colors)

# Add perplexity values above bars
for bar, val in zip(bars, perplexities):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=8)

# Labels and formatting
ax.set_xlabel("Layer Index", fontsize=12)
ax.set_ylabel("Eval Set Perplexity", fontsize=10)
ax.set_title("Layer-wise Perplexity Heatmap", fontsize=14)
ax.set_xticks(np.arange(len(layers)))
ymax = max(perplexities) * 1.1  # 10% higher than tallest bar
ax.set_ylim(0, ymax)


# Add colorbar linked to the colormap
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # required for older matplotlib versions
fig.colorbar(sm, ax=ax, label="Perplexity Scale")

plt.tight_layout()
# plt.show()
plt.savefig("layer_perplexity_eval_heatmap.png", dpi=200)

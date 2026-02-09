import json
import matplotlib.pyplot as plt
import numpy as np

# ----------- Set global font family to Times New Roman -----------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 32

# ---------- Load both JSON files ----------    
with open("Results/layer_selection_3epoch_test_results.json", "r") as f:
    test_results = json.load(f)

with open("Results/layer_selection_3epoch_eval_results.json", "r") as f:
    eval_results = json.load(f)

# ---------- Extract layers & perplexities ----------
test_layers = [int(r["layer"].split("_")[1]) for r in test_results]
test_ppl = [r["perplexity"] for r in test_results]

eval_layers = [int(r["layer"].split("_")[1]) for r in eval_results]
eval_ppl = [r["perplexity"] for r in eval_results]

# ---------- Sort by layer index ----------
test_layers, test_ppl = zip(*sorted(zip(test_layers, test_ppl)))
eval_layers, eval_ppl = zip(*sorted(zip(eval_layers, eval_ppl)))

# ---------- Shared color normalization ----------
all_ppl = list(test_ppl) + list(eval_ppl)
norm = plt.Normalize(min(all_ppl), max(all_ppl))
cmap = plt.cm.viridis_r

# ---------- Figure with 2 subplots ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

# ---------- Plot TEST ----------
colors_test = cmap(norm(test_ppl))
bars1 = ax1.bar(test_layers, test_ppl, color=colors_test, width=0.6)

for bar, val in zip(bars1, test_ppl):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.1,
             f"{val:.2f}",
             ha="center", va="bottom", rotation=90)

ax1.set_ylabel("Test", )
# ax1.set_title("Layer-wise Perplexity", fontsize=24)
ax1.tick_params(axis="both", )
ax1.set_ylim(min(test_ppl)*0.98, max(test_ppl)*1.05)

# ---------- Plot EVAL ----------
colors_eval = cmap(norm(eval_ppl))
bars2 = ax2.bar(eval_layers, eval_ppl, color=colors_eval, width=0.6)

for bar, val in zip(bars2, eval_ppl):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.1,
             f"{val:.2f}",
             ha="center", va="bottom", rotation=90)

ax2.set_ylabel("Validation", )
ax2.set_xlabel("Layer Index", )
ax2.tick_params(axis="both", )
ax2.set_ylim(min(eval_ppl)*0.98, max(eval_ppl)*1.05)

# ---------- Shared colorbar ----------
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax1, ax2], fraction=0.025, pad=0.02, location='right')
cbar.set_label("Perplexity Scale", )
cbar.ax.tick_params()

plt.subplots_adjust(right=0.8, hspace=0.3)
plt.savefig("layer_perplexity_test_eval_combined.png", dpi=200)

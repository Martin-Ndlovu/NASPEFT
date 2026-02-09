# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# # ---------------------------------------
# # Define file paths
# # ---------------------------------------
# image_paths = {
#     "WikiText-2": {
#         "Train": "figures/WikiText-2_train_token_length_dist.png",
#         "Validation": "figures/WikiText-2_validation_token_length_dist.png",
#         "Test": "figures/WikiText-2_test_token_length_dist.png"
#     },
#     "WikiText-103": {
#         "Train": "figures/WikiText-103_train_token_length_dist.png",
#         "Validation": "figures/WikiText-103_validation_token_length_dist.png",
#         "Test": "figures/WikiText-103_test_token_length_dist.png"
#     }
# }

# # ---------------------------------------
# # Create high-quality 3x2 grid
# # ---------------------------------------
# fig, axes = plt.subplots(3, 2, figsize=(18, 12))
# plt.subplots_adjust(wspace=0.02, hspace=0.05)

# splits = ["Train", "Validation", "Test"]
# datasets = ["WikiText-2", "WikiText-103"]

# for row_idx, split in enumerate(splits):
#     for col_idx, dataset in enumerate(datasets):
#         ax = axes[row_idx, col_idx]
#         img = mpimg.imread(image_paths[dataset][split])
#         ax.imshow(img)
#         ax.axis("off")  # remove ticks, frames, and labels

# # ---------------------------------------
# # Save high-resolution figure
# # ---------------------------------------
# plt.tight_layout(pad=0)
# plt.savefig(
#     "figures/combined_token_length_distributions_clean.png",
#     dpi=600,  # increase DPI for sharpness
#     bbox_inches="tight",
#     pad_inches=0
# )
# plt.show()







#============================================================================================================================='''
'''

# Now plotting the figure of the training and validation loss curves and results
import json
import matplotlib.pyplot as plt
import os

# -------------------------------
# Load the trainer_state.json file
# -------------------------------
json_path = "/root/Martin/NasPEFT/naspeft/output/NAS_wikitext_random_seed_42_bs_4/test/seed_42/0_0_0_1_0_0_0_1_1_0_1_1_1_0_1_1_32_8_8_2/trainer_state.json"
save_path = "/root/Martin/NasPEFT/naspeft/Analysis/figures/training_diagnostics/pareto_5/"

with open(json_path, "r") as f:
    data = json.load(f)

log_history = data.get("log_history", [])

# -------------------------------
# Extract metrics safely
# -------------------------------
steps, train_loss, eval_loss, eval_ppl, lr, grad_norm = [], [], [], [], [], []

for entry in log_history:
    if "step" in entry:
        step = entry["step"]
        if "loss" in entry:
            steps.append(step)
            train_loss.append(entry["loss"])
        if "eval_perplexity" in entry:
            eval_ppl.append((step, entry["eval_perplexity"]))
        if "eval_loss" in entry:
            eval_loss.append((step, entry["eval_loss"]))
        if "learning_rate" in entry:
            lr.append((step, entry["learning_rate"]))
        if "grad_norm" in entry:
            grad_norm.append((step, entry["grad_norm"]))

# -------------------------------
# Output directory for figures
# -------------------------------
os.makedirs(save_path, exist_ok=True)

# -------------------------------
# 1️⃣ Training Loss vs Steps
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(steps[:len(train_loss)], train_loss, label="Training Loss", color="royalblue")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss vs Steps")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(save_path+"training_loss_vs_steps.png", dpi=300)
plt.close()

# -------------------------------
# 2️⃣ Evaluation Perplexity vs Steps
# -------------------------------
if eval_ppl:
    eval_steps, eval_perplexity = zip(*eval_ppl)
    plt.figure(figsize=(8, 5))
    plt.plot(eval_steps, eval_perplexity, marker='o', color="darkorange", label="Eval Perplexity")
    plt.xlabel("Step")
    plt.ylabel("Perplexity")
    plt.title("Evaluation Perplexity vs Steps")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path+"eval_perplexity_vs_steps.png", dpi=300)
    plt.close()

# -------------------------------
# 3️⃣ Learning Rate Schedule
# -------------------------------
if lr:
    lr_steps, lr_values = zip(*lr)
    plt.figure(figsize=(8, 5))
    plt.plot(lr_steps, lr_values, color="green", label="Learning Rate")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule during Training")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path+"learning_rate_schedule.png", dpi=300)
    plt.close()

# -------------------------------
# 4️⃣ Gradient Norm vs Steps
# -------------------------------
if grad_norm:
    grad_steps, grad_values = zip(*grad_norm)
    plt.figure(figsize=(8, 5))
    plt.plot(grad_steps, grad_values, color="red", label="Gradient Norm")
    plt.xlabel("Step")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm vs Steps")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path+"gradient_norm_vs_steps.png", dpi=300)
    plt.close()

print("All training diagnostic figures saved in: {}".format(save_path))
'''










#============================================================================================================================='''

import matplotlib.pyplot as plt

#--------------------- Set font global settings ---------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 22
#---------------------------------------------------------------------

# --- Data ---
methods = [
    "LoRA",
    "Prefix-Tuning",
    "Parallel Adapter",
    "NAS-PEFT [64,32,1,2]",
    "NAS-PEFT [16,32,1,4]",
    "NAS-PEFT [8,32,8,2]"
]

perplexity = [16.6, 18.7, 12.94, 4.33, 5.60, 9.35]
trainable_params = [0.15, 1.10, 5.34, 5.47, 6.69, 1.15]
categories = ["Baseline", "Baseline", "Baseline", "NAS", "NAS", "NAS"]

# --- Separate Baseline and NAS points ---
baseline_x = [trainable_params[i] for i in range(len(methods)) if categories[i] == "Baseline"]
baseline_y = [perplexity[i] for i in range(len(methods)) if categories[i] == "Baseline"]
nas_x = [trainable_params[i] for i in range(len(methods)) if categories[i] == "NAS"]
nas_y = [perplexity[i] for i in range(len(methods)) if categories[i] == "NAS"]

# --- Plot ---
plt.figure(figsize=(12, 8))  # ⬅️ MUCH BIGGER, prevents squeezing

plt.scatter(baseline_x, baseline_y, color="#1f77b4", s=200, label="Baseline Methods", marker="o")
plt.scatter(nas_x, nas_y, color="#d62728", s=250, label="NAS-Optimized Models", marker="^")

# Pareto frontier
sorted_nas = sorted(zip(nas_x, nas_y))
front_x, front_y = zip(*sorted_nas)
plt.plot(front_x, front_y, '--', color='gray', linewidth=2, label="Pareto Frontier")

# --- Labels & Formatting ---
plt.xlabel("Trainable Parameters (%)")
plt.ylabel("Perplexity (↓)")
plt.legend()

plt.xlim(0, 8)
plt.ylim(0, 20)

# --- Smarter label placement (no overlap) ---
for i, txt in enumerate(methods):
    plt.annotate(
        txt,
        (trainable_params[i], perplexity[i]),
        xytext=(10, 10),              # offset in points, prevents squeezing
        textcoords="offset points",
        ha="left",
        fontsize=20
    )

# More padding around plot
plt.tight_layout(pad=2.0)

# --- Save BEFORE show ---
save_path = "/root/Martin/NasPEFT/naspeft/Analysis/figures/"
plt.savefig(save_path + "perplexity_vs_trainable_params.png", dpi=300)
plt.show()



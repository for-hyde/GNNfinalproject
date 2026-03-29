import torch
import numpy as np
import os
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from models.vae_atac import InfoVAE_ATAC
from utils.data_loading import SingleDatasetVAE, separate_loader, get_atac_pos_weights, threshold_to_match_sparsity
from utils.device import get_free_gpu
from collections import OrderedDict
from scipy.stats import pearsonr, spearmanr, ks_2samp
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score, jaccard_score
)

plt.rcParams['image.cmap'] = 'PiYG'
CUSTOM_CMAP = plt.get_cmap('PiYG')


####################################################################################################
#                                                                                                  #
# Evaluation of ATAC VAE model                                                                     #
# After every run, adjust the config section accordingly.                                          #
#                                                                                                  #
####################################################################################################


######################################## Config ########################################

DATA_DIR = "/workspace/data/preprocessed_data/integrated_uniform_split"
#MODEL_PATH = "/workspace/runs/atac_vae_training_run_integrated/2026-03-23 20:52:24.405348_vae_model_weights.pth"
MODEL_PATH = "/workspace/runs/2026-03-25 22:34:44.782109_vae_model_weights.pth"
EVAL_OUT_DIR = "/workspace/final_evaluation/atac_vae_uniform_kl"
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

######################################## Load Data and Model ########################################

train_atac, val_atac, test_atac = separate_loader(DATA_DIR, "ATAC")
atac_pos_weights = get_atac_pos_weights(test_atac.X).squeeze(0)

test_dataset = SingleDatasetVAE(test_atac)
test_loader  = DataLoader(test_dataset, batch_size=512, shuffle=False)
input_size = test_atac.shape[1]
device = get_free_gpu()

model = InfoVAE_ATAC(
    input_size=input_size, 
    latent_size=128, 
    lr=0, wd=0, 
    mode="", 
    device=device, 
    pos_weight=atac_pos_weights.to(device)
    )

state_dict = torch.load(MODEL_PATH)
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k.replace("_orig_mod.", "")
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
print("Model weights loaded successfully!")
model.eval()

######################################## Get all reconstructions ########################################

original = []
reconstructed = []
latent_coords = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        _, mu, logvar = model.encode(batch) 
        recon = model.decode(mu)
        recon_probs = torch.sigmoid(recon)
        
        original.append(batch.cpu().numpy())
        reconstructed.append(recon_probs.cpu().numpy())
        latent_coords.append(mu.cpu().numpy())

original = np.concatenate(original)
reconstructed = np.concatenate(reconstructed)
latent_coords = np.concatenate(latent_coords)
cell_types = test_atac.obs["cell_type"].values

######################################## Compute Per-Cell Metrics ########################################

recon_binary, threshold = threshold_to_match_sparsity(reconstructed, original)

per_cell = []
for i in range(len(original)):
    true = original[i].astype(int)
    pred = recon_binary[i].astype(int)
    prob = reconstructed[i]

    if true.sum() == 0:
        continue

    per_cell.append({
        "cell_type": cell_types[i],
        "recall": recall_score(true, pred, zero_division=0),       
        "precision": precision_score(true, pred, zero_division=0),
        "f1": f1_score(true, pred, zero_division=0),
        "jaccard": jaccard_score(true, pred, zero_division=0),      
        "auprc": average_precision_score(true, prob),             
        "auroc": roc_auc_score(true, prob),
    })

df = pd.DataFrame(per_cell)
print(df[["recall", "precision", "f1", "jaccard", "auprc", "auroc"]].describe())
df.to_csv(f"{EVAL_OUT_DIR}/per_cell_recon_metrics.csv", index=False)


######################################## Binary Comparison & KS Depiction ########################################

# 1. Confusion Matrix
cm = confusion_matrix(original.flatten(), recon_binary.flatten(), normalize='true')
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="PiYG", ax=ax, xticklabels=["Closed", "Open"], yticklabels=["Closed", "Open"])
ax.set_title("Normalized Confusion Matrix (Sparsity Matched)")
ax.set_xlabel("Reconstructed")
ax.set_ylabel("Original")
fig.tight_layout()
fig.savefig(f"{EVAL_OUT_DIR}/confusion_matrix.png", dpi=150)
fig.savefig(f"{EVAL_OUT_DIR}/confusion_matrix.svg")

# 2. Probability Separation Plot
flat_orig = original.flatten()
flat_probs = reconstructed.flatten()
indices = np.random.choice(len(flat_orig), size=min(1000000, len(flat_orig)), replace=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(flat_probs[indices][flat_orig[indices] == 0], bins=100, alpha=0.6, label="True Closed (0)", color=CUSTOM_CMAP(0.2), density=True)
ax.hist(flat_probs[indices][flat_orig[indices] == 1], bins=100, alpha=0.6, label="True Open (1)", color=CUSTOM_CMAP(0.8), density=True)
ax.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.3f})')
ax.set_xlabel("Reconstruction Probability (Sigmoid Output)")
ax.set_ylabel("Density")
ax.set_title("Model Discrimination Power")
ax.legend()
fig.tight_layout()
fig.savefig(f"{EVAL_OUT_DIR}/probability_separation.png", dpi=150)
fig.savefig(f"{EVAL_OUT_DIR}/probability_separation.svg")

######################################## Check Mean Accessibility ########################################

orig_mean = original.mean(axis=0)       
recon_mean = reconstructed.mean(axis=0)
r2 = r2_score(orig_mean, recon_mean)
r_peak, _ = pearsonr(orig_mean, recon_mean)

fig, ax = plt.subplots(figsize=(5, 5))
# Use PiYG for density-based scatter or just fixed color from palette
ax.scatter(orig_mean, recon_mean, s=1, alpha=0.3, color=CUSTOM_CMAP(0.1), rasterized=True)
lims = [min(orig_mean.min(), recon_mean.min()), max(orig_mean.max(), recon_mean.max())]
ax.plot(lims, lims, "k--", linewidth=0.8)
ax.set_xlabel("Mean accessibility (original)")
ax.set_ylabel("Mean accessibility (reconstructed)")
ax.set_title(f"Per-peak mean  r={r_peak:.3f}, R²={r2:.3f}")
fig.tight_layout()
fig.savefig(f"{EVAL_OUT_DIR}/peak_mean_scatter.png", dpi=150) # Fixed f-string bug
fig.savefig(f"{EVAL_OUT_DIR}/peak_mean_scatter.svg")
plt.close(fig)


# ######################################## Latent UMAP ########################################

# adata_latent = sc.AnnData(latent_coords, obs=test_atac.obs.copy())

# sc.pp.neighbors(adata_latent)
# sc.tl.umap(adata_latent)
# sc.pl.umap(adata_latent, color=["cell_type"], title="Latent Space UMAP", show=False, palette="PiYG")
# plt.tight_layout()
# plt.savefig(f"{EVAL_OUT_DIR}/latent_umap.png")
# plt.savefig(f"{EVAL_OUT_DIR}/latent_umap.svg")

# recon_binary_fixed = (reconstructed > threshold).astype(np.float32)
# recon_tensor = torch.tensor(recon_binary_fixed).to(device)
# latent_recon = []

# with torch.no_grad():
#     for i in range(0, len(recon_tensor), 512):
#         batch = recon_tensor[i:i+512]
#         _, mu, _ = model.encode(batch)
#         latent_recon.append(mu.cpu().numpy())

# latent_recon = np.concatenate(latent_recon)


# ######################################## Compare Original to Recon ########################################

# adata_compare = sc.AnnData(np.vstack([latent_coords, latent_recon]))
# adata_compare.obs["source"] = ["Original"] * len(latent_coords) + ["Reconstructed"] * len(latent_recon)
# adata_compare.obs["cell_type"] = np.concatenate([test_atac.obs["cell_type"].values, test_atac.obs["cell_type"].values])

# sc.pp.neighbors(adata_compare)
# sc.tl.umap(adata_compare)

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# sc.pl.umap(adata_compare, color="source", ax=axes[0], palette="PiYG", title="Original vs. Reconstructed (latent)", show=False)

# sc.pl.umap(adata_compare, color="cell_type", ax=axes[1], palette="Set2", title="Cell type structure (latent)", show=False)

# fig.tight_layout()
# fig.savefig(f"{EVAL_OUT_DIR}/recon_vs_input_umap.png", dpi=150)
# fig.savefig(f"{EVAL_OUT_DIR}/recon_vs_input_umap.svg")
# plt.close(fig)

# print(f"Evaluation plots saved to {EVAL_OUT_DIR}")

######################################## Latent UMAP ########################################

adata_latent = sc.AnnData(latent_coords, obs=test_atac.obs.copy())
sc.pp.neighbors(adata_latent)
sc.tl.umap(adata_latent)

adata_recon = sc.AnnData(np.vstack([original, reconstructed]))
adata_recon.obs["type"] = ["Original"] * len(original) + ["Reconstructed"] * len(reconstructed)
sc.pp.pca(adata_recon)
sc.pp.neighbors(adata_recon)
sc.tl.umap(adata_recon)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sc.pl.umap(
    adata_latent,
    color=["cell_type"],
    title="Latent Space UMAP",
    show=False,
    palette="PiYG",
    ax=ax1
)

sc.pl.umap(
    adata_recon,
    color="type",
    title="Input vs. Reconstructed",
    show=False,
    palette="Set2",
    ax=ax2
)

plt.tight_layout()
plt.savefig(f"{EVAL_OUT_DIR}/recon_vs_input_umap.png", bbox_inches='tight')
plt.savefig(f"{EVAL_OUT_DIR}/recon_vs_input_umap.svg", bbox_inches='tight')
plt.close()

print(f"Evaluation plots saved to {EVAL_OUT_DIR}")
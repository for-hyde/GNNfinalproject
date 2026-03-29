import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.vae_rna import InfoVAE_RNA
from utils.data_loading import separate_loader, SingleDatasetVAE
from utils.device import get_free_gpu
from collections import OrderedDict
from scipy.stats import pearsonr, spearmanr, ks_2samp
from sklearn.metrics import r2_score
import pandas as pd
import seaborn as sns
import os

# Set Global Plotting Style
plt.rcParams['image.cmap'] = 'PiYG'
CUSTOM_CMAP = plt.get_cmap('PiYG')

######################################## Config ########################################

DATA_DIR = "/workspace/data/preprocessed_data/integrated_celltype_split"
#MODEL_PATH = "/workspace/runs/rna_vae_training_run_integrated/2026-03-23 19:44:16.127080_vae_model_weights.pth"
MODEL_PATH = "/workspace/final_evaluation/final_models/RNA_vae_model_celltype.pth"
EVAL_OUT_DIR = "/workspace/final_evaluation/rna_vae_celltype_kl"
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

######################################## Load Data and Model ########################################

train_rna, val_rna, test_rna = separate_loader(DATA_DIR, "RNA")

test_dataset = SingleDatasetVAE(test_rna)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
input_size = test_rna.shape[1]
device = get_free_gpu()

model = InfoVAE_RNA(input_size=input_size, latent_size=128, lr=0, wd=0, device=device, mode="RNA")

state_dict = torch.load(MODEL_PATH)
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
print("Model weights loaded successfully!")
model.eval()

original, reconstructed, latent_coords = [], [], []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        _, mu, _ = model.encode(batch)
        recon = model.decode(mu)
        original.append(batch.cpu().numpy())
        reconstructed.append(recon.cpu().numpy())
        latent_coords.append(mu.cpu().numpy())

original = np.concatenate(original)
reconstructed = np.concatenate(reconstructed)
latent_coords = np.concatenate(latent_coords)
cell_types = test_rna.obs["cell_type"].values


######################################## Per-cell Metrics ########################################

per_cell = []
for i in range(len(original)):
    r, _ = pearsonr(original[i], reconstructed[i])
    rho, _ = spearmanr(original[i], reconstructed[i])
    mse = np.mean((original[i] - reconstructed[i]) ** 2)
    per_cell.append({"cell_type": cell_types[i], "pearson_r": r, "spearman_r": rho, "mse": mse})

df = pd.DataFrame(per_cell)
df.to_csv(f"{EVAL_OUT_DIR}/per_cell_recon_metrics.csv", index=False)

# Histogram of Pearson Correlations
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df["pearson_r"], bins=80, color=CUSTOM_CMAP(0.2), kde=True, ax=ax)
ax.axvline(df["pearson_r"].median(), color=CUSTOM_CMAP(0.8), linestyle="--", 
            label=f"Median = {df['pearson_r'].median():.3f}")
ax.set_title("Per-cell Reconstruction Correlation (RNA)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{EVAL_OUT_DIR}/per_cell_pearson_dist.png", dpi=150)
fig.savefig(f"{EVAL_OUT_DIR}/per_cell_pearson_dist.svg")


######################################## Per-celltype Boxplot ########################################

fig, ax = plt.subplots(figsize=(10, 5))
order = df.groupby("cell_type")["pearson_r"].median().sort_values(ascending=False).index
# Using PiYG palette for the boxes
sns.boxplot(data=df, x="cell_type", y="pearson_r", order=order, ax=ax, palette="PiYG", flierprops={"markersize": 1})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_title("Reconstruction Quality by Cell Type")
fig.tight_layout()
fig.savefig(f"{EVAL_OUT_DIR}/per_celltype_pearson.png", dpi=150)
fig.savefig(f"{EVAL_OUT_DIR}/per_celltype_pearson.svg")


######################################## IMPROVED Global KS/Distribution ########################################

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 1. Density Plot
sns.kdeplot(original.flatten()[:100000], label="Original", ax=ax1, color=CUSTOM_CMAP(0.1), fill=True)
sns.kdeplot(reconstructed.flatten()[:100000], label="Reconstructed", ax=ax1, color=CUSTOM_CMAP(0.9), fill=True)
ax1.set_title("Global Density Comparison")

# 2. CDF Plot
ax2.hist(original.flatten()[:100000], bins=200, cumulative=True, density=True, 
        histtype='step', label="Original", color=CUSTOM_CMAP(0.1), linewidth=2)
ax2.hist(reconstructed.flatten()[:100000], bins=200, cumulative=True, density=True, 
        histtype='step', label="Reconstructed", color=CUSTOM_CMAP(0.9), linewidth=2)
ks_stat, _ = ks_2samp(original.flatten(), reconstructed.flatten())
ax2.set_title(f"Cumulative Distribution (KS Stat: {ks_stat:.4f})")
ax2.legend()

fig.tight_layout()
fig.savefig(f"{EVAL_OUT_DIR}/global_distribution_improved.png", dpi=150)
fig.savefig(f"{EVAL_OUT_DIR}/global_distribution_improved.svg")

######################################## Per-gene Mean Density Scatter ########################################

orig_mean = original.mean(axis=0)
recon_mean = reconstructed.mean(axis=0)

fig, ax = plt.subplots(figsize=(5, 5))
hb = ax.hexbin(orig_mean, recon_mean, gridsize=50, cmap='PiYG', mincnt=1, bins='log')
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Log10(Gene Count)')

lims = [0, max(orig_mean.max(), recon_mean.max())]
ax.plot(lims, lims, "k--", alpha=0.5, linewidth=0.8)
ax.set_xlabel("Mean expression (original)")
ax.set_ylabel("Mean expression (reconstructed)")
ax.set_title(f"Per-gene Mean (R²={r2_score(orig_mean, recon_mean):.3f})")

fig.tight_layout()
fig.savefig(f"{EVAL_OUT_DIR}/gene_mean_scatter.png", dpi=150)
fig.savefig(f"{EVAL_OUT_DIR}/gene_mean_scatter.svg")

######################################## Latent & Comparison UMAPs ########################################

# Latent Space
adata_latent = sc.AnnData(latent_coords, obs=test_rna.obs.copy())
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
    palette="Set2", # Changed for contrast, feel free to keep PiYG
    ax=ax2
)

plt.tight_layout()
plt.savefig(f"{EVAL_OUT_DIR}/recon_vs_input_umap.png", bbox_inches='tight')
plt.savefig(f"{EVAL_OUT_DIR}/recon_vs_input_umap.svg", bbox_inches='tight')
plt.show()


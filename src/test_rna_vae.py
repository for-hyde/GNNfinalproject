import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.vae_rna import InfoVAE_RNA
from utils.data_loading import separate_loader, SingleDatasetVAE, get_gene_weight, get_gene_weight_alt
from utils.device import get_free_gpu
from collections import OrderedDict
from scipy.stats import pearsonr, spearmanr, ks_2samp
from sklearn.metrics import r2_score
import pandas as pd
import seaborn as sns

train_rna, val_rna, test_rna = separate_loader(
    "/workspace/data/preprocessed_data/bmmc_celltype_split", "RNA"
)
gene_weight = get_gene_weight_alt(test_rna)

test_dataset = SingleDatasetVAE(test_rna)
test_loader  = DataLoader(test_dataset, batch_size=512, shuffle=False)
input_size   = test_rna.shape[1]
device       = get_free_gpu()

model = InfoVAE_RNA(
    input_size=input_size,
    latent_size=128,
    lr=0, wd=0,
    device=device,
    # gene_weight=gene_weight,
    mode="RNA",
)
state_dict = torch.load("/workspace/runs/2026-03-22 22:26:13.913327_vae_model_weights.pth")
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
print("Model weights loaded successfully!")
model.eval()

original, reconstructed, latent_coords = [], [], []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        _, mu, _ = model.encode(batch)
        recon    = model.decode(mu)
        original.append(batch.cpu().numpy())
        reconstructed.append(recon.cpu().numpy())
        latent_coords.append(mu.cpu().numpy())

original      = np.concatenate(original)
reconstructed = np.concatenate(reconstructed)
latent_coords = np.concatenate(latent_coords)

# CHANGED: test_rna already is the test split — no index filtering needed
cell_types = test_rna.obs["cell_type"].values

# ── 1. Per-cell metrics ───────────────────────────────────────────────────────
per_cell = []
for i in range(len(original)):
    r,   _ = pearsonr(original[i], reconstructed[i])
    rho, _ = spearmanr(original[i], reconstructed[i])
    mse    = np.mean((original[i] - reconstructed[i]) ** 2)
    per_cell.append({"cell_type": cell_types[i], "pearson_r": r, "spearman_r": rho, "mse": mse})

df = pd.DataFrame(per_cell)
print(df[["pearson_r", "spearman_r", "mse"]].describe())
df.to_csv("/workspace/runs/per_cell_recon_metrics.csv", index=False)

# ── 2. Per-cell Pearson R distribution ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["pearson_r"], bins=80, color="steelblue", edgecolor="none")
ax.axvline(df["pearson_r"].median(), color="red", linestyle="--",
           label=f"Median = {df['pearson_r'].median():.3f}")
ax.set_xlabel("Pearson r (per cell, across genes)")
ax.set_ylabel("Count")
ax.set_title("Per-cell Reconstruction Correlation")
ax.legend()
fig.tight_layout()
fig.savefig("/workspace/runs/per_cell_pearson_dist.png", dpi=150)
plt.close(fig)

# ── 3. Per-cell-type breakdown ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
order = df.groupby("cell_type")["pearson_r"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="cell_type", y="pearson_r", order=order, ax=ax,
            palette="tab20", flierprops={"markersize": 2})
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_title("Reconstruction Pearson r by Cell Type")
ax.set_ylabel("Pearson r")
fig.tight_layout()
fig.savefig("/workspace/runs/per_celltype_pearson.png", dpi=150)
plt.close(fig)

# ── 4. Global distribution comparison ────────────────────────────────────────
ks_stat, ks_pval = ks_2samp(original.flatten(), reconstructed.flatten())
print(f"KS statistic: {ks_stat:.4f}  p-value: {ks_pval:.4e}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(original.flatten(),      bins=200, alpha=0.5, color="steelblue",
        density=True, label="Original")
ax.hist(reconstructed.flatten(), bins=200, alpha=0.5, color="darkorange",
        density=True, label="Reconstructed")
ax.set_xlabel("Expression value")
ax.set_ylabel("Density")
ax.set_title(f"Global Value Distribution  (KS={ks_stat:.4f}, p={ks_pval:.2e})")
ax.legend()
fig.tight_layout()
fig.savefig("/workspace/runs/global_distribution.png", dpi=150)
plt.close(fig)

# ── 5. Per-gene mean scatter ──────────────────────────────────────────────────
orig_mean  = original.mean(axis=0)
recon_mean = reconstructed.mean(axis=0)
r2         = r2_score(orig_mean, recon_mean)
r_gene, _  = pearsonr(orig_mean, recon_mean)

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(orig_mean, recon_mean, s=1, alpha=0.3, color="mediumpurple", rasterized=True)
lims = [min(orig_mean.min(), recon_mean.min()), max(orig_mean.max(), recon_mean.max())]
ax.plot(lims, lims, "k--", linewidth=0.8)
ax.set_xlabel("Mean expression (original)")
ax.set_ylabel("Mean expression (reconstructed)")
ax.set_title(f"Per-gene mean  r={r_gene:.3f}, R²={r2:.3f}")
fig.tight_layout()
fig.savefig("/workspace/runs/gene_mean_scatter.png", dpi=150)
plt.close(fig)

# ── 6. UMAP of latent space ───────────────────────────────────────────────────
# CHANGED: use test_rna.obs directly — no index lookup needed
adata_latent = sc.AnnData(latent_coords, obs=test_rna.obs.copy())
sc.pp.neighbors(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(adata_latent, color=["cell_type"], title="Latent Space UMAP", show=False)
plt.savefig("/workspace/runs/latent_umap.png")
plt.close()

# ── 7. Input vs reconstructed UMAP ───────────────────────────────────────────
adata_recon = sc.AnnData(np.vstack([original, reconstructed]))
adata_recon.obs["type"] = ["Original"] * len(original) + ["Reconstructed"] * len(reconstructed)
sc.pp.pca(adata_recon)
sc.pp.neighbors(adata_recon)
sc.tl.umap(adata_recon)
sc.pl.umap(adata_recon, color="type", title="Input vs. Reconstructed", show=False)
plt.savefig("/workspace/runs/recon_vs_input_umap.png")
plt.close()

print("Evaluation plots saved to /workspace/runs/")
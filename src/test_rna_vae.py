import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.vae_rna import InfoVAE_RNA  # Adjust based on your actual class name
from utils.data_loading import load_data, SingleDatasetVAE, uniform_split_dataset, cell_type_split_dataset
from utils.device import get_free_gpu
from collections import OrderedDict



rna, _ = load_data("bmmc_rna_highly_variable.h5ad", "bmmc_atac_highly_variable.h5ad", multiome=False)
#_, _, test_idxs = uniform_split_dataset(rna, val_ratio=0.2, test_ratio=0.1)
_, _, test_idxs = cell_type_split_dataset(rna, annot=True, cell_col='cell_type', cluster_col='leiden', test_ratio=0.1, val_ratio=0.2, seed=19193)


test_dataset = SingleDatasetVAE(rna, test_idxs)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

input_size = rna.shape[1]

model_params = {
    "input_size": input_size,
    "latent_size": 512,
    "lr": 1e-3,
    "wd": 1e-5,
    "device": get_free_gpu(),
    "mode": "rna",
    "lambda_mmd": 0.1
}

model = InfoVAE_RNA(input_size=input_size, latent_size=512, lr=0, wd=0, mode="", device=model_params["device"])
state_dict = torch.load("/workspace/runs/2026-03-16 18:27:04.293336_vae_model_weights.pth")
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k.replace("_orig_mod.", "")
    new_state_dict[name] = v

# 3. Now load it into your model
model.load_state_dict(new_state_dict)
print("Model weights loaded successfully!")
model.eval()

original = []
reconstructed = []
latent_coords = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(model_params["device"])
        
        _, mu, logvar = model.encode(batch) 
        recon = model.decode(mu)
        
        original.append(batch.cpu().numpy())
        reconstructed.append(recon.cpu().numpy())
        latent_coords.append(mu.cpu().numpy())

original = np.concatenate(original)
reconstructed = np.concatenate(reconstructed)
latent_coords = np.concatenate(latent_coords)

from scipy.stats import pearsonr, spearmanr, ks_2samp
from sklearn.metrics import r2_score
import pandas as pd
import seaborn as sns

cell_types = rna.obs["cell_type"].values[
    np.where(rna.obs_names.isin(test_idxs))[0]
]

# ── 1. Per-cell metrics ───────────────────────────────────────────────────────

per_cell = []
for i in range(len(original)):
    r, _  = pearsonr(original[i], reconstructed[i])
    rho,_ = spearmanr(original[i], reconstructed[i])
    mse   = np.mean((original[i] - reconstructed[i]) ** 2)
    per_cell.append({"cell_type": cell_types[i], "pearson_r": r, "spearman_r": rho, "mse": mse})

df = pd.DataFrame(per_cell)
print(df[["pearson_r", "spearman_r", "mse"]].describe())
df.to_csv("/workspace/runs/per_cell_recon_metrics.csv", index=False)


# ── 2. Per-cell Pearson R distribution ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["pearson_r"], bins=80, color="steelblue", edgecolor="none")
ax.axvline(df["pearson_r"].median(), color="red", linestyle="--",
           label=f"Median = {df['pearson_r'].median():.3f}")
ax.set_xlabel("Pearson r (per cell, across peaks)")
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

# KS test on flattened values — tests if the two distributions are the same
ks_stat, ks_pval = ks_2samp(original.flatten(), reconstructed.flatten())
print(f"KS statistic: {ks_stat:.4f}  p-value: {ks_pval:.4e}")
# Ideally: small KS stat, large p-value (fail to reject H0 = distributions are similar)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(original.flatten(),       bins=200, alpha=0.5, color="steelblue",
        density=True, label="Original")
ax.hist(reconstructed.flatten(),  bins=200, alpha=0.5, color="darkorange",
        density=True, label="Reconstructed")
ax.set_xlabel("Accessibility value")
ax.set_ylabel("Density")
ax.set_title(f"Global Value Distribution  (KS={ks_stat:.4f}, p={ks_pval:.2e})")
ax.legend()
fig.tight_layout()
fig.savefig("/workspace/runs/global_distribution.png", dpi=150)
plt.close(fig)


# ── 5. Scatter: mean accessibility per peak (original vs reconstructed) ───────
# Collapses across cells — tells you if peak-level statistics are preserved

orig_mean = original.mean(axis=0)       # [n_peaks]
recon_mean = reconstructed.mean(axis=0)
r2 = r2_score(orig_mean, recon_mean)
r_peak, _ = pearsonr(orig_mean, recon_mean)

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(orig_mean, recon_mean, s=1, alpha=0.3, color="mediumpurple", rasterized=True)
lims = [min(orig_mean.min(), recon_mean.min()), max(orig_mean.max(), recon_mean.max())]
ax.plot(lims, lims, "k--", linewidth=0.8)
ax.set_xlabel("Mean accessibility (original)")
ax.set_ylabel("Mean accessibility (reconstructed)")
ax.set_title(f"Per-peak mean  r={r_peak:.3f}, R²={r2:.3f}")
fig.tight_layout()
fig.savefig("/workspace/runs/peak_mean_scatter.png", dpi=150)
plt.close(fig)


#switched from iloc to loc, as uniform splitting returns indices, but cell type splitting returns labels
adata_latent = sc.AnnData(latent_coords, obs=rna.obs.loc[test_idxs])
adata_recon = sc.AnnData(np.vstack([original, reconstructed]))
adata_recon.obs["type"] = ["Original"] * len(original) + ["Reconstructed"] * len(reconstructed)

sc.pp.neighbors(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(adata_latent, color=["cell_type"], title="Latent Space UMAP", show=False)
plt.savefig("/workspace/runs/latent_umap.png")

sc.pp.pca(adata_recon)
sc.pp.neighbors(adata_recon)
sc.tl.umap(adata_recon)
sc.pl.umap(adata_recon, color="type", title="Input vs. Reconstructed", show=False)
plt.savefig("/workspace/runs/recon_vs_input_umap.png")

print("Evaluation plots saved to /workspace/plots/")
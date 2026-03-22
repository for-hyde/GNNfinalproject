import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.vae_atac import InfoVAE_ATAC  # Adjust based on your actual class name
from utils.data_loading import  SingleDatasetVAE, separate_loader, get_gene_weight_alt, get_atac_pos_weights
from utils.device import get_free_gpu
from collections import OrderedDict
from scipy.stats import pearsonr, spearmanr, ks_2samp
from sklearn.metrics import r2_score
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score, jaccard_score
)

train_atac, val_atac, test_atac = separate_loader(
    "/workspace/data/preprocessed_data/bmmc_celltype_split", "ATAC"
)
#gene_weight = get_gene_weight_alt(test_atac)
atac_pos_weights = get_atac_pos_weights(test_atac.X).squeeze(0)

test_dataset = SingleDatasetVAE(test_atac)
test_loader  = DataLoader(test_dataset, batch_size=512, shuffle=False)

input_size = test_atac.shape[1]

device = get_free_gpu()

model = InfoVAE_ATAC(input_size=input_size, latent_size=128, lr=0, wd=0, mode="", device=device, pos_weight=atac_pos_weights.to(device))
state_dict = torch.load("/workspace/runs/2026-03-22 23:00:20.310592_vae_model_weights.pth")
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k.replace("_orig_mod.", "")
    new_state_dict[name] = v

# 3. Now load it into your model
model.load_state_dict(new_state_dict)
print("Model weights loaded successfully!")
model.eval()

def threshold_to_match_sparsity(recon_probs, original, tolerance=0.005):
    target_density = original.mean()  # ~0.18 for your data
    
    lo, hi = 0.0, 1.0
    for _ in range(50):  # binary search
        mid = (lo + hi) / 2
        pred_density = (recon_probs > mid).mean()
        if abs(pred_density - target_density) < tolerance:
            break
        if pred_density > target_density:
            lo = mid
        else:
            hi = mid
    
    print(f"Threshold: {mid:.4f}, Pred density: {pred_density:.4f}, Target: {target_density:.4f}")
    return (recon_probs > mid).astype(np.float32), mid


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

# ── 1. Per-cell metrics ───────────────────────────────────────────────────────

recon_binary, threshold = threshold_to_match_sparsity(reconstructed, original)

per_cell = []
for i in range(len(original)):
    true  = original[i].astype(int)
    pred  = recon_binary[i].astype(int)
    prob  = reconstructed[i]

    # Skip cells with no open peaks (edge case)
    if true.sum() == 0:
        continue

    per_cell.append({
        "cell_type":   cell_types[i],
        # Core binary metrics
        "recall":      recall_score(true, pred, zero_division=0),       # open peak recovery
        "precision":   precision_score(true, pred, zero_division=0),
        "f1":          f1_score(true, pred, zero_division=0),
        "jaccard":     jaccard_score(true, pred, zero_division=0),      # |intersection| / |union|
        # Ranking quality — doesn't require thresholding
        "auprc":       average_precision_score(true, prob),             # most informative for sparse data
        "auroc":       roc_auc_score(true, prob),
    })

df = pd.DataFrame(per_cell)
print(df[["recall", "precision", "f1", "jaccard", "auprc", "auroc"]].describe())
df.to_csv("/workspace/runs/per_cell_recon_metrics.csv", index=False)


# # ── 2. Per-cell Pearson R distribution ───────────────────────────────────────

# fig, ax = plt.subplots(figsize=(7, 4))
# ax.hist(df["pearson_r"], bins=80, color="steelblue", edgecolor="none")
# ax.axvline(df["pearson_r"].median(), color="red", linestyle="--",
#            label=f"Median = {df['pearson_r'].median():.3f}")
# ax.set_xlabel("Pearson r (per cell, across peaks)")
# ax.set_ylabel("Count")
# ax.set_title("Per-cell Reconstruction Correlation")
# ax.legend()
# fig.tight_layout()
# fig.savefig("/workspace/runs/per_cell_pearson_dist.png", dpi=150)
# plt.close(fig)


# # ── 3. Per-cell-type breakdown ────────────────────────────────────────────────

# fig, ax = plt.subplots(figsize=(10, 5))
# order = df.groupby("cell_type")["pearson_r"].median().sort_values(ascending=False).index
# sns.boxplot(data=df, x="cell_type", y="pearson_r", order=order, ax=ax,
#             palette="tab20", flierprops={"markersize": 2})
# ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
# ax.set_title("Reconstruction Pearson r by Cell Type")
# ax.set_ylabel("Pearson r")
# fig.tight_layout()
# fig.savefig("/workspace/runs/per_celltype_pearson.png", dpi=150)
# plt.close(fig)


# ── 4. Global distribution comparison ────────────────────────────────────────

# KS test on flattened values — tests if the two distributions are the same
ks_stat, ks_pval = ks_2samp(original.flatten(), recon_binary.flatten())
print(f"KS statistic: {ks_stat:.4f}  p-value: {ks_pval:.4e}")
# Ideally: small KS stat, large p-value (fail to reject H0 = distributions are similar)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(original.flatten(),       bins=200, alpha=0.5, color="steelblue",
        density=True, label="Original")
ax.hist(recon_binary.flatten(),  bins=200, alpha=0.5, color="darkorange",
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


# ── Latent UMAP (original data, colored by cell type) ────────────────────────
adata_latent = sc.AnnData(latent_coords, obs=test_atac.obs.copy())

sc.pp.neighbors(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(adata_latent, color=["cell_type"], title="Latent Space UMAP", show=False)
plt.savefig("/workspace/runs/latent_umap.png")


# ── Encode thresholded reconstructions into latent space ─────────────────────
recon_binary = (reconstructed > threshold).astype(np.float32)
recon_tensor = torch.tensor(recon_binary).to(device)
latent_recon = []

with torch.no_grad():
    for i in range(0, len(recon_tensor), 512):
        batch = recon_tensor[i:i+512]
        _, mu, _ = model.encode(batch)
        latent_recon.append(mu.cpu().numpy())

latent_recon = np.concatenate(latent_recon)


# ── Compare original vs reconstructed in the same latent space ───────────────
adata_compare = sc.AnnData(
    np.vstack([latent_coords, latent_recon]),
)
adata_compare.obs["source"] = (
    ["Original"] * len(latent_coords) + ["Reconstructed"] * len(latent_recon)
)
# Carry cell type through so you can check per-type structure preservation
adata_compare.obs["cell_type"] = np.concatenate([
    test_atac.obs["cell_type"].values,
    test_atac.obs["cell_type"].values,
])

sc.pp.neighbors(adata_compare)
sc.tl.umap(adata_compare)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: are original and reconstructed overlapping?
sc.pl.umap(adata_compare, color="source", ax=axes[0],
           title="Original vs. Reconstructed (latent)", show=False)

# Right: does cell type structure survive in the reconstructions?
sc.pl.umap(adata_compare, color="cell_type", ax=axes[1],
           title="Cell type structure (latent)", show=False)

fig.tight_layout()
fig.savefig("/workspace/runs/recon_vs_input_umap.png", dpi=150)
plt.close(fig)

print("Evaluation plots saved to /workspace/plots/")
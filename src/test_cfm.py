import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
import json
import os
from collections import OrderedDict

from sklearn.decomposition import PCA
import matplotlib.cm as cm
import seaborn as sns


from models.cfm import ModalityConverter
from models.vae_rna import InfoVAE_RNA
from models.vae_atac import InfoVAE_ATAC
from utils.device import load_model, get_free_gpu
from utils.data_loading import load_data, MultiomeDataset, cell_type_split_dataset, separate_loader, get_atac_pos_weights, threshold_to_match_sparsity

# Set global colormap
plt.rcParams['image.cmap'] = 'managua'
CUSTOM_CMAP = plt.get_cmap('managua')

# from matplotlib.colors import LinearSegmentedColormap

# custom_map_obj = sns.diverging_palette(300, 120, s=80, l=40, as_cmap=True)

# # 2. Register it so Matplotlib recognizes the name "my_piyg"
# try:
#     cm.register_cmap("PiYG", custom_map_obj)
# except:
#     # If already registered in this session, just get it
#     pass

# # 3. Set the global rcParam to the string name you just registered
# plt.rcParams['image.cmap'] = "PiYG"
# CUSTOM_CMAP = plt.get_cmap("PiYG")


def threshold_to_match_sparsity(probs: np.ndarray, original: np.ndarray,
                                tolerance: float = 0.005) -> tuple[np.ndarray, float]:
    """Binary-search threshold so predicted density ≈ original density."""
    target = original.mean()
    lo, hi = 0.0, 1.0
    mid = 0.5
    for _ in range(50):
        mid = (lo + hi) / 2
        if abs((probs > mid).mean() - target) < tolerance:
            break
        if (probs > mid).mean() > target:
            lo = mid
        else:
            hi = mid
    print(f"  threshold={mid:.4f}  pred_density={(probs>mid).mean():.4f}  "
          f"target_density={target:.4f}")
    return (probs > mid).astype(np.float32), mid


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """Unbiased MMD² estimate with RBF kernel. Small = distributions are close."""
    from sklearn.metrics.pairwise import rbf_kernel
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    n, m = len(X), len(Y)
    np.fill_diagonal(XX, 0); np.fill_diagonal(YY, 0)
    return XX.sum() / (n*(n-1)) + YY.sum() / (m*(m-1)) - 2*XY.mean()


def per_dim_kl_gaussian(mu1, sigma1, mu2, sigma2):
    """KL(N(mu1,σ1²) ‖ N(mu2,σ2²)) per dimension (scalar arrays of length D)."""
    return (np.log(sigma2/sigma1)
            + (sigma1**2 + (mu1-mu2)**2) / (2*sigma2**2)
            - 0.5)


def encode_batched(model_encode_fn, tensor: torch.Tensor,
                   batch_size: int = 512, device=None) -> np.ndarray:
    """Run encoder over a large tensor in mini-batches, return mu as numpy."""
    mus = []
    for i in range(0, len(tensor), batch_size):
        b = tensor[i:i+batch_size].to(device)
        with torch.no_grad():
            _, mu, _ = model_encode_fn(b)
        mus.append(mu.cpu().numpy())
    return np.concatenate(mus)


def load_state(model, path, device):
    state_dict = torch.load(path, map_location=device, weights_only=False)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print("Model weights loaded successfully!")
    model.eval()
    return model


EVAL_OUT_DIR = "/workspace/final_evaluation/cfm_uniform_final_kl/"
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

MODEL_PARAMS = {
    "latent_dim":    128,
    "rna_vae_path":  "/workspace/final_evaluation/final_models/RNA_vae_model_celltype.pth",
    "atac_vae_path": "/workspace/final_evaluation/final_models/ATAC_vae_model_celltype.pth",
    "cfm_path":      "/workspace/final_evaluation/final_models/CFM_model_celltype.pth",
    "device":        get_free_gpu(),
}

DEVICE     = MODEL_PARAMS["device"]
BATCH_SIZE = 512
CFM_STEPS  = 100

print("\n── Loading data ──")
train_rna,  val_rna,  test_rna  = separate_loader(
    "/workspace/data/preprocessed_data/integrated_uniform_split", "RNA")
train_atac, val_atac, test_atac = separate_loader(
    "/workspace/data/preprocessed_data/integrated_uniform_split", "ATAC")

test_dataset = MultiomeDataset(test_rna, test_atac)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

input_size_rna  = train_rna.shape[1]
input_size_atac = train_atac.shape[1]
cell_types      = test_rna.obs["cell_type"].values

print(f"  test cells={len(test_rna)}  RNA features={input_size_rna}  "
      f"ATAC features={input_size_atac}")

print("\n── Loading models ──")
atac_pos_weights = get_atac_pos_weights(test_atac.X).squeeze(0)

rna_vae = load_state(
    InfoVAE_RNA(input_size=input_size_rna,  latent_size=MODEL_PARAMS["latent_dim"],
                lr=0, wd=0, mode="rna",  device=DEVICE),
    MODEL_PARAMS["rna_vae_path"], DEVICE)

atac_vae = load_state(
    InfoVAE_ATAC(input_size=input_size_atac, latent_size=MODEL_PARAMS["latent_dim"],
                 lr=0, wd=0, mode="atac", device=DEVICE,
                 pos_weight=torch.ones(input_size_atac).to(DEVICE)),
    MODEL_PARAMS["atac_vae_path"], DEVICE)

cfm_model = ModalityConverter(
    latent_dim=MODEL_PARAMS["latent_dim"],
    rna_vae=rna_vae, atac_vae=atac_vae, device=DEVICE)
cfm_model = load_state(cfm_model, MODEL_PARAMS["cfm_path"], DEVICE)


with torch.no_grad():
    x_test = next(iter(test_loader))[1].to(DEVICE)
    x_recon, _, _, _ = atac_vae.forward(x_test)
    vae_r2 = r2_score(x_test.cpu().numpy().flatten(), x_recon.cpu().numpy().flatten())
    print(f"ATAC VAE reconstruction R²: {vae_r2:.4f}")


# ── Inference ─────────────────────────────────────────────────────────────────
print("\n── Running Flow & Evaluation ──")
all_rna_raw    = []
all_atac_raw   = []
all_atac_probs = []   # sigmoid probabilities
z_rna_list     = []
z_atac_list    = []
z_cfm_list     = []

with torch.no_grad():
    for x_rna, x_atac in test_loader:
        x_rna  = x_rna.to(DEVICE)
        x_atac = x_atac.to(DEVICE)

        # Encode both modalities for latent-space visualisations
        mu_rna_norm  = cfm_model._encode_rna(x_rna)
        mu_atac_norm = cfm_model._encode_atac(x_atac)
        z_rna_list.append(mu_rna_norm.cpu().numpy())
        z_atac_list.append(mu_atac_norm.cpu().numpy())

        # Run the flow: return_trajectory=True gives us the final latent z as well
        x_atac_hat_logits, trajectory = cfm_model.predict(
            x_rna, n_steps=CFM_STEPS, return_trajectory=True)

        z_cfm_hat = trajectory[-1]            # [N, latent_dim] — final flow state
        x_atac_probs = torch.sigmoid(x_atac_hat_logits)

        all_atac_probs.append(x_atac_probs.cpu().numpy())
        all_atac_raw.append(x_atac.cpu().numpy())
        all_rna_raw.append(x_rna.cpu().numpy())
        z_cfm_list.append(z_cfm_hat.cpu().numpy())

predicted_probs = np.vstack(all_atac_probs)   # [N, atac_features]  probabilities
targets         = np.vstack(all_atac_raw)      # [N, atac_features]  true binary ATAC
z_rna           = np.vstack(z_rna_list)        # [N, latent_dim]
z_atac          = np.vstack(z_atac_list)       # [N, latent_dim]
z_cfm           = np.vstack(z_cfm_list)        # [N, latent_dim]

print(f"  Probs: {predicted_probs.shape}  |  Targets: {targets.shape}")

# ── Sparsity-matched binarisation ─────────────────────────────────────────────
print("\n── Applying sparsity-matching threshold ──")
recon_binary, threshold = threshold_to_match_sparsity(predicted_probs, targets)
print(f"  Binary preds shape: {recon_binary.shape}")

# ── Metrics on binarised predictions ─────────────────────────────────────────
print("\n── Computing metrics ──")

mse = np.mean((recon_binary - targets) ** 2)
mae = np.mean(np.abs(recon_binary - targets))

# Per-feature Pearson R (across cells, for each peak/feature)
feature_pearson = np.array([
    pearsonr(recon_binary[:, i], targets[:, i])[0]
    for i in range(targets.shape[1])
])

# Per-cell Pearson R (across features, for each cell)
cell_pearson = np.array([
    pearsonr(recon_binary[i], targets[i])[0]
    for i in range(targets.shape[0])
])

# R² across all elements (flattened)
r2 = r2_score(targets.flatten(), recon_binary.flatten())

metrics = {
    "binarisation_threshold":   float(threshold),
    "pred_density":             float(recon_binary.mean()),
    "target_density":           float(targets.mean()),
    "mse":                      float(mse),
    "mae":                      float(mae),
    "r2_global":                float(r2),
    "mean_feature_pearson_r":   float(np.nanmean(feature_pearson)),
    "median_feature_pearson_r": float(np.nanmedian(feature_pearson)),
    "mean_cell_pearson_r":      float(np.nanmean(cell_pearson)),
    "median_cell_pearson_r":    float(np.nanmedian(cell_pearson)),
}

print("\n Evaluation Metrics (on binarised predictions) ")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

with open(f"{EVAL_OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)


# ── PCA trajectory plot ───────────────────────────────────────────────────────
N_TRAJ = 500
n_steps_traj = 200

x_rna_sub = torch.tensor(np.vstack(all_rna_raw)[:N_TRAJ]).to(DEVICE)

with torch.no_grad():
    _, trajectory = cfm_model.predict(x_rna_sub, n_steps=n_steps_traj,
                                      return_trajectory=True)

traj_np = trajectory.cpu().numpy()   # [n_steps, N_TRAJ, latent_dim]

anchors  = np.vstack([z_rna[:N_TRAJ], z_atac[:N_TRAJ]])
pca      = PCA(n_components=2)
pca.fit(anchors)

flat_traj = traj_np.reshape(-1, traj_np.shape[-1])
proj_traj = pca.transform(flat_traj).reshape(n_steps_traj, N_TRAJ, 2)

proj_rna  = pca.transform(z_rna[:N_TRAJ])
proj_atac = pca.transform(z_atac[:N_TRAJ])

N_LINES   = 30
ct_labels = cell_types[:N_TRAJ]
unique_cts = np.unique(ct_labels)
ct_palette = dict(zip(unique_cts, sns.color_palette("PiYG", len(unique_cts))))

fig, ax = plt.subplots(figsize=(10, 10))
# Replaced hardcoded colors with PiYG fractions
ax.scatter(proj_rna[:, 0],  proj_rna[:, 1],  s=4, alpha=0.3,
           color=CUSTOM_CMAP(0.1),   label="z_RNA (source)")
ax.scatter(proj_atac[:, 0], proj_atac[:, 1], s=4, alpha=0.3,
           color=CUSTOM_CMAP(0.9),  label="z_ATAC true (target)")
ax.scatter(proj_traj[:, :, 0], proj_traj[:, :, 1],
           s=0.3, alpha=0.05, color=CUSTOM_CMAP(0.5), rasterized=True)

for i in range(N_LINES):
    xs = proj_traj[:, i, 0]
    ys = proj_traj[:, i, 1]
    for t in range(len(xs) - 1):
        alpha = 0.5 + 0.5 * (t / len(xs))
        # Gradient mapping over the trajectory using PiYG
        color_fraction = 0.1 + (0.8 * (t / len(xs)))
        ax.plot(xs[t:t+2], ys[t:t+2], lw=0.8, alpha=alpha, color=CUSTOM_CMAP(color_fraction))

ax.scatter(proj_traj[-1, :N_LINES, 0], proj_traj[-1, :N_LINES, 1],
           s=20, color=CUSTOM_CMAP(0.8),  zorder=5, label="CFM endpoint (z_ATAC pred)")
ax.scatter(proj_traj[0,  :N_LINES, 0], proj_traj[0,  :N_LINES, 1],
           s=20, color=CUSTOM_CMAP(0.2), zorder=5, label="Start (z_RNA)")

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("CFM latent trajectories (RNA → ATAC, projected to PCA)")
ax.legend(markerscale=3, fontsize=8)
fig.tight_layout()
fig.savefig(f"{EVAL_OUT_DIR}/M_cfm_trajectories_pca.png", dpi=150)
fig.savefig(f"{EVAL_OUT_DIR}/M_cfm_trajectories_pca.svg") # Added SVG
plt.close(fig)


# ── UMAP ─────────────────────────────────────────────────────────────────────
print("\n── Building UMAP (this may take a minute) ──")
N = len(z_atac)
combined      = np.vstack([z_rna, z_atac, z_cfm])
source_labels = ["RNA z"] * N + ["True ATAC z"] * N + ["CFM z"] * N
ct_labels     = np.tile(cell_types, 3)

adata_all = sc.AnnData(combined)
adata_all.obs["source"]    = source_labels
adata_all.obs["cell_type"] = ct_labels

sc.pp.neighbors(adata_all, n_neighbors=15, use_rep="X")
sc.tl.umap(adata_all)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sc.pl.umap(adata_all, color="source",    ax=axes[0],
           title="Latent clouds: RNA z / true ATAC z / CFM z", show=False,
           palette={"RNA z": CUSTOM_CMAP(0.1),
                    "True ATAC z": CUSTOM_CMAP(0.9),
                    "CFM z": CUSTOM_CMAP(0.5)}) # PiYG Mapping
# Fallback to general PiYG mapping for cell types
sc.pl.umap(adata_all, color="cell_type", ax=axes[1],
           title="Cell type structure across latent clouds", show=False, palette="PiYG")
fig.tight_layout()
fig.savefig(f"{EVAL_OUT_DIR}/J_trilatent_umap.png", dpi=150)
fig.savefig(f"{EVAL_OUT_DIR}/J_trilatent_umap.svg") # Added SVG
plt.close(fig)


adata_cmp = sc.AnnData(np.vstack([z_atac, z_cfm]))
adata_cmp.obs["source"]    = ["True ATAC z"] * N + ["CFM z"] * N
adata_cmp.obs["cell_type"] = np.tile(cell_types, 2)

sc.pp.neighbors(adata_cmp, n_neighbors=15, use_rep="X")
sc.tl.umap(adata_cmp)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.pl.umap(adata_cmp, color="source",    ax=axes[0],
           title="True ATAC z vs CFM predicted z", show=False,
           palette={"True ATAC z": CUSTOM_CMAP(0.9), "CFM z": CUSTOM_CMAP(0.5)}) # PiYG mapping
sc.pl.umap(adata_cmp, color="cell_type", ax=axes[1],
           title="Cell type (true vs predicted)", show=False, palette="PiYG")
fig.tight_layout()
fig.savefig(f"{EVAL_OUT_DIR}/K_atac_vs_cfm_umap.png", dpi=150)
fig.savefig(f"{EVAL_OUT_DIR}/K_atac_vs_cfm_umap.svg") # Added SVG
plt.close(fig)
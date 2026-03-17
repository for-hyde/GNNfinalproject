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


from models.cfm import ModalityConverter
from models.vae_rna import InfoVAE
from utils.device import load_model, get_free_gpu
from utils.data_loading import load_data, MultiomeDataset, cell_type_split_dataset



EVAL_OUT_DIR = "/workspace/runs/cfm_eval"
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

MODEL_PARAMS = {
    "latent_dim": 128,
    "rna_vae_path":  "/workspace/runs/rna_vae_run_1/2026-03-15 15:22:17.448786_vae_model_weights.pth",  # adjust as needed! 
    "atac_vae_path": "/workspace/runs/atac_vae_run_1/2026-03-15 15:56:11.264790_vae_model_weights.pth",
    "cfm_path":      "/workspace/runs/2026-03-15 18:01:39.442836_CFM_model_weights.pth",
    "device":        get_free_gpu(),
}

DEVICE = MODEL_PARAMS["device"]
BATCH_SIZE = 512

rna, atac = load_data(
    'bmmc_rna_highly_variable.h5ad',
    'bmmc_atac_highly_variable.h5ad',
    multiome=False
)

_, _, test_idxs = cell_type_split_dataset(
    rna, annot=True, cell_col='cell_type', cluster_col='leiden',
    test_ratio=0.1, val_ratio=0.2, seed=19193
)

test_dataset = MultiomeDataset(rna, atac, test_idxs)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

input_size_rna  = rna.shape[1]
input_size_atac = atac.shape[1]


rna_vae_raw  = InfoVAE(input_size=input_size_rna,  latent_size=MODEL_PARAMS["latent_dim"],
                       lr=0, wd=0, mode="rna",  device=DEVICE)
atac_vae_raw = InfoVAE(input_size=input_size_atac, latent_size=MODEL_PARAMS["latent_dim"],
                       lr=0, wd=0, mode="atac", device=DEVICE)

rna_vae  = load_model(rna_vae_raw,  MODEL_PARAMS["rna_vae_path"])
atac_vae = load_model(atac_vae_raw, MODEL_PARAMS["atac_vae_path"])

model = ModalityConverter(
    latent_dim=MODEL_PARAMS["latent_dim"],
    rna_vae=rna_vae,
    atac_vae=atac_vae,
    device=DEVICE,
)


state_dict = torch.load(MODEL_PARAMS["cfm_path"], map_location=DEVICE)
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k.replace("_orig_mod.", "")
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
print("Model weights loaded successfully!")
model.eval()


with torch.no_grad():
    x_test = next(iter(test_loader))[1].to(DEVICE)  # grab true ATAC
    x_recon, _, _, _ = atac_vae.forward(x_test)
    vae_r2 = r2_score(x_test.cpu().numpy().flatten(), x_recon.cpu().numpy().flatten())
    print(f"ATAC VAE reconstruction R²: {vae_r2:.4f}")


all_preds  = []
all_targets = []

with torch.no_grad():
    for x_rna, x_atac in test_loader:
        x_rna = x_rna.to(DEVICE)
        x_atac_hat = model.predict(x_rna, n_steps=100)

        all_preds.append(x_atac_hat.cpu().numpy())
        all_targets.append(x_atac.numpy())

preds   = np.vstack(all_preds)    # [n_test, atac_features]
targets = np.vstack(all_targets)  # [n_test, atac_features]
print(f"Inference complete. Predictions shape: {preds.shape}")


mse = np.mean((preds - targets) ** 2)
mae = np.mean(np.abs(preds - targets))

# Per-feature Pearson R (across cells, for each peak/feature)
feature_pearson = np.array([
    pearsonr(preds[:, i], targets[:, i])[0]
    for i in range(targets.shape[1])
])

# Per-cell Pearson R (across features, for each cell)
cell_pearson = np.array([
    pearsonr(preds[i], targets[i])[0]
    for i in range(targets.shape[0])
])

# R² across all elements (flattened)
r2 = r2_score(targets.flatten(), preds.flatten())

metrics = {
    "mse":                    float(mse),
    "mae":                    float(mae),
    "r2_global":              float(r2),
    "mean_feature_pearson_r": float(np.nanmean(feature_pearson)),
    "median_feature_pearson_r": float(np.nanmedian(feature_pearson)),
    "mean_cell_pearson_r":    float(np.nanmean(cell_pearson)),
    "median_cell_pearson_r":  float(np.nanmedian(cell_pearson)),
}

print("\n Evaluation Metrics ")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

with open(f"{EVAL_OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)


# Per-cell-type mean Pearson R, so you can see which populations transfer well

test_cell_types = rna.obs["cell_type"].values[test_idxs]
ct_metrics = {}

for ct in np.unique(test_cell_types):
    mask = test_cell_types == ct
    if mask.sum() < 5:               # skip tiny groups
        continue
    r_vals = np.array([
        pearsonr(preds[mask][i], targets[mask][i])[0]
        for i in range(mask.sum())
    ])
    ct_metrics[ct] = {
        "n_cells": int(mask.sum()),
        "mean_cell_pearson_r": float(np.nanmean(r_vals)),
    }

print("\n Per Cell Type ")
for ct, m in sorted(ct_metrics.items(), key=lambda x: -x[1]["mean_cell_pearson_r"]):
    print(f"  {ct:30s}  n={m['n_cells']:4d}  r={m['mean_cell_pearson_r']:.4f}")

with open(f"{EVAL_OUT_DIR}/celltype_metrics.json", "w") as f:
    json.dump(ct_metrics, f, indent=2)
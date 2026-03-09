import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.vae import InfoVAE  # Adjust based on your actual class name
from utils.data_loading import load_data, SingleDatasetVAE, uniform_split_dataset
from utils.device import get_free_gpu
from collections import OrderedDict



rna, _ = load_data("bmmc_rna_highly_variable.h5ad", "bmmc_atac_highly_variable.h5ad", multiome=False)
_, _, test_idxs = uniform_split_dataset(rna, val_ratio=0.2, test_ratio=0.1)

test_dataset = SingleDatasetVAE(rna, test_idxs)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

input_size = rna.shape[1]

model_params = {
    "input_size": input_size,
    "latent_size": 64,
    "lr": 1e-3,
    "wd": 1e-5,
    "device": get_free_gpu(),
    "mode": "rna",
    "lambda_mmd": 0.1
}

model = InfoVAE(input_size=input_size, latent_size=64, lr=0, wd=0, mode="", device=model_params["device"])
state_dict = torch.load("/workspace/models/2026-03-09 11:02:59.695223_vae_model_weights.pth")
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


adata_latent = sc.AnnData(latent_coords, obs=rna.obs.iloc[test_idxs])
adata_recon = sc.AnnData(np.vstack([original, reconstructed]))
adata_recon.obs["type"] = ["Original"] * len(original) + ["Reconstructed"] * len(reconstructed)

sc.pp.neighbors(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(adata_latent, color=["cell_type"], title="Latent Space UMAP", show=False)
plt.savefig("/workspace/plots/latent_umap.png")

sc.pp.pca(adata_recon)
sc.pp.neighbors(adata_recon)
sc.tl.umap(adata_recon)
sc.pl.umap(adata_recon, color="type", title="Input vs. Reconstructed", show=False)
plt.savefig("/workspace/plots/recon_vs_input_umap.png")

print("Evaluation plots saved to /workspace/plots/")
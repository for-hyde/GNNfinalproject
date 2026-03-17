from torch.utils.data import DataLoader

from models.cfm import train_modality_converter
from utils.device import get_free_gpu
from utils.data_loading import load_data, MultiomeDataset, cell_type_split_dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json

# Still load both rna and atac (_) data to ensure that we only use the rna data for which we also have the paired atac data! 
rna, atac = load_data('bmmc_rna_highly_variable.h5ad', 'bmmc_atac_highly_variable.h5ad', multiome=False)

#train_idxs, val_idxs = split_dataset(rna)

#train_idxs, val_idxs, test_idxs = uniform_split_dataset(rna, val_ratio=0.2, test_ratio=0.1)

# or

train_idxs, val_idxs, test_idxs = cell_type_split_dataset(rna, annot=True, cell_col='cell_type', cluster_col='leiden', test_ratio=0.1, val_ratio=0.2, seed=19193)
# Because the rna and atac datasets contain only common cells and are perfectly aligned, we can use the indices on only the rna, to subset also the atac data 

train_dataset = MultiomeDataset(rna, atac, train_idxs)
val_dataset   = MultiomeDataset(rna, atac, val_idxs)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

input_size_atac = atac.shape[1]
input_size_rna = rna.shape[1]

model_params = {
    "latent_dim": 128,
    "rna_vae_input": input_size_rna,
    "atac_vae_input": input_size_atac,
    "rna_vae_path": "/workspace/runs/rna_vae_run_1/2026-03-15 15:22:17.448786_vae_model_weights.pth",  # just placeholder state dictionary! 
    "atac_vae_path": "/workspace/runs/atac_vae_run_1/2026-03-15 15:56:11.264790_vae_model_weights.pth",  # just placeholder state dictionary! 
    "device": get_free_gpu(),
}

trained_model, train_loss, val_loss = train_modality_converter(
    model_params=model_params,
    train_loader=train_loader,
    valid_loader=val_loader,
    epochs=500,
    patience=50
)

print(f"Final Training Loss: {train_loss[-1]:.4f}")

plt.figure()
plt.plot(train_loss, c="red")
plt.plot(val_loss, c="blue")
plt.savefig("/workspace/runs/test_cfm_training_losses.png")
plt.savefig("/workspace/runs/test_cfm_training_losses.svg")


with open("/workspace/runs/losses.json", "w") as f:
    json.dump({"train": train_loss, "val": val_loss}, f)

# python train_rna_vae.py > /workspace/logs/rna_vae_test_training.log 2>&1 &
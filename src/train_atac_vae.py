import torch
from torch.utils.data import DataLoader

from models.vae_rna import train_infoVAE
from utils.device import get_free_gpu
from utils.data_loading import load_data, split_dataset, SingleDatasetVAE, uniform_split_dataset, cell_type_split_dataset
from utils.logging_utils import log, log_section, start_log

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json

# Still load both rna and atac (_) data to ensure that we only use the rna data for which we also have the paired atac data! 
_, atac = load_data('bmmc_rna_highly_variable.h5ad', 'bmmc_atac_highly_variable.h5ad', multiome=False)

#train_idxs, val_idxs = split_dataset(rna)

train_idxs, val_idxs, test_idxs = uniform_split_dataset(atac, val_ratio=0.2, test_ratio=0.1)

# or

# train_idxs, val_idxs, test_idxs = cell_type_split_dataset(rna, annot=True, cell_col='cell_type', cluster_col='leiden', test_ratio=0.1, val_ratio=0.2, seed=19193)

train_dataset = SingleDatasetVAE(atac, train_idxs)
val_dataset   = SingleDatasetVAE(atac, val_idxs)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

data_iter = iter(train_loader)
inputs = next(data_iter)
input_size = inputs.shape[1]

print(f"Input Size: {input_size}")

# Layer number is now hard coded into the model!
model_params = {
    "input_size": input_size,
    "latent_size": 128,
    "lr": 0.00034398502974696964,
    "wd": 4.839134126781686e-05,
    "device": get_free_gpu(),
    "mode": "atac",
    "lambda_mmd": 0.14479044679925782
}

trained_model, train_loss, val_loss = train_infoVAE(
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
plt.savefig("/workspace/runs/test_atac_training_losses.png")
plt.savefig("/workspace/runs/test_atac_training_losses.svg")


with open("/workspace/runs/losses_atac.json", "w") as f:
    json.dump({"train": train_loss, "val": val_loss}, f)

# python train_atac_vae.py > /workspace/logs/atac_vae_test_training.log 2>&1 &
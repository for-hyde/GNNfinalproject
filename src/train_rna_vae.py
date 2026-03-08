import torch
from torch.utils.data import DataLoader

from models.vae import train_infoVAE
from utils.device import get_free_gpu
from utils.data_loading import load_data, split_dataset, SingleDatasetVAE
from utils.logging_utils import log, log_section, start_log

import matplotlib.pyplot as plt

# Still load both rna and atac (_) data to ensure that we only use the rna data for which we also have the paired atac data! 
rna, _ = load_data('bmmc_rna_highly_variable.h5ad', 'bmmc_atac_highly_variable.h5ad', multiome=False)

train_idxs, val_idxs = split_dataset(rna)

train_dataset = SingleDatasetVAE(rna, train_idxs)
val_dataset   = SingleDatasetVAE(rna, val_idxs)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

data_iter = iter(train_loader)
inputs = next(data_iter)
input_size = inputs.shape[1]

print(f"Input Size: {input_size}")

model_params = {
    "input_size": input_size,
    "latent_size": 64,
    "layers": 5,
    "lr": 1e-3,
    "wd": 1e-5,
    "device": get_free_gpu(),
    "mode": "rna"
}

trained_model, train_loss, val_loss = train_infoVAE(
    model_params=model_params,
    train_loader=train_loader,
    valid_loader=val_loader,
    epochs=5,
    patience=50
)

print(f"Final Training Loss: {train_loss[-1]:.4f}")

plt.figure()
plt.plot(train_loss, c="red")
plt.plot(val_loss, c="blue")
plt.savefig("/workspace/plots/test_rna_training_losses.png")
plt.savefig("/workspace/plots/test_rna_training_losses.svg")

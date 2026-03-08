import torch
from torch.utils.data import DataLoader, TensorDataset

from models.vae import InfoVAE, train_infoVAE
from utils.device import get_free_gpu

import matplotlib.pyplot as plt


def get_test_loaders(n_samples=1000, input_dim=32, batch_size=64):
    cluster1 = torch.randn(n_samples // 2, input_dim) + 2.0
    cluster2 = torch.randn(n_samples // 2, input_dim) - 2.0
    
    data = torch.cat([cluster1, cluster2], dim=0)
    data = data[torch.randperm(data.size(0))]

    split = int(0.8 * n_samples)
    train_data, val_data = data[:split], data[split:]
    
    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


model_params = {
    "input_size": 32,
    "latent_size": 4,
    "layers": 6,
    "lr": 1e-3,
    "wd": 1e-5,
    "device": get_free_gpu(),
}

# 1. Get Data
train_loader, val_loader = get_test_loaders(input_dim=model_params["input_size"])

# 2. Run Training
# (Make sure your logging_utils functions are defined or mocked)
trained_model, train_loss, val_loss = train_infoVAE(
    model_params=model_params,
    train_loader=train_loader,
    valid_loader=val_loader,
    epochs=100
)

print(f"Final Training Loss: {train_loss[-1]:.4f}")

plt.figure()
plt.plot(train_loss, c="red")
plt.plot(val_loss, c="blue")
plt.savefig("/workspace/test_img.png")
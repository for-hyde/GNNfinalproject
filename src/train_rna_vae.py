from torch.utils.data import DataLoader
from models.vae_rna import train_infoVAE_RNA
from utils.device import get_free_gpu
from utils.data_loading import separate_loader, SingleDatasetVAE, get_gene_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

####################################################################################################
#                                                                                                  #
# Short helper script for training RNA VAE model                                                   #
# Parameters can be adjusted in model_params.                                                      #
# Currently it is set up to use the cell type split during training, changing to a uniform split   #
# requires a change of the data path. Evaluation is performed in the test_rna_vae.py script.       #
#                                                                                                  #
####################################################################################################


#################### Create Training and Validation Dataloaders ####################

train_rna, val_rna, test_rna = separate_loader("/workspace/data/preprocessed_data/integrated_celltype_split", "RNA")
#gene_weight = get_gene_weight(train_rna)

train_dataset = SingleDatasetVAE(train_rna)
val_dataset   = SingleDatasetVAE(val_rna)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

data_iter = iter(train_loader)
inputs = next(data_iter)
input_size = inputs.shape[1]

print(f"Input Size: {input_size}")

#################### Define Model Parameters ####################

model_params = {
    "input_size": input_size,
    "latent_size": 128,
    "lr": 0.0005,  # tuned with optuna
    "wd": 3e-05,  # tuned with optuna
    "device": get_free_gpu(),
    "mode": "rna",
    "lambda_mmd": 15.0,
    #"gene_weight": gene_weight
}

#################### Train Model ####################

trained_model, train_loss, val_loss = train_infoVAE_RNA(
    model_params=model_params,
    train_loader=train_loader,
    valid_loader=val_loader,
    epochs=500,
    patience=50
)  # Saving of model is handled within train_infoVAE_RNA itself
print(f"Final Training Loss: {train_loss[-1]:.4f}")

#################### Plot Training and Validation Losses ####################

plt.figure()
plt.plot(train_loss, c="red")
plt.plot(val_loss, c="blue")
plt.savefig("/workspace/runs/test_rna_training_losses.png")
plt.savefig("/workspace/runs/test_rna_training_losses.svg")


with open("/workspace/runs/losses.json", "w") as f:
    json.dump({"train": train_loss, "val": val_loss}, f)
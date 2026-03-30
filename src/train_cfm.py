from torch.utils.data import DataLoader

from models.cfm import train_modality_converter
from utils.device import get_free_gpu
from utils.data_loading import MultiomeDataset, separate_loader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json


####################################################################################################
#                                                                                                  #
# Short helper script for training CFM model                                                       #
# Parameters can be adjusted in model_params. Most importantly load here the RNA and ATAC mdoels.  #
# Currently it is set up to use the cell type split during training, changing to a uniform split   #
# requires a change of the data path. Evaluation is performed in the test_rna_vae.py script.       #
#                                                                                                  #
####################################################################################################

DATA_PATH = "/workspace/data/preprocessed_data/integrated_uniform_split"
RNA_MODEL_PATH = "/workspace/final_evaluation/final_models/RNA_vae_model_uniform.pth"
ATAC_MODEL_PATH = "/workspace/final_evaluation/final_models/ATAC_vae_model_uniform.pth"

#################### Create Training and Validation Dataloaders ####################


train_rna, val_rna, test_rna = separate_loader(DATA_PATH, "RNA")
train_atac, val_atac, test_atac = separate_loader(DATA_PATH, "ATAC")

train_dataset = MultiomeDataset(train_rna, train_atac)
val_dataset   = MultiomeDataset(val_rna, val_atac)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

input_size_atac = train_atac.shape[1]
input_size_rna = train_rna.shape[1]

model_params = {
    "latent_dim": 128,
    "rna_vae_input": input_size_rna,
    "atac_vae_input": input_size_atac,
    "rna_vae_path": RNA_MODEL_PATH,  
    "atac_vae_path": ATAC_MODEL_PATH,
    "device": get_free_gpu(),
}


#################### Train Model ####################

trained_model, train_loss, val_loss = train_modality_converter(
    model_params=model_params,
    train_loader=train_loader,
    valid_loader=val_loader,
    epochs=1000,
    patience=100
)

print(f"Final Training Loss: {train_loss[-1]:.4f}")

plt.figure()
plt.plot(train_loss, c="red")
plt.plot(val_loss, c="blue")
plt.savefig("/workspace/runs/test_cfm_training_losses.png")
plt.savefig("/workspace/runs/test_cfm_training_losses.svg")


with open("/workspace/runs/losses.json", "w") as f:
    json.dump({"train": train_loss, "val": val_loss}, f)

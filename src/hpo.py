import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import KFold
import optuna
from optuna.samplers import TPESampler
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.vae import train_infoVAE
from utils.device import get_free_gpu
from utils.data_loading import load_data, cell_type_split_dataset, SingleDatasetVAE, uniform_split_dataset
from utils.logging_utils import (start_log, log, log_section)

start_log("/workspace/runs/rna_vae_hpo", "RNA-VAE-HPO_log")

# --- Data loading (done ONCE, outside the objective) ---
rna, _ = load_data('bmmc_rna_highly_variable.h5ad', 'bmmc_atac_highly_variable.h5ad', multiome=False)

# train_idxs, val_idxs, test_idxs = cell_type_split_dataset(
#     rna, annot=True, cell_col='cell_type', cluster_col='leiden',
#     test_ratio=0.1, val_ratio=0.2, seed=19193
# )

log_section("LOADING DATA")
train_idxs, val_idxs, test_idxs = uniform_split_dataset(rna, val_ratio=0.2, test_ratio=0.1)

train_dataset = SingleDatasetVAE(rna, train_idxs)
#val_dataset   = SingleDatasetVAE(rna, val_idxs)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
#val_loader   = DataLoader(val_dataset,   batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

log("Data Loading Successful!")

input_size = next(iter(train_loader)).shape[1]
DEVICE = get_free_gpu()

def objective(trial):
    model_params = {
        "input_size": input_size,
        "latent_size": trial.suggest_int("latent_size", 32, 128, step=16),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        "device": DEVICE,
        "mode": "rna",
        "lambda_mmd": trial.suggest_float("lambda_mmd", 0.1, 0.5),
    }

    # Use only train+val indices for CV (keep test_idxs held out entirely)
    cv_indices = np.concatenate([train_idxs, val_idxs])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_val_losses = []

    for fold, (fold_train_idx, fold_val_idx) in enumerate(kf.split(cv_indices)):
        # Map back to actual dataset indices
        fold_train = cv_indices[fold_train_idx]
        fold_val   = cv_indices[fold_val_idx]

        fold_train_loader = DataLoader(
            SingleDatasetVAE(rna, fold_train),
            batch_size=512, shuffle=True, num_workers=4, pin_memory=True
        )
        fold_val_loader = DataLoader(
            SingleDatasetVAE(rna, fold_val),
            batch_size=512, shuffle=False, num_workers=4, pin_memory=True
        )

        _, _, val_loss = train_infoVAE(
            model_params=model_params,
            train_loader=fold_train_loader,
            valid_loader=fold_val_loader,
            epochs=30,
            patience=5,
            log_path="/workspace/runs/rna_hpo",
            save=False,
            restart_log=False
        )
        fold_val_losses.append(min(val_loss))
    
    mean_loss = np.mean(fold_val_losses)
    return mean_loss


sampler = TPESampler(seed=42)
study = optuna.create_study(
    direction="minimize",
    sampler=sampler,
    study_name="RNA_infoVAE_hpo",  #hpo for hyperparameter optimization
)

study.optimize(objective, n_trials=50, gc_after_trial=True)

best = study.best_trial
log(f"Best val loss : {best.value:.4f}")
log(f"Best params   : {best.params}")

# Save best params
with open("/workspace/runs/best_rnavae_hpo_params.json", "w") as f:
    json.dump(best.params, f, indent=2)

try:
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image("/workspace/runs/hpo_param_importances.png")

    fig2 = optuna.visualization.plot_optimization_history(study)
    fig2.write_image("/workspace/runs/hpo_history.png")
except Exception as e:
    log(f"Visualization skipped: {e}")
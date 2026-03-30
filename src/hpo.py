import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import KFold
import optuna
from optuna.samplers import TPESampler
import json
import matplotlib
matplotlib.use('Agg')

from models.vae_rna import train_infoVAE_RNA
from utils.device import get_free_gpu
from utils.data_loading import SingleDatasetVAE, separate_loader
from utils.logging_utils import (start_log, log, log_section)



DATA_PATH = "/workspace/data/preprocessed_data/integrated_uniform_split"
MODALITY = "RNA"


start_log(f"/workspace/runs/{MODALITY}_vae_hpo", f"{MODALITY}-VAE-HPO_log")

train_data, val_data, _ = separate_loader(DATA_PATH, MODALITY)

train_dataset = SingleDatasetVAE(train_data)
val_dataset   = SingleDatasetVAE(val_data)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)


log("Data Loading Successful!")

input_size = next(iter(train_loader)).shape[1]
DEVICE = get_free_gpu()

def objective(trial):
    model_params = {
        "input_size": input_size,
        "latent_size": 128,
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        "device": DEVICE,
        "mode": "atac",
        "lambda_mmd": trial.suggest_float("lambda_mmd", 0.1, 0.5),
    }

    # Combine train + val into one CV pool (test_rna stays held out)
    cv_dataset = SingleDatasetVAE(np.concatenate([train_data, val_data], axis=0))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_val_losses = []

    for fold, (fold_train_idx, fold_val_idx) in enumerate(kf.split(range(len(cv_dataset)))):
        fold_train_loader = DataLoader(
            Subset(cv_dataset, fold_train_idx),
            batch_size=512, shuffle=True, num_workers=4, pin_memory=True
        )
        fold_val_loader = DataLoader(
            Subset(cv_dataset, fold_val_idx),
            batch_size=512, shuffle=False, num_workers=4, pin_memory=True
        )

        _, _, val_loss = train_infoVAE_RNA(
            model_params=model_params,
            train_loader=fold_train_loader,
            valid_loader=fold_val_loader,
            epochs=200,
            patience=50,
            log_path=f"/workspace/runs/{MODALITY}_hpo",
            save=False,
            restart_log=False
        )
        fold_val_losses.append(min(val_loss))

    return np.mean(fold_val_losses)


sampler = TPESampler(seed=42)
study = optuna.create_study(
    direction="minimize",
    sampler=sampler,
    study_name=f"{MODALITY}_infoVAE_hpo",  #hpo for hyperparameter optimization
)

study.optimize(objective, n_trials=30, gc_after_trial=True)

best = study.best_trial
log(f"Best val loss : {best.value:.4f}")
log(f"Best params   : {best.params}")

# Save best params
with open("/workspace/runs/best_atacvae_hpo_params.json", "w") as f:
    json.dump(best.params, f, indent=2)

try:
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image("/workspace/runs/hpo_param_importances_atac.png")

    fig2 = optuna.visualization.plot_optimization_history(study)
    fig2.write_image("/workspace/runs/hpo_history_atac.png")
except Exception as e:
    log(f"Visualization skipped: {e}")
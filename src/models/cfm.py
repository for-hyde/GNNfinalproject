import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader

from torchcfm.models import MLP
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchdiffeq import odeint

from utils.logging_utils import (start_log, log, log_section)
from utils.device import load_model
from models.vae_rna import InfoVAE_RNA
from models.vae_atac import InfoVAE_ATAC
import copy
from datetime import datetime
from tqdm import tqdm 
import sys


class ModalityConverter(nn.Module):
    def __init__(self, latent_dim, rna_vae, atac_vae, device, sigma=0.1):
        super().__init__()
        self.sigma = sigma
        self.latent_dim = latent_dim
        self.cfm_model = MLP(dim=latent_dim, time_varying=True, w=64)
        self.fm = ExactOptimalTransportConditionalFlowMatcher(sigma=self.sigma)

        self.encoder_rna = rna_vae.encoder
        self.fc_mu_rna = rna_vae.fc_mu        
        self.decoder_rna = rna_vae.decoder

        self.encoder_atac = atac_vae.encoder
        self.fc_mu_atac = atac_vae.fc_mu       
        self.decoder_atac= atac_vae.mu_head #atac_vae.decoder

        for module in [self.encoder_rna, self.fc_mu_rna, self.decoder_rna,
                    self.encoder_atac, self.fc_mu_atac, self.decoder_atac]:
            self._freeze_module(module)

        self.device = device
        self.to(self.device)

    @staticmethod
    def _freeze_module(module: nn.Module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def _encode_rna(self, x):
        return self.fc_mu_rna(self.encoder_rna(x))   

    def _encode_atac(self, x):
        return self.fc_mu_atac(self.encoder_atac(x)) 

    def forward(self, x_rna, x_atac):
        with torch.no_grad():
            z_rna  = self._encode_rna(x_rna)
            z_atac = self._encode_atac(x_atac)

        t, zt, ut = self.fm.sample_location_and_conditional_flow(z_rna, z_atac)
        vt = self.cfm_model(torch.cat([zt, t[:, None]], dim=-1))
        return vt, ut


    def convert(self, x_rna):
        z_rna = self._encode_rna(x_rna)
        t_span = torch.linspace(0, 1, 100, device=x_rna.device)
        z_atac_hat = odeint(self.vector_field, z_rna, t_span)[-1]
        return self.decoder_atac(z_atac_hat)


    def training_step(self, x_source: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        vt, ut = self.forward(x_source, x_target)
        return torch.mean((vt - ut) ** 2)
    
    def vector_field(self, t, z):
        t_batch = t.expand(z.shape[0], 1)
        return self.cfm_model(torch.cat([z, t_batch], dim=-1))


    def train_one_epoch(self, train_loader, optimizer: torch.optim.Optimizer) -> float:
        self.cfm_model.train()

        for module in [self.encoder_rna, self.decoder_rna, self.encoder_atac, self.decoder_atac]:
            self._freeze_module(module)

        total_loss = 0.0
        for x_source, x_target in train_loader: 
            x_source = x_source.to(next(self.parameters()).device)
            x_target = x_target.to(next(self.parameters()).device)

            optimizer.zero_grad()
            loss = self.training_step(x_source, x_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss


    @torch.no_grad()
    def validate(self, valid_loader) -> float:

        self.eval()
        total_loss = 0.0

        for x_source, x_target in valid_loader:

            x_source = x_source.to(next(self.parameters()).device)
            x_target = x_target.to(next(self.parameters()).device)
            loss = self.training_step(x_source, x_target)
            total_loss += loss.item()

        return total_loss
    

    @torch.no_grad()
    def predict(self, x_rna, n_steps=100, return_trajectory=False):
        self.eval()

        x_rna = x_rna.to(self.device)

        z_rna = self._encode_rna(x_rna)

        t_span = torch.linspace(0, 1, n_steps, device=self.device)
        trajectory = odeint(self.vector_field, z_rna, t_span)
        z_atac_hat = trajectory[-1]
        x_atac_hat = self.decoder_atac(z_atac_hat)

        if return_trajectory:
            return x_atac_hat, trajectory
        
        return x_atac_hat
    

def train_modality_converter(
    model_params: dict, 
    train_loader: DataLoader, 
    valid_loader: DataLoader, 
    epochs: int, 
    patience: int = 50,
    save: bool = True,
    log_path: str = "/workspace/runs",
    restart_log: bool = True,
    ):
    
    if restart_log:
        start_log(log_path, "otcfm_training_run")
    log_section("LOADING MODEL")

    # adjust latent dimension as needed! 
    rna_model_raw = InfoVAE_RNA(
        input_size=model_params["rna_vae_input"], 
        latent_size=model_params["latent_dim"], 
        lr=0, wd=0, mode="", 
        device=model_params["device"],
        lambda_mmd=None,
        )
    
    atac_model_raw = InfoVAE_ATAC(
        input_size=model_params["atac_vae_input"], 
        latent_size=model_params["latent_dim"], 
        lr=0, wd=0, mode="", 
        device=model_params["device"],
        pos_weight=torch.ones(model_params["atac_vae_input"]).to(model_params["device"]))

    rna_model = load_model(rna_model_raw, model_params["rna_vae_path"])
    atac_model = load_model(atac_model_raw, model_params["atac_vae_path"])

    modality_converter = ModalityConverter(
        latent_dim=model_params["latent_dim"],
        rna_vae=rna_model,
        atac_vae=atac_model,
        device=model_params["device"]
        )
    
    modality_converter = torch.compile(modality_converter)
    log(str(modality_converter.parameters))

    ot_cfm_optimizer = torch.optim.Adam(modality_converter.parameters(), 1e-4)

    training_losses = []
    validation_losses = []
    min_loss = np.inf
    patience_counter = 0
    
    log_section("TRAINING START")
    for epoch in tqdm(
        range(0, epochs),
        initial=0,
        ncols=100,
        desc="\nEpochs",
        file=sys.stdout
        ):
    
        # First validate, then train, so that both use the same model state!
        validation_loss = modality_converter.validate(valid_loader)
        validation_losses.append(validation_loss)

        training_loss = modality_converter.train_one_epoch(train_loader, ot_cfm_optimizer)
        training_losses.append(training_loss)

        if validation_loss < min_loss:
            min_loss = validation_loss
            best_state = copy.deepcopy(modality_converter._orig_mod.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch+1)%10 == 0:
            log(f"Epoch [{epoch+1}/{epochs}]: Training Loss: {training_loss}, Validation Loss: {validation_loss}")
        
        if patience_counter >= patience:
            log(f"Early stopping triggered at epoch {epoch+1}!")
            break

    log_section("FINISHED TRAINING, SAVING MODEL")
    modality_converter._orig_mod.load_state_dict(best_state)

    if save:
        save_path = f"/workspace/runs/{datetime.now()}_CFM_model_weights.pth"
        torch.save(modality_converter.state_dict(), save_path)
        log(f"Model saved to {save_path}")

    return modality_converter, training_losses, validation_losses
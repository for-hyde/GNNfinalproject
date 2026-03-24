import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchcfm.models import MLP
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdiffeq import odeint

from utils.logging_utils import (start_log, log, log_section)
from utils.device import load_model
from models.vae_rna import InfoVAE_RNA
from models.vae_atac import InfoVAE_ATAC
import copy
from datetime import datetime
from tqdm import tqdm 
import sys
import math


class TimeEmbedding(nn.Module):
    """Maps a scalar t to a high-dimensional vector using sine/cosine."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t shape: (batch,)
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, dim, time_emb_dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.time_bias = nn.Linear(time_emb_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.SiLU() # SiLU (Swish) is standard for flows

    def forward(self, x, t_emb):
        # Inject time via addition
        out = self.linear(x) + self.time_bias(t_emb)
        out = self.norm(out)
        out = self.activation(out)
        return x + out # Residual connection


class ExpressiveCFM(nn.Module):
    def __init__(self, dim, w=512, time_emb_dim=64, num_layers=4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        self.input_proj = nn.Linear(dim, w)
        
        self.layers = nn.ModuleList([
            ResidualBlock(w, time_emb_dim) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(w, dim)

    def forward(self, x_t_concat):
        # torchcfm sends (batch, dim + 1)
        x = x_t_concat[:, :-1]
        t = x_t_concat[:, -1]
        
        t_emb = self.time_mlp(t)
        h = self.input_proj(x)
        
        for layer in self.layers:
            h = layer(h, t_emb)
            
        return self.output_proj(h)



class ModalityConverter(nn.Module):
    def __init__(self, latent_dim, rna_vae, atac_vae, device, sigma=0.01, proj_dim=128):
        super().__init__()
        self.sigma = sigma
        self.latent_dim = latent_dim
        self.cfm_model = ExpressiveCFM(dim=latent_dim, w=1024, num_layers=8)
        self.fm = ExactOptimalTransportConditionalFlowMatcher(sigma=self.sigma)
        #self.fm = ConditionalFlowMatcher(sigma=self.sigma)

        self.encoder_rna = rna_vae.encoder
        self.fc_mu_rna = rna_vae.fc_mu        
        self.decoder_rna = rna_vae.decoder

        self.encoder_atac = atac_vae.encoder
        self.fc_mu_atac = atac_vae.fc_mu

        self.decoder_atac= atac_vae.decoder #atac_vae.decoder
        self.decoder_mu_head_atac= atac_vae.mu_head #atac_vae.decoder

        for module in [self.encoder_rna, self.fc_mu_rna, self.decoder_rna,
                    self.encoder_atac, self.fc_mu_atac, self.decoder_atac, self.decoder_mu_head_atac]:
            self._freeze_module(module)
        
        self.proj_rna  = nn.Sequential(nn.Linear(latent_dim, proj_dim), nn.LayerNorm(proj_dim))
        self.proj_atac = nn.Sequential(nn.Linear(latent_dim, proj_dim), nn.LayerNorm(proj_dim))
        
        self.register_buffer("rna_mean", torch.zeros(latent_dim))
        self.register_buffer("rna_std",  torch.ones(latent_dim))
        self.register_buffer("atac_mean", torch.zeros(latent_dim))
        self.register_buffer("atac_std",  torch.ones(latent_dim))

        self.device = device
        self.to(self.device)
    

    def alignment_loss(self, z_rna, z_atac, temperature=0.1):
        # Project and L2-normalize
        p_rna  = F.normalize(self.proj_rna(z_rna),  dim=-1)  # (B, proj_dim)
        p_atac = F.normalize(self.proj_atac(z_atac), dim=-1)  # (B, proj_dim)

        # Similarity matrix
        logits = (p_rna @ p_atac.T) / temperature  # (B, B)

        # Diagonal = matched pairs = positive examples
        labels = torch.arange(len(logits), device=logits.device)
        
        # Symmetric cross-entropy (same as CLIP)
        loss = (F.cross_entropy(logits, labels) + 
                F.cross_entropy(logits.T, labels)) / 2
        return loss
    
    
    def compute_latent_norm(self, train_loader):
        z_rna_all, z_atac_all = [], []
        with torch.no_grad():
            for x_rna, x_atac in train_loader:
                z_rna = self.fc_mu_rna(self.encoder_rna(x_rna.to(self.device)))
                z_atac = self.fc_mu_atac(self.encoder_atac(x_atac.to(self.device)))
                z_rna_all.append(z_rna)
                z_atac_all.append(z_atac)

        z_rna_all = torch.cat(z_rna_all)
        z_atac_all = torch.cat(z_atac_all)

        self.register_buffer("rna_mean", z_rna_all.mean(0))
        self.register_buffer("rna_std",  z_rna_all.std(0) + 1e-6)
        self.register_buffer("atac_mean", z_atac_all.mean(0))
        self.register_buffer("atac_std",  z_atac_all.std(0) + 1e-6)

    @staticmethod
    def _freeze_module(module: nn.Module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    # def _encode_rna(self, x):
    #     return self.fc_mu_rna(self.encoder_rna(x)) 
    
    def _encode_rna(self, x):
        z = self.fc_mu_rna(self.encoder_rna(x))
        return (z - self.rna_mean) / self.rna_std

    # def _encode_atac(self, x):
    #     return self.fc_mu_atac(self.encoder_atac(x)) 

    def _encode_atac(self, x):
        z = self.fc_mu_atac(self.encoder_atac(x))
        return (z - self.atac_mean) / self.atac_std

    def forward(self, x_rna, x_atac):
        with torch.no_grad():
            z_rna  = self._encode_rna(x_rna)
            z_atac = self._encode_atac(x_atac)

        t, zt, ut = self.fm.sample_location_and_conditional_flow(z_rna, z_atac)
        vt = self.cfm_model(torch.cat([zt, t[:, None]], dim=-1))
        return vt, ut


    # def convert(self, x_rna):
    #     z_rna = self._encode_rna(x_rna)
    #     t_span = torch.linspace(0, 1, 100, device=x_rna.device)
    #     z_atac_hat = odeint(self.vector_field, z_rna, t_span)[-1]
    #     h = self.decoder_atac(z_atac_hat)
    #     return self.decoder_mu_head_atac(h)

    def convert(self, x_rna, n_steps=100):
        self.cfm_model.eval()
        with torch.no_grad():
            z_rna = self._encode_rna(x_rna)
            t_span = torch.linspace(0, 1, n_steps, device=x_rna.device)
            
            # Predict the trajectory
            trajectory = odeint(
                self.vector_field, 
                z_rna, 
                t_span,
                method="dopri5",          
                rtol=1e-4, 
                atol=1e-5
            )
            z_atac_hat = trajectory[-1] # This is NORMALIZED
            
            z_atac_unnorm = (z_atac_hat * self.atac_std) + self.atac_mean
            
            h = self.decoder_atac(z_atac_unnorm)
            return self.decoder_mu_head_atac(h)


    # def training_step(self, x_source: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
    #     vt, ut = self.forward(x_source, x_target)
    #     return torch.mean((vt - ut) ** 2)

    # NT-Xent / InfoNCE alginment loss included! 
    def training_step(self, x_rna: torch.Tensor, x_atac: torch.Tensor, alpha=0) -> torch.Tensor:
        with torch.no_grad():
            z_rna  = self._encode_rna(x_rna)
            z_atac = self._encode_atac(x_atac)

        t, zt, ut = self.fm.sample_location_and_conditional_flow(z_rna, z_atac)
        vt = self.cfm_model(torch.cat([zt, t[:, None]], dim=-1))
        loss_cfm = torch.mean((vt - ut) ** 2)

        #loss_align = self.alignment_loss(z_rna, z_atac)

        return loss_cfm #+ alpha * loss_align

    
    def vector_field(self, t, z):
        t_batch = torch.full((z.shape[0], 1), t.item(), device=z.device)
        return self.cfm_model(torch.cat([z, t_batch], dim=-1))


    def train_one_epoch(self, train_loader, optimizer: torch.optim.Optimizer, scheduler) -> float:
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
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)


    @torch.no_grad()
    def validate(self, valid_loader) -> float:

        self.cfm_model.eval()
        total_loss = 0.0

        for x_source, x_target in valid_loader:

            x_source = x_source.to(next(self.parameters()).device)
            x_target = x_target.to(next(self.parameters()).device)
            loss = self.training_step(x_source, x_target)
            total_loss += loss.item()
            
        self.cfm_model.train()
        return total_loss / len(valid_loader)
    

    @torch.no_grad()
    def predict(self, x_rna, n_steps=100, return_trajectory=False):
        self.cfm_model.eval()

        x_rna = x_rna.to(self.device)

        z_rna = self._encode_rna(x_rna)

        t_span = torch.linspace(0, 1, n_steps, device=self.device)
        #trajectory = odeint(self.vector_field, z_rna, t_span)
        trajectory = odeint(
            self.vector_field, 
            z_rna, 
            t_span,
            method="dopri5",          # adaptive — much more accurate for large gaps
            rtol=1e-4, 
            atol=1e-5,
        )
        z_atac_hat = trajectory[-1]

        z_atac_unnorm = (z_atac_hat * self.atac_std) + self.atac_mean
        
        h_atac_hat = self.decoder_atac(z_atac_unnorm)
        x_atac_hat = self.decoder_mu_head_atac(h_atac_hat)

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
    
    modality_converter.compute_latent_norm(train_loader)

    modality_converter = torch.compile(modality_converter)
    log(str(modality_converter.parameters))

    ot_cfm_optimizer = torch.optim.AdamW(modality_converter.parameters(), lr=1e-3, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        ot_cfm_optimizer, 
        max_lr=1e-3, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs
    )

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
    
        training_loss = modality_converter.train_one_epoch(train_loader, ot_cfm_optimizer, scheduler)
        training_losses.append(training_loss)

        validation_loss = modality_converter.validate(valid_loader)
        validation_losses.append(validation_loss)

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
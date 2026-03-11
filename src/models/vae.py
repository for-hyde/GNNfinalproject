import torch
import numpy as np
import sys
from torch import nn
from torch.amp.grad_scaler import GradScaler
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from datetime import datetime


from utils.logging_utils import (start_log, log, log_section)


# def get_hidden_dim_steps(input, output, steps):
#     return list(np.linspace(output, input, num=steps))


# def init_encoder(layers, input_dim, latent_dim, dropout_prob=None):
#     encoder_layers = []

#     layer_dims = get_hidden_dim_steps(input_dim, latent_dim, layers+1)

#     for i in range(1, layers+1):

#         encoder_layers.append(nn.Linear(int(layer_dims[-i]), int(layer_dims[-i-1])))
#         encoder_layers.append(nn.ReLU(inplace=True))
#         if dropout_prob:
#             encoder_layers.append(nn.Dropout(dropout_prob))
    
#     encoder = nn.Sequential(*encoder_layers)

#     return encoder 


# def init_decoder(layers, input_dim, latent_dim, dropout_prob=None):
#     decoder_layers = []

#     layer_dims = get_hidden_dim_steps(input_dim, latent_dim, layers+1)

#     for i in range(layers-1):

#         decoder_layers.append(nn.Linear(int(layer_dims[i]), int(layer_dims[i+1])))
#         decoder_layers.append(nn.ReLU(inplace=True))
#         if dropout_prob:
#             decoder_layers.append(nn.Dropout(dropout_prob))
    
#     decoder_layers.append(nn.Linear(int(layer_dims[-2]), int(layer_dims[-1])))

#     decoder = nn.Sequential(*decoder_layers)

#     return decoder 


def init_encoder(input_dim, latent_dim, hidden_dims=[1024, 512, 256]):
    layers = []
    curr_dim = input_dim
    
    for h_dim in hidden_dims:
        layers.append(nn.Linear(curr_dim, h_dim))
        layers.append(nn.BatchNorm1d(h_dim))
        layers.append(nn.ReLU())
        curr_dim = h_dim
        
    encoder = nn.Sequential(*layers)

    return encoder, curr_dim 


def init_decoder(latent_dim, input_dim, hidden_dims=[256, 512, 1024]):
    layers = []
    curr_dim = latent_dim
    
    for h_dim in hidden_dims:
        layers.append(nn.Linear(curr_dim, h_dim))
        layers.append(nn.BatchNorm1d(h_dim))
        layers.append(nn.ReLU())
        curr_dim = h_dim
        
    # Final layer to map back to gene expression
    layers.append(nn.Linear(curr_dim, input_dim))
    
    return nn.Sequential(*layers)


class InfoVAE(nn.Module):
    def __init__(
            self, 
            input_size: int, 
            latent_size: int, 
            lr: float,
            wd: float,
            mode: str,  # either "rna" or "atac"
            device,
            lambda_mmd: float = 0.1,
            ):
        
        super(InfoVAE, self).__init__()
        
        self.input_size = int(input_size)
        self.latent_size = int(latent_size)
        self.lambda_mmd = lambda_mmd
        self.mode = mode

        self.encoder, final_enc_layer_size = init_encoder(self.input_size, self.latent_size)
        self.decoder = init_decoder(self.latent_size, self.input_size)

        self.fc_mu = nn.Linear(final_enc_layer_size, self.latent_size)
        self.fc_logvar = nn.Linear(final_enc_layer_size, self.latent_size)

        self.apply(self.weights_init)
        self.device = device
        self.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=wd,
            )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
        )

        
    def weights_init(self, param):
        if isinstance(param, nn.Linear):
            nn.init.kaiming_uniform_(param.weight, nonlinearity="relu")
            nn.init.constant_(param.bias, 0)


    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor):
        logvar = logvar.clamp(-30, 20)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar
    

    def decode(self, z: torch.Tensor):
        return self.decoder(z)


    def forward(self, x: torch.Tensor):
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z, mu, logvar

    def compute_mmd(self, z: torch.Tensor):

        z = z.to(torch.float32) 
        prior_z = torch.randn_like(z)

        def rbf_kernel(x1, x2):
            sigma = 2.0 * x1.size(1)
            # torch.cdist is highly optimized and numerically stable
            dist = torch.cdist(x1, x2, p=2.0).pow(2)
            return torch.exp(-dist / sigma)

        z_kernel = rbf_kernel(z, z)
        prior_kernel = rbf_kernel(prior_z, prior_z)
        cross_kernel = rbf_kernel(z, prior_z)

        return z_kernel.mean() + prior_kernel.mean() - 2 * cross_kernel.mean()

    def loss_function(self, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        mmd_loss = self.compute_mmd(z)
        
        return recon_loss + (self.lambda_mmd * mmd_loss)


    def train_one_epoch(self, train_loader):
        self.train()
        total_loss = 0

        for batch_idx, batch_data in enumerate(train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                batch_data = batch_data.to(self.device, non_blocking=True)
                
                x_recon, z, mu, logvar = self.forward(batch_data)
                loss = self.loss_function(batch_data, x_recon, z, mu, logvar)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        
        total_loss /= len(train_loader)
        
        return total_loss


    def validate(self, valid_loader):
        self.eval()

        with torch.no_grad():
            total_loss = 0

            for batch_idx, batch_data in enumerate(valid_loader):
                
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    batch_data = batch_data.to(self.device, non_blocking=True)
                    
                    x_recon, z, mu, logvar = self.forward(batch_data)
                    loss = self.loss_function(batch_data, x_recon, z, mu, logvar)
                
                total_loss += loss.item()
        
        total_loss /= len(valid_loader)
        
        return total_loss



def train_infoVAE(
        model_params: dict, 
        train_loader: DataLoader, 
        valid_loader: DataLoader, 
        epochs: int,
        patience: int = 50,
        save: bool = True,
        log_path: str = "/workspace/logs",
        restart_log: bool = True,
        ):
    
    if restart_log:
        start_log(log_path, "infoVAE_training_run")
    log_section("LOADING MODEL")

    model = InfoVAE(
        input_size=model_params["input_size"],
        latent_size=model_params["latent_size"],
        lr=model_params["lr"],
        wd=model_params["wd"],
        device=model_params["device"],
        mode=model_params["mode"],
        lambda_mmd=model_params["lambda_mmd"]
    )
    model = torch.compile(model)
    log(str(model.parameters))

    training_losses = []
    validation_losses = []
    min_loss = np.inf
    patience_counter = 0
    
    log_section("TRAINING START")
    for epoch in tqdm(range(0, epochs), initial=0, ncols=100, desc="\nEpochs", file=sys.stdout):
        
        # 1. Train first
        training_loss = model.train_one_epoch(train_loader)
        training_loss_avg = training_loss #/ len(train_loader) already divided by N in loss function!
        training_losses.append(training_loss_avg)

        # 2. Then validate
        validation_loss = model.validate(valid_loader)
        validation_loss_avg = validation_loss #/ len(valid_loader)
        validation_losses.append(validation_loss_avg)
        
        # 3. Step scheduler based on the actual trained epoch
        model.scheduler.step(validation_loss_avg)

        if validation_loss_avg < min_loss:
            min_loss = validation_loss_avg
            best_state = copy.deepcopy(model._orig_mod.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        log(f"Epoch [{epoch+1}/{epochs}]: Train: {training_loss_avg:.4f}, Val: {validation_loss_avg:.4f}")
        
        if patience_counter >= patience:
            log(f"Early stopping triggered at epoch {epoch+1}!")
            break
        
    log_section("FINISHED TRAINING, SAVING MODEL")
    model._orig_mod.load_state_dict(best_state)
    
    if save:
        save_path = f"/workspace/models/{datetime.now()}_vae_model_weights.pth"
        torch.save(model.state_dict(), save_path)
        log(f"Model saved to {save_path}")

    return model, training_losses, validation_losses





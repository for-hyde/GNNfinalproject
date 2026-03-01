import torch
import numpy as np
import sys
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from datetime import datetime


from utils.logging_utils import (start_log, log, log_section)


def get_hidden_dim_steps(input, output, steps):
    return list(np.linspace(output, input, num=steps))


def init_encoder(layers, input_dim, latent_dim, dropout_prob=None):
    encoder_layers = []

    layer_dims = get_hidden_dim_steps(input_dim, latent_dim, layers+1)

    for i in range(1, layers+1):

        encoder_layers.append(nn.Linear(int(layer_dims[-i]), int(layer_dims[-i-1])))
        encoder_layers.append(nn.ReLU(inplace=True))
        if dropout_prob:
            encoder_layers.append(nn.Dropout(dropout_prob))
    
    encoder = nn.Sequential(*encoder_layers)

    return encoder 


def init_decoder(layers, input_dim, latent_dim, dropout_prob=None):
    decoder_layers = []

    layer_dims = get_hidden_dim_steps(input_dim, latent_dim, layers+1)

    for i in range(layers):

        decoder_layers.append(nn.Linear(int(layer_dims[i]), int(layer_dims[i+1])))
        decoder_layers.append(nn.ReLU(inplace=True))
        if dropout_prob:
            decoder_layers.append(nn.Dropout(dropout_prob))
    
    decoder = nn.Sequential(*decoder_layers)

    return decoder 


class InfoVAE(nn.Module):
    def __init__(
            self, 
            input_size: int, 
            latent_size: int, 
            layers: int,
            lr: float,
            wd: float,
            device,
            lambda_mmd: float = 100.0,
            ):
        
        super(InfoVAE, self).__init__()
        
        self.input_size = int(input_size)
        self.latent_size = int(latent_size)
        self.layers = int(layers)
        self.lambda_mmd = lambda_mmd

        self.encoder = init_encoder(self.layers, self.input_size, self.latent_size)
        self.decoder = init_decoder(self.layers, self.input_size, self.latent_size)

        self.fc_mu = nn.Linear(self.latent_size, self.latent_size)
        self.fc_logvar = nn.Linear(self.latent_size, self.latent_size)

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

        self.apply(self.weights_init)
        self.device = device
        self.to(self.device)
    

    def weights_init(self, param):
        if isinstance(param, nn.Linear):
            nn.init.xavier_uniform_(param.weight)
            nn.init.constant_(param.bias, 0)


    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor):
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
        prior_z = torch.randn_like(z)
        
        def rbf_kernel(x1, x2):

            z_dim = x1.size(1)
            sigma = 2.0 * z_dim
            diff = x1.unsqueeze(1) - x2.unsqueeze(0)
            dist = torch.pow(diff, 2).sum(2)
           
            return torch.exp(-dist / sigma)

        z_kernel = rbf_kernel(z, z)
        prior_kernel = rbf_kernel(prior_z, prior_z)
        cross_kernel = rbf_kernel(z, prior_z)

        mmd = z_kernel.mean() + prior_kernel.mean() - 2 * cross_kernel.mean()
        return mmd


    def loss_function(self, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor, mu: torch.Tensor, logvar:torch.Tensor):
        
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        mmd_loss = self.compute_mmd(z)
        
        # InfoVAE objective: Recon + KL + lambda*MMD
        return recon_loss + kl_loss + (self.lambda_mmd * mmd_loss)


    def train_one_epoch(self, train_loader):
        self.train()
        total_loss = 0

        for batch_idx, (batch_data,) in enumerate(train_loader):
            self.optimizer.zero_grad()
            batch_data = batch_data.to(self.device)
            
            x_recon, z, mu, logvar = self.forward(batch_data)
            loss = self.loss_function(batch_data, x_recon, z, mu, logvar)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        if self.scheduler:
            self.scheduler.step(total_loss)
        
        return total_loss


    def validate(self, valid_loader):
        self.eval()

        with torch.no_grad():
            total_loss = 0

            for batch_idx, (batch_data,) in enumerate(valid_loader):
                self.optimizer.zero_grad()
                batch_data = batch_data.to(self.device)
                
                x_recon, z, mu, logvar = self.forward(batch_data)
                loss = self.loss_function(batch_data, x_recon, z, mu, logvar)
                
                total_loss += loss.item()
        
        return total_loss



def train_infoVAE(
        model_params: dict, 
        train_loader: DataLoader, 
        valid_loader: DataLoader, 
        epochs: int, 
        ):
    
    start_log("/workspace/logs", "infoVAE_training_run")
    log_section("LOADING MODEL")

    model = InfoVAE(
        input_size=model_params["input_size"],
        latent_size=model_params["latent_size"],
        layers=model_params["layers"],
        lr=model_params["lr"],
        wd=model_params["wd"],
        device=model_params["device"],
    )
    log(str(model.parameters))

    training_losses = []
    validation_losses = []
    min_loss = np.inf
    trained_model = copy.deepcopy(model)
    
    log_section("TRAINING START")
    for epoch in tqdm(
        range(0, epochs),
        initial=0,
        ncols=100,
        desc="\nEpochs",
        file=sys.stdout
        ):
    
        # First validate, then train, so that both use the same model state!
        validation_loss = model.validate(valid_loader)
        validation_losses.append(validation_loss)

        training_loss = model.train_one_epoch(train_loader)
        training_losses.append(training_loss)

        if validation_loss < min_loss:
            trained_model = copy.deepcopy(model)

        if (epoch+1)%100 == 0:
            log(f"Epoch [{epoch+1}/{epochs}]: Training Loss: {training_loss}, Validation Loss: {validation_loss}")
        
    log_section("FINISHED TRAINING, SAVING MODEL")
    save_path = f"/workspace/models/{datetime.now()}_vae_model_weights.pth"
    torch.save(trained_model.state_dict(), save_path)
    log(f"Model saved to {save_path}")

    return trained_model, training_losses, validation_losses





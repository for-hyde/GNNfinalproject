from torchcfm.models import MLP
from torchcfm.utils import plot_trajectories, torch_wrapper

import torch
from torch import nn
import numpy as np


# From the single cell example
def get_batch(FM, X, batch_size, n_times, device, return_noise=False):
    """
    Construct a batch with points from each timepoint pair
    """
    ts = []
    xts = []
    uts = []
    noises = []
    for t_start in range(n_times - 1):
        x0 = (
            torch.from_numpy(X[t_start][np.random.randint(X[t_start].shape[0], size=batch_size)])
            .float()
            .to(device)
        )
        x1 = (
            torch.from_numpy(
                X[t_start + 1][np.random.randint(X[t_start + 1].shape[0], size=batch_size)]
            )
            .float()
            .to(device)
        )
        if return_noise:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(
                x0, x1, return_noise=return_noise
            )
            noises.append(eps)
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)
    
    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)
    
    if return_noise:
        noises = torch.cat(noises)
        return t, xt, ut, noises
    
    return t, xt, ut


class ModalityConverter(nn.Module):
    def __init__(
            self, 
            latent_dim: int,
            encoder_path: str, 
            decoder_path: str,
            ):
        
        self.ot_cfm_model = MLP(dim=latent_dim)
        # Get the encoder and decoder from the corresponding trained models
        self.encoder = torch.load(encoder_path, weights_only=False).encoder()
        self.decoder = torch.load(decoder_path, weights_only=False).decoder()

        return
    
    def forward(self, x):
        return




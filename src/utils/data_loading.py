import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import anndata as ad 
import numpy as np
import random
from torch.utils.data import DataLoader
import os 


def load_data(rna, atac, multiome=False):
    
    #rna = ad.read_h5ad(os.path.join('/workspace/data', rna))
    #atac = ad.read_h5ad(os.path.join('/workspace/data', atac))
    
    rna = ad.read_h5ad(os.path.join('/workspace/data', rna))
    atac = ad.read_h5ad(os.path.join('/workspace/data', atac))

    common_cells = rna.obs_names.intersection(atac.obs_names)
        
    rna = rna[common_cells].copy()
    atac = atac[common_cells].copy()
    atac = atac[sorted(atac.obs_names), :]
    atac = atac[ :, sorted(atac.var_names),]

    rna = rna[sorted(rna.obs_names), :]
    rna = rna[ :, sorted(rna.var_names),]

    if multiome==False: 
        return rna, atac 
    
    else: 
        multiome_dataset = ad.concat([rna, atac], axis=1, merge='same')
        
        return multiome_dataset

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, val_ratio=0.2, seed=19193):
    
    if seed is not None:
        setup_seed(seed)
        
    train_indices = []
    val_indices = []

    for ct in dataset.obs['cell_type'].unique():
        
        ct_idx = np.where(dataset.obs['cell_type'] == ct)[0]
        np.random.shuffle(ct_idx)

        n_val = int(len(ct_idx) * val_ratio)

        val_indices.extend(ct_idx[:n_val])
        train_indices.extend(ct_idx[n_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    return  train_indices, val_indices


class MultiomeDatasetVAE(Dataset):
    def __init__(self, rna, atac, indices):
        self.atac = atac 
        self.rna = rna 
        self.indices = indices
    
        assert all(rna.obs_names == atac.obs_names)

        rna = rna[indices]
        atac = atac[indices]
        
        X_rna = rna.X
        if sp.issparse(X_rna):
            X_rna = X_rna.toarray()

        X_atac = atac.X
        if sp.issparse(X_atac):
            X_atac = X_atac.toarray()

        self.X_rna = torch.tensor(X_rna, dtype=torch.float32)
        self.X_atac = torch.tensor(X_atac, dtype=torch.float32)

    def __len__(self):
        return self.X_rna.shape[0]

    def __getitem__(self, idx):
        return self.X_rna[idx], self.X_atac[idx]


class SingleDatasetVAE(Dataset):
    def __init__(self, data, indices):
        subset = data[indices]
        X = subset.X
        if sp.issparse(X):
            X = X.toarray()
        self.X = torch.tensor(X, dtype=torch.float32)  # dense tensor, in memory once

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]  # now just a fast tensor index
    
    
class MultiomeDatasetCMF(Dataset):
    def __init__(self, multiome, indices):
        self.multiome = multiome
        self.indices = indices
                
        X_multiome = multiome.X
        if sp.issparse(X_multiome):
            X_multiome = X_multiome.toarray()

        self.X_multiome = torch.tensor(X_multiome, dtype=torch.float32)

    def __len__(self):
        return self.X_multiome.shape[0]

    def __getitem__(self, idx):
        return self.X_multiome[idx]



def main():
    
    rna, atac = load_data('bmmc_atac_highly_variable.h5ad', 'bmmc_rna_highly_variable.h5ad', multiome=False)

    train_idx, val_idx = split_dataset(rna)

    train_dataset = MultiomeDatasetVAE(rna, atac, train_idx)
    val_dataset   = MultiomeDatasetVAE(rna, atac, val_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    
    # multiome = load_data('bmmc_atac_highly_variable.h5ad', 'bmmc_rna_highly_variable.h5ad', multiome=True)
    # train_idx, val_idx = split_dataset(multiome)

    # train_dataset = MultiomeDatasetCMF(multiome, train_idx)
    # val_dataset   = MultiomeDatasetCMF(multiome, val_idx)

    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

if __name__ == '__main__':
    main()
    

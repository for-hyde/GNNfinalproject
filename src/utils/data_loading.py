import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import anndata as ad 
import numpy as np
import random
from torch.utils.data import DataLoader
import os 
from collections import OrderedDict


####################################################################################################
#                                                                                                  #
# DATA LOADING                                                                                     #
# Collection of functions that help with data loading and processing prior to model training       #
# Some functions are deprecated and not used anymore                                               #
#                                                                                                  #
####################################################################################################


def threshold_to_match_sparsity(recon_probs, original, tolerance=0.005):
    target_density = original.mean()  
    
    lo, hi = 0.0, 1.0
    for _ in range(50):  
        mid = (lo + hi) / 2
        pred_density = (recon_probs > mid).mean()
        if abs(pred_density - target_density) < tolerance:
            break
        if pred_density > target_density:
            lo = mid
        else:
            hi = mid
    
    print(f"Threshold: {mid:.4f}, Pred density: {pred_density:.4f}, Target: {target_density:.4f}")
    return (recon_probs > mid).astype(np.float32), mid


def get_gene_weight(data):
    hvg_mask = data.var["highly_variable"]
    adata_hvg = data[:, hvg_mask]

    raw_weights = adata_hvg.var["dispersions_norm"].values.astype(np.float32)

    gene_weights = raw_weights / (raw_weights.sum() + 1e-8) * adata_hvg.n_vars
    gene_weights = torch.from_numpy(gene_weights)
    return gene_weights

def get_gene_weight_alt(data):
    X = data.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    gene_var = X.var(axis=0).astype(np.float32)
    w = gene_var / (gene_var.sum() + 1e-8) * data.n_vars
    return torch.from_numpy(w)


def get_atac_pos_weights(train_data, epsilon=1e-6, max_weight=20.0):
    # train_data should be your raw binary numpy array or tensor (N, Peaks)
    if torch.is_tensor(train_data):
        train_data = train_data.cpu().numpy()
        
    n_cells = train_data.shape[0]
    peak_sums = train_data.sum(axis=0) # How many cells have this peak open?
    
    # Ratio of closed cells to open cells for each peak
    # If a peak is only open in 1% of cells, weight it ~100x more
    pos_weights = (n_cells - peak_sums) / (peak_sums + epsilon)
    pos_weights = np.clip(pos_weights, 1.0, max_weight)
    
    return torch.tensor(pos_weights, dtype=torch.float32)


def separate_loader(data_dir, modality):

    if modality=='RNA':
        train_rna = ad.read_h5ad(os.path.join(data_dir, 'train_rna.h5ad'))
        val_rna = ad.read_h5ad(os.path.join(data_dir, 'val_rna.h5ad'))
        test_rna = ad.read_h5ad(os.path.join(data_dir, 'test_rna.h5ad'))
        return train_rna, val_rna, test_rna

    elif modality=='ATAC':
        train_atac = ad.read_h5ad(os.path.join(data_dir,'train_atac.h5ad'))
        val_atac = ad.read_h5ad(os.path.join(data_dir, 'val_atac.h5ad'))
        test_atac = ad.read_h5ad(os.path.join(data_dir, 'test_atac.h5ad'))
        return train_atac, val_atac, test_atac

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

def uniform_split_dataset(dataset, test_ratio, val_ratio=0.2, seed=19193):
    '''Set test_ratio to None if you don't wish to have a testing dataset, otherwise set it to any float like 0.1 and enjoy a fresh testing'''
    
    if seed is not None:
        setup_seed(seed)
    
    if test_ratio == None: 
        
        train_indices = []
        val_indices = []
        test_indices = []

        for ct in dataset.obs['cell_type'].unique():
            
            ct_idx = np.where(dataset.obs['cell_type'] == ct)[0]
            np.random.shuffle(ct_idx)

            n_val = int(len(ct_idx) * val_ratio)

            val_indices.extend(ct_idx[:n_val])
            train_indices.extend(ct_idx[n_val:])

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)

        return  train_indices, val_indices, test_indices

    if test_ratio != None: 
        
        train_indices = []
        val_indices = []
        test_indices = []

        for ct in dataset.obs['cell_type'].unique():
            
            ct_idx = np.where(dataset.obs['cell_type'] == ct)[0]
            np.random.shuffle(ct_idx)

            n_val = int(len(ct_idx) * val_ratio)
            n_test = int(len(ct_idx) * test_ratio)

            val_indices.extend(ct_idx[:n_val])
            test_indices.extend(ct_idx[n_val:n_val+n_test])
            train_indices.extend(ct_idx[n_val+n_test:])

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)

        return  train_indices, val_indices, test_indices
    
def cell_type_split_dataset(dataset, annot, cell_col, cluster_col, test_ratio, val_ratio=0.2, seed=19193):
    ''' This function lets you choose if you want to split the datasets by cell type (annot==True) or by whatever clusters is available (annot==False). 
    cell_col should be a string - how the cell types column is names in your data. Sadly, you'll have to check that manually. 
    For BMMC it's just 'cell_type' tho.  
    cluster_col is the same. For BMMC it's 'leiden' '''
    
    import math 
    
    if not seed is None:
        setup_seed(seed)
    
    if annot == True: #run on the cell type annotations 
        cells = list(dataset.obs_names)
        cell_types = dataset.obs[cell_col].unique().to_list()
        random.shuffle(cell_types)
        type_to_id = {ct: i for i, ct in enumerate(cell_types)}
        id_to_type = {i: ct for ct, i in type_to_id.items()}
        cell_types_num = np.array([type_to_id[ct] for ct in cell_types])
        
        val_count = math.floor(len(cell_types) * val_ratio)
        test_count = math.floor(len(cell_types) * test_ratio) 
        train_count = len(cell_types) - val_count - test_count 
        
        test_cells = list(cell_types_num[: test_count])
        val_cells = cell_types_num[test_count: test_count + val_count]
        train_cells = cell_types_num[test_count + val_count:]

        test_types = {id_to_type[i] for i in test_cells}
        mask = dataset.obs[cell_col].isin(test_types)
        test_indices = dataset[mask]
        
        val_types = {id_to_type[i] for i in val_cells}
        mask = dataset.obs[cell_col].isin(val_types)
        val_indices = dataset[mask]
        
        mask = ~dataset.obs[cell_col].isin(test_types | val_types) 
        train_indices = dataset[mask]
    
        return train_indices.obs.index, val_indices.obs.index, test_indices.obs.index
        
    # if annot == False: #divide by Leiden clusters - throw an error if there are no clusters & run some 
    #     cells = list(dataset.obs_names)
    #     clusters = dataset.obs[cluster_col].unique().to_list()
        
    #     return train_indices, val_indices, test_indices
        
    
class MultiomeDataset(Dataset):
    def __init__(self, rna, atac):
        self.atac = atac 
        self.rna = rna 
        #self.indices = indices
    
        assert all(rna.obs_names == atac.obs_names)

        #rna = rna[indices]
        #atac = atac[indices]
        
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
    def __init__(self, data):
        X = data.X
        if sp.issparse(X):
            X = X.toarray()
        self.X = torch.tensor(X, dtype=torch.float32)  # dense tensor, in memory once

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]  # now just a fast tensor index

#class SingleDatasetVAE(Dataset):
#    def __init__(self, data, indices):
#        subset = data[indices]
#        X = subset.X
#        if sp.issparse(X):
#            X = X.toarray()
#        self.X = torch.tensor(X, dtype=torch.float32)  # dense tensor, in memory once
#
#    def __len__(self):
#        return self.X.shape[0]
#
#    def __getitem__(self, idx):
#        return self.X[idx]  # now just a fast tensor index
    
    
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


def threshold_to_match_sparsity_cfm(probs: np.ndarray, original: np.ndarray,
                                tolerance: float = 0.005) -> tuple[np.ndarray, float]:
    """Binary-search threshold so predicted density ≈ original density."""
    target = original.mean()
    lo, hi = 0.0, 1.0
    mid = 0.5
    for _ in range(50):
        mid = (lo + hi) / 2
        if abs((probs > mid).mean() - target) < tolerance:
            break
        if (probs > mid).mean() > target:
            lo = mid
        else:
            hi = mid
    print(f"  threshold={mid:.4f}  pred_density={(probs>mid).mean():.4f}  "
          f"target_density={target:.4f}")
    return (probs > mid).astype(np.float32), mid


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """Unbiased MMD² estimate with RBF kernel. Small = distributions are close."""
    from sklearn.metrics.pairwise import rbf_kernel
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    n, m = len(X), len(Y)
    np.fill_diagonal(XX, 0); np.fill_diagonal(YY, 0)
    return XX.sum() / (n*(n-1)) + YY.sum() / (m*(m-1)) - 2*XY.mean()


def per_dim_kl_gaussian(mu1, sigma1, mu2, sigma2):
    """KL(N(mu1,σ1²) ‖ N(mu2,σ2²)) per dimension (scalar arrays of length D)."""
    return (np.log(sigma2/sigma1)
            + (sigma1**2 + (mu1-mu2)**2) / (2*sigma2**2)
            - 0.5)


def encode_batched(model_encode_fn, tensor: torch.Tensor,
                   batch_size: int = 512, device=None) -> np.ndarray:
    """Run encoder over a large tensor in mini-batches, return mu as numpy."""
    mus = []
    for i in range(0, len(tensor), batch_size):
        b = tensor[i:i+batch_size].to(device)
        with torch.no_grad():
            _, mu, _ = model_encode_fn(b)
        mus.append(mu.cpu().numpy())
    return np.concatenate(mus)


def load_state(model, path, device):
    state_dict = torch.load(path, map_location=device, weights_only=False)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print("Model weights loaded successfully!")
    model.eval()
    return model
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import anndata as ad 
import numpy as np
import random
from torch.utils.data import DataLoader
import os 


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

    train_rna = ad.read_h5ad(os.path.join(data_dir, 'train_rna.h5ad'))
    val_rna = ad.read_h5ad(os.path.join(data_dir, 'val_rna.h5ad'))
    test_rna = ad.read_h5ad(os.path.join(data_dir, 'test_rna.h5ad'))

    train_atac = ad.read_h5ad(os.path.join(data_dir,'train_atac.h5ad'))
    val_atac = ad.read_h5ad(os.path.join(data_dir, 'val_atac.h5ad'))
    test_atac = ad.read_h5ad(os.path.join(data_dir, 'test_atac.h5ad'))

    if modality=='RNA':
        return train_rna, val_rna, test_rna

    elif modality=='ATAC':
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



def main():
    
    rna, atac = load_data('bmmc_atac_highly_variable.h5ad', 'bmmc_rna_highly_variable.h5ad', multiome=False)

    train_idx, val_idx = split_dataset(rna)

    train_dataset = MultiomeDataset(rna, atac, train_idx)
    val_dataset   = MultiomeDataset(rna, atac, val_idx)
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
    

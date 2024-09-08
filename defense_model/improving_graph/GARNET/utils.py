import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

import numpy as np
import yaml
import random
from scipy.sparse import identity, diags
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import normalize

def normal_adj(adj):
    adj = SparseTensor.from_scipy(adj)
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float('inf')] = 0
    DAD = D_isqrt.view(-1,1) * adj * D_isqrt.view(1,-1)

    return DAD.to_scipy(layout='csr')

def embedding_normalize(embedding, norm):
    if norm == "unit_vector":
        return normalize(embedding, axis=1)
    elif norm == "standardize":
        scaler = StandardScaler()
        return scaler.fit_transform(embedding)
    elif norm == "minmax":
        scaler = MinMaxScaler()
        return scaler.fit_transform(embedding)
    else:
        return embedding
    
def adj2laplacian(A):
    norm_adj = normal_adj(A)
    L = identity(norm_adj.shape[0]).multiply(1+1e-6) - norm_adj

    return L

def hnsw(features, k=10, ef=100, M=48):
    num_samples, dim = features.shape

    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_samples, ef_construction=ef, M=M)
    labels_index = np.arange(num_samples)
    p.add_items(features, labels_index)
    p.set_ef(ef)

    neighs, _ = p.knn_query(features, k+1)
    adj = construct_adj(neighs)

    return adj
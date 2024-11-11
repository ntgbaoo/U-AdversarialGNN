from typing import Sequence, Tuple, Union

import numpy as np
import torch
import torch_scatter
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_sparse import SparseTensor, SparseStorage, coalesce

from .util import ppr_topk

# from torchtyping import TensorType, patch_typeguard
# from typeguard import typechecked



def get_approx_topk_ppr_matrix(edge_idx: torch.Tensor,
                               n: int,
                               alpha: float = 0.15,
                               k: float = 64,
                               ppr_err: float = 1e-4,
                               **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates the personalized page rank diffusion of the adjacency matrix as proposed in Johannes Klicpera,
    Stefan Weißenberger, and Stephan Günnemann. Diffusion Improves Graph Learning.

    Parameters
    ----------
    edge_idx : torch.Tensor
        Sparse (unweighted) adjacency matrix.
    alpha : float, optional
        Teleport probability, by default 0.15.
    k : int, optional
        Neighborhood for sparsification, by default 32.
    ppr_err : bool, optional
        Admissible error, by default 1e-4

    Returns
    -------
    torch.Tensor
        Preprocessed adjacency matrix.

    """
    edge_weight = torch.ones_like(edge_idx[0], dtype=torch.float32)
    edge_idx, edge_weight = coalesce(edge_idx, edge_weight, n, n)

    row, col = edge_idx

    deg = torch_scatter.scatter_add(edge_weight, col, dim=0, dim_size=n)

    edge_weight = edge_weight.cpu()
    row, col = row.cpu(), col.cpu()

    ppr = ppr_topk(
        adj_matrix=sp.csr_matrix((edge_weight, (row, col)), (n, n)),
        alpha=alpha,
        epsilon=ppr_err,
        nodes=np.arange(n),
        topk=k
    ).tocsr()

    edge_idx, edge_weight = [tensor.to(edge_idx.device) for tensor in from_scipy_sparse_matrix(ppr)]

    # Uncommented normalization
    row, col = edge_idx
    deg_inv = deg.sqrt()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight = deg_inv[row] * edge_weight * deg_inv_sqrt[col]

    # Row norm
    norm = torch_scatter.scatter_sum(edge_weight, edge_idx[0], dim=-1, dim_size=n)
    norm[norm <= 0] = 1
    edge_weight /= norm[edge_idx[0]]

    return edge_idx, edge_weight
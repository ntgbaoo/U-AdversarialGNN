import warnings
import torch
from torch.nn.parameter import Parameter
from deeprobust.graph import utils
from deeprobust.graph.targeted_attack import Nettack
import torch.nn.functional as F
from torch import optim
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
import scipy.sparse as sp
import copy
from copy import deepcopy
from numba import jit
from torch import spmm
from greatx.nn.models.surrogate import Surrogate
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_scipy_sparse_matrix, to_edge_index, to_dense_adj
import os
import sys
from numpy.linalg import norm
from torch_geometric.utils import k_hop_subgraph


# Add path for importing from other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import DataLoader
from utility.config import args
from utility.util import set_random, logger, init_logger, mkdirs, prepare_dir
from evaluation import ModelSupervisor
from utility import ConfigHandler
from static import *

# Detect device (MPS for Apple Silicon GPU or CPU fallback)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

class PopularityAttack:
    def __init__(self, data: Data, device=device, budget=1, attack_structure=True, attack_features=True, seed=720):
        # export PYTORCH_ENABLE_MPS_FALLBACK=1
        # set_random(seed)
        assert isinstance(data, Data)
        assert data.x is not None
        assert data.edge_index is not None
        assert data.edge_weight is None

        self.ori_data = data
        self.structure_perturbations = []
        self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []
        self.features = data.x.to(device)
        self.adj: sp.csr_matrix = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].to(device)
        self.labels = data.y.to(device)
        self.modified_edgelist = None

        self.cooc_constraint = None
        self.nnodes = data.num_nodes
        self.device = device

        self.modified_adj = None
        self.modified_features = None
        self.budget = budget

    def set_max_perturbations(self, budget):
        self.n_perturbations = budget

    def data(self):
        modified_data = copy.deepcopy(self.ori_data)
        modified_data.edge_index = self.modified_edgelist
        return modified_data

    def reset(self):
        self.modified_adj = None
        self.modified_features = None
        self.cooc_constraint = None
        self.structure_perturbations = []
        self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []

    def attack(self, target, num_budgets=None, attack_structure=True, attack_features=False, direct=True, verbose=False, n_influencers=0, ll_cutoff=0.004, **kwargs):
        if type(self.adj) is torch.Tensor:
            self.ori_adj = utils.to_scipy(self.adj).tolil()
            self.modified_adj = utils.to_scipy(self.adj).tolil()
            self.modified_edgelist = self.adj  # Initialize with original edge list
        else:
            self.ori_adj = self.adj.tolil()
            self.modified_adj = self.adj.tolil()
            self.modified_edgelist = self.adj

        self.target_node = target
        assert n_influencers < self.nnodes - 1, "number of influencers cannot be >= number of nodes in the graph!"
        neighbors = self.ori_adj[self.target_node].nonzero()[1]
        assert self.target_node not in neighbors 
        

        res = []
        target_val = self.labels[self.target_node].cpu().numpy()
        for node in range(self.nnodes):
            if node == target:
                continue

            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                node, 3, self.ori_data.edge_index
            )
            neighbors_labels = self.labels[subset].cpu().numpy()

            count = 0
            for label in neighbors_labels:
                if label == target_val:
                    count += 1

            res.append([node, count])
        

        res.sort(key=lambda x: x[1])
        final_potential_connections = res[:self.budget]
        modified_data = self.data()

        target = torch.tensor([self.target_node])
        for node, _ in final_potential_connections:
            new_edge = torch.tensor([node])
            new_edge = new_edge

            new_connection = torch.cat((target, new_edge), dim=0)

            modified_data.edge_index = new_connection
        
        self.modified_edgelist = modified_data.edge_index

        # for node, _ in final_potential_connections:
        #     print(f"Connecting target node {self.target_node} to candidate node {node}")

        #     # Create the new edge tensor
        #     new_edge = torch.tensor([[self.target_node], [node]], dtype=torch.long).to(self.device)

        #     # Check the shapes before concatenation
        #     if modified_data.edge_index is None:
        #         modified_data.edge_index = new_edge
        #     else:
        #         # Ensure the existing edge index and new edge have compatible sizes
        #         print("Current edge_index shape:", modified_data.edge_index.shape)
        #         print("New edge shape:", new_edge.shape)

        #         # Concatenate only if dimensions are compatible
        #         if modified_data.edge_index.shape[0] == new_edge.shape[0]:  # Must match
        #             modified_data.edge_index = torch.cat([modified_data.edge_index, new_edge], dim=1)
        #         else:
        #             raise ValueError("Edge dimensions do not match for concatenation.")

        # self.modified_edgelist = modified_data.edge_index
        # print("Modified Edge Index:", self.modified_edgelist)


if __name__ == "__main__":
    target_node = 1

    # Load dataset and config
    args.dataset = "cora"
    data, split = DataLoader.load(args.dataset)
    config = ConfigHandler.load_model_config(model_type="GCN", config_name="max_best_config", dataset="cora", split=0)
    model_config = config['model_config']
    split_config = config['split']
    set_random(720)

    # Initialize and train model supervisor
    runner = ModelSupervisor(data, device=device, **model_config)
    idx_train, idx_val, idx_test = split_config['idx_train'], split_config['idx_val'], split_config['idx_test']
    runner.train_model(idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    model = PopularityAttack(data=data, budget=10, attack_structure=True, attack_features=True, device=device)
    model.attack(target_node, n_influencers=5)
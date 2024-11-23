from torch_geometric.data import Data
import torch
import os
import sys
from torch_geometric.utils import degree, to_scipy_sparse_matrix
import copy
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utility.util import set_random

class BaselineAttack:
    def __init__(self, data:Data , device='cpu',seed = 720):
        set_random(seed)
        self.ori_data = data
        self.device = device
        self.deg = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long).numpy()

    def set_max_perturbations(self,budget):
        self.n_perturbations = budget

    def reset(self):
        self.modified_edgelist = None

    def attack(self, target,num_budgets=None,structure_attack = True, feature_attack = False):
        self.n_perturbations = num_budgets
        data = self.ori_data
        if structure_attack:
            deg = self.deg
            candidate_nodes = np.argsort(deg)[::-1]
            modified_edgelist = copy.deepcopy(data.edge_index).cpu()
            for i in range(self.n_perturbations):
                new_edge = torch.tensor([[target],[candidate_nodes[i]]])
                modified_edgelist = torch.cat((modified_edgelist,new_edge),dim=1)
            self.modified_edgelist = modified_edgelist

    def data(self):
        assert self.modified_edgelist is not None, "attack() method must be called before calling data()"
        modified_data = copy.deepcopy(self.ori_data)
        modified_data.edge_index = self.modified_edgelist
        return modified_data
            
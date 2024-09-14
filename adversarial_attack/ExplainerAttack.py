from torch_geometric.explain import Explainer, GNNExplainer
import torch

from torch_geometric.data import Data
from greatx.utils import BunchDict, add_edges, remove_edges
from copy import copy
from functools import lru_cache

import sys
import os

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from greatx.attack.targeted.targeted_attacker import TargetedAttacker
from utility.config import args
from utility.util import set_random




class ExplainerAttack(TargetedAttacker):
        
    def update_config(self,epoch):
        self.epoch = epoch

    def setup_surrogate(self,model):
        self.surr_model = model

    

    def attack(self, target,num_budgets=None,structure_attack =True,feature_attack=False):
        explainer = Explainer(
            model=self.surr_model,
            algorithm=GNNExplainer(epochs=self.epoch),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',
            )
        )

        super().attack(target, target_label=None, num_budgets=num_budgets,
                       direct_attack=True,
                       structure_attack=structure_attack,
                       feature_attack=feature_attack)
        
        explanation = explainer(self.ori_data.x.to(self.device), self.ori_data.edge_index.to(self.device), index=target)
        potential_edge_index = sorted(range(len(explanation.edge_mask)),key=lambda i :explanation.edge_mask[i],reverse=True)
        
        row = []
        col = []
        removed_count = 0
        edge_index = 0
        while removed_count < num_budgets:
            u = self.ori_data.edge_index[0][edge_index]
            v = self.ori_data.edge_index[1][edge_index]
            if not self.is_singleton_edge(u, v):
                row.append(u)
                col.append(v)
                removed_count +=1

            edge_index += 1
            if edge_index >= len(potential_edge_index): break
        
        self.potential_edges = torch.stack([torch.tensor(row),torch.tensor(col)],dim=0)

    @lru_cache(maxsize=1)
    def data(
        self,
        edge_ratio: float = 1.0,
        feat_ratio: float = 1.0,
        coalesce: bool = True,
        symmetric: bool = True,
    ) -> Data:
        

        data = copy(self.ori_data)
        edge_index = data.edge_index
        edge_weight = data.edge_weight
       
        removed = self.potential_edges

        if removed is not None:
            edge_index = remove_edges(edge_index, removed, symmetric=symmetric)

        data.edge_index = edge_index

        return data
    
    def reset(self):
        super().reset()
        self.potential_edges = None
    
        

    

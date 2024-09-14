from torch_geometric.explain import Explainer, GNNExplainer


from torch_geometric.data import Data
from greatx.utils import BunchDict, add_edges, remove_edges
from copy import copy

import sys
import os

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utility.config import args
from utility.util import set_random




class ExplainerAttack:
    def __init__(self, data:Data , device='cpu',seed = 720):
        set_random(seed)
        self.ori_data = data
        self.devide =device
        
        
    def update_config(self,epoch):
        self.epoch = epoch

    def setup_surrogate(self,model):
        self.surr_model = model

    def set_max_perturbations(self,budget):
        self.num_budgets = budget

    def attack(self, target,num_budgets=None,attack_structure =True,attack_features=False):
        explainer = Explainer(
            model=self.surr_model,
            algorithm=GNNExplainer(epochs=10),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',
            )
        )
        explanation = explainer(self.ori_data.x.to(self.device), self.ori_data.edge_index.to(self.device), index=target)
        self.potential_edge_index = sorted(range(len(explanation.edge_mask)),key=lambda i :explanation.edge_mask[i],reverse=True)[:num_budgets]
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
       
        removed = self.potential_edge_index

        if removed is not None:
            edge_index = remove_edges(edge_index, removed, symmetric=symmetric)

        return data
    
    def reset(self):
        self.potential_edge_index = None
    
        

    

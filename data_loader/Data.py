import sys
import os
from scipy.sparse import csr_matrix
import torch
import numpy as np

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader.IDataLoader import IDataLoader
from data_loader.IData import IData
from data_loader.utils import sparse_adjacency_to_edge_list,edge_list_to_sparse_adjacency



class Data(IData):
    def __init__(self,edgelist: np.ndarray,
                 adj: csr_matrix,
                 feature: np.ndarray,
                 label:np.ndarray,
                 idx_train:np.ndarray,
                 idx_val:np.ndarray,
                 idx_test:np.ndarray) -> None:
        super().__init__()
        self.adj = adj
        self.edgelist = edgelist
        self.feature = feature
        self.label = label
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

        self.nclass = self.label.max().item() + 1
        self.nfeat = self.feature.shape[1]
        self.nnode = len(self.label)
        
        self.folds = []

    def update_adversarial_adj(self,adversarial_adj: csr_matrix):
        self.adj = adversarial_adj
        self.edgelist = sparse_adjacency_to_edge_list(adversarial_adj)

    def update_adversarial_edgelist(self,adversarial_edgelist:np.ndarray):
        self.adj = edge_list_to_sparse_adjacency(adversarial_edgelist)
        self.edgelist = adversarial_edgelist

    def update_adversarial_feature(self,adversarial_feature:np.ndarray):
        self.feature = adversarial_feature

    # TODO: implement random splitting k-fold train, validation and test
    def init_train_test_splits(self,k):
        raise Exception("Not yet implemented")

    def get_split(self,fold_kth):
        if len(self.folds) != 0: raise Exception("Folds is not yet initialized")
        if fold_kth < len(self.folds): raise "Given index {} out of bound of {}-fold".format(fold_kth,len(self.folds))
        k_fold = self.folds[fold_kth]
        return k_fold.train,k_fold.val,k_fold.test
    
    def get_tensor_edgelist(self):
        return torch.tensor(self.edgelist,dtype=torch.long)

    def get_tensor_label(self):
        return torch.tensor(self.label,dtype=torch.long)

    def get_tensor_feature(self):
        return torch.tensor(self.feature,dtype=torch.float)


    
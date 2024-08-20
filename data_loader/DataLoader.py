from abc import ABC, abstractmethod
import sys
import os
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T





# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from greatx.datasets import GraphDataset
from data_loader.utils import sparse_adjacency_to_edge_list,edge_list_to_sparse_adjacency,edge_list_to_sysmetric_sparse_adjacency
from data_loader.IDataLoader import IDataLoader
from data_loader.Splitter import Splitter
from static import *
from greatx.utils import split_nodes


DATASETS_greatX = {
    'citeseer',
    'citeseer_full',
    'cora',
    'cora_ml',
    'cora_full',
    'amazon_cs',
    'amazon_photo',
    'coauthor_cs',
    'coauthor_phy',
    'polblogs',
    'karate_club',
    'pubmed',
    'flickr',
    'blogcatalog',
    'dblp',
    'acm',
    'uai',
    'pdn',
}


class DataLoader(IDataLoader):
    @abstractmethod
    def load(dataset:str) :
        if dataset in DATASETS_greatX:
            if not os.path.exists(PATH_DATASET):
                os.makedirs(PATH_DATASET)

            dataset = GraphDataset(root=PATH_DATASET, name=dataset,
                       transform=T.LargestConnectedComponents())
            data = dataset[0]
            split = split_nodes(data.y,random_state=720)
            default_split = {IDX_TRAIN:split.train_nodes,IDX_VAL:split.val_nodes,IDX_TEST:split.test_nodes}
        elif dataset in ['ogbn-products','ogbn-proteins','ogbn-arxiv','ogbn-papers100M','ogbn-mag']:
            ogb_dataset = PygNodePropPredDataset(name = dataset, root = PATH_DATASET,transform=T.ToUndirected())
            default_split  = ogb_dataset.get_idx_split()
            data = ogb_dataset[0]
            data.y = data.y.squeeze()
        else:
            raise Exception("Unsupport datasets")
        return data,default_split
    

if __name__ == '__main__':
    data_ogbn = DataLoader.load("ogbn-products")    
    idx_train,idx_val,idx_test = data_ogbn.idx_train, data_ogbn.idx_val, data_ogbn.idx_test
    adj, features, labels = data_ogbn.adj, data_ogbn.feature, data_ogbn.label
    


    
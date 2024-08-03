from abc import ABC, abstractmethod
import sys
import os
from deeprobust.graph.data import Dataset, Dpr2Pyg
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset




# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader.IData import IData
from data_loader.utils import sparse_adjacency_to_edge_list,edge_list_to_sparse_adjacency,edge_list_to_sysmetric_sparse_adjacency
from data_loader.Data import Data
from data_loader.IDataLoader import IDataLoader
from data_loader.Splitter import Splitter
from static import *



class DataLoader(IDataLoader):
    @abstractmethod
    def load(dataset:str) -> IData:
        if dataset in ['cora', 'citeseer', 'cora_ml', 'polblogs','pubmed', 'acm', 'blogcatalog', 'uai', 'flickr']:
            if not os.path.exists(PATH_DATASET):
                os.makedirs(PATH_DATASET)
            orgin_data = Dataset(root=PATH_DATASET,name = dataset)
            data = Data(
                        dataset_name= dataset,
                        edgelist = np.array(sparse_adjacency_to_edge_list(orgin_data.adj)),
                        adj=orgin_data.adj,
                        feature = np.array(orgin_data.features.toarray()),
                        label = orgin_data.labels,
                        idx_train = orgin_data.idx_train,
                        idx_val = orgin_data.idx_val,
                        idx_test = orgin_data.idx_test)
        elif dataset in ['ogbn-products','ogbn-proteins','ogbn-arxiv','ogbn-papers100M','ogbn-mag']:
            ogb_data = PygNodePropPredDataset(name = dataset, root = PATH_DATASET)
            split_idx  = ogb_data.get_idx_split()
            train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

            labels = np.array([int(tensor) for tensor in ogb_data[0].y],dtype=np.int8)
            data = Data(
                        dataset_name= dataset,
                        edgelist = ogb_data.edge_index.numpy().astype('int32'),
                        adj=edge_list_to_sysmetric_sparse_adjacency(ogb_data.edge_index.numpy()).astype('float32'),
                        feature = ogb_data[0].x.numpy(),
                        label = labels,
                        idx_train = train_idx.numpy().astype('int32'),
                        idx_val = valid_idx.numpy().astype('int32'),
                        idx_test = test_idx.numpy().astype('int32'))
        else:
            raise Exception("Unsupport datasets")
        return data

if __name__ == '__main__':
    data_ogbn = DataLoader.load("ogbn-products")    
    idx_train,idx_val,idx_test = data_ogbn.idx_train, data_ogbn.idx_val, data_ogbn.idx_test
    adj, features, labels = data_ogbn.adj, data_ogbn.feature, data_ogbn.label
    


    
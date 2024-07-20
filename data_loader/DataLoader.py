from abc import ABC, abstractmethod
import sys
import os
from deeprobust.graph.data import Dataset, Dpr2Pyg
import numpy as np



# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader.IData import IData
from data_loader.utils import sparse_adjacency_to_edge_list
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
        else:
            raise Exception("Unsupport datasets")
        return data

if __name__ == '__main__':
    dataset = DataLoader.load("cora")     
    
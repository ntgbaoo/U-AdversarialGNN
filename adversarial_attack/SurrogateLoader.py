from abc import ABC, abstractmethod
import sys
import os
import warnings
import numpy as np
from deeprobust.graph.defense import GCN


    # Hide all warnings
warnings.filterwarnings("ignore")
    
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader.IData import IData
from model.SGC import SGC
from data_loader.DataLoader import DataLoader



class SurrogateLoader:
    @abstractmethod
    def load_surrogate(data:IData,idx_train,idx_val,model = 'gcn',device = 'cpu',lr = 0.01):
        if model == 'gcn':
            surrogate = GCN(nfeat=data.nfeat, nclass=data.nclass, nhid=16, dropout=0, with_relu=False,
                    with_bias=False, device=device,lr=lr)
            surrogate.to(device)
            surrogate.fit(data.feature, data.adj, data.label, idx_train, idx_val, patience=30)
            
        elif model == 'sgc':
            surrogate = SGC(nfeat=data.nfeat, nclass=data.nclass,K=2,lr=0.01,
                    with_bias=True, device=device)
            surrogate.to(device)
            surrogate.fit(data.get_tensor_feature(), data.get_tensor_edgelist(), data.get_tensor_label(), idx_train, idx_val)
        else:
            raise Exception("{} is not support as surrogate models. Choosing gcn/sgc".format(model))
        
        return surrogate


if __name__ == '__main__':
    data = DataLoader.load('cora')
    LoadSurrogateModel.load_surrogate(data,data.idx_train,data.idx_val,device = 'cuda',model = 'sgc')
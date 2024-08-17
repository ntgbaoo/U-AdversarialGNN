from abc import ABC, abstractmethod
import sys
import os
import warnings
import numpy as np
from torch_geometric.data import Data
from greatx.nn.models import GCN
from greatx.nn.models import SGC
from greatx.training import Trainer

    # Hide all warnings
warnings.filterwarnings("ignore")
    
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.util import set_random
from data_loader.DataLoader import DataLoader
from static import *



class SurrogateLoader:
    @abstractmethod
    def load_surrogate(data:Data,idx_train,idx_val,model = 'GCN',device = 'cpu',model_configs:dict = dict(),seed = 720):
        num_features = data.x.size(-1)
        num_classes = data.y.max().item() + 1
        set_random(seed)

        if model == 'GCN':
            trainer = Trainer(GCN(num_features, num_classes),device=device)
            trainer.fit(data, mask=(idx_train, idx_val),verbose=0)
            
        elif model == 'SGC':
            # nlayer = 1
            trainer = Trainer(SGC(num_features, num_classes),device=device)
            # trainer = Trainer(SGC(num_features, num_classes), device=device, lr=lr,
            #              weight_decay=weight_decay)
            trainer.fit(data, mask=(idx_train, idx_val),verbose=0)
        else:
            raise Exception("{} is not support as surrogate models. Choosing gcn/sgc".format(model))
        
        return trainer.model


    
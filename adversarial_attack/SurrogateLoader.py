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
        lr = model_configs.get(LEARNING_RATE,0.1)
        nhid = model_configs.get(HIDDEN_UNITS,16)
        nlayer = model_configs.get(NUM_LAYERS,2)
        weight_decay = model_configs.get(WEIGHT_DECAY,1e-5)
        dropout = model_configs.get(DROPOUT,0.5)
        set_random(seed)

        if model == 'GCN':
            trainer = Trainer(GCN(num_features, num_classes,acts = ['relu'],dropout = dropout, hids = [nhid] *nlayer), device=device,weight_decay =weight_decay,lr =  lr)
            trainer.fit(data, mask=(idx_train, idx_val),verbose=0)
            
        elif model == 'SGC':
            trainer = Trainer(SGC(num_features, num_classes,acts = ['relu'],dropout = dropout,hids = [nhid] *nlayer), device=device, lr=lr,
                         weight_decay=weight_decay)
            trainer.fit(data, mask=(idx_train, idx_val),verbose=0)
        else:
            raise Exception("{} is not support as surrogate models. Choosing gcn/sgc".format(model))
        
        return trainer.model


    
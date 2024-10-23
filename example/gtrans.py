import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


        
if __name__ == '__main__':
    import sys
    import os
    
    # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utility.config import args
    from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
    from evaluation import ModelSupervisor
    from data_loader import DataLoader
    from utility import ConfigHandler
    from static import *
    from defense_model.improving_graph import GTransPurification

    args.dataset = "cora"
    
    data,split = DataLoader.load(args.dataset)
    config = ConfigHandler.load_model_config(model_type="GCN",config_name="max_best_config",dataset="cora",split=0)
    model_config = config['model_config']
    split_config = config['split']
    set_random(720)
    runner = ModelSupervisor(data,device=args.device,**model_config)
    idx_train,idx_val,idx_test = split_config['idx_train'],split_config['idx_val'],split_config['idx_test']
    runner.train_model(idx_train=idx_train,idx_val=idx_val,idx_test= idx_test)


    purification = GTransPurification(model_name="GCN",dataset="cora",split_indx=0,device=args.device,epochs=5)
    new_data= purification(data)
    purification = GTransPurification(model_name="GCN",dataset="cora",split_indx=0,device=args.device,epochs=5)
    new_data= purification(data)

    

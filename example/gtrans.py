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
    from evaluation.ModelSupervisor import ModelSupervisor
    from data_loader.DataLoader import DataLoader
    from utility.ConfigHandler import ConfigHandler
    from static import *
    from defense_model.improving_graph.GTrans.gtrans import GraphAgent

    args.dataset = "cora"
    
    data,split = DataLoader.load(args.dataset)
    idx_train,idx_val,idx_test = split['idx_train'],split['idx_val'],split['idx_test']

    purification = GraphAgent(device=args.device)
    purification.update_idx_val(idx_val)
    new_feat, new_edge_index, new_edge_weight = purification(data.edge_idx)

    

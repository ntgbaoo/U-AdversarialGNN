from torch_geometric.contrib.nn import GRBCDAttack, PRBCDAttack
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch



        
if __name__ == '__main__':
    import sys
    import os
    
    # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from adversarial_attack import NodeSelector
    from utility.config import args
    from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
    from evaluation import ModelSupervisor
    from data_loader import DataLoader
    from utility import ConfigHandler
    from static import *

    args.dataset = "blogcatalog"
    
    data,split = DataLoader.load(args.dataset)
    config = ConfigHandler.load_model_config(model_type="GIN")
    model_config = config['model_config']

    config_surrogate = ConfigHandler.load_model_config(model_type="GCN")
    config_surrogate = config_surrogate['model_config']
    set_random(720)
    idx_train,idx_val,idx_test = split['idx_train'],split['idx_val'],split['idx_test']

    runner_surrogate = ModelSupervisor(data,device=args.device,**config_surrogate)
    runner_surrogate.train_model(idx_train=idx_train,idx_val=idx_val,idx_test= idx_test)


    bucket_target_nodes= {'high_margin': [1554, 929, 1163, 1347, 1504, 1541, 1340, 2406, 1049,2082],
                          'low_margin': [1183, 692, 390, 1120, 1455, 523,2153,227, 877, 2221], 
                          'other': [1779, 1243, 1264, 2436, 1931, 1222, 1126, 123, 1713, 2225, 852, 2074, 161, 377, 2271, 1182, 2192, 1921, 1286, 872]}
    target_nodes = bucket_target_nodes['high_margin'] + bucket_target_nodes['low_margin'] + bucket_target_nodes['other']
    
    miss = 0
    for node in target_nodes:
        runner_surrogate.model.eval()
        prbcd = PRBCDAttack(runner_surrogate.model, block_size=250_000, lr=2_000)
        if isinstance(node, torch.Tensor):
            node = node.item()

        pert_edge_index, perts = prbcd.attack(
            data.x.to(args.device),
            data.edge_index.to(args.device),
            data.y.to(args.device),
            budget=1,
            idx_attack=[node],
        )

        data_modified,_ = DataLoader.load(args.dataset)
        data_modified.edge_index = pert_edge_index
        runner_poison = ModelSupervisor(data_modified,device=args.device,**model_config)
        runner_poison.train_model(idx_train=idx_train,idx_val=idx_val,idx_test= idx_test)
        output = runner_poison.get_model_prediction(data_modified)[node]
        if data.y[node].item() != output.item():
            print(data.y[node].item())
            print(output.item())
            miss += 1



    print(miss/40)
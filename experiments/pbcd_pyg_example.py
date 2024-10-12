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

    args.dataset = "cora"
    
    data,split = DataLoader.load(args.dataset)
    config = ConfigHandler.load_model_config(model_type="GCN")
    model_config = config['model_config']
    set_random(720)
    runner = ModelSupervisor(data,device=args.device,**model_config)
    idx_train,idx_val,idx_test = split['idx_train'],split['idx_val'],split['idx_test']
    runner.train_model(idx_train=idx_train,idx_val=idx_val,idx_test= idx_test)

    test_acc = runner.get_prediction_accuracy(idx_test,data)

    bucket_target_nodes= NodeSelector.select_nodes_margin(runner,data,idx_test,seed=720)
    print(bucket_target_nodes)
    target_nodes = bucket_target_nodes['high_margin'] + bucket_target_nodes['low_margin'] + bucket_target_nodes['other']
    
    miss = 0
    for node in target_nodes:
        prbcd = PRBCDAttack(runner.model, block_size=250_000, lr=2_000)
        if isinstance(node, torch.Tensor):
            node = node.item()

        pert_edge_index, perts = prbcd.attack(
            data.x,
            data.edge_index,
            data.y,
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
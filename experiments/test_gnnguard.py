import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import degree
import torch
import numpy as np

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
    from greatx.attack.targeted import FGAttack, SGAttack

    args.dataset = "chameleon"
    node = 2

    data,split = DataLoader.load(args.dataset)
    print(data)


    config = ConfigHandler.load_model_config(model_type="GNNGuard",split=0,config_name="best_config",dataset="chameleon")
    split_config = config['split']
    model_config = config['model_config']
    idx_train,idx_val,idx_test = split_config['idx_train'],split_config['idx_val'],split_config['idx_test']
    # print(idx_test)

    split_model_surrogate = ConfigHandler.load_model_config("SGC","best_config",args.dataset,0)
    configs_surrogate = split_model_surrogate.get('model_config')
    
    attacker = SGAttack(data, device=args.device,seed=720)

    surrogate = ModelSupervisor(data,args.device,seed=720,**configs_surrogate)
    surrogate.train_model(idx_train,idx_val,idx_test)
    attacker.setup_surrogate(surrogate.model)
    attacker.reset()
    attacker.set_max_perturbations(100)
    attacker.attack(node,num_budgets = 100,structure_attack = True, feature_attack = False)

    init_logger(
                prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(
                    args.seed) + '_log.txt')
    
    

    # print(config)
    set_random(720)
    modified_data = attacker.data()
    print(modified_data)
    runner = ModelSupervisor(modified_data,device=args.device,**model_config)
    runner.train_model(idx_train=idx_train,idx_val=idx_val,idx_test= idx_test)
    
   
    pred_evasion = runner.get_model_prediction(modified_data)[node].item()
    print(pred_evasion)
    print(data.y[node])
   
    
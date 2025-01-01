import os.path as osp

import torch
import torch_geometric.transforms as T
import sys
import os
import statistics as stats
import pandas as pd
import timeit


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
    from greatx.attack.targeted import PRBCDAttack

    from static import *

    args.dataset = "cora"
    # init_logger(
    #             prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(
    #                 args.seed) + '_log.txt')
    
    data,split = DataLoader.load(args.dataset)
    # data.edge_attr = torch.ones(data.edge_index.shape[1])
    print(data)
    
    config = ConfigHandler.load_model_config(model_type="GNNGuard",config_name="best_config",dataset="cora",split=0)
    # print(config)
    model_config = config['model_config']
    split_config = config['split']
    set_random(720)
    runner = ModelSupervisor(data,device=args.device,**model_config)
    idx_train,idx_val,idx_test = split_config['idx_train'],split_config['idx_val'],split_config['idx_test']
    runner.train_model(idx_train=idx_train,idx_val=idx_val,idx_test= idx_test)

    test_acc = runner.get_prediction_accuracy(idx_test,data)
    prbcd = PRBCDAttack(data,device=args.device,seed=args.seed)
    prbcd.reset()
    runner.model.eval()
    prbcd.setup_surrogate(runner.model)
    prbcd.set_max_perturbations(5)
    prbcd.attack(0,num_budgets=5)
    print(test_acc)
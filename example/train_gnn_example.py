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

    args.dataset = "cora"
    args.seed = 710
    set_random(args.seed)


    init_logger(
                prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(
                    args.seed) + '_log.txt')
    
    data,split = DataLoader.load_v1(args.dataset)
    # model_type="GCN",config_name = "default",dataset = None,split = None
    config = ConfigHandler.load_model_config(model_type="GCN",config_name="best_config",dataset="cora",split=0)
    model_config = config['model_config']
    split_config = config['split']
    runner = ModelSupervisor(data,device=args.device,**model_config)
    idx_train,idx_val,idx_test = split_config['idx_train'],split_config['idx_val'],split_config['idx_test']
    runner.train_model(idx_train=idx_train,idx_val=idx_val,idx_test= idx_test)

    test_acc = runner.get_prediction_accuracy(idx_test,data)
    print(test_acc)
   
    
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
    from utility.util import load_model_config
    from static import *

    args.dataset = "cora"
    args.seed = 710
    set_random(args.seed)


    init_logger(
                prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(
                    args.seed) + '_log.txt')
    
    data = DataLoader.load(args.dataset)
    config = load_model_config()
    runner = ModelSupervisor(data,device=args.device,**config)
    runner.train_model()

    train_acc, val_acc, test_acc = runner.get_prediction_accuracy()
    print(train_acc, val_acc, test_acc)
   
    
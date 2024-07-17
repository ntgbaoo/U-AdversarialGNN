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
    from static import *



    args.model = "GCN"
    args.dataset = "cora"
    args.seed = 710
    args.train_early_stopping = True
    set_random(args.seed)

    
    

    init_logger(
                prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(
                    args.seed) + '_log.txt')
    
    data = DataLoader.load(args.dataset)
    config = {
        MODEL:'GCN',
        LEARNING_RATE:0.01,
        OPTIMIZER:ADAM,
        HIDDEN_UNITS:8,
        NUM_LAYERS:2,
        AGGREGATION:"add",
        EPOCHS:500,
        PATIENCE:40,
    }
    runner = ModelSupervisor(data,args,**config)
    runner.train_model()

    train_acc, val_acc, test_acc = runner.get_prediction_accuracy()
    print(train_acc, val_acc, test_acc)
   
    
import wandb
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import timeit

        
if __name__ == '__main__':
    import sys
    import os
    
    # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utility.config import args
    from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
    from utility.data_util import dataset_loader,sparse_adjacency_to_edge_list
    from utility.load_model import load_model
    from model.Model_Runner import Runner

    args.model = "GCN"
    args.dataset = "cora"
    args.seed = 710
    args.train_early_stopping = False
    set_random(args.seed)
    



    # data = dataset_loader(args)
    init_logger(
                prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(
                    args.seed) + '_log.txt')
    runner = Runner()
    runner.run()

    train_acc, val_acc, test_acc = runner.compute_acc(runner.get_predict())
    print(train_acc, val_acc, test_acc)
   
    
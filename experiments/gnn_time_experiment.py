import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import degree
import torch
import numpy as np

def save_results(dataset, model, purification, mean,std,time,time_std):
    partial_path = "data/results"
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    result_path = f"{partial_path}/model_evaluation.csv"
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(
            columns=["dataset", "model", "purification", "mean", "std","time","std_time"])
    else:
        result_df = pd.read_csv(result_path)

    if purification is None:
        purification = "None"
    result_df = result_df.append({"dataset":dataset,
                                  "model":model,
                                  "purification":purification, 
                                  "mean":mean, 
                                  "std":std,
                                  "time":time,
                                  "std_time":time_std
                                  }, ignore_index=True)
    result_df.to_csv(result_path, index=False)

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

    args.dataset = "cora"
    print(args.model)

    # init_logger(
    #             prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(
    #                 args.seed) + '_log.txt')
    
    # data,split = DataLoader.load(args.dataset)
    
    # config = ConfigHandler.load_model_config(model_type="NoisyGNN")
    # model_config = config['model_config']
    # split_config = split
    # set_random(720)
    # runner = ModelSupervisor(data,device=args.device,**model_config)
    # idx_train,idx_val,idx_test = split_config['idx_train'],split_config['idx_val'],split_config['idx_test']
    # runner.train_model(idx_train=idx_train,idx_val=idx_val,idx_test= idx_test)

    # test_acc = runner.get_prediction_accuracy(idx_test,data)
    # print(test_acc)
   
    
import sys
import os
import statistics as stats
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.ConfigHandler import ConfigHandler
from data_loader.DataLoader import DataLoader
from utility.config import args
from evaluation.ModelSupervisor import ModelSupervisor


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

    result_df = result_df.append({"dataset":dataset,
                                  "model":model,
                                  "purification":purification, 
                                  "mean":mean, 
                                  "std":std,
                                  "time":time,
                                  "std_time":time_std
                                  }, ignore_index=True)
    result_df.to_csv(result_path, index=False)

if __name__ == "__main__":
    
    data,split = DataLoader.load(args.dataset)
    for split_idx in range(5):

        split_model = ConfigHandler.load_model_config(args.model,"best_config",args.dataset,split_idx)
        split = split_model['split']
        configs = split_model.get('model_config')
        idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_test']

        for idx_run in range(args.num_run):
            seed = args.seed + idx_run
            node_selector = ModelSupervisor(data,args.device,seed=seed,**configs)
            node_selector.train_model(idx_train,idx_val,idx_test)
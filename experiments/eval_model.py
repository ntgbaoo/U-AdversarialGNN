import sys
import os
import statistics as stats
import pandas as pd
import timeit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility import ConfigHandler
from data_loader import DataLoader
from utility.config import args
from evaluation import ModelSupervisor


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

if __name__ == "__main__":
    
    data,split = DataLoader.load(args.dataset)
    performance_list = []
    time_list = []
    total = 5*3
    count = 0
    for split_idx in range(5):
        use_purification = False
        purification_config = None
        if args.purification is not None:
            use_purification = True
            purification_config = ConfigHandler.load_purification_config(args.purification,args.model,"best_config",args.dataset,split=split_idx)['model_config']
            split_model = ConfigHandler.load_model_config("GCN","best_config",args.dataset,split_idx)
           
        else:
            split_model = ConfigHandler.load_model_config(args.model,"best_config",args.dataset,split_idx)

        split = split_model['split']
        configs = split_model.get('model_config')
        idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_test']

        for idx_run in range(args.num_run):
            seed = args.seed + idx_run
            start_time = timeit.default_timer()
            classifier = ModelSupervisor(data,args.device,seed=seed,use_purification= use_purification,purification_config=purification_config,**configs)
            classifier.train_model(idx_train,idx_val,idx_test)
            time_list.append(timeit.default_timer() - start_time)
            test_acc = classifier.get_prediction_accuracy(idx_test,data)
            performance_list.append(test_acc)
            count += 1
            print(f'\rINFO : finished {count}/{total}', end='')
    
    save_results(args.dataset,args.model,args.purification,
                 stats.mean(performance_list),stats.stdev(performance_list),
                 stats.mean(time_list),stats.stdev(time_list))

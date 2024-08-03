import sys
import os
import warnings
import numpy as np
import timeit
# Hide all warnings
warnings.filterwarnings("ignore")
    
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader.IData import IData

from utility.config import args
from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
from deeprobust.graph.targeted_attack import Nettack,FGA,RND,SGAttack,IGAttack
from evaluation.ModelSupervisor import ModelSupervisor
from evaluation.ModelSelector import ModelSelector
from data_loader.DataLoader import DataLoader
from data_loader.Splitter import Splitter
from utility.ConfigHandler import ConfigHandler
from evaluation.Grid import Grid

if __name__ == "__main__":
    data = DataLoader.load(args.dataset)
    split_config = ConfigHandler.load_split_config(config_name="default")
   
    data.init_train_test_splits(**split_config)
    for split_idx in range(5):
        idx_train,idx_val,idx_test = data.get_split(split_idx)
        hyper_grid_config = ConfigHandler.load_hyper_grid(model_type=args.model,grid_name="hyper_grid")
        modelSelector = ModelSelector(
            config_dict= hyper_grid_config,
            idx_train=idx_train,
            idx_val= idx_val,
            idx_test= idx_test,
            data=data,
            device=args.device
        )
        start = timeit.default_timer()
        modelSelector.process_model_selection()
        end = timeit.default_timer()
        modelSelector.save_config(split_idx)
        print("Total time:{}".format(end -start))
        print(modelSelector.get_best_config())
    
     

    
   
    



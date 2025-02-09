import sys
import os
import warnings
import numpy as np
import timeit
# Hide all warnings
warnings.filterwarnings("ignore")
    
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utility.config import args
from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
from deeprobust.graph.targeted_attack import Nettack,FGA,RND,SGAttack,IGAttack
from evaluation import ModelSupervisor
from evaluation import ModelSelector
from data_loader import DataLoader
from data_loader import Splitter
from utility import ConfigHandler
from evaluation import ModelGrid

if __name__ == "__main__":
    hyper_grid_config = ConfigHandler.load_hyper_grid(model_type=args.model,grid_name="hyper_grid")
    grid_generator = ModelGrid(hyper_grid_config)
    all_configs = grid_generator.init_grid()
    data,default_split = DataLoader.load(args.dataset)
    if args.use_default_split:
        idx_train,idx_val,idx_test = default_split['train'], default_split['valid'],default_split['test']
    else:
        split_config = ConfigHandler.load_split_config(config_name="default")

        splitter = Splitter(**split_config)
        folds = splitter.split(data.x.numpy(),data.y.numpy())
        
        fold = folds[args.split_idx]
        idx_train,idx_val,idx_test =fold.idx_train,fold.idx_val,fold.idx_test
    
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
    if args.use_default_split:
        modelSelector.save_config("default",args.dataset)
    else:
        modelSelector.save_config(args.split_idx,args.dataset)
    print("Total time:{}".format(end -start))
    print(modelSelector.get_best_config())
    
     

    
   
    



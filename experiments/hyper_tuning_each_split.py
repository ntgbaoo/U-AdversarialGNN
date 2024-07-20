import sys
import os
import warnings
import numpy as np
# Hide all warnings
warnings.filterwarnings("ignore")
    
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader.IData import IData

from utility.config import args
from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
from deeprobust.graph.targeted_attack import Nettack,FGA,RND,SGAttack,IGAttack
from evaluation.ModelSupervisor import ModelSupervisor
from data_loader.DataLoader import DataLoader
from data_loader.Splitter import Splitter
from utility.ConfigHandler import ConfigHandler

if __name__ == "__main__":
    args.dataset = "cora"
    data = DataLoader.load(args.dataset)
    split_config = ConfigHandler.load_split_config(config_name="default")
   
    data.init_train_test_splits(**split_config)
    hyper_grid = ConfigHandler.load_hyper_grid(model_type="GCN",grid_name="toy_grid")
    print(hyper_grid)

    
   
    



import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.Grid import Grid
from evaluation.ModelSupervisor import ModelSupervisor
from data_loader.IData import IData
from adversarial_attack.utils import select_nodes
from utility.util import set_random



class ModelSelector:
    def __init__(self,
                 config_dict:dict,
                 idx_train:np.ndarray,
                 idx_val :np.ndarray,
                 idx_test:np.ndarray,
                 data: IData,
                 device = 'cpu'
                 ) -> None:
        self.config_dict = config_dict
        self.grid_generator = Grid(self.config_dict)
        self.all_configs = self.grid_generator.init_grid()
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.data = data
        self.results = [0]*len(self.all_configs)
        self.device = device

    def process_model_selection(self):
        for index,config in enumerate(self.all_configs):
            self.results[index] = self._run_model_on_config(config)

    def get_best_config(self):
        return self.results.index(max(self.results))
    
    def _run_model_on_config(self,config):
        runner = ModelSupervisor(self.data,device=self.device,**config)
        runner.train_model()
        _, _, test_acc = runner.get_prediction_accuracy()
        return test_acc

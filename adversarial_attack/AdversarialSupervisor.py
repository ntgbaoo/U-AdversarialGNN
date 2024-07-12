import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader.IData import IData
from evaluation import IModelSupervisor


class IAdversarial:
    def attack(**kwargs):
        pass

class AdversarialSupervisor:
    def __init__(self,
                 adversarial: IAdversarial, 
                 model_supervisor:IModelSupervisor,
                 data:IData,
                 surrogate_model,
                 target_nodes,
                 idx_train,
                 idx_val,
                 idx_test,
                 **kwargs
                 ) -> None:
        self.adversarial = adversarial
        self.model_supervisor = model_supervisor
        self.data = data
        self.target_nodes = target_nodes
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.surrogate_model = surrogate_model
    
    # @TODO: implement to attack on graph 
    def attack(budget = 1, attack_structure = True, feature_structure = False, verbose = False):
        pass

    # @TODO: return attack success rate
    def get_miss_rate():
        pass

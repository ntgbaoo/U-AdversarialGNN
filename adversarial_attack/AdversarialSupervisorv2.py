import sys
import os
import copy
import timeit
import statistics as stats
import io
import timeit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.ModelSupervisor import ModelSupervisor
from utility.util import set_random
from torch_geometric.data import Data
from greatx.attack.attacker import Attacker
from typing import List, Tuple
from static import *
class AdversarialSupervisor:
    def __init__(self,data:Data,
                 adversarial:Attacker,
                 idx_train,
                 idx_val,
                 idx_test,
                 device = "cpu",
                 **victim_configs
                 ) -> None:
        
        self.data = data
        self.adversarial = adversarial
        self.device = device
        self.victim_configs = victim_configs
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

    def target_attack(self,bucket_target_nodes:Tuple[List[int],...],budget,evasion = False, poision = True,
                      evasion_model:ModelSupervisor= None,seed = 720,structure_attack = True, feature_attack = False):
        assert evasion or poision, "Either evasion or posion setting"
        return_tuple = ()
        for target_nodes in bucket_target_nodes:
            result = {
                'time' : 0
            }
            if evasion:
                assert evasion_model is not None, "Evasion setting requires re-train models"
                result[EVASION] = 0
            if poision:
                result[POSION] = 0
            
            total = 0
            for node in target_nodes:
                truth_label = self.data.y[node].item()

                self.adversarial.reset()
                self.adversarial.set_max_perturbations(budget)
                start_attack = timeit.default_timer()
                self.adversarial.attack(node,num_budgets = budget,structure_attack = structure_attack, feature_attack = feature_attack)
                duration = timeit.default_timer() - start_attack
                total += duration

                if evasion:
                    pred_evasion = evasion_model.get_model_prediction(self.adversarial.data())[node].item()
                    if pred_evasion != truth_label:
                        result[EVASION] += 1
                
                if poision:
                    victim_model = ModelSupervisor(self.adversarial.data(), device= self.device,seed = 720,**self.victim_configs)

                    victim_model.train_model(self.idx_train,self.idx_val,self.idx_test)
                    pred_poision = victim_model.get_model_prediction(self.adversarial.data())[node].item()

                    if pred_poision != truth_label:
                        result[POSION] += 1

            result['time'] = total

            return_tuple += (result,)
        return return_tuple




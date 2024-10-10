import sys
import os
import copy
import timeit
import statistics as stats
import io
import timeit
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation import ModelSupervisor
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
                 use_purification = False,
                 purification_config = None,
                 **victim_configs
                 ) -> None:
        self.use_purification = use_purification
        self.purification_config = purification_config
        self.data = data
        self.adversarial = adversarial
        self.device = device
        self.victim_configs = victim_configs
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.labels = []
        self.evasion_preds = []
        self.poision_preds = []        

    def target_attack(self,bucket_target_nodes:Tuple[List[int],...],budget,evasion = False, poision = True,
                      evasion_model:ModelSupervisor= None,seed = 720,structure_attack = True, feature_attack = False):
        assert evasion or poision, "Either evasion or posion setting"
        return_tuple = ()
        counter = 0     

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
                self.labels.append(truth_label)

                self.adversarial.reset()
                self.adversarial.set_max_perturbations(budget)
                start_attack = timeit.default_timer()
                try:
                    self.adversarial.attack(node,num_budgets = budget,structure_attack = structure_attack, feature_attack = feature_attack)
                except Exception as e:
                    duration = timeit.default_timer() - start_attack
                    total += duration
                    counter += 1
                    continue
                duration = timeit.default_timer() - start_attack
                total += duration

                if evasion:
                    pred_evasion = evasion_model.get_model_prediction(self.adversarial.data())[node].item()
                    self.evasion_preds.append(pred_evasion)
                    if pred_evasion != truth_label:
                        result[EVASION] += 1
                
                if poision:
                    victim_model = ModelSupervisor(self.adversarial.data(), device= self.device,seed = seed,use_purification=self.use_purification,purification_config=self.purification_config,**self.victim_configs)

                    victim_model.train_model(self.idx_train,self.idx_val,self.idx_test)
                    pred_poision = victim_model.get_model_prediction(self.adversarial.data())[node].item()
                    self.poision_preds.append(pred_poision)

                    if pred_poision != truth_label:
                        result[POSION] += 1
                
                counter += 1
                print(f'\rINFO : finished {counter}/{len(target_nodes)}', end='')
                

            result['time'] = total

            return_tuple += (result,)
        return return_tuple

    def save_result(self,adversarial_name, seed, dataset_name,model_name,budget ):
        saved_path = "{}{}/{}/budget{}/".format(PATH_CACHED_PRED,model_name,dataset_name,budget)
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        
        saved_file = "{}{}_{}.json".format(saved_path,adversarial_name,seed)
        saved_dict = {
            'labels':self.labels,
            'evasion_preds': self.evasion_preds,
            'poision_preds':self.poision_preds
        }
        with open(saved_file, 'w') as json_file:
            json.dump(saved_dict, json_file,indent=4)



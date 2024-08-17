import sys
import os
import copy
import timeit
import statistics as stats
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.ModelSupervisor import ModelSupervisor
from utility.util import set_random
from torch_geometric.data import Data
from greatx.attack.attacker import Attacker


class AdversarialSupervisor:
    def __init__(self,
                 adversarial:Attacker, 
                 data:Data,
                 idx_train,
                 idx_val,
                 idx_test,
                 attack_structure: bool = True,
                 attack_feature: bool = False,
                 device = 'cpu',
                 **model_kwargs
                 ) -> None:
        self.attack_structure = attack_structure
        self.attack_feature = attack_feature
        self.adversarial = adversarial
        self.data = data
        self.device= device
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.mode_config = model_kwargs
    
    def target_attack(self,target_nodes,seed,budget,poison = True, evasion = False,train_before_model = None,**kwargs):
        assert poison or evasion, "Either poision or evasion or both."
        set_random(seed)
        
        if evasion:
            assert train_before_model is not None
            evasion_model = train_before_model
            miss_evasion = 0
        
        if poison:
           miss_poison = 0

        attack_time = []
        counter = 0     
        for target_node in target_nodes:
            self.adversarial.reset()
            self.adversarial.set_max_perturbations(budget)

            start = timeit.default_timer()
            
            self.adversarial.attack(target = target_node,num_budgets = budget, structure_attack = self.attack_structure, feature_attack = self.attack_feature)
            
            duration = timeit.default_timer() - start
            attack_time.append(duration)

            modiefied_data = copy.deepcopy(self.adversarial.data())
            if poison:
                poison_model =  ModelSupervisor(modiefied_data,device=self.device,seed=seed,**self.mode_config)
                poison_model.train_model(self.idx_train,self.idx_val,self.idx_test)
                poison_prediction = poison_model.get_model_prediction(modiefied_data)
                if (poison_prediction[target_node] == self.data.y[target_node]).item() == 0:
                    miss_poison += 1

            if evasion:
                evasion_prediction = evasion_model.get_model_prediction(modiefied_data)
                if (evasion_prediction[target_node] == self.data.y[target_node]).item() == 0:
                    miss_evasion += 1
            
            counter += 1
            print(f'\rINFO : finished {counter}/{len(target_nodes)}', end='')

        print("\n")
        result_dict = {
            'time': stats.mean(attack_time)
        }

        if poison:
            result_dict['poision'] = miss_poison/len(target_nodes)

        if evasion:
            result_dict['evasion'] = miss_evasion/len(target_nodes)

        return result_dict

        
        

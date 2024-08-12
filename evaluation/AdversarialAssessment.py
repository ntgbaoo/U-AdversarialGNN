
import sys
import os
import numpy as np
import statistics as stats
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from greatx.attack.attacker import Attacker
from adversarial_attack.SurrogateLoader import SurrogateLoader
from adversarial_attack.AdversarialSupervisor import AdversarialSupervisor
from torch_geometric.data import Data
from adversarial_attack.NodeSelector import NodeSelector

from evaluation.ModelSupervisor import ModelSupervisor
from utility.util import set_random

class AdversarialAssessment:
    def __init__(self,
                 data: Data,
                 fold_config_list : list,
                 adversarial_class:Attacker,
                 device = 'cpu',
                 base_seed = 710,
                 R = 3,
                 surogate_name = None,
                 surogate_model_configs = None,
                 **adversarial_args
                 ) -> None:
        self.data = data
        self.fold_config_list = fold_config_list
        self.adversarial_class = adversarial_class
        self.adversarial_method = adversarial_class(data)
        self.device = device
        self.base_seed = base_seed
        self.R = R
        self.adversarial_args = adversarial_args
        self.surogate_name = surogate_name
        self.surogate_model_configs = surogate_model_configs
        

    def evaluate(self,budget,attack_structure = True,attack_feature = False,poision = True,evasion = False):
        
        result = {
            'time':[]
        }
        if poision:
            result['poision'] = {
                'high_margin':[],
                'low_margin':[],
                'high_degree':[],
                'low_degree':[],
                'other':[],
                'total':[]
            }
        
        if evasion:
            result['evasion'] = {
                'high_margin':[],
                'low_margin':[],
                'high_degree':[],
                'low_degree':[],
                'other':[],
                'total':[]
            }

        
        for index,fold_config in enumerate(self.fold_config_list):
            split = fold_config.get('split')
            best_config = fold_config.get('model_config')
            idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_train']
                        
            for r in range(0,self.R):
                seed = r + self.base_seed
                self.adversarial_method = self.adversarial_class(self.data,device = self.device)
                if self.surogate_name is not None:
                    surrogate = SurrogateLoader.load_surrogate(self.data,idx_train, idx_val,device = self.device,
                                                               seed = seed,model = self.surogate_name,model_configs = self.surogate_model_configs[index])
                    self.adversarial_method.reset()
                    self.adversarial_method.setup_surrogate(surrogate)
                
                set_random(r + self.base_seed)
                node_select_model = ModelSupervisor(self.data,device=self.device,seed= seed,**best_config)
                node_select_model.train_model(idx_train,idx_val,idx_test)
                high,low,high_degree,low_degree,others = NodeSelector.select_nodes_margin_degree(node_select_model,self.data,idx_test,seed=seed)
                adv_super =AdversarialSupervisor(adversarial = self.adversarial_method, 
                                                data = self.data,
                                                idx_train =  idx_train,
                                                idx_val = idx_val,
                                                idx_test = idx_test,
                                                attack_structure = attack_structure,
                                                attack_feature = attack_feature,
                                                device = self.device,
                                                **best_config)
                
                high_margin_acc = adv_super.target_attack(high,seed=seed,poison=poision,evasion=evasion,budget=budget,train_before_model=node_select_model)
                low_margin_acc = adv_super.target_attack(low,seed=seed,poison=poision,evasion=evasion,budget=budget,train_before_model=node_select_model)
                high_degree_acc = adv_super.target_attack(high_degree,seed=seed,poison=poision,evasion=evasion,budget=budget,train_before_model=node_select_model)
                low_degree_acc = adv_super.target_attack(low_degree,seed=seed,poison=poision,evasion=evasion,budget=budget,train_before_model=node_select_model)
                other_acc = adv_super.target_attack(others,seed=seed,poison=poision,evasion=evasion,budget=budget,train_before_model=node_select_model)

                total_time = high_margin_acc['time'] + low_margin_acc['time'] + high_degree_acc['time'] + low_degree_acc['time'] + other_acc['time']
                result['time'].append(total_time)


                if poision:
                    result['poision']['high_margin'].append(high_margin_acc['poision'])
                    result['poision']['low_margin'].append(low_margin_acc['poision'])
                    result['poision']['high_degree'].append(high_degree_acc['poision'])
                    result['poision']['low_degree'].append(low_degree_acc['poision'])
                    result['poision']['other'].append(other_acc['poision'])
                    result['poision']['total'].append((high_margin_acc['poision'] + low_margin_acc['poision'] + high_degree_acc['poision'] + low_degree_acc['poision'] + other_acc['poision'])*10/50)

                if evasion:
                    result['evasion']['high_margin'].append(high_margin_acc['evasion'])
                    result['evasion']['low_margin'].append(low_margin_acc['evasion'])
                    result['evasion']['high_degree'].append(high_degree_acc['evasion'])
                    result['evasion']['low_degree'].append(low_degree_acc['evasion'])
                    result['evasion']['other'].append(other_acc['evasion'])
                    result['evasion']['total'].append((high_margin_acc['evasion'] + low_margin_acc['evasion'] + high_degree_acc['evasion'] + low_degree_acc['evasion'] + other_acc['evasion'])*10/50)

                
        return_dict = {
            'time':(stats.mean(result['time']),stats.stdev(result['time']))
        }
        if poision:
            return_dict['poision'] = {}
            poision_dict = result['poision']
            for key,value in poision_dict.items():
                return_dict['poision'][key] = (stats.mean(value),stats.mean(value))

            
        if evasion:
            return_dict['evasion'] = {}
            poision_dict = result['evasion']
            for key,value in poision_dict.items():
                return_dict['evasion'][key] = (stats.mean(value),stats.mean(value))


               
        return return_dict


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
        

    def evaluate(self,budget,attack_structure = True,attack_feature = False,poision = True,evasion = False):
        high_margin_acc_list = []
        low_margin_acc_list = []
        high_degree_acc_list = []
        low_degree_acc_list  = []
        other_acc_list = []

        high_degree = []
        for fold_config in self.fold_config_list:
            split = fold_config.get('split')
            best_config = fold_config.get('model_config')
            idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_train']
                        
            for r in range(0,self.R):
                seed = r + self.base_seed
                self.adversarial_method.reset()
                if self.surogate_name is not None:
                    self.adversarial_method = self.adversarial_class(self.data,device = self.device)
                    surrogate = SurrogateLoader.load_surrogate(self.data,idx_train, idx_val,device = self.device,seed = seed,model = self.surogate_name)
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
                
                high_margin_acc_list.append(adv_super.target_attack(high,seed=seed,poison=poision,evasion=evasion,budget=budget,train_before_model=node_select_model))
                low_margin_acc_list.append(adv_super.target_attack(low,seed=seed,poison=poision,evasion=evasion,budget=budget,train_before_model=node_select_model))
                high_degree_acc_list.append(adv_super.target_attack(high_degree,seed=seed,poison=poision,evasion=evasion,budget=budget,train_before_model=node_select_model))
                low_degree_acc_list.append(adv_super.target_attack(low_degree,seed=seed,poison=poision,evasion=evasion,budget=budget,train_before_model=node_select_model))
                other_acc_list.append(adv_super.target_attack(others,seed=seed,poison=poision,evasion=evasion,budget=budget,train_before_model=node_select_model))

                
                
                
        result_dict = {
            'high_margin':(stats.mean(high_margin_acc_list),stats.stdev(high_margin_acc_list)),
            'low_margin':(stats.mean(low_margin_acc_list),stats.stdev(low_margin_acc_list)),
            'high_degree':(stats.mean(high_degree_acc_list),stats.stdev(high_degree_acc_list)),
            'low_degree':(stats.mean(low_degree_acc_list),stats.stdev(low_degree_acc_list)),
            'other':(stats.mean(other_acc_list),stats.stdev(other_acc_list)),
        }
               
        return result_dict

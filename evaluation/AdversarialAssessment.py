
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
from adversarial_attack.utils import select_nodes
from utility.util import load_model_config

class AdversarialAssessment:
    def __init__(self,
                 data: Data,
                 fold_config_list : list,
                 adversarial_method:Attacker,
                 device = 'cpu',
                 base_seed = 710,
                 R = 3,
                 surogate = 'gcn',
                 **adversarial_args
                 ) -> None:
        self.data = data
        self.fold_config_list = fold_config_list
        self.adversarial_method = adversarial_method
        self.device = device
        self.base_seed = base_seed
        self.R = R
        self.adversarial_args = adversarial_args
        self.surogate = surogate
        

    def evaluate(self,budget,attack_structure,attack_feature):
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

            # data:Data,idx_train,idx_val,model = 'gcn',device = 'cpu',model_configs:dict = dict(),seed = 720
            
            fold_accuracies = []
            for r in range(0,self.R):
                self.adversarial_method.reset()
                if self.surogate is not None:
                    surrogate = SurrogateLoader.load_surrogate(self.data,idx_train, idx_val,device = self.device,seed = r + self.base_seed,model = self.surogate)
                    self.adversarial_method.reset()
                    self.adversarial_method.setup_surrogate(surrogate)
                set_random(r + self.base_seed)
                node_select_model = ModelSupervisor(self.data,device=self.device,seed= r + self.base_seed,**best_config)
                node_select_model.train_model(self.idx_train,self.idx_val,self.idx_test)
                high,low,high_degree,low_degree,others = NodeSelector.select_nodes_margin_degree(node_select_model)
                adv_super =AdversarialSupervisor(adversarial = self.adversarial_method, 
                 data = self.data,
                 idx_train =  idx_train,
                 idx_val = idx_val,
                 idx_test = idx_test,
                 attack_structure = attack_structure,
                 attack_feature = attack_feature,
                 device = self.device,
                 **best_config)
                high_margin_acc_list.append(AdversarialSupervisor.target_attack(high))
                low_margin_acc_list.append(AdversarialSupervisor.target_attack(low))
                high_degree_acc_list.append(AdversarialSupervisor.target_attack(high_degree))
                low_degree_acc_list.append(AdversarialSupervisor.target_attack(low_degree))
                other_acc_list.append(AdversarialSupervisor.target_attack(others))

                
                
                
        result_dict = {
            'high_margin':(stats.mean(high_margin_acc_list),stats.stdev(high_margin_acc_list)),
            'low_margin':(stats.mean(low_margin_acc_list),stats.stdev(low_margin_acc_list)),
            'high_degree':(stats.mean(high_degree_acc_list),stats.stdev(high_degree_acc_list)),
            'low_degree':(stats.mean(low_degree_acc_list),stats.stdev(low_degree_acc_list)),
            'other':(stats.mean(other_acc_list),stats.stdev(other_acc_list)),
        }
               
        return result_dict

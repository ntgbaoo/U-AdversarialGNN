
import sys
import os
import numpy as np
import statistics as stats
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from adversarial_attack.SurrogateLoader import SurrogateLoader
from adversarial_attack.AdversarialSupervisor import AdversarialSupervisor

from evaluation.ModelSupervisor import ModelSupervisor
from data_loader.IData import IData
from utility.util import set_random
from adversarial_attack.utils import select_nodes
from utility.util import load_model_config

class AdversarialAssessment:
    def __init__(self,
                 data: IData,
                 fold_config_list : list,
                 adversarial_method,
                 device = 'cpu',
                 base_seed = 710,
                 R = 3,
                 **adversarial_args
                 ) -> None:
        self.data = data
        self.fold_config_list = fold_config_list
        self.adversarial_method = adversarial_method
        self.device = device
        self.base_seed = base_seed
        self.R = R
        self.adversarial_args = adversarial_args

    def evaluate(self,budget):
        method_accuracy = []
        for fold_config in self.fold_config_list:
            fold = fold_config.get('fold')
            best_config = fold_config.get('model_config')
            idx_train, idx_val, idx_test = fold['idx_train'],fold['idx_val'],fold['idx_train']

            gcn_surrogate = SurrogateLoader.load_surrogate(self.data,idx_train, idx_val,device = self.device)
            fold_accuracies = []
            for r in range(0,self.R):
                set_random(r + self.base_seed)
                node_list = select_nodes(gcn_surrogate)
                miss = 0
                false_nodes = []
                for node in node_list:
                   adversarial_method = self.adversarial_method(**self.adversarial_args)
                   supervisor_adversarial = ModelSupervisor(self.data,device=self.device,**best_config)

                   adversarial_supervisor = AdversarialSupervisor(adversarial = adversarial_method,
                                                           model_supervisor = supervisor_adversarial,
                                                           data = self.data,
                                                           surrogate_model = gcn_surrogate,
                                                           target_nodes = node,
                                                           attack_structure= True,
                                                           attack_feature= False
                                                           )
                   success = 0
                   # @TODO:**FGA_config need to be changed to fixed input
                   # success = adversarial_supervisor.attack(node,r + self.base_seed,**FGA_config)
                   if success == 0:
                    miss += 1
                    false_nodes.append(node)
                accuracy = miss /len(node_list)
                fold_accuracies.append(accuracy)
            method_accuracy.append(stats.mean(fold_accuracies))
        return stats.mean(method_accuracy),stats.stdev(method_accuracy)

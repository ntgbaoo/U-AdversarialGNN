import sys
import os
import statistics as stats
import pandas as pd
from greatx.attack.targeted import SGAttack,RandomAttack,DICEAttack,GFAttack,FGAttack,PGDAttack,IGAttack
from adversarial_attack.Nettack import Nettack
from adversarial_attack.GOttack import OrbitAttack
from adversarial_attack.PRBCDAttack import PRBCDAttack
from adversarial_attack.SurrogateLoader import SurrogateLoader
from static import *
from utility.ConfigHandler import ConfigHandler
from evaluation.ModelSupervisor import ModelSupervisor
from adversarial_attack.NodeSelector import NodeSelector
from adversarial_attack.AdversarialSupervisor import AdversarialSupervisor


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader.DataLoader import DataLoader


class AdversarialAssessment:
    def __init__(self,
                 model:str,
                 dataset : str,
                 adversarial : str,
                 budgets: list,
                 num_runs = 3,
                 num_split= 5,
                 base_seed = 720,
                 device = "cpu") -> None:
        
        self.model = model
        self.device = device
        self.dataset_name = dataset
        self.data,_ =  DataLoader.load(dataset)
        self.adversarial_name = adversarial
        self.budgets = budgets
        self.num_runs = num_runs
        self.num_splits = num_split
        self.base_seed = base_seed
    
    def _get_stat(results_list: list):
        return (stats.mean(results_list),stats.stdev(results_list))
    
    def _load_attack(adversarial_name:str):
        if adversarial_name == SGA_NAME:
            return SGAttack
        elif adversarial_name == NETTACK_NAME:
            return Nettack
        elif adversarial_name == FGA_NAME:
            return FGAttack
        elif adversarial_name == RND_NAME:
            return RandomAttack
        elif adversarial_name == DICE_NAME:
            return DICEAttack
        elif adversarial_name == GFATTACK_NAME:
            return GFAttack
        elif adversarial_name == PGDATTACK_NAME:
            return PGDAttack
        elif adversarial_name == IGATTACK_NAME:
            return IGAttack
        elif adversarial_name == PRBCD_NAME:
            return PRBCDAttack
        elif adversarial_name == GOTTACK:
            return OrbitAttack
        else:
            raise Exception("Unsupport adversarial method")
        
    def _get_surrogate_model(adversarial_name):
        if adversarial_name == SGA_NAME:
            return 'SGC'
        elif adversarial_name in [NETTACK_NAME,FGA_NAME,PGDATTACK_NAME,IGATTACK_NAME]:
            return 'GCN'
        else:
            return None
        
    def evaluate(self):
        columns = ['budget','high_margin','low_margin','high_degree','low_degree','other','total','time']
        rowlist_evasion = []
        rowlist_poision = []

        for budget in range(1,6):
            evasion_dict = {
                'high_margin':[],
                'low_margin':[],
                'high_degree':[],
                'low_degree':[],
                'other':[],
                'total':[]
            }
            poision_dict = {
                'high_margin':[],
                'low_margin':[],
                'high_degree':[],
                'low_degree':[],
                'other':[],
                'total':[]
            }
            time = []

            for split_idx in range(5):

                split_model = ConfigHandler.load_model_config(self.model,"best_config",self.dataset_name,split_idx)
                split = split_model['split']
                configs = split_model.get('model_config')
                idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_train']

                for idx_run in range(self.num_runs):
                    seed = self.base_seed + idx_run
                    node_selector = ModelSupervisor(self.data,self.device,**configs)
                    node_selector.train_model(idx_train,idx_val,idx_test)
                    bucket_target_nodes= NodeSelector.select_nodes_margin_degree(node_selector,self.data,idx_test,seed=seed)
                    # node_list = high +low + highest_degree + lowest_degree + other


                    # surrogate_supervisor = 
                    attacker = self._load_attack(self.adversarial_name)(self.data, device=self.device,seed=seed)

                    if self.adversarial_name == PRBCD_NAME:
                        attacker.update_config(10000,500)
                    
                    if self.adversarial_name == PRBCD_NAME:
                        if self.model == "GCN":
                            attacker.setup_surrogate(node_selector.model)
                        else:
                            split_model_surrogate = ConfigHandler.load_model_config("GCN","best_config",self.dataset_name,split_idx)
                            configs_surrogate = split_model_surrogate.get('model_config')
                            surrogate = ModelSupervisor(self.data,self.device,**configs_surrogate)
                            surrogate.train_model(idx_train,idx_val,idx_test)
                            attacker.setup_surrogate(surrogate.model)

                    if self._get_surrogate_model(self.adversarial_name) is not None:
                        
                        adversarial_surrogate = SurrogateLoader.load_surrogate(self.data,idx_train,idx_val,device=self.device,seed=seed,model=self._get_surrogate_model(self.adversarial_name))
                        attacker.setup_surrogate(adversarial_surrogate)


                    adversarial_supervisor = AdversarialSupervisor(self.data,attacker,idx_train,idx_val,idx_test,self.device,**configs)
                    high_result,low_result, highest_degree_result,lowest_degree_result,other_result = adversarial_supervisor.target_attack(bucket_target_nodes,budget,evasion=True,poision=True, evasion_model=node_selector,seed = seed)
                    
                    evasion_dict['high_margin'].append( high_result[EVASION]/10)
                    poision_dict['high_margin'].append( high_result[POSION]/10)

                    evasion_dict['low_margin'].append( low_result[EVASION]/10)
                    poision_dict['low_margin'].append( low_result[POSION]/10)

                    evasion_dict['high_degree'].append( highest_degree_result[EVASION]/10)
                    poision_dict['high_degree'].append( highest_degree_result[POSION]/10)

                    evasion_dict['low_degree'].append( lowest_degree_result[EVASION]/10)
                    poision_dict['low_degree'].append( lowest_degree_result[POSION]/10)

                    evasion_dict['other'].append( other_result[EVASION]/10)
                    poision_dict['other'].append( other_result[POSION]/10)
                    
                    evasion_count = high_result[EVASION] + high_result[EVASION] + low_result[EVASION] + highest_degree_result[EVASION] + lowest_degree_result[EVASION] + other_result[EVASION]
                    poision_count = high_result[POSION] + high_result[EVASION] + low_result[POSION] + highest_degree_result[POSION] + lowest_degree_result[POSION] + other_result[POSION]
                    evasion_dict['total'].append(evasion_count/50)
                    poision_dict['total'].append(poision_count/50)

                    time_total = high_result['time'] + high_result['time'] + low_result['time'] + highest_degree_result['time'] + lowest_degree_result['time'] + other_result['time']
                    time.append(time_total)

            print(evasion_dict)
            print(poision_dict)
            print(time) 

            row_evasion = [
                budget,
                self._get_stat(evasion_dict['high_margin']),
                self._get_stat(evasion_dict['low_margin']),
                self._get_stat(evasion_dict['high_degree']),
                self._get_stat(evasion_dict['low_degree']),
                self._get_stat(evasion_dict['other']),
                self._get_stat(evasion_dict['total']),
                self._get_stat(time),
            ]
            rowlist_evasion.append(row_evasion)

            row_poision = [
                budget,
                self._get_stat(poision_dict['high_margin']),
                self._get_stat(poision_dict['low_margin']),
                self._get_stat(poision_dict['high_degree']),
                self._get_stat(poision_dict['low_degree']),
                self._get_stat(poision_dict['other']),
                self._get_stat(poision_dict['total']),
                self._get_stat(time),
            ]
            rowlist_poision.append(row_poision)
            print(evasion_dict)
            print(poision_dict)

        df_poision = pd.DataFrame(rowlist_poision,columns=columns)
        df_evasion = pd.DataFrame(rowlist_evasion,columns=columns)

        return df_poision,df_evasion


import sys
import os
import statistics as stats
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader.DataLoader import DataLoader
from data_loader.Splitter import Splitter
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



class AdversarialAssessment:
    def __init__(self,
                 model:str,
                 dataset : str,
                 adversarial : str,
                 budgets: list,
                 num_runs = 3,
                 num_split= 5,
                 base_seed = 720,
                 use_tune_model = True,
                 use_degree_node_selection =True,
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
        self.use_tune_model = use_tune_model
        self.use_degree_node_selection = use_degree_node_selection
        self.node_selector = NodeSelector.select_nodes_margin_degree if use_degree_node_selection else NodeSelector.select_nodes_margin
        self.node_category = ['high_margin','low_margin','high_degree','low_degree','other'] if use_degree_node_selection else ['high_margin','low_margin','other']
        self.config_name = "best_config" if use_tune_model else "default"
        
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
        
    def _prepare_dict(self):
        all_categories = self.node_category + ['total']
        return_dict = {}
        for category in all_categories:
            return_dict[category] = []

        return return_dict
    
    def _load_config_split(self,data):
        all_split_config = []
        if not self.use_tune_model:
            split_config = ConfigHandler.load_split_config(config_name="default")
            folds = splitter.split(data.x.numpy(),data.y.numpy())


            splitter = Splitter(**split_config)
        for split_idx in range(self.num_splits):
            config = ConfigHandler.load_model_config(self.model,self.config_name,self.data)
            if not self.use_tune_model:
                config['split'] ={
                    IDX_TRAIN: folds[split_idx].idx_train,
                    IDX_VAL:folds[split_idx].idx_val,
                    IDX_TEST:folds[split_idx].idx_test
                }
            all_split_config.append(config)
        return all_split_config
    
    def _get_surrogate_name(attack_method):
        if attack_method == SGA_NAME:
            return 'SGC'
        elif attack_method in [NETTACK_NAME,FGA_NAME,PGDATTACK_NAME,IGATTACK_NAME]:
            return 'GCN_surrogate'
        else:
            return None


        
   
    def evaluate(self,evasion = True, poision = True, budgets_list = [1,2,3,4,5]):
        assert evasion or poision, "At least on of evasion and poision setting"
        num_rand = 10 if self.use_degree_node_selection else 30
        data,_ = DataLoader.load(self.data)
        rowlist_evasion = [] if evasion else None
        rowlist_poision = [] if poision else None
        all_split_configs = self._load_config_split(data)

        for budget in budgets_list:
            evasion_dict = self._prepare_dict() if evasion else None
            poision_dict = self._prepare_dict() if poision else None
            time = []

            for split_idx in range(self.num_splits):
                split_model = all_split_configs[split_idx]
                split = split_model['split']
                configs = split_model.get('model_config')
                idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_test']
                
                for idx_run in range(self.num_runs):
                    seed = self.base_seed + idx_run
                    node_selector = ModelSupervisor(data,self.device,seed=seed,**configs)
                    node_selector.train_model(idx_train,idx_val,idx_test)
                    bucket_target_nodes= self.node_selector(node_selector,data,idx_test,seed=seed,num_rand=num_rand)
        
                    attacker = self._load_attack(args.adversarial)(data, device=self.device,seed=seed)
                    surrogate_name = self._get_surrogate_name(self.adversarial)

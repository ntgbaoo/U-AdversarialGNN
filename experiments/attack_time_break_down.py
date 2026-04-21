import os.path as osp

import torch
import torch_geometric.transforms as T
import sys
import os
import statistics as stats
import pandas as pd
import timeit
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility import ConfigHandler
from data_loader import DataLoader
from greatx.attack.targeted import SGAttack,RandomAttack,DICEAttack,GFAttack,FGAttack,PGDAttack,IGAttack
from adversarial_attack import Nettack
from greatx.datasets import GraphDataset
from greatx.nn.models import GCN,SGC
from greatx.training import Trainer
from greatx.training.callbacks import ModelCheckpoint
from evaluation import ModelSupervisor
from adversarial_attack import NodeSelector
from adversarial_attack import SurrogateLoader
from adversarial_attack import AdversarialSupervisor
from static import *
from utility.config import args
from adversarial_attack import PRBCDAttack,GRBCDAttack, DegreeAttack,L1NormAttack,OrbitAttack,RandomDegreeAttack,RandomOnlyAddAttack, RandomOnlyRemoveAttack
from data_loader import Splitter
import statistics as stats
from greatx.utils import mark, split_nodes

def save_results(experiment_id, results, intermediate_path = ''):
    partial_path = f'{PATH_RESULT}/{intermediate_path}'
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)

    result_path = f'{partial_path}/{experiment_id}.csv'
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=results.keys())
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df._append(results, ignore_index=True)
    result_df.to_csv(result_path, index=False)

def load_attack(attack_method):
    if attack_method == SGA_NAME:
        return SGAttack
    elif attack_method == NETTACK_NAME:
        return Nettack
    elif attack_method == FGA_NAME:
        return FGAttack
    elif attack_method == RND_NAME:
        return RandomAttack
    elif attack_method == DICE_NAME:
        return DICEAttack
    elif attack_method == GFATTACK_NAME:
        return GFAttack
    elif attack_method == PGDATTACK_NAME:
        return PGDAttack
    elif attack_method == IGATTACK_NAME:
        return IGAttack
    elif attack_method == PRBCD_NAME:
        from adversarial_attack import PRBCDAttack,GRBCDAttack
        return PRBCDAttack
    elif attack_method == GRBCD_NAME:
        return GRBCDAttack
    elif attack_method == PRBCD_greatX:
        from greatx.attack.targeted import PRBCDAttack
        return PRBCDAttack
    elif attack_method == DEGREE_ATTACK:
        return DegreeAttack
    elif attack_method == L1_NORM_ATTACK:
        return L1NormAttack
    elif attack_method == GOTTACK:
        return OrbitAttack
    elif attack_method == "random_degree_attack":
        return RandomDegreeAttack
    elif attack_method == "random_degree_add_attack":
        return RandomOnlyAddAttack
    elif attack_method == "random_degree_remove_attack":
        return RandomOnlyRemoveAttack
    

    

def get_surrogate_name(attack_method):
    if attack_method == SGA_NAME:
        return 'SGC'
    elif attack_method in [NETTACK_NAME,FGA_NAME,PGDATTACK_NAME,IGATTACK_NAME,PRBCD_greatX,GOTTACK]:
        return 'GCN_surrogate'
    elif attack_method == PRBCD_NAME:
        return 'GCN'
    else:
        return None

if __name__ == "__main__":
    classifier_config_version = args.config_setting
    

    data,split = DataLoader.load(args.dataset)
    if classifier_config_version == "default":
        split_config = ConfigHandler.load_split_config(config_name="default")
        splitter = Splitter(**split_config)
        folds = splitter.split(data.x.numpy(),data.y.numpy())
        
    seed = 720
    split_idx = 0
    budget =1
    time_break_down = {}
    time_break_down['surrogate'] = []
    time_break_down['pre_compute'] = []
    time_break_down['finding_best_edges'] = []

    split_model = ConfigHandler.load_model_config(args.model,classifier_config_version,args.dataset,split_idx)
    if classifier_config_version != "default":
        split = split_model['split']
        idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_test']
    else:
        fold = folds[split_idx]
        idx_train,idx_val,idx_test =fold.idx_train,fold.idx_val,fold.idx_test

    configs = split_model.get('model_config')
    # print(configs)

    # Target node selection
    node_selector = ModelSupervisor(data,args.device,seed=seed,**configs)
    node_selector.train_model(idx_train,idx_val,idx_test)
    target_node_dict= NodeSelector.select_nodes_margin_degree(node_selector,data,idx_test,seed=seed)
    category_list = ['high_margin','low_margin','highest_degree','lowest_degree','other']
    target_nodes = []
    for cate in category_list:
        target_nodes.extend(target_node_dict[cate])

    # Attack set up
    
    surrogate_name = get_surrogate_name(args.adversarial)
    if classifier_config_version.split("_")[0] in AVAILABLE_AGGRE and "best" in classifier_config_version:
        surrogate_config_setting = "best_config"
    else:
        surrogate_config_setting = classifier_config_version

    if args.adversarial is GFATTACK_NAME and args.dataset == "cora":
        attacker.T = data.num_nodes//2   
    
    if args.adversarial == PRBCD_NAME or args.adversarial == GRBCD_NAME:
        # attacker.update_config(10000,200)
        split_model_surrogate = ConfigHandler.load_model_config("GCN",surrogate_config_setting,args.dataset,split_idx)
        configs_surrogate = split_model_surrogate.get('model_config')
        surrogate = ModelSupervisor(data,args.device,seed=seed,**configs_surrogate)

        ##### Set up surrogate
        start_train = timeit.default_timer()
        surrogate.train_model(idx_train,idx_val,idx_test)
        time_surrogate = timeit.default_timer() - start_train
        # attacker.setup_surrogate(surrogate.model) 

    if surrogate_name is not None:
        split_model_surrogate = ConfigHandler.load_model_config(surrogate_name,surrogate_config_setting,args.dataset,split_idx)
        configs_surrogate = split_model_surrogate.get('model_config')
        surrogate = ModelSupervisor(data,args.device,seed=seed,**configs_surrogate)
        
        ##### Set up surrogate
        start_train = timeit.default_timer()
        surrogate.train_model(idx_train,idx_val,idx_test)
        time_surrogate = timeit.default_timer() - start_train
        # attacker.setup_surrogate(surrogate.model) 

    dummy_node = True
    for node in tqdm([1] + target_nodes):
        
        attacker = load_attack(args.adversarial)(data, device=args.device,seed=seed)
        
        # attacker.reset()
        if args.adversarial == PRBCD_NAME or args.adversarial == GRBCD_NAME:
            attacker.update_config(10000,200)
        start_time = timeit.default_timer()
        if surrogate_name is not None: attacker.setup_surrogate(surrogate.model) 
        time_set_up = timeit.default_timer() - start_time + (time_surrogate if surrogate_name is not None else 0)
        
        
        attacker.set_max_perturbations(budget)
        attacker.reset()
        try:
            attacker.attack(node,num_budgets = budget,structure_attack = True, feature_attack = False)
        except Exception as e:
            print(e)
            if 'pre_compute' in attacker.time_break_down:
                time_break_down['pre_compute'].append(attacker.time_break_down['pre_compute'])
            if 'pre_compute' in attacker.time_break_down:
                time_break_down['pre_compute'].append(attacker.time_break_down['pre_compute'])
        

        if not dummy_node:
            time_break_down_per_attack = attacker.time_break_down
            # print(time_break_down_per_attack)
            # print(time_set_up)
            time_break_down['surrogate'].append(time_set_up)

            for time_type in time_break_down_per_attack:
                time_break_down[time_type].append(time_break_down_per_attack[time_type])
        
        if dummy_node:
            dummy_node = False
    results_dict = {}
    results_dict['attack'] = args.adversarial
    results_dict['dataset'] =  args.dataset
    print(results_dict)
    for time_type in time_break_down.keys():
        all_time = time_break_down[time_type]
        results_dict[time_type] = stats.mean(all_time) if len(all_time) > 1 else all_time[0]
        results_dict[f'{time_type}_std'] = stats.stdev(all_time) if len(all_time) > 1 else 0

    save_results(
        f'results_{args.dataset}',
        results_dict,
        'time_break_down'
    )
    

import os.path as osp

import torch
import torch_geometric.transforms as T
import sys
import os
import statistics as stats
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.ConfigHandler import ConfigHandler
from data_loader.DataLoader import DataLoader
from greatx.attack.targeted import SGAttack,RandomAttack,DICEAttack,GFAttack,FGAttack,PGDAttack,IGAttack
from adversarial_attack.Nettack import Nettack
from greatx.datasets import GraphDataset
from greatx.nn.models import GCN,SGC
from greatx.training import Trainer
from greatx.training.callbacks import ModelCheckpoint
from evaluation.ModelSupervisor import ModelSupervisor
from adversarial_attack.NodeSelector import NodeSelector
from adversarial_attack.SurrogateLoader import SurrogateLoader
from adversarial_attack.AdversarialSupervisor import AdversarialSupervisor
from static import *
from utility.config import args
from adversarial_attack.PRBCDAttack import PRBCDAttack


from greatx.utils import mark, split_nodes

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
        return PRBCDAttack
    

    

def get_surrogate_name(attack_method):
    if attack_method == SGA_NAME:
        return 'SGC'
    elif attack_method in [NETTACK_NAME,FGA_NAME,PGDATTACK_NAME,IGATTACK_NAME]:
        return 'GCN_surrogate'
    else:
        return None

def get_stat(results_list):
    return (stats.mean(results_list),stats.stdev(results_list))

if __name__ == "__main__":
    # args.dataset = 'cora'
    # args.model = "GCN"
    # args.seed = 720
    # args.adversarial = SGA_NAME

    data,split = DataLoader.load(args.dataset)
    columns = ['budget','high_margin','low_margin','other','total','time']
    rowlist_evasion = []
    rowlist_poision = []

    for budget in range(1,6):
        evasion_dict = {
            'high_margin':[],
            'low_margin':[],
            'other':[],
            'total':[]
        }
        poision_dict = {
            'high_margin':[],
            'low_margin':[],
            'other':[],
            'total':[]
        }
        time = []

        for split_idx in range(5):

            split_model = ConfigHandler.load_model_config(args.model,"best_config",args.dataset,split_idx)
            split = split_model['split']
            configs = split_model.get('model_config')
            idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_test']

            for idx_run in range(args.num_run):
                seed = args.seed + idx_run
                node_selector = ModelSupervisor(data,args.device,seed=seed,**configs)
                node_selector.train_model(idx_train,idx_val,idx_test)
                bucket_target_nodes= NodeSelector.select_nodes_margin(node_selector,data,idx_test,seed=seed,num_rand=30)

                # surrogate_supervisor = 
                attacker = load_attack(args.adversarial)(data, device=args.device,seed=seed)
                surrogate_name = get_surrogate_name(args.adversarial)

                if args.adversarial is GFATTACK_NAME and args.dataset == "cora":
                    attacker.T = data.num_nodes//2                    
                
                if args.adversarial == PRBCD_NAME:
                    attacker.update_config(10000,200)
                    if args.model == "GCN":
                        attacker.setup_surrogate(node_selector.model)
                    else:
                        split_model_surrogate = ConfigHandler.load_model_config("GCN","best_config",args.dataset,split_idx)
                        configs_surrogate = split_model_surrogate.get('model_config')
                        surrogate = ModelSupervisor(data,args.device,seed=seed,**configs_surrogate)
                        surrogate.train_model(idx_train,idx_val,idx_test)
                        attacker.setup_surrogate(surrogate.model)

                if surrogate_name is not None:
                    split_model_surrogate = ConfigHandler.load_model_config(surrogate_name,"best_config",args.dataset,split_idx)
                    configs_surrogate = split_model_surrogate.get('model_config')
                    surrogate = ModelSupervisor(data,args.device,seed=seed,**configs_surrogate)
                    surrogate.train_model(idx_train,idx_val,idx_test)
                    attacker.setup_surrogate(surrogate.model)


                adversarial_supervisor = AdversarialSupervisor(data,attacker,idx_train,idx_val,idx_test,args.device,**configs)
                high_result,low_result,other_result = adversarial_supervisor.target_attack(bucket_target_nodes,budget,evasion=True,poision=True, evasion_model=node_selector,seed = seed)
                adversarial_supervisor.save_result(args.adversarial,seed,args.dataset,args.model,budget)
                evasion_dict['high_margin'].append( high_result[EVASION]/10)
                poision_dict['high_margin'].append( high_result[POSION]/10)

                evasion_dict['low_margin'].append( low_result[EVASION]/10)
                poision_dict['low_margin'].append( low_result[POSION]/10)

                evasion_dict['other'].append( other_result[EVASION]/30)
                poision_dict['other'].append( other_result[POSION]/30)
                
                evasion_count = high_result[EVASION]  + low_result[EVASION] + other_result[EVASION]
                poision_count = high_result[POSION]+ low_result[POSION]  + other_result[POSION]
                evasion_dict['total'].append(evasion_count/50)
                poision_dict['total'].append(poision_count/50)

                time_total = high_result['time']  + low_result['time'] + other_result['time']
                time.append(time_total)

        print(evasion_dict)
        print(poision_dict)
        print(time) 

        row_evasion = [
            budget,
            get_stat(evasion_dict['high_margin']),
            get_stat(evasion_dict['low_margin']),
            get_stat(evasion_dict['other']),
            get_stat(evasion_dict['total']),
            get_stat(time),
        ]
        rowlist_evasion.append(row_evasion)

        row_poision = [
            budget,
            get_stat(poision_dict['high_margin']),
            get_stat(poision_dict['low_margin']),
            get_stat(poision_dict['other']),
            get_stat(poision_dict['total']),
            get_stat(time),
        ]
        rowlist_poision.append(row_poision)
        print(evasion_dict)
        print(poision_dict)

    df_poision = pd.DataFrame(rowlist_poision,columns=columns)
    df_evasion = pd.DataFrame(rowlist_evasion,columns=columns)

    if not os.path.exists("{}/ablation/no_degree/{}/{}/".format(PATH_RESULT,args.model,args.dataset)):
        os.makedirs("{}/ablation/no_degree/{}/{}/".format(PATH_RESULT,args.model,args.dataset))
    df_poision.to_csv("{}/ablation/no_degree/{}/{}/{}_poision.csv".format(PATH_RESULT,args.model,args.dataset,args.adversarial))
    df_evasion.to_csv("{}/ablation/no_degree/{}/{}/{}_evasion.csv".format(PATH_RESULT,args.model,args.dataset,args.adversarial))




  
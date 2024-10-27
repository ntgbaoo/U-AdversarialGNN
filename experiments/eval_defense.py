import os.path as osp

import torch
import torch_geometric.transforms as T
import sys
import os
import statistics as stats
import pandas as pd
import timeit

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
from adversarial_attack import PRBCDAttack,GRBCDAttack
from data_loader import Splitter



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
        from adversarial_attack import PRBCDAttack,GRBCDAttack
        return PRBCDAttack
    elif attack_method == GRBCD_NAME:
        return GRBCDAttack
    elif attack_method == "PRBCD_greatx":
        from greatx.attack.targeted import PRBCDAttack
        return PRBCDAttack
    

    

def get_surrogate_name(attack_method):
    if attack_method == SGA_NAME:
        return 'SGC'
    elif attack_method in [NETTACK_NAME,FGA_NAME,PGDATTACK_NAME,IGATTACK_NAME,"PRBCD_greatx"]:
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

    classifier_config_version = args.config_setting
    data,split = DataLoader.load(args.dataset)

    if classifier_config_version == "default":
        split_config = ConfigHandler.load_split_config(config_name="default")
        splitter = Splitter(**split_config)
        folds = splitter.split(data.x.numpy(),data.y.numpy())

    columns = ['budget','high_margin','low_margin','high_degree','low_degree','other','total','time','time_defense']
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
        time_defense = []
        purification_config = None
        use_purification = False

        

        for split_idx in range(5):
            if args.purification is not None:
                use_purification = True
                purification_config = ConfigHandler.load_purification_config(args.purification,args.model,"best_config",args.dataset,split=split_idx)['model_config']
                print(purification_config)

            split_model = ConfigHandler.load_model_config(args.model,classifier_config_version,args.dataset,split_idx)
            if classifier_config_version != "default":
                split = split_model['split']
                idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_test']
            else:
                fold = folds[split_idx]
                idx_train,idx_val,idx_test =fold.idx_train,fold.idx_val,fold.idx_test

            configs = split_model.get('model_config')
            print(configs)
            

            for idx_run in range(args.num_run):
                seed = args.seed + idx_run
                node_selector = ModelSupervisor(data,args.device,seed=seed,use_purification=use_purification,purification_config=purification_config,**configs)
                time_defense_start = timeit.default_timer()
                node_selector.train_model(idx_train,idx_val,idx_test)
                time_defense.append(timeit.default_timer() - time_defense_start)
                target_node_dict= NodeSelector.select_nodes_margin_degree(node_selector,data,idx_test,seed=seed)
                category_list = ['high_margin','low_margin','highest_degree','lowest_degree','other']
                bucket_target_nodes = []
                for cate in category_list:
                    bucket_target_nodes.append(target_node_dict[cate])


                # surrogate_supervisor = 
                attacker = load_attack(args.adversarial)(data, device=args.device,seed=seed)
                surrogate_name = get_surrogate_name(args.adversarial)
                if classifier_config_version.split("_")[0] in AVAILABLE_AGGRE and "best" in classifier_config_version:
                    surrogate_config_setting = "best_config"
                else:
                    surrogate_config_setting = classifier_config_version

                if args.adversarial is GFATTACK_NAME and args.dataset == "cora":
                    attacker.T = data.num_nodes//2   

                
                if args.adversarial == PRBCD_NAME or args.adversarial == GRBCD_NAME:
                    attacker.update_config(10000,200)
                    

                    split_model_surrogate = ConfigHandler.load_model_config("GCN",surrogate_config_setting,args.dataset,split_idx)
                    configs_surrogate = split_model_surrogate.get('model_config')
                    surrogate = ModelSupervisor(data,args.device,seed=seed,**configs_surrogate)
                    surrogate.train_model(idx_train,idx_val,idx_test)
                    attacker.setup_surrogate(surrogate.model)
                    
                if  args.adversarial == GRBCD_NAME:
                        attacker.update_config(10000,125)

                if surrogate_name is not None:
                    split_model_surrogate = ConfigHandler.load_model_config(surrogate_name,surrogate_config_setting,args.dataset,split_idx)
                    configs_surrogate = split_model_surrogate.get('model_config')
                    surrogate = ModelSupervisor(data,args.device,seed=seed,**configs_surrogate)
                    surrogate.train_model(idx_train,idx_val,idx_test)
                    attacker.setup_surrogate(surrogate.model)

                


                adversarial_supervisor = AdversarialSupervisor(data,attacker,idx_train,idx_val,idx_test,args.device,use_purification=use_purification,purification_config=purification_config,**configs)
                high_result,low_result, highest_degree_result,lowest_degree_result,other_result = adversarial_supervisor.target_attack(bucket_target_nodes,budget,evasion=True,poision=True, evasion_model=node_selector,seed = seed)
                adversarial_supervisor.save_result(args.adversarial,seed,args.dataset,args.model,budget,args.purification)
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
                
                evasion_count = high_result[EVASION]  + low_result[EVASION] + highest_degree_result[EVASION] + lowest_degree_result[EVASION] + other_result[EVASION]
                poision_count = high_result[POSION] + low_result[POSION] + highest_degree_result[POSION] + lowest_degree_result[POSION] + other_result[POSION]
                evasion_dict['total'].append(evasion_count/50)
                poision_dict['total'].append(poision_count/50)

                time_total = high_result['time']  + low_result['time'] + highest_degree_result['time'] + lowest_degree_result['time'] + other_result['time']
                time.append(time_total)

            print(evasion_dict)
            print(poision_dict)
            print(time) 

        row_evasion = [
            budget,
            get_stat(evasion_dict['high_margin']),
            get_stat(evasion_dict['low_margin']),
            get_stat(evasion_dict['high_degree']),
            get_stat(evasion_dict['low_degree']),
            get_stat(evasion_dict['other']),
            get_stat(evasion_dict['total']),
            get_stat(time),
            get_stat(time_defense)
        ]
        rowlist_evasion.append(row_evasion)

        row_poision = [
            budget,
            get_stat(poision_dict['high_margin']),
            get_stat(poision_dict['low_margin']),
            get_stat(poision_dict['high_degree']),
            get_stat(poision_dict['low_degree']),
            get_stat(poision_dict['other']),
            get_stat(poision_dict['total']),
            get_stat(time),
            get_stat(time_defense)
        ]
        rowlist_poision.append(row_poision)
        print(evasion_dict)
        print(poision_dict)

    df_poision = pd.DataFrame(rowlist_poision,columns=columns)
    df_evasion = pd.DataFrame(rowlist_evasion,columns=columns)


    if args.purification is not None:
        if not os.path.exists("{}{}/{}_{}/{}".format(PATH_RESULT,classifier_config_version,args.model,args.purification ,args.dataset)):
            os.makedirs("{}{}/{}_{}/{}".format(PATH_RESULT,classifier_config_version,args.model,args.purification ,args.dataset))

        df_poision.to_csv("{}{}/{}_{}/{}/{}_poision.csv".format(PATH_RESULT,classifier_config_version,args.model,args.purification ,args.dataset,args.adversarial))
        df_evasion.to_csv("{}{}/{}_{}/{}/{}_evasion.csv".format(PATH_RESULT,classifier_config_version,args.model,args.purification,args.dataset,args.adversarial))
    else:
        if not os.path.exists("{}{}/{}/{}/".format(PATH_RESULT,classifier_config_version,args.model,args.dataset)):
            os.makedirs("{}{}/{}/{}/".format(PATH_RESULT,classifier_config_version,args.model,args.dataset))

        df_poision.to_csv("{}{}/{}/{}/{}_poision.csv".format(PATH_RESULT,classifier_config_version,args.model,args.dataset,args.adversarial))
        df_evasion.to_csv("{}{}/{}/{}/{}_evasion.csv".format(PATH_RESULT,classifier_config_version,args.model,args.dataset,args.adversarial))




  
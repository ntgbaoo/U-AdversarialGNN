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
from adversarial_attack import PRBCDAttack,GRBCDAttack, DegreeAttack,L1NormAttack,OrbitAttack,RandomDegreeAttack,RandomOnlyAddAttack, RandomOnlyRemoveAttack
from data_loader import Splitter
import pickle



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
    else:
        return None

def get_stat(results_list):
    return (stats.mean(results_list),stats.stdev(results_list))

def compute_average_missclassification_rate(target_nodes, missclassified_evasion, missclassified_poison,node_degree):
    all_target_nodes = []

    for sub_list in target_nodes:
        all_target_nodes.extend(sub_list)

    target_node_degree_count = {}
    missclassified_evasion_degree_count = {}
    missclassified_poison_degree_count = {}

    for node in all_target_nodes:
        degree = node_degree[node]
        if degree not in target_node_degree_count:
            target_node_degree_count[degree] = 0
        target_node_degree_count[degree] += 1

    for node in missclassified_evasion:
        degree = node_degree[node]
        if degree not in missclassified_evasion_degree_count:
            missclassified_evasion_degree_count[degree] = 0
        missclassified_evasion_degree_count[degree] += 1

    for node in missclassified_poison:
        degree = node_degree[node]
        if degree not in missclassified_poison_degree_count:
            missclassified_poison_degree_count[degree] = 0
        missclassified_poison_degree_count[degree] += 1

    missclassified_evasion_degree_average = {}
    missclassified_poison_degree_average = {}
    
    for degree in target_node_degree_count.keys():
        total = target_node_degree_count[degree]
        missclassified_evasion_degree_average[degree] = missclassified_evasion_degree_count.get(degree,0)/total
        missclassified_poison_degree_average[degree] = missclassified_poison_degree_count.get(degree,0)/total

    return missclassified_evasion_degree_average,missclassified_poison_degree_average

    
        

if __name__ == "__main__":
    # args.dataset = 'cora'
    # args.model = "GCN"
    # args.seed = 720
    # args.adversarial = SGA_NAME

    classifier_config_version = args.config_setting
    data,split = DataLoader.load(args.dataset)
    edge_index = data.edge_index

    node_degree = {}
    evasion_results = {}
    poison_results = {}

    for src,dst in edge_index.t():
        src, dst = src.item(),dst.item()
        if src not in node_degree:
            node_degree[src] = set()
        if dst not in node_degree:
            node_degree[dst] = set()

        node_degree[src].add(dst)
        node_degree[dst].add(src)

    for node in node_degree.keys():
        node_degree[node] = len(node_degree[node])


    


    if classifier_config_version == "default":
        split_config = ConfigHandler.load_split_config(config_name="default")
        splitter = Splitter(**split_config)
        folds = splitter.split(data.x.numpy(),data.y.numpy())

    for budget in [5]:
        
        purification_config = None
        use_purification = False

        

        for split_idx in range(5):
            if args.purification is not None:
                use_purification = True
                purification_config = ConfigHandler.load_purification_config(args.purification,args.model,classifier_config_version,args.dataset,split=split_idx)['model_config']
                print(purification_config)

            split_model = ConfigHandler.load_model_config(args.model,classifier_config_version,args.dataset,split_idx)
            if classifier_config_version != "default":
                split = split_model['split']
                idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_test']
            else:
                fold = folds[split_idx]
                idx_train,idx_val,idx_test =fold.idx_train,fold.idx_val,fold.idx_test

            configs = split_model.get('model_config')
            

            for idx_run in range(args.num_run):
                seed = args.seed + idx_run
                node_selector = ModelSupervisor(data,args.device,seed=seed,use_purification=use_purification,purification_config=purification_config,**configs)
                time_defense_start = timeit.default_timer()
                node_selector.train_model(idx_train,idx_val,idx_test)
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
                adversarial_supervisor.target_attack(bucket_target_nodes,budget,evasion=True,poision=True, evasion_model=node_selector,seed = seed)
                missclassified_nodes_evasion, missclassified_nodes_poison = adversarial_supervisor.get_missclassified_nodes()
                run_evasion_result, run_poison_result = compute_average_missclassification_rate(bucket_target_nodes,missclassified_nodes_evasion,missclassified_nodes_poison,node_degree)

                for degree in run_evasion_result.keys():
                    if degree not in evasion_results:
                        evasion_results[degree] = []

                    evasion_results[degree].append(run_evasion_result[degree])

                for degree in run_poison_result.keys():
                    if degree not in poison_results:
                        poison_results[degree] = []

                    poison_results[degree].append(run_poison_result[degree])


    for degree in evasion_results.keys():
        if len(evasion_results[degree]) > 1:
            evasion_results[degree] = stats.mean(evasion_results[degree])
        else:
            evasion_results[degree] = evasion_results[degree][0]

    for degree in poison_results.keys():
        if len(poison_results[degree]) > 1:
            poison_results[degree] = stats.mean(poison_results[degree])
        else:
            poison_results[degree] = poison_results[degree][0]


    print(evasion_results)
    print(poison_results)


    
    if not os.path.exists("{}{}/{}/{}/".format(PATH_RESULT,classifier_config_version,args.model,args.dataset)):
        os.makedirs("{}{}/{}/{}/".format(PATH_RESULT,classifier_config_version,args.model,args.dataset))

    with open("{}{}/{}/{}/{}_evasion_degree.pkl".format(PATH_RESULT,classifier_config_version,args.model,args.dataset,args.adversarial), 'wb') as f:
        pickle.dump(evasion_results, f)

    with open("{}{}/{}/{}/{}_poison_degree.pkl".format(PATH_RESULT,classifier_config_version,args.model,args.dataset,args.adversarial), 'wb') as f:
        pickle.dump(poison_results, f)





  
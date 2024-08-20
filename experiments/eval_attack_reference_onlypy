
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch_geometric.transforms as T
import statistics as stats
import pandas as pd

from greatx.attack.targeted import Nettack,SGAttack
from evaluation.ModelSupervisor import ModelSupervisor
from data_loader.DataLoader import DataLoader
from utility.ConfigHandler import ConfigHandler
from static import *
from adversarial_attack.SurrogateLoader import SurrogateLoader
from adversarial_attack.NodeSelector import NodeSelector
from adversarial_attack.AdversarialSupervisor import AdversarialSupervisor
from greatx.attack.targeted import Nettack,SGAttack,RandomAttack,DICEAttack,GFAttack,FGAttack,PGDAttack,IGAttack

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

def get_surrogate_name(attack_method):
    if attack_method == SGA_NAME:
        return 'SGC'
    elif attack_method in [NETTACK_NAME,FGA_NAME,PGDATTACK_NAME,IGATTACK_NAME]:
        return 'GCN'
    else:
        return None

if __name__=="__main__":
    from utility.config import args

    # args.model = "GCN"
    # args.dataset = "cora"
    # args.adversarial =FGA_NAME
    args.num_run = 3

    
    data,_ = DataLoader.load(args.dataset)

    columns = ['high_margin','low_margin','high_degree','low_degree','others','total','time']
    rowlist_poision = []
    rowlist_evasion = []
    for budget in range(1,6):
        result = {
            'poision':{
                'high_margin':[],
                'low_margin':[],
                'high_degree':[],
                'low_degree':[],
                'other':[],
                'total':[]
            },
            'evasion':{
                'high_margin':[],
                'low_margin':[],
                'high_degree':[],
                'low_degree':[],
                'other':[],
                'total':[]
            },
            'time':[]
        }
        for split_indx in range(5):
            split_model = ConfigHandler.load_model_config(args.model,"best_config",args.dataset,split_indx)
            split = split_model['split']
            best_config = split_model.get('model_config')
            idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_train']
        
            for run_idx in range(args.num_run):
                seed = args.seed + run_idx
                miss_poision_count = 0
                miss_evasion_count = 0
                time = 0
            
                surrogate_supervisor = ModelSupervisor(data,args.device,**best_config,seed=seed)
                surrogate_supervisor.train_model(idx_train,idx_val,idx_test)
                high,low, highest_degree,lowest_degree,other = NodeSelector.select_nodes_margin_degree(surrogate_supervisor,data,idx_test,seed=seed)

                if get_surrogate_name(args.adversarial) is not None:
                    adversarial_surrogate = SurrogateLoader.load_surrogate(data,idx_train,idx_val,device= args.device,seed=seed,model=get_surrogate_name(args.adversarial))
                
                attack = load_attack(args.adversarial)(data = data,device = args.device,seed = seed)
                if get_surrogate_name(args.adversarial) is not None:
                    attack.setup_surrogate(adversarial_surrogate)
                adv_supervisor = AdversarialSupervisor(attack,data,idx_train,idx_val,idx_test,device=args.device,**best_config)
                miss_rate_dict = adv_supervisor.target_attack(high,seed,budget,evasion=True,train_before_model=surrogate_supervisor)
                result['poision']['high_margin'].append(miss_rate_dict['poision'])
                result['evasion']['high_margin'].append(miss_rate_dict['evasion'])
                miss_poision_count += miss_rate_dict['poision'] *10
                miss_evasion_count += miss_rate_dict['evasion'] *10
                time += miss_rate_dict['time']

                attack = load_attack(args.adversarial)(data = data,device = args.device,seed = seed)
                if get_surrogate_name(args.adversarial) is not None:
                    attack.setup_surrogate(adversarial_surrogate)
                adv_supervisor = AdversarialSupervisor(attack,data,idx_train,idx_val,idx_test,device=args.device,**best_config)
                miss_rate_dict = adv_supervisor.target_attack(low,seed,budget,evasion=True,train_before_model=surrogate_supervisor)
                result['poision']['low_margin'].append(miss_rate_dict['poision'])
                result['evasion']['low_margin'].append(miss_rate_dict['evasion'])
                miss_poision_count += miss_rate_dict['poision'] *10
                miss_evasion_count += miss_rate_dict['evasion'] *10
                time += miss_rate_dict['time']

                attack = load_attack(args.adversarial)(data = data,device = args.device,seed = seed)
                if get_surrogate_name(args.adversarial) is not None:
                    attack.setup_surrogate(adversarial_surrogate)
                adv_supervisor = AdversarialSupervisor(attack,data,idx_train,idx_val,idx_test,device=args.device,**best_config)
                miss_rate_dict = adv_supervisor.target_attack(highest_degree,seed,budget,evasion=True,train_before_model=surrogate_supervisor)
                result['poision']['high_degree'].append(miss_rate_dict['poision'])
                result['evasion']['high_degree'].append(miss_rate_dict['evasion'])
                miss_poision_count += miss_rate_dict['poision'] *10
                miss_evasion_count += miss_rate_dict['evasion'] *10
                time += miss_rate_dict['time']

                attack = load_attack(args.adversarial)(data = data,device = args.device,seed = seed)
                if get_surrogate_name(args.adversarial) is not None:
                    attack.setup_surrogate(adversarial_surrogate)
                adv_supervisor = AdversarialSupervisor(attack,data,idx_train,idx_val,idx_test,device=args.device,**best_config)
                miss_rate_dict = adv_supervisor.target_attack(lowest_degree,seed,budget,evasion=True,train_before_model=surrogate_supervisor)
                result['poision']['low_degree'].append(miss_rate_dict['poision'])
                result['evasion']['low_degree'].append(miss_rate_dict['evasion'])
                miss_poision_count += miss_rate_dict['poision'] *10
                miss_evasion_count += miss_rate_dict['evasion'] *10
                time += miss_rate_dict['time']

                attack = load_attack(args.adversarial)(data = data,device = args.device,seed = seed)
                if get_surrogate_name(args.adversarial) is not None:
                    attack.setup_surrogate(adversarial_surrogate)
                adv_supervisor = AdversarialSupervisor(attack,data,idx_train,idx_val,idx_test,device=args.device,**best_config)
                miss_rate = adv_supervisor.target_attack(other,seed,budget,evasion=True,train_before_model=surrogate_supervisor)
                result['poision']['other'].append(miss_rate_dict['poision'])
                result['evasion']['other'].append(miss_rate_dict['evasion'])
                miss_poision_count += miss_rate_dict['poision'] *10
                miss_evasion_count += miss_rate_dict['evasion'] *10
                time += miss_rate_dict['time']

                result['time'].append(time)
                result['poision']['total'].append(miss_poision_count)
                result['evasion']['total'].append(miss_evasion_count)

        
        rowlist_poision.append([
            (stats.mean(result['poision']['high_margin']),stats.stdev(result['poision']['high_margin'])),
            (stats.mean(result['poision']['low_margin']),stats.stdev(result['poision']['low_margin'])),
            (stats.mean(result['poision']['high_degree']),stats.stdev(result['poision']['high_degree'])),
            (stats.mean(result['poision']['low_degree']),stats.stdev(result['poision']['low_degree'])),
            (stats.mean(result['poision']['other']),stats.stdev(result['poision']['other'])),
            (stats.mean(result['poision']['total']),stats.stdev(result['poision']['total'])),
            (stats.mean(result['time']),stats.stdev(result['time']))
        ])

        rowlist_evasion.append([
            (stats.mean(result['evasion']['high_margin']),stats.stdev(result['evasion']['high_margin'])),
            (stats.mean(result['evasion']['low_margin']),stats.stdev(result['evasion']['low_margin'])),
            (stats.mean(result['evasion']['high_degree']),stats.stdev(result['evasion']['high_degree'])),
            (stats.mean(result['evasion']['low_degree']),stats.stdev(result['evasion']['low_degree'])),
            (stats.mean(result['evasion']['other']),stats.stdev(result['evasion']['other'])),
            (stats.mean(result['evasion']['total']),stats.stdev(result['evasion']['total'])),
            (stats.mean(result['time']),stats.stdev(result['time']))
        ])

    row_indx = [1,2,3,4,5]
    if not os.path.exists("{}{}/{}/".format(PATH_RESULT,args.model,args.dataset)):
        os.makedirs("{}{}/{}/".format(PATH_RESULT,args.model,args.dataset))
    poision_df = pd.DataFrame(rowlist_poision,columns=columns)
    poision_df.index = row_indx
    poision_df.to_csv("{}{}/{}/{}_poision.csv".format(PATH_RESULT,args.model,args.dataset,args.adversarial))

    evasion_df = pd.DataFrame(rowlist_evasion,columns=columns)
    evasion_df.index = row_indx
    evasion_df.to_csv("{}{}/{}/{}_evasion.csv".format(PATH_RESULT,args.model,args.dataset,args.adversarial))




                


   
    
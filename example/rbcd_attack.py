
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch_geometric.transforms as T

from greatx.attack.targeted import SGAttack,FGAttack,Nettack
from adversarial_attack.Nettack import Nettack
from evaluation.ModelSupervisor import ModelSupervisor
from data_loader.DataLoader import DataLoader
from utility.ConfigHandler import ConfigHandler
from static import *
from adversarial_attack.SurrogateLoader import SurrogateLoader
from adversarial_attack.NodeSelector import NodeSelector
from adversarial_attack.AdversarialSupervisor import AdversarialSupervisor
from adversarial_attack.PRBCDAttack import PRBCDAttack

if __name__=="__main__":
    from utility.config import args

    args.dataset = 'cora'
    args.seed = 722
    data,split = DataLoader.load(args.dataset)
    
    split_0_model = ConfigHandler.load_model_config("GCN","best_config","cora",0)
    split = split_0_model['split']
    best_config = split_0_model.get('model_config')
    idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_train']

    surrogate_supervisor = ModelSupervisor(data,args.device,**best_config)
    surrogate_supervisor.train_model(idx_train,idx_val,idx_test)
    target_bucket= NodeSelector.select_nodes_margin_degree(surrogate_supervisor,data,idx_test,seed=args.seed)

    prbcd = PRBCDAttack(data,args.device,args.seed)
    prbcd.update_config(10000,500)
    prbcd.setup_surrogate(surrogate_supervisor.model)

    adv_supervisor =  AdversarialSupervisor(data,prbcd,idx_train,idx_val,idx_test,args.device,**best_config)
    high_result,low_result, highest_degree_result,lowest_degree_result,other_result = adv_supervisor.target_attack(target_bucket,5,evasion=True,poision=True, evasion_model=surrogate_supervisor,seed = 720)
    evasion_count = high_result[EVASION] + high_result[EVASION] + low_result[EVASION] + highest_degree_result[EVASION] + lowest_degree_result[EVASION] + other_result[EVASION]
    poision_count = high_result[POSION] + high_result[EVASION] + low_result[POSION] + highest_degree_result[POSION] + lowest_degree_result[POSION] + other_result[POSION]
    print(evasion_count,poision_count)

   
    

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch_geometric.transforms as T
import pandas as pd

from adversarial_attack import PRBCDAttack,GRBCDAttack, DegreeAttack,L1NormAttack,OrbitAttack,RandomDegreeAttack,RandomOnlyAddAttack, RandomOnlyRemoveAttack
from greatx.attack.targeted import SGAttack,RandomAttack,DICEAttack,GFAttack,FGAttack,PGDAttack,IGAttack
from adversarial_attack import Nettack
from evaluation import ModelSupervisor
from data_loader import DataLoader
from utility import ConfigHandler
from static import *
from adversarial_attack import SurrogateLoader
from adversarial_attack import NodeSelector
from adversarial_attack import AdversarialSupervisor

def save_results(dataset, attack, budget, mean_gpu_mem_usage, std_gpu_mem_usage):
    partial_path = 'data/results/computational_experiment'
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    result_path = f"{partial_path}/gpu_memory_usage.csv"
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(
            columns=["dataset", "attack", "budget", "mean_gpu_mem_usage", "std_gpu_mem_usage"])
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df.append({'dataset': dataset,
                                  'attack': attack,
                                  'budget': budget,
                                  "mean_gpu_mem_usage": mean_gpu_mem_usage,
                                  "std_gpu_mem_usage": std_gpu_mem_usage,
                                  }, ignore_index=True)
    result_df.to_csv(result_path, index=False)

def load_attack(attack_method):
    if attack_method == SGA_NAME:
        return SGAttack
    elif attack_method == NETTACK_NAME:
        return Nettack
    elif attack_method == FGA_NAME:
        return FGAttack
    elif attack_method == PGDATTACK_NAME:
        return PGDAttack
    elif attack_method == PRBCD_NAME:
        from adversarial_attack import PRBCDAttack,GRBCDAttack
        return PRBCDAttack
    elif attack_method == GRBCD_NAME:
        return GRBCDAttack
    elif attack_method == PRBCD_greatX:
        from greatx.attack.targeted import PRBCDAttack
        return PRBCDAttack
    elif attack_method == "random_degree_attack":
        return RandomDegreeAttack
    elif attack_method == GOTTACK:
        return OrbitAttack
    
def get_surrogate_name(attack_method):
    if attack_method == SGA_NAME:
        return 'SGC'
    elif attack_method in [NETTACK_NAME,FGA_NAME,PGDATTACK_NAME,IGATTACK_NAME,PRBCD_greatX,GOTTACK]:
        return 'GCN_surrogate'
    elif attack_method in [PRBCD_NAME, GRBCD_NAME]:
        return 'GCN'
    else:
        return None

if __name__=="__main__":
    from utility.config import args

    datasets = ['cora','citeseer','pubmed','chameleon','squirrel']
    attacks = [SGA_NAME,NETTACK_NAME,FGA_NAME,PGDATTACK_NAME,PRBCD_NAME,'random_degree_attack',GOTTACK]
    budgets = [2,3,4]
    args.seed = 722
    data,split = DataLoader.load(args.dataset)
    split_index = 0

    split_0_model = ConfigHandler.load_model_config(args.model,"best_config",args.dataset,split_index)
    split = split_0_model['split']
    best_config = split_0_model.get('model_config')
    idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_train']
    fail = []

    for dataset in datasets:
        args.dataset = dataset

        for budget in budgets:
            for attack in attacks:
                    args.adversarial = attack
                # try:
                    evasion_model_supervisor = ModelSupervisor(data,args.device,**best_config)
                    evasion_model_supervisor.train_model(idx_train,idx_val,idx_test)
                    target_bucket= NodeSelector.select_nodes_margin_degree(evasion_model_supervisor,data,idx_test,seed=args.seed)

                    bucket_target_nodes = []
                    for category in target_bucket:
                        bucket_target_nodes.append(target_bucket[category])

                    adversarial_surrogate = SurrogateLoader.load_surrogate(data,idx_train,idx_val,device= args.device,seed=args.seed,model=args.model)
                    gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

                    attacker = load_attack(args.adversarial)(data, device=args.device,seed=args.seed)
                    surrogate_name = get_surrogate_name(args.adversarial)
                    
                    if args.adversarial == PRBCD_NAME:
                        attacker.update_config(10000,200)
                        
                    if  args.adversarial == GRBCD_NAME:
                            attacker.update_config(10000,125)

                    if surrogate_name is not None:
                        split_model_surrogate = ConfigHandler.load_model_config(surrogate_name,"best_config",args.dataset,split_index)
                        configs_surrogate = split_model_surrogate.get('model_config')
                        surrogate = ModelSupervisor(data,args.device,seed=args.seed,**configs_surrogate)
                        surrogate.train_model(idx_train,idx_val,idx_test)
                        attacker.setup_surrogate(surrogate.model)
                    

                    adv_supervisor =  AdversarialSupervisor(data,attacker,idx_train,idx_val,idx_test,args.device,**best_config)
                    high_result,low_result, highest_degree_result,lowest_degree_result,other_result = adv_supervisor.target_attack(bucket_target_nodes,budget,evasion=True,poision=False, evasion_model=evasion_model_supervisor,seed = 720)
                    mean,std = adv_supervisor.get_gpu_mem_usage()
                    save_results(args.dataset,args.adversarial,budget,mean,std)
                # except Exception as e:
                    # print(e)
                    fail.append(f"{args.dataset}_{budget}_{args.adversarial}")


    print(fail)
    
   
    
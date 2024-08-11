
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch_geometric.transforms as T

from greatx.attack.targeted import Nettack,SGAttack
from evaluation.ModelSupervisor import ModelSupervisor
from data_loader.DataLoader import DataLoader
from utility.ConfigHandler import ConfigHandler
from static import *
from adversarial_attack.SurrogateLoader import SurrogateLoader
from adversarial_attack.NodeSelector import NodeSelector
from adversarial_attack.AdversarialSupervisor import AdversarialSupervisor


if __name__=="__main__":
    from utility.config import args

    args.dataset = 'cora'
    args.seed = 720
    data,split = DataLoader.load(args.dataset)
    idx_train,idx_val,idx_test = split[IDX_TRAIN],split[IDX_VAL],split[IDX_TEST]
    config = ConfigHandler.load_model_config()

    surrogate_supervisor = ModelSupervisor(data,args.device,**config)
    surrogate_supervisor.train_model(idx_train,idx_val,idx_test)
    high,low, highest_degree,lowest_degree,other = NodeSelector.select_nodes_margin_degree(surrogate_supervisor,data,idx_test,seed=args.seed)
    node_list = high + low + highest_degree +lowest_degree + other

    adversarial_surrogate = SurrogateLoader.load_surrogate(data,idx_train,idx_val,device= args.device,seed=args.seed,model='SGC')
    nettack = SGAttack(data,args.device,args.seed)
    nettack.setup_surrogate(adversarial_surrogate)

    adv_supervisor = AdversarialSupervisor(nettack,data,idx_train,idx_val,idx_test,device=args.device,**config)
    miss_rate = adv_supervisor.target_attack(node_list,720,1,evasion=True,train_before_model=surrogate_supervisor)
    print(miss_rate)


   
    
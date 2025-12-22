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
    

    




    
        

if __name__ == "__main__":
    # args.dataset = 'cora'
    # args.model = "GCN"
    # args.seed = 720
    # args.adversarial = SGA_NAME
    data_name = 'citeseer'
    args.dataset = data_name

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

    
    all_degree = sorted(set(node_degree.values()))

    average_misclassification = {degree : [] for degree in all_degree}

    # adversarial = [FGA_NAME, NETTACK_NAME, PRBCD_NAME]
    # model = ['GCN','GIN','GSAGE','PNA']

    adversarials = [FGA_NAME, NETTACK_NAME, PRBCD_NAME]
    models = ['GCN','GIN','GSAGE','PNA']

    # adversarials = [FGA_NAME]
    # models = ['GCN','GIN','GSAGE']

    for attack in adversarials:
        for model in models:
            file_path = "{}{}/{}/{}/{}_poison_degree.pkl".format(PATH_RESULT,'best_config',model,data_name,attack)
            with open(file_path, 'rb') as f:
                result = pickle.load(f)
                for degree in all_degree:
                    average_misclassification[degree].append(result.get(degree,0))

    print("Node degree & Average Misclassification rate")
    for degree in all_degree:
        mean = stats.mean(average_misclassification[degree])
        std = stats.stdev(average_misclassification[degree])
        print("{} $\\pm$ {:.2f} $\\pm$  {:.2f}".format(degree,mean,std))

    





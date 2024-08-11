import os
import warnings
import numpy as np
import timeit
import sys
# Hide all warnings
warnings.filterwarnings("ignore")
    
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from greatx.attack.targeted import Nettack,SGAttack,RandomAttack,DICEAttack,GFAttack,FGAttack,PGDAttack,IGAttack
from utility.config import args
from utility.util import set_random
from static import * 
from evaluation.AdversarialAssessment import AdversarialAssessment
from data_loader.DataLoader import DataLoader
from utility.ConfigHandler import ConfigHandler



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
    elif attack_method in [NETTACK_NAME,FGA_NAME]:
        return 'GCN'
    

if __name__ == "__main__":
    data,_ = DataLoader.load(args.dataset)
    args.num_run = 2
    args.adversarial = SGA_NAME
    split_0_model = ConfigHandler.load_model_config("GCN","best_config","cora",0)
    split_1_model = ConfigHandler.load_model_config("GCN","best_config","cora",1)
    adversarial_asse = AdversarialAssessment(data,
                                             [split_0_model,split_1_model],
                                             load_attack(args.adversarial),
                                             args.device,args.seed,
                                             args.num_run,
                                             get_surrogate_name(args.adversarial)
                                             )
    result = adversarial_asse.evaluate(budget=1)
    print(result)
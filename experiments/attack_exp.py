import os
import warnings
import numpy as np
import timeit
import sys
import pandas as pd
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
    else:
        return None
    

if __name__ == "__main__":
    data,_ = DataLoader.load(args.dataset)
    # args.num_run = 3
    # args.adversarial = SGA_NAME
    split_0_model = ConfigHandler.load_model_config(args.model,"best_config",args.dataset,0)
    split_1_model = ConfigHandler.load_model_config(args.model,"best_config",args.dataset,1)
    split_2_model = ConfigHandler.load_model_config(args.model,"best_config",args.dataset,2)
    split_3_model = ConfigHandler.load_model_config(args.model,"best_config",args.dataset,3)
    split_4_model = ConfigHandler.load_model_config(args.model,"best_config",args.dataset,4)

    surrogate_model_name = get_surrogate_name(args.adversarial)
    surrogate_model_configs = None
    if surrogate_model_name is not None:
        split_0_surrogate = ConfigHandler.load_model_config(surrogate_model_name,"best_config",args.dataset,0)
        split_1_surrogate = ConfigHandler.load_model_config(surrogate_model_name,"best_config",args.dataset,1)
        split_2_surrogate = ConfigHandler.load_model_config(surrogate_model_name,"best_config",args.dataset,2)
        split_3_surrogate = ConfigHandler.load_model_config(surrogate_model_name,"best_config",args.dataset,3)
        split_4_surrogate = ConfigHandler.load_model_config(surrogate_model_name,"best_config",args.dataset,4)
        surrogate_model_configs = [split_0_surrogate,split_1_surrogate,split_2_surrogate,split_3_surrogate,split_4_surrogate]

    results = []
    columns = ['high_margin','low_margin','high_degree','low_degree','other','total','time']
    row_indx = [1,2,3,4,5]
    rowlist_poision = []
    rowlist_evasion = []
    for budget in range(1,6):
       
        adversarial_asse = AdversarialAssessment(data,
                                                [split_0_model,split_1_model,split_2_model,split_3_model,split_4_model],
                                                load_attack(args.adversarial),
                                                args.device,args.seed,
                                                args.num_run,
                                                surrogate_model_name,
                                                surrogate_model_configs
                                                )
        result = adversarial_asse.evaluate(budget=budget,evasion=True,poision=True)
        poision_result = result['poision']
        rowlist_poision.append([poision_result['high_margin'],poision_result['low_margin'],poision_result['high_degree'],
                                poision_result['low_degree'],poision_result['other'],poision_result['total'],result['time']])

        evasion_result = result['evasion']
        rowlist_evasion.append([evasion_result['high_margin'],evasion_result['low_margin'],evasion_result['high_degree'],
                                evasion_result['low_degree'],evasion_result['other'],evasion_result['total'],result['time']])
        
    if not os.path.exists("{}{}/{}/".format(PATH_RESULT,args.model,args.dataset)):
        os.makedirs("{}{}/{}/".format(PATH_RESULT,args.model,args.dataset))
    poision_df = pd.DataFrame(rowlist_poision,columns=columns)
    poision_df.index = row_indx
    poision_df.to_csv("{}{}/{}/{}_poision.csv".format(PATH_RESULT,args.model,args.dataset,args.adversarial))

    evasion_df = pd.DataFrame(rowlist_evasion,columns=columns)
    evasion_df.index = row_indx
    evasion_df.to_csv("{}{}/{}/{}_evasion.csv".format(PATH_RESULT,args.model,args.dataset,args.adversarial))




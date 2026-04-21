
import torch
if __name__ == '__main__':
    import sys
    import os
    
    # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utility.config import args
    from data_loader import DataLoader
    from static import *
    from adversarial_attack import RandomDegreeAttack


    args.dataset = "cora"    
    data,split = DataLoader.load(args.dataset)
    # print(data.y.shape)
    # print(torch.norm(data.x[[1,2,3,4],:]))
    
    attack = RandomDegreeAttack(data,seed=800)
    attack.reset()
    attack.set_max_perturbations(1)
    
    attack.attack(target=0,num_budgets=1)

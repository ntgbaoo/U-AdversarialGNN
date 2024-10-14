import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch



        
if __name__ == '__main__':
    import sys
    import os
    
    # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from adversarial_attack import AdversarialSupervisor
    from adversarial_attack import PRBCDAttack
    from adversarial_attack import NodeSelector
    from utility.config import args
    from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
    from evaluation import ModelSupervisor
    from data_loader import DataLoader
    from utility import ConfigHandler
    from static import *

    args.dataset = "blogcatalog"
    
    data,split = DataLoader.load(args.dataset)
    config = ConfigHandler.load_model_config(model_type="GIN")
    model_config = config['model_config']

    config_surrogate = ConfigHandler.load_model_config(model_type="GCN")
    config_surrogate = config_surrogate['model_config']
    set_random(720)
    idx_train,idx_val,idx_test = split['idx_train'],split['idx_val'],split['idx_test']

    runner_surrogate = ModelSupervisor(data,device=args.device,**config_surrogate)
    runner_surrogate.train_model(idx_train=idx_train,idx_val=idx_val,idx_test= idx_test)


    bucket_target_nodes= {'high_margin': [1554, 929, 1163, 1347, 1504, 1541, 1340, 2406, 1049,2082],
                          'low_margin': [1183, 692, 390, 1120, 1455, 523,2153,227, 877, 2221], 
                          'other': [1779, 1243, 1264, 2436, 1931, 1222, 1126, 123, 1713, 2225, 852, 2074, 161, 377, 2271, 1182, 2192, 1921, 1286, 872]}
    target_nodes = [bucket_target_nodes['high_margin'] ,bucket_target_nodes['low_margin'], 
                        bucket_target_nodes['other']]
    
    prbcd = PRBCDAttack(data,args.device)
    prbcd.update_config(block_size=250_000, lr=2_000)
    prbcd.setup_surrogate(runner_surrogate.model)

    adversarial_supervisor = AdversarialSupervisor(data,prbcd,idx_train,idx_val,idx_test,args.device,**model_config)
    high_result,low_result,other_result = adversarial_supervisor.target_attack(target_nodes,1,evasion=False,poision=True,seed = 720)
    print(high_result)
    print(low_result)
    print(other_result)
    total = high_result['poision'] + low_result['poision'] +  other_result['poision']   
    print(total)        
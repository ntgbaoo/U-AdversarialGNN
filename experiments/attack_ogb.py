import timeit
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from scipy.sparse import csr_matrix
import statistics as stats
import pandas as pd

if __name__ == '__main__':
    import sys
    import os
    import warnings
    # Hide all warnings
    warnings.filterwarnings("ignore")
    
    # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from adversarial_attack.AdversarialSupervisor import AdversarialSupervisor
    from adversarial_attack.SurrogateLoader import SurrogateLoader
    from utility.config import args
    from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
    from adversarial_attack.utils import select_nodes,select_nodes_margin

    from deeprobust.graph.targeted_attack import Nettack,FGA,RND,SGAttack,IGAttack
    from evaluation.ModelSupervisor import ModelSupervisor

    from data_loader.DataLoader import DataLoader
    from static import *
    from utility.ConfigHandler import ConfigHandler
    from adversarial_attack.NodeSelector import NodeSelector
    from adversarial_attack.GOttack.OrbitAttack import OrbitAttack
    from utility.orbit_table_generator import OrbitTableGenerator
    from adversarial_attack.Nettack import Nettack
    from adversarial_attack.SGA import SGAttack
    from adversarial_attack.Nettack import Nettack


    args.dataset ="ogbn-arxiv"
    data = DataLoader.load(args.dataset)
    
    gcn_config = ConfigHandler.load_model_config()
    
    #Set up surrogate model
    supervisor_surrogate = ModelSupervisor(data,device=args.device,**gcn_config)
    supervisor_surrogate.train_model()

    adj, features, labels = data.adj, data.feature, data.label
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test


    #Select target nodes
    high,low,other = NodeSelector.select_nodes_margin(supervisor_surrogate)    
    node_list = high + low  + other
    

    gcn_surrogate = SurrogateLoader.load_surrogate(data,idx_train, idx_val,device = args.device,model='sgc')

    adversarial = SGAttack(model = gcn_surrogate, features = csr_matrix(features), adj = adj,
                           labels = labels, n_perturbations = 1, nnodes=data.nnode, attack_structure=True, attack_features=False, device=args.device)
    supervisor_adversarial = ModelSupervisor(data,device= args.device,**gcn_config)
    adversarial_supervisor = AdversarialSupervisor(adversarial = adversarial,
                                                           model_supervisor = supervisor_adversarial,
                                                           data = data,
                                                           surrogate_model = gcn_surrogate,
                                                           target_nodes = node_list[0],
                                                           attack_structure= True,
                                                           attack_feature= False
                                                           )
    
    classified = adversarial_supervisor.attack(node_list[0],720)
    print(classified)


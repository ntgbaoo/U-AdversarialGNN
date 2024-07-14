import timeit
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from scipy.sparse import csr_matrix
from deeprobust.graph.defense import SGC
from deeprobust.graph.data import Dataset, Dpr2Pyg







        
if __name__ == '__main__':
    import sys
    import os
    import warnings
    import numpy as np
    # Hide all warnings
    warnings.filterwarnings("ignore")
    
    # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data_loader.utils import sparse_adjacency_to_edge_list


    from utility.config import args
    from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
    from utility.attack_util import select_nodes_margin
    from deeprobust.graph.targeted_attack import Nettack,FGA,RND,SGAttack,IGAttack
    from evaluation.ModelSupervisor import ModelSupervisor
    from data_loader.DataLoader import DataLoader
    from data_loader.Splitter import Splitter



    args.model = "GCN"
    args.dataset = "cora"
    args.seed = 713
    args.train_early_stopping = True
    args.hidden_channels = 16
    set_random(args.seed)
    
    
    data = Dataset(root='/tmp/', name=args.dataset)
    adj, features, labels = data.adj, data.features, data.labels

    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    idx_unlabeled = np.union1d(idx_val, idx_test)

    # Setup Surrogate model
    surrogate = SGC(nfeat=features.shape[1],
                    nclass=labels.max().item() + 1, K=2,
                    lr=0.01, device=args.device).to(args.device)

    pyg_data = Dpr2Pyg(data)
    surrogate.fit(pyg_data, verbose=False)  # train with earlystopping
    surrogate.test()
        
    data = DataLoader.load(args.dataset)
   
   
    idx_train, idx_val, idx_test = data.idx_train,data.idx_val,data.idx_test
    #Set up surrogate model
    supervisor_surrogate = ModelSupervisor(args,data)
    supervisor_surrogate.train_model()

    adj, features, labels = data.adj, data.feature, data.label
    for seed in (713,720):
        args.seed = seed
        set_random(args.seed)

        # Select target nodes
        node_list = select_nodes_margin(supervisor_surrogate,data.get_tensor_label())
        miss = 0
        counter = 0
        for node in node_list:
            set_random(args.seed)
            adversarial = SGAttack(surrogate, attack_structure=True, attack_features=False, device=args.device).to(args.device)
            adversarial.attack( csr_matrix(features),  adj,  labels, node,1,direct=True, verbose=False)
            modified_adj = adversarial.modified_adj
            # print(len(sparse_adjacency_to_edge_list(adj)[0]),len(sparse_adjacency_to_edge_list(modified_adj)[0]))
            supervisor_adversarial = ModelSupervisor(args,data)
            supervisor_adversarial.update_structure_with_adj(modified_adj)
            # supervisor_adversarial.update_split(idx_train, idx_val, idx_test )
            supervisor_adversarial.train_model()
            prediction = supervisor_adversarial.get_model_prediction()
            if not (prediction[node] == labels[node]).item():
                    miss += 1
                    

            counter+=1
            # print(f'\rINFO : finished {counter}/{len(node_list)}', end='')
            
            
        
        print(miss/40)

            
       


    
    
    
    
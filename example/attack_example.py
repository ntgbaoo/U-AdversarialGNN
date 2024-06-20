import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
        
if __name__ == '__main__':
    import sys
    import os
    import warnings

    # Hide all warnings
    warnings.filterwarnings("ignore")
    
    # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utility.config import args
    from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
    from utility.data_util import dataset_loader,sparse_adjacency_to_edge_list
    from utility.load_model import load_model
    from model.Model_Runner import Runner
    from utility.attack_util import select_nodes_margin,set_up_surrogate_model
    from attack.FGA_rework import FGA
    from deeprobust.graph.targeted_attack import Nettack,FGA,RND,SGAttack,IGAttack

    from utility.data_util import dataset_loader,sparse_adjacency_to_edge_list



    args.model = "GSAGE"
    args.dataset = "cora"
    args.seed = 710
    args.train_early_stopping = True
    set_random(args.seed)
    
    init_logger(
                prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(
                    args.seed) + '_log.txt')
    
    #Set up surrogate model
    runner_surrogate = Runner()
    runner_surrogate.run()

    data = runner_surrogate.data
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    train_acc, val_acc, test_acc = runner_surrogate.compute_acc(runner_surrogate.get_predict())
    node_list = select_nodes_margin(runner_surrogate)
    print(node_list)
    # adversarial = FGA(args,runner_surrogate,runner_surrogate.data.adj.shape[0])

    gcn_surrogate = set_up_surrogate_model(features, adj, labels, idx_train, idx_val,args)
    adversarial = Nettack(gcn_surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=args.device)

   
    # adversarial.attack(features, adj,labels,idx_train, 2307, 1, verbose=True)
    adversarial.attack(features, adj, labels, 2307, 1, verbose=False)
    modified_adj = adversarial.modified_adj
    modified_edgelist = sparse_adjacency_to_edge_list(modified_adj)
    print(modified_edgelist.shape)

    
    
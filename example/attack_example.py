import timeit
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from scipy.sparse import csr_matrix



        
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
    from utility.attack_util import select_nodes_margin,set_up_surrogate_model
    from deeprobust.graph.targeted_attack import Nettack,FGA,RND,SGAttack,IGAttack
    from evaluation.ModelSupervisor import ModelSupervisor
    from data_loader.DataLoader import DataLoader


    args.model = "GCN"
    args.dataset = "cora"
    args.seed = 720
    args.train_early_stopping = True
    set_random(args.seed)
    
    
    if args.verbose:
        init_logger(
                    prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(
                        args.seed) + '_log.txt')
        
    data = DataLoader.load(args.dataset)

    
    #Set up surrogate model
    supervisor_surrogate = ModelSupervisor(args,data)
    supervisor_surrogate.train_model()

    adj, features, labels = data.adj, data.feature, data.label
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    #Select target nodes
    node_list = select_nodes_margin(supervisor_surrogate)
    
    
    #Set up gcn model
    gcn_surrogate = set_up_surrogate_model(features, adj, labels, idx_train, idx_val,args)
    miss = 0
    budgets = [1]
    counter = 0
    false_nodes = []
    for budget in budgets:
        print("=="*30)
        print("INFO: Apply Nettack on {} on {} dataset with budget {}".format(args.model, args.dataset,budget))
        start = timeit.default_timer()
        for node in node_list:
            adversarial = Nettack(gcn_surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=args.device)
            adversarial.to(args.device)
            adversarial.attack(csr_matrix(features), adj, labels, node, budget, verbose=False)
            modified_adj = adversarial.modified_adj
            
            supervisor_adversarial = ModelSupervisor(args,data)
            supervisor_adversarial.update_structure_with_adj(modified_adj)
            supervisor_adversarial.train_model()
            prediction = supervisor_adversarial.get_model_prediction()

            if node in false_nodes:
                miss += 1
            elif not (prediction[node] == labels[node]).item():
                miss += 1
                false_nodes.append(node)
                
            counter += 1
            # print("INFO: finished {}/{}".format(counter,len(node_list)))
            print(f'\rINFO : finished {counter}/{len(node_list)}', end='')
        
        duration = timeit.default_timer() - start
        print("\nWrong classified ratio:{:.3f}".format(miss/len(node_list)))
        print("Total time:{:.2f}".format(duration))



    
    
    
    
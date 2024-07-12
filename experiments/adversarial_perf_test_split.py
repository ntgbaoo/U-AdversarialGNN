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

    from utility.config import args
    from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
    from utility.attack_util import select_nodes_margin,set_up_surrogate_model
    from deeprobust.graph.targeted_attack import Nettack,FGA,RND,SGAttack,IGAttack
    from evaluation.ModelSupervisor import ModelSupervisor
    from data_loader.DataLoader import DataLoader
    from data_loader.Splitter import Splitter



    args.model = "GCN"
    args.dataset = "cora"
    args.seed = 720
    args.train_early_stopping = True
    args.hidden_channels = 16

    set_random(args.seed)
    
    
    if args.verbose:
        init_logger(
                    prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(
                        args.seed) + '_log.txt')
        
    data = DataLoader.load(args.dataset)
    split = Splitter(True,5,0.1,0.1,0.8,100)
    data.init_train_test_splits(split)
    budget = 4
    seeds = [710,720,800]
    for i in range(5):
        print("==="*30)
        print("Working on fold {}".format(i))
        idx_train, idx_val, idx_test = data.get_split(i)
        for seed in seeds:
            args.seed = seed
            set_random(args.seed)
            print("SEED:{}".format(seed))
            #Set up surrogate model
            supervisor_surrogate = ModelSupervisor(args,data)
            supervisor_surrogate.update_split(idx_train, idx_val, idx_test )
            supervisor_surrogate.train_model()

            adj, features, labels = data.adj, data.feature, data.label

            # Select target nodes
            node_list = select_nodes_margin(supervisor_surrogate)
        
        
            # Set up gcn & sgc model
            gcn_surrogate = set_up_surrogate_model(features, adj, labels, idx_train, idx_val,args)

            data_surrogate = Dataset(root='../dataset/', name=args.dataset)
            data_surrogate = Dpr2Pyg(data_surrogate)
            data_surrogate.train_mask, data_surrogate.val_mask, data_surrogate.test_mask = idx_train, idx_val, idx_test
            sgc_surrogate = SGC(nfeat=features.shape[1],
                    nclass=labels.max().item() + 1, K=2,
                    lr=0.01, device=args.device).to(args.device)
            
            sgc_surrogate.fit(data_surrogate, verbose=False)  # train with earlystopping

            miss = [0,0,0]
            counter = 0
            false_nodes = [[],[],[]]
            for node in node_list:
                #Nettack
                adversarial = Nettack(gcn_surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=args.device)
                adversarial.to(args.device)
                adversarial.attack(csr_matrix(features), adj, labels, node, budget, verbose=False)
                modified_adj = adversarial.modified_adj

                supervisor_adversarial = ModelSupervisor(args,data)
                supervisor_adversarial.update_structure_with_adj(modified_adj)
                supervisor_adversarial.update_split(idx_train, idx_val, idx_test )
                supervisor_adversarial.train_model()
                prediction = supervisor_adversarial.get_model_prediction()

                if node in false_nodes[0]:
                    miss[0] += 1
                elif (prediction[node] == labels[node]).item() == 0:
                    miss[0] += 1
                    false_nodes[0].append(node)

                #FGA
                adversarial = FGA(gcn_surrogate, nnodes=data.nnode, attack_structure=True, attack_features=False, device=args.device).to(args.device)
                adversarial.attack(ori_features = csr_matrix(features), ori_adj = adj, labels = labels, idx_train = idx_train, target_node = node, n_perturbations = budget)
                modified_adj = adversarial.modified_adj

                supervisor_adversarial = ModelSupervisor(args,data)
                supervisor_adversarial.update_structure_with_adj(modified_adj)
                supervisor_adversarial.update_split(idx_train, idx_val, idx_test )
                supervisor_adversarial.train_model()
                prediction = supervisor_adversarial.get_model_prediction()

                if node in false_nodes[1]:
                    miss[1] += 1
                elif (prediction[node] == labels[node]).item() == 0:
                    miss[1] += 1
                    false_nodes[1].append(node)


                #SGA
                adversarial = SGAttack(sgc_surrogate, attack_structure=True, attack_features=False, device=args.device).to(args.device)
                adversarial.attack( csr_matrix(features),  adj,  labels, node,budget,direct=True, verbose=False)
                modified_adj = adversarial.modified_adj

                supervisor_adversarial = ModelSupervisor(args,data)
                supervisor_adversarial.update_structure_with_adj(modified_adj)
                supervisor_adversarial.update_split(idx_train, idx_val, idx_test )
                supervisor_adversarial.train_model()
                prediction = supervisor_adversarial.get_model_prediction()

                if node in false_nodes[2]:
                    miss[2] += 1
                elif  (prediction[node] == labels[node]).item() == 0:
                    miss[2] += 1
                    false_nodes[2].append(node)

                counter += 1
                
                print(f'\rINFO : finished {counter}/{len(node_list)}', end='')
            
            miss = np.array(miss)/len(node_list)
            print(miss)

            
       


    
    
    
    
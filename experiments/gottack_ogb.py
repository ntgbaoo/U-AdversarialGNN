import timeit
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from scipy.sparse import csr_matrix
import pandas as pd
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
    from adversarial_attack.utils import select_nodes_margin,select_nodes_margin_v1

    from deeprobust.graph.targeted_attack import Nettack,FGA,RND,SGAttack,IGAttack
    from evaluation.ModelSupervisor import ModelSupervisor
    from adversarial_attack.GOttack.OrbitAttack import OrbitAttack
    from utility.ConfigHandler import ConfigHandler



    from data_loader.DataLoader import DataLoader
    from static import *

    args.dataset = "ogbn-arxiv"
    args.seed = 722
    set_random(args.seed)
    
    data = DataLoader.load(args.dataset)
    gcn_config = ConfigHandler.load_model_config(model_type="GCN")
    
    if args.verbose:
        init_logger(
                    prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(
                        args.seed) + '_log.txt')
        
    orbit_dict = pd.read_csv("data/orbit/cora_orbit_table.csv")
    adj, features, labels = data.adj, data.feature, data.label
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    row = []
    row_time = []
    for budget in range(1,6):
        budget_result = []
        budget_time = []
        for i in range(5):
            #Set up surrogate model
            supervisor_surrogate = ModelSupervisor(data,device=args.device,**gcn_config)
            supervisor_surrogate.train_model()

            #Select target nodes
            node_list = select_nodes_margin_v1(supervisor_surrogate)        
            
            #Set up gcn model
            gcn_surrogate = SurrogateLoader.load_surrogate(data,idx_train, idx_val,device = args.device)
            miss = 0
            counter = 0
            false_nodes = []
            gottack_config = {
                        'features':csr_matrix(features),
                        'adj':adj, 
                        'labels':labels,
                        'n_perturbations':budget,
                        
            }
        
            start = timeit.default_timer()
            for node in node_list:
                adversarial =  OrbitAttack(model = gcn_surrogate,orbit_dict = orbit_dict, nnodes=data.nnode, device=args.device).to(args.device)
                supervisor_adversarial = ModelSupervisor(data,device= args.device,**gcn_config)
                adversarial_supervisor = AdversarialSupervisor(adversarial = adversarial,
                                                                model_supervisor = supervisor_adversarial,
                                                                data = data,
                                                                surrogate_model = gcn_surrogate,
                                                                target_nodes = node,
                                                                attack_structure= True,
                                                                attack_feature= False
                                                                )
                    
                success = adversarial_supervisor.attack(node,args.seed,**gottack_config)
                if success == 0:
                    miss += 1
                    false_nodes.append(node)
                counter += 1
                print(f'\rINFO : finished {counter}/{len(node_list)}', end='')
                
            print("\n")
            duration = timeit.default_timer() - start
            budget_time.append(duration)
            budget_result.append(miss/40)

        row.append("{:.3f}{:.3f}".format(stats.mean(budget_result),stats.stdev(budget_result)))
        row_time.append("{:.3f}{:.3f}".format(stats.mean(budget_time),stats.stdev(budget_time)))
    df_result = pd.DataFrame([row,row_time],columns=[1,2,3,4,5])
    print(df_result)
    df_result.to_csv("gottack_ogb.csv")
        

    



    
    
    
    
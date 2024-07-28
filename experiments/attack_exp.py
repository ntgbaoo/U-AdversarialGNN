import timeit
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from scipy.sparse import csr_matrix
import statistics as stats
import pandas as pd

def attack_sga(list_node,sgc_surrogate,args,model_config,data,idx_train,idx_val,idx_test,features,adj,labels,budget,seed,attack_structure = True, attack_feature = False):
    miss = 0
    for node in list_node:
        adversarial = SGAttack(sgc_surrogate, attack_structure=attack_structure, attack_features=attack_feature, device=args.device).to(args.device)
        supervisor_adversarial = ModelSupervisor(data,device= args.device,seed= seed,**model_config)
        supervisor_adversarial.update_split(idx_train,idx_val,idx_test)
        adversarial_supervisor = AdversarialSupervisor(adversarial = adversarial,
                                                        model_supervisor = supervisor_adversarial,
                                                        data = data,
                                                        surrogate_model = sgc_surrogate,
                                                        target_nodes = node,
                                                        attack_structure= attack_structure,
                                                        attack_feature= attack_feature
                                                        )
        SGA_config = {
            'features':csr_matrix(features),
            'adj':adj, 
            'labels':labels,
            'n_perturbations':budget,
        }

        success = adversarial_supervisor.attack(node,seed,**SGA_config)
        if success == 0:
            miss += 1
    return miss

def attack_nettack(list_node,gcn_surrogate,args,model_config,data,idx_train,idx_val,idx_test,features,adj,labels,budget,seed,attack_structure = True, attack_feature = False):
    miss = 0
    for node in list_node:
        adversarial = Nettack(gcn_surrogate, nnodes=data.nnode, attack_structure=attack_structure, attack_features=attack_feature, device=args.device)
        supervisor_adversarial = ModelSupervisor(data,device= args.device,seed= seed,**model_config)
        supervisor_adversarial.update_split(idx_train,idx_val,idx_test)
        adversarial_supervisor = AdversarialSupervisor(adversarial = adversarial,
                                                        model_supervisor = supervisor_adversarial,
                                                        data = data,
                                                        surrogate_model = gcn_surrogate,
                                                        target_nodes = node,
                                                        attack_structure= attack_structure,
                                                        attack_feature= attack_feature
                                                        )
        nettack_config = {
            'features':csr_matrix(features),
            'adj':adj, 
            'labels':labels,
            'n_perturbations':budget,
                
        }

        success = adversarial_supervisor.attack(node,seed,**nettack_config)
        if success == 0:
            miss += 1
    return miss

def attack_fga(list_node,gcn_surrogate,args,model_config,data,idx_train,idx_val,idx_test,features,adj,labels,budget,seed,attack_structure = True, attack_feature = False):
    miss = 0
    for node in list_node:
        adversarial = FGA(gcn_surrogate, nnodes=data.nnode, attack_structure=True, attack_features=False, device=args.device).to(args.device)
        supervisor_adversarial = ModelSupervisor(data,device= args.device,seed= seed,**model_config)
        supervisor_adversarial.update_split(idx_train,idx_val,idx_test)
        adversarial_supervisor = AdversarialSupervisor(adversarial = adversarial,
                                                        model_supervisor = supervisor_adversarial,
                                                        data = data,
                                                        surrogate_model = gcn_surrogate,
                                                        target_nodes = node,
                                                        attack_structure= attack_structure,
                                                        attack_feature= attack_feature
                                                        )
        FGA_config = {
            'ori_features': csr_matrix(features),
            'ori_adj' : adj,
            'labels':labels,
            'idx_train':idx_train,
            'n_perturbations' :budget,
        }

        success = adversarial_supervisor.attack(node,seed,**FGA_config)
        if success == 0:
            miss += 1
    return miss



        
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




    data = DataLoader.load(args.dataset)
    adj, features, labels = data.adj, data.feature, data.label
    rowlist = []
    for budget in range(1,6):
        print("===================Budget {}===================".format(budget))
        acc_high_margin = []
        acc_low_margin = []
        acc_high_degree = []
        acc_low_degree = []
        acc_other = []
        acc_total = []
        for split_idx in range(args.num_split):
            print("Split-{}".format(split_idx))
            model_split_config = ConfigHandler.load_model_config(model_type=args.model,config_name="{}{}".format(BEST_CONFIG,split_idx),dataset=args.dataset)
            model_config = model_split_config['model_config']
            split_config = model_split_config['split']
            idx_train,idx_val,idx_test = split_config[IDX_TRAIN],split_config[IDX_VAL],split_config[IDX_TEST]

            surrogate_node_model = ModelSupervisor(data,device=args.device,seed=args.seed,**model_config)
            surrogate_node_model.update_split(idx_train,idx_val,idx_test)
            
            surrogate_node_model.train_model()
            # sgc_surrogate = SurrogateLoader.load_surrogate(data,idx_train, idx_val,device = args.device,model = 'sgc')
            gcn_surrogate = SurrogateLoader.load_surrogate(data,idx_train, idx_val,device = args.device,model = 'gcn')

            
            for run_idx in range(3):
                seed = args.seed +  run_idx
                set_random(seed)
                high_margin,low_margin,high_degree,low_degree,other = NodeSelector.select_nodes_margin_degree(surrogate_node_model)
                miss_high_margin = attack_fga(high_margin,gcn_surrogate,args,model_config,data,idx_train,idx_val,idx_test,
                                features,adj,labels,budget,seed)
                
                miss_low_margin = attack_fga(low_margin,gcn_surrogate,args,model_config,data,idx_train,idx_val,idx_test,
                                features,adj,labels,budget,seed)
                
                miss_high_degree = attack_fga(high_degree,gcn_surrogate,args,model_config,data,idx_train,idx_val,idx_test,
                                features,adj,labels,budget,seed)
                
                miss_low_degree = attack_fga(low_degree,gcn_surrogate,args,model_config,data,idx_train,idx_val,idx_test,
                                features,adj,labels,budget,seed)
                
                miss_other = attack_fga(other,gcn_surrogate,args,model_config,data,idx_train,idx_val,idx_test,
                                features,adj,labels,budget,seed)
                
                total_miss = miss_high_margin + miss_low_margin + miss_high_degree + miss_low_degree + miss_other

                acc_high_margin.append(miss_high_margin/10)
                acc_low_margin.append(miss_low_margin/10)
                acc_high_degree.append(miss_high_degree/10)
                acc_low_degree.append(miss_low_degree/10)
                acc_other.append(miss_other/10)
                acc_total.append(total_miss/50)
            
        
        row = []
        row.append("{:.3f},{:.3f}".format(stats.mean(acc_high_margin),stats.stdev(acc_high_margin)))
        row.append("{:.3f},{:.3f}".format(stats.mean(acc_low_margin),stats.stdev(acc_low_margin)))
        row.append("{:.3f},{:.3f}".format(stats.mean(acc_high_degree),stats.stdev(acc_high_degree)))
        row.append("{:.3f},{:.3f}".format(stats.mean(acc_low_degree),stats.stdev(acc_low_degree)))
        row.append("{:.3f},{:.3f}".format(stats.mean(acc_other),stats.stdev(acc_other)))
        row.append("{:.3f},{:.3f}".format(stats.mean(acc_total),stats.stdev(acc_total)))
        rowlist.append(row)
    df_result = pd.DataFrame(rowlist,columns=["high_margin","low_margin","high_degree","low_degree","other","total"])
    df_result.index =[1,2,3,4,5]
    print(df_result)
    df_result.to_csv("results/{}/{}/{}.csv".format(args.model,args.dataset,"fga"))
        



        

                   

        
            






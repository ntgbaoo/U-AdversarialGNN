import timeit
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from scipy.sparse import csr_matrix
import statistics as stats
import pandas as pd


def attack_sga(orbit_dict,list_node,sgc_surrogate,args,model_config,data,idx_train,idx_val,idx_test,features,adj,labels,budget,seed,attack_structure = True, attack_feature = False):
    miss = 0
    counter = 1
    start_time = timeit.default_timer()
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
        try:
            success = adversarial_supervisor.attack(node,seed,**SGA_config)
        except Exception as e:
            success = 0
        print(success)
        if success == 0:
            miss += 1
        print(f'\rINFO : finished {counter}/{len(node_list)}', end='')
        counter +=1
    return miss/(len(node_list)),timeit.default_timer() - start_time

def attack_nettack(orbit_dict,list_node,gcn_surrogate,args,model_config,data,idx_train,idx_val,idx_test,features,adj,labels,budget,seed,attack_structure = True, attack_feature = False):
    miss = 0
    counter = 1
    start_time = timeit.default_timer()
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
            'verbose':False
                
        }
        try:
            success = adversarial_supervisor.attack(node,seed,**nettack_config)
        except Exception as e:
            success = 1
        if success == 0:
            miss += 1
        print(f'\rINFO : finished {counter}/{len(node_list)}', end='')
        counter +=1

    print("\n")
    print(miss/(len(node_list)))
    return miss/(len(node_list)), timeit.default_timer() - start_time

def attack_fga(orbit_dict,list_node,gcn_surrogate,args,model_config,data,idx_train,idx_val,idx_test,features,adj,labels,budget,seed,attack_structure = True, attack_feature = False):
    miss = 0
    counter = 1
    start_time = timeit.default_timer()
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
            'verbose':False
        }

        success = adversarial_supervisor.attack(node,seed,**FGA_config)
        if success == 0:
            miss += 1
        print(f'\rINFO : finished {counter}/{len(node_list)}', end='')
        counter +=1
    return miss/(len(node_list)),timeit.default_timer() - start_time

def attack_gottack(orbit_dict,list_node,gcn_surrogate,args,model_config,data,idx_train,idx_val,idx_test,features,adj,labels,budget,seed,attack_structure = True, attack_feature = False):
    miss = 0
    counter = 1
    start_time = timeit.default_timer()
    for node in list_node:
        adversarial = OrbitAttack(model = gcn_surrogate,orbit_dict = orbit_dict, nnodes=data.nnode, device=args.device).to(args.device)
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
            'verbose':False
                
        }

        success = adversarial_supervisor.attack(node,seed,**nettack_config)
        if success == 0:
            miss += 1
        print(f'\rINFO : finished {counter}/{len(node_list)}', end='')
        counter +=1

    print("\n")
    print(miss/(len(node_list)))
    return miss/(len(node_list)),timeit.default_timer() - start_time


def attack_rnd(orbit_dict,list_node,gcn_surrogate,args,model_config,data,idx_train,idx_val,idx_test,features,adj,labels,budget,seed,attack_structure = True, attack_feature = False):
    miss = 0
    counter = 1
    start_time = timeit.default_timer()
    for node in list_node:        
        adversarial = RND(device=args.device)
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
        
        # ori_adj, labels, idx_train, target_node, n_perturbations
        rnd_config = {
            'ori_adj':adj, 
            'labels':labels,
            'n_perturbations':budget,
            'idx_train': idx_train,
                
        }

        success = adversarial_supervisor.attack(node,seed,**rnd_config)
        if success == 0:
            miss += 1
        print(f'\rINFO : finished {counter}/{len(node_list)}', end='')
        counter +=1
    return miss/(len(node_list)),timeit.default_timer() - start_time
def load_attack(attack_method):
    if attack_method == SGA_NAME:
        return attack_sga
    elif attack_method == NETTACK_NAME:
        return attack_nettack
    elif attack_method == FGA_NAME:
        return attack_fga
    elif attack_method == GOTTACK_NAME:
        return attack_gottack
    elif attack_method == RND_NAME:
        return attack_rnd
    
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



    data = DataLoader.load(args.dataset)
    idx_train,idx_val,idx_test = data.idx_train, data.idx_val, data.idx_test

    adj, features, labels = data.adj, data.feature, data.label
    orbit_dict = pd.read_csv('data/orbit/arxiv_orbit_df.csv')
    attack_method = load_attack(args.adversarial)

    rowlist = []
    row_acc = []
    row_time = []
    for budget in range(1,6):
        print("===================Budget {}===================".format(budget))
        acc_list = []
        time_list = []

        model_config = ConfigHandler.load_model_config(model_type=args.model)

        surrogate_node_model = ModelSupervisor(data,device=args.device,seed=args.seed,**model_config)
        surrogate_node_model.train_model()
        if args.adversarial == SGA_NAME:
            surrogate_model = SurrogateLoader.load_surrogate(data,idx_train, idx_val,device = args.device,model = 'sgc')
        else:
            surrogate_model = SurrogateLoader.load_surrogate(data,idx_train, idx_val,device = args.device,model = 'gcn')

        for run_idx in range(3):
            seed = args.seed +  run_idx
            set_random(seed)
            high,low,other = NodeSelector.select_nodes_margin(surrogate_node_model)
            node_list = high + low + other

            print("=={}==".format(args.adversarial))
            acc,time = attack_method(orbit_dict,node_list,surrogate_model,args,model_config,data,idx_train,idx_val,idx_test,
                            features,adj,labels,budget,seed)
            acc_list.append(acc)
            time_list.append(time)
        
        row_acc.append("{},{}".format(stats.mean(acc_list),stats.stdev(acc_list)))
        row_time.append("{},{}".format(stats.mean(time_list),stats.stdev(time_list)))
        
      
    rowlist.append(row_acc)
    rowlist.append(row_time)
    df_result = pd.DataFrame(rowlist,columns=['1','2','3','4','5'])
    df_result.index =[args.adversarial,"{}_time".format(args.adversarial)]
    print(df_result)
    if not os.path.exists("data/results_gottack/{}/{}/".format(args.model,args.dataset)):
        os.makedirs("data/results_gottack/{}/{}/".format(args.model,args.dataset))
    df_result.to_csv("data/results_gottack/{}/{}/{}.csv".format(args.model,args.dataset,args.adversarial))
        



        

                   

        
            






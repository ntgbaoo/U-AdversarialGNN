import sys
import os
import warnings
from scipy.sparse import csr_matrix
import timeit
import pandas as pd

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
from utility.ConfigHandler import ConfigHandler
from adversarial_attack.NodeSelector import NodeSelector
from adversarial_attack.GOttack.OrbitAttack import OrbitAttack
from data_loader.DataLoader import DataLoader
from static import *

def attack_FGA(node_list,features,adj,labels,idx_train,budget,gcn_surrogate,data,args,gcn_config):
    FGA_config = {
        'ori_features': csr_matrix(features),
        'ori_adj' : adj,
        'labels':labels,
        'idx_train':idx_train,
        'n_perturbations' :budget,
    }
    counter = 0
    miss = 0
    false_nodes = []
    for node in node_list:
        adversarial =  FGA(gcn_surrogate, nnodes=data.nnode, attack_structure=True, attack_features=False, device=args.device).to(args.device)
        supervisor_adversarial = ModelSupervisor(data,device= args.device,**gcn_config)
        adversarial_supervisor = AdversarialSupervisor(adversarial = adversarial,
                                                           model_supervisor = supervisor_adversarial,
                                                           data = data,
                                                           surrogate_model = gcn_surrogate,
                                                           target_nodes = node,
                                                           attack_structure= True,
                                                           attack_feature= False
                                                           )
        success = adversarial_supervisor.attack(node,args.seed,**FGA_config)
        if success == 0:
            miss += 1
            false_nodes.append(node)
        counter += 1
        print(f'\rINFO : finished {counter}/{len(node_list)}', end='')
    return miss/len(node_list)   

def attack_nettack(node_list,features,adj,labels,idx_train,budget,gcn_surrogate,data,args,gcn_config):
    nettack_config = {
                'features':csr_matrix(features),
                'adj':adj, 
                'labels':labels,
                'n_perturbations':budget,     
                'verbose':False     
    }
    counter = 0
    miss = 0
    false_nodes = []
    for node in node_list:
        adversarial = Nettack(gcn_surrogate, nnodes=data.nnode, attack_structure=True, attack_features=False, device=args.device)
        supervisor_adversarial = ModelSupervisor(data,device= args.device,**gcn_config)
        adversarial_supervisor = AdversarialSupervisor(adversarial = adversarial,
                                                           model_supervisor = supervisor_adversarial,
                                                           data = data,
                                                           surrogate_model = gcn_surrogate,
                                                           target_nodes = node,
                                                           attack_structure= True,
                                                           attack_feature= False
                                                           )
        success = adversarial_supervisor.attack(node,args.seed,**nettack_config)
        if success == 0:
            miss += 1
            false_nodes.append(node)
        counter += 1
        print(f'\rINFO : finished {counter}/{len(node_list)}', end='')
    return miss/len(node_list)   
        
def attack_sga(node_list,features,adj,labels,idx_train,budget,sgc_surrogate,data,args,gcn_config):
    counter = 0
    miss = 0
    false_nodes = []
    SGA_config = {
                'features':csr_matrix(features),
                'adj':adj, 
                'labels':labels,
                'n_perturbations':budget,
    }
    for node in node_list:
        adversarial = SGAttack(sgc_surrogate, attack_structure=True, attack_features=False, device=args.device).to(args.device)
        supervisor_adversarial = ModelSupervisor(data,device= args.device,**gcn_config)
        adversarial_supervisor = AdversarialSupervisor(adversarial = adversarial,
                                                           model_supervisor = supervisor_adversarial,
                                                           data = data,
                                                           surrogate_model = sgc_surrogate,
                                                           target_nodes = node,
                                                           attack_structure= True,
                                                           attack_feature= False
                                                           )
        success = adversarial_supervisor.attack(node,args.seed,**SGA_config)
        if success == 0:
            miss += 1
            false_nodes.append(node)
        counter += 1
        print(f'\rINFO : finished {counter}/{len(node_list)}', end='')
    return miss/len(node_list)

def attack_gottack(node_list,features,adj,labels,idx_train,budget,gcn_surrogate,data,args,gcn_config,orbit_dict):
    gottack_config = {
                'features':csr_matrix(features),
                'adj':adj, 
                'labels':labels,
                'n_perturbations':budget,
                
    }
    counter = 0
    miss = 0
    false_nodes = []
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
    return miss/len(node_list)   


    



if __name__ == "__main__":
    args.dataset = "cora"
    args.seed = 724
    set_random(args.seed)
    
    data = DataLoader.load(args.dataset)
    
    gcn_config = ConfigHandler.load_model_config()
    orbit_dict = pd.read_csv("data/orbit/cora_orbit_table.csv")

    
    #Set up surrogate model
    supervisor_surrogate = ModelSupervisor(data,device=args.device,**gcn_config)
    supervisor_surrogate.train_model()

    adj, features, labels = data.adj, data.feature, data.label
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    budget =1
    
    #Set up gcn model
    gcn_surrogate = SurrogateLoader.load_surrogate(data,idx_train, idx_val,device = args.device)
    sgc_surrogate = SurrogateLoader.load_surrogate(data,idx_train, idx_val,device = args.device,model = 'sgc')

    #Select target nodes
    high_degree,low_degree = NodeSelector.select_node_degree(supervisor_surrogate,top_n=40)   
    high_centra,low_centra = NodeSelector.select_node_centrality(supervisor_surrogate,top_n=40)   
    
    columns = ['highest_degree','lowest_degree','highest_centra','lowest_centra']
    row_indx = ['Nettack','FGA','SGA','Gottack']
    row_list = []
    
    print("INFO: Running Nettack")
    row= []
    result_net = attack_nettack(high_degree,features,adj,labels,idx_train,budget,gcn_surrogate,data,args,gcn_config)
    row.append(result_net)
    result_net = attack_nettack(low_degree,features,adj,labels,idx_train,budget,gcn_surrogate,data,args,gcn_config)
    row.append(result_net)
    result_net = attack_nettack(high_centra,features,adj,labels,idx_train,budget,gcn_surrogate,data,args,gcn_config)
    row.append(result_net)
    result_net = attack_nettack(low_centra,features,adj,labels,idx_train,budget,gcn_surrogate,data,args,gcn_config)
    row.append(result_net)
    row_list.append(row)

    print("INFO: Running FGA")
    row = []
    result_fga = attack_FGA(high_degree,features,adj,labels,idx_train,budget,gcn_surrogate,data,args,gcn_config)
    row.append(result_fga)
    result_fga = attack_FGA(low_degree,features,adj,labels,idx_train,budget,gcn_surrogate,data,args,gcn_config)
    row.append(result_fga)
    result_fga = attack_FGA(high_centra,features,adj,labels,idx_train,budget,gcn_surrogate,data,args,gcn_config)
    row.append(result_fga)
    result_fga = attack_FGA(low_centra,features,adj,labels,idx_train,budget,gcn_surrogate,data,args,gcn_config)
    row.append(result_fga)
    row_list.append(row)

    print("INFO: Running SGA")
    row = []
    result_sga = attack_sga(high_degree,features,adj,labels,idx_train,budget,sgc_surrogate,data,args,gcn_config)
    row.append(result_sga)
    result_sga = attack_sga(low_degree,features,adj,labels,idx_train,budget,sgc_surrogate,data,args,gcn_config)
    row.append(result_sga)
    result_sga = attack_sga(high_centra,features,adj,labels,idx_train,budget,sgc_surrogate,data,args,gcn_config)
    row.append(result_sga)
    result_sga = attack_sga(low_centra,features,adj,labels,idx_train,budget,sgc_surrogate,data,args,gcn_config)
    row.append(result_sga)
    row_list.append(row)

    print("INFO: Running GOttack")
    row = []
    result_gottack = attack_gottack(high_degree,features,adj,labels,idx_train,budget,gcn_surrogate,data,args,gcn_config,orbit_dict)
    row.append(result_gottack)
    result_gottack = attack_gottack(low_degree,features,adj,labels,idx_train,budget,gcn_surrogate,data,args,gcn_config,orbit_dict)
    row.append(result_gottack)
    result_gottack = attack_gottack(high_centra,features,adj,labels,idx_train,budget,gcn_surrogate,data,args,gcn_config,orbit_dict)
    row.append(result_gottack)
    result_gottack = attack_gottack(low_centra,features,adj,labels,idx_train,budget,gcn_surrogate,data,args,gcn_config,orbit_dict)
    row.append(result_gottack)
    row_list.append(row)


    df_result = pd.DataFrame(row_list,columns=columns)
    df_result.index = row_indx
    print("\n")
    print(df_result)
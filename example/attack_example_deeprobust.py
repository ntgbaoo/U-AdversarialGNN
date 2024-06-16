import pandas as pd
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore")


import random
from deeprobust.graph.data import Dataset, Dpr2Pyg

# from deeprobust.graph.targeted_attack import Nettack

"""
Author: Anonymous Authors
Credits:    
            - The OrbitAttack is implemented with reference to Nettack's source code,
            a method proposed in the paper: 'Adversarial Attacks on Neural Networks for Graph Data'
            by Daniel Zügner, Amir Akbarnejad and Stephan Günnemann
            - Gottack is built on top of DeepRobust - A PyTorch Library for Adversarial
            Attacks and Defenses developed by Yaxin Li, Wei Jin, Han Xu and Jiliang Tang
"""



def test_acc_GCN(adj, features, data,target_node):
    idx_train, idx_val = data.idx_train, data.idx_val
    labels = data.labels
    ''' test on GCN '''
    gcn = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    gcn.eval()
    output = gcn.predict()
    probs = torch.exp(output[[target_node]])[0]
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()



def select_nodes(target_gcn=None):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''

    if target_gcn is None:
        target_gcn = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5, device=device)
        target_gcn = target_gcn.to(device)
        target_gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    target_gcn.eval()
    output = target_gcn.predict()

    margin_dict = {}
    for idx in idx_test:
        margin = classification_margin(output[idx], labels[idx])
        if margin < 0: # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10: ]]
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high + low + other

def set_up_surrogate_model(features, adj, labels, idx_train, idx_val):
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16, dropout=0, with_relu=False,
                    with_bias=False, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    return surrogate



def attack_net(surrogate,data, target_node_list,budget,features,adj,labels,test_model,verbose = False):
    miss = 0
    false_class_node = []
    for target_node in tqdm(target_node_list):
        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
        model = model.to(device)
        model.attack(features, adj, labels, target_node, budget, verbose=verbose)
        modified_adj = model.modified_adj
        if test_model == 'GCN':
            acc = test_acc_GCN(modified_adj, features, data,
                               target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
       
        if target_node in false_class_node:
            miss += 1
        elif acc == 0:
            miss += 1
            false_class_node.append(target_node)
    return miss / len(target_node_list)


if __name__ == '__main__':
    import sys
    import os
     # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utility.data_util import dataset_loader
    from utility.config import args
    from deeprobust.graph.targeted_attack import Nettack,FGA,RND,SGAttack,IGAttack

    '''
       Config attack method and dataset
       method : list of adversarial techniques (1518 is Gottack)
       budget list: Perturbation budget list
       dataset_name : cora or citeseer or pubmed or polblogs or Blogcatalog 
       
    '''
    method = ['1518', 'Nettack']
    budget_list = [1]
    random.seed(102)
    dataset_name = 'cora'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    print("INFO: Applying adversarial techniques {} on {} dataset with perturbation budget {} ".format(method, dataset_name,budget_list))

    ######################### Loading dataset  #########################

    data = dataset_loader(args)
    adj, features, labels = data.adj, data.features, data.labels

    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test


    for budget in budget_list:
        target_node = select_nodes()
        print(target_node)

      
        # Nettack
        surrogate = set_up_surrogate_model(features, adj, labels, idx_train, idx_val)
       
        
        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
        model = model.to(device)
        miss_percentage = attack_net(surrogate,data, target_node, budget, features, adj, labels,"GCN")
      

     
     




    

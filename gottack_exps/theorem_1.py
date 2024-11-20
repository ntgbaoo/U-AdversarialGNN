import pandas as pd
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from tqdm import tqdm
import warnings
import statistics


warnings.filterwarnings("ignore")
import random
from deeprobust.graph.data import Dataset, Dpr2Pyg
from deeprobust.graph.targeted_attack import Nettack

"""
Author: Anonymous Authors
Credits:    
            - The OrbitAttack is implemented with reference to Nettack's source code,
            a method proposed in the paper: 'Adversarial Attacks on Neural Networks for Graph Data'
            by Daniel Zügner, Amir Akbarnejad and Stephan Günnemann
            - Gottack is built on top of DeepRobust - A PyTorch Library for Adversarial
            Attacks and Defenses developed by Yaxin Li, Wei Jin, Han Xu and Jiliang Tang
"""


def test_acc_GCN(adj, features, data, target_node):
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
        if margin < 0:  # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x: x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10:]]
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high + low + other


def set_up_surrogate_model(features, adj, labels, idx_train, idx_val):
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16, dropout=0, with_relu=False,
                    with_bias=False, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    return surrogate



import torch
import numpy as np

def set_random(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def random_walk_to_target(graph, source, target, max_steps=100):
    if source not in graph or target not in graph:
        raise ValueError("Source or target node not in graph.")
    
    current = source
    for steps in range(max_steps):
        if current == target:
            return steps
        neighbors = list(graph.neighbors(current))
        if not neighbors:  
            break
        current = random.choice(neighbors)
    return max_steps  

    

def average_path_length_to_target(graph, source, target, trials=1000, max_steps=100):
    lengths = []
    for _ in range(trials):
        length = random_walk_to_target(graph, source, target, max_steps)
        if length is not None:
            lengths.append(length)
    return sum(lengths) / len(lengths) if lengths else float('inf')


def generate_orbit_tables_from_sratch(dataset):
    file_path = 'data/orbit/'
    filename = file_path +"dpr_"+ dataset + '.out'
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Process the lines to create a list of lists
    data_list = [list(map(int, line.split())) for line in lines]

    # Convert the list of lists to a NumPy array

    graphlet_features = np.array(data_list)
    mylist = []
    for i in range(len(graphlet_features)):
        arr = graphlet_features[i]
        sorted_indices = np.argsort(arr)[::-1]
        if sorted_indices[0] < sorted_indices[1]:
            s1 = str(sorted_indices[0]) + str(sorted_indices[1])
        else:
            s1 = str(sorted_indices[1]) + str(sorted_indices[0])

        mylist.append([i, str(sorted_indices[0]), str(sorted_indices[1]), s1])

    my_array = np.array(mylist)
    df_2d = pd.DataFrame(my_array, columns=['node_number', 'Orbit_type_I', 'Orbit_type_II', 'two_Orbit_type'])
    return df_2d

if __name__ == '__main__':
    import networkx as nx
    '''
       Config attack method and dataset
       method : list of adversarial techniques (1518 is Gottack)
       budget list: Perturbation budget list
       dataset_name : cora or citeseer or pubmed or polblogs or Blogcatalog 

    '''
    random.seed(102)
    dataset_name = 'cora'
    df_orbit = generate_orbit_tables_from_sratch(dataset_name)
    
    device = "cpu"


    ######################### Loading dataset  #########################
    data = Dataset(root='data/dataset', name=dataset_name)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    cora_graph = nx.from_scipy_sparse_array(adj)


    all_target_nodes = set()
    for i in  range(1):
        seed= 102+i
        set_random(seed)
        target_node = select_nodes()
        all_target_nodes.update(target_node)

    result_dict = {}
    # for source in tqdm(all_target_nodes):
    #     for index, row in df_orbit.iterrows():
    #         target = int(row['node_number'])
    #         orbit = int(row['two_Orbit_type'])
    #         len_shortest_path = nx.shortest_path_length(cora_graph,source,target)
    #         if not orbit in result_dict:
    #             result_dict[orbit] =  []
    #         result_dict[orbit].append(len_shortest_path)

    for source in tqdm(all_target_nodes):
        for index, row in df_orbit.iterrows():
            target = int(row['node_number'])
            orbit = int(row['two_Orbit_type'])
            hit_times = average_path_length_to_target(cora_graph,source,target,trials=10,max_steps=1000)
            if hit_times != float('inf'):
                if not orbit in result_dict:
                    result_dict[orbit] =  []
                
                result_dict[orbit].append(hit_times)

    
    for key in result_dict.keys():
        values = result_dict[key]
        if len(values) > 1:
            result_dict[key]=statistics.mean(values)
        else:
            result_dict[key]= values[0]

    sorted_list = sorted(result_dict,key= lambda k:result_dict[k],reverse=True)
    for element in sorted_list:
        if element == 1518:
            print("======")    
        print(element,result_dict[element])


        
 
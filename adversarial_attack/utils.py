import numpy as np
import networkx
import os 
import sys
from scipy.sparse import coo_matrix
from deeprobust.graph.utils import *


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from DeepRobust.deeprobust.graph.utils import classification_margin
from evaluation import IModelSupervisor

def calculate_degrees(adj_matrix, directed=False):
    if directed:
        # For directed graphs
        out_degrees = np.array(adj_matrix.sum(axis=1)).flatten()  # Sum of rows
        in_degrees = np.array(adj_matrix.sum(axis=0)).flatten()   # Sum of columns
        return in_degrees, out_degrees
    else:
        # For undirected graphs
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()  # Sum of rows (or columns, as it's symmetric)
        return degrees
    
def select_nodes_margin_degree(adj_matrix,node_list,top_n = 10):
    degrees = calculate_degrees(adj_matrix)
    filter_degrees = [(node,degrees[node]) for node in node_list]
    filter_degrees.sort(key=lambda x: x[1])
    highest = [node for node,_ in filter_degrees[-top_n:]]
    lowest = [node for node,_ in filter_degrees[:top_n]]
    return highest + lowest

   
def select_nodes(target_runner:IModelSupervisor):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''

    target_runner.model.eval()
    output = target_runner.get_model_output()
    
    margin_dict = {}
    for idx in target_runner.idx_test:
        margin = classification_margin(output[idx], target_runner.labels_torch[idx])
        if margin < 0: # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10: ]]
    other = [x for x, y in sorted_margins[10: -10]]
    # other = np.random.choice(other, 20, replace=False).tolist()
    other = select_nodes_margin_degree(target_runner.adj,other)

    return high + low + other

if __name__ == '__main__':
    sparse_adj_matrix  =    parse_adj_matrix = coo_matrix([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])

    degrees = calculate_degrees(sparse_adj_matrix)
    degrees = degrees[0,2]
    print("Degrees (undirected):", degrees)

    values_array = np.array([1, 5, 6, 7, 8])

    # Array with indices
    indices_array = np.array([0, 3])

    # Extract values
    extracted_values = values_array[indices_array]

    print("Extracted values:", extracted_values)
    
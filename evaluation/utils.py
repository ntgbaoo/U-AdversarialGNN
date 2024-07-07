import numpy as np


def sparse_adjacency_to_edge_list(sparse_adj_matrix):
     # Find the indices of the non-zero elements in the matrix
    row_indices, col_indices = sparse_adj_matrix.nonzero()
    
    # Pair the indices to form the edge list
    edge_list = list(zip(row_indices, col_indices))

    edge_list =np.array(edge_list).T
    
    return edge_list

def optimizer_loader(optimizer_name, model_parameters):
    pass
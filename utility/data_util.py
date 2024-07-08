from deeprobust.graph.data import Dataset, Dpr2Pyg
import numpy as np


DATASET_ROOT = "../dataset/"

# def dataset_loader(args):
#     if args.dataset in ['cora', 'citeseer', 'cora_ml', 'polblogs','pubmed', 'acm', 'blogcatalog', 'uai', 'flickr']:
#         print("INFO: load {} dataset from DeepRobust".format(args.dataset))
#         data = Dataset(root=DATASET_ROOT,name = args.dataset)
#     else:
#         raise Exception("Unsupport datasets")
    
#     return data

# def sparse_adjacency_to_edge_list(sparse_adj_matrix):
#      # Find the indices of the non-zero elements in the matrix
#     row_indices, col_indices = sparse_adj_matrix.nonzero()
    
#     # Pair the indices to form the edge list
#     edge_list = list(zip(row_indices, col_indices))

#     edge_list =np.array(edge_list).T
    
#     return edge_list
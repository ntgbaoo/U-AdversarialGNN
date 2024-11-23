import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch
from tqdm import tqdm
from torch_geometric.utils import degree, to_scipy_sparse_matrix


# Function to compute 5-hop neighbors
def get_k_hop_subgraph(node, k, edge_index):
    visited = set([node.item()])
    frontier = [node.item()]
    for _ in range(k):
        neighbors = edge_index[1][torch.isin(edge_index[0], torch.tensor(frontier))].tolist()
        frontier = list(set(neighbors) - visited)
        visited.update(frontier)
    return torch.tensor(list(visited))

        
if __name__ == '__main__':
    import sys
    import os
    import pandas as pd
    
    # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utility.config import args
    from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
    from evaluation import ModelSupervisor
    from data_loader import DataLoader
    from utility import ConfigHandler
    from static import *
    

    args.dataset = "ogbn-arxiv"
    
    data,split = DataLoader.load(args.dataset)
    orbit_df = pd.read_csv("data/orbit/ogbn-arxiv_orbit_table.csv")
    node_1518 = orbit_df[orbit_df['two_Orbit_type'] == 1518]['node_number'].tolist()
    print(len(node_1518))
    edge_index = data.edge_index

    # Compute the degree of each node
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)

    # Count leaf nodes (nodes with degree 1)
    leaf_count = (deg == 1).sum().item()

    leaf_nodes = torch.where(deg == 1)[0].numpy().tolist()
    print(len(leaf_nodes)/data.num_nodes)
    counter = 0
    for orbit_node in node_1518:
        if orbit_node in leaf_nodes:
            counter += 1
    print(counter/len(node_1518))


# # Find edges in 5-hop neighborhood of all leaf nodes
#     total_edges = 0
#     for leaf in tqdm(leaf_nodes):
#         k_hop_nodes = get_k_hop_subgraph(leaf, 5, edge_index)
#         mask = torch.isin(edge_index[0], k_hop_nodes) & torch.isin(edge_index[1], k_hop_nodes)
#         sub_edge_index = edge_index[:, mask]
#         total_edges += sub_edge_index.size(1)

#     print("Total number of edges in 5-hop neighborhoods:", total_edges)
#     print("Number of leaf nodes:", leaf_count)
#     print(leaf_count/data.num_nodes)
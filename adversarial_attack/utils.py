import numpy as np
import networkx
import os 
import sys
from scipy.sparse import coo_matrix
from deeprobust.graph.utils import *
import networkx as nx


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from DeepRobust.deeprobust.graph.utils import classification_margin
from evaluation import IModelSupervisor

def calculate_centrality(adj_matrix,direct= False):
    row_indices,col_indices = adj_matrix.nonzero()
    edgelists = list(zip(row_indices,col_indices))
    G = nx.Graph(edgelists)
    centrality_node = nx.degree_centrality(G)
    centrality_values = [(key,value) for key,value in centrality_node.items()]
    return centrality_values


def select_nodes_margin_centrality(adj_matrix,top_n = 10):
    centrality_values = calculate_centrality(adj_matrix)
    centrality_values.sort(key=lambda x: x[1])
    highest = [node for node,_ in centrality_values[-top_n:]]
    lowest = [node for node,_ in centrality_values[:top_n]]
    return highest + lowest



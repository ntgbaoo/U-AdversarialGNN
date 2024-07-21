import numpy as np
from deeprobust.graph.utils import *
import os 
import sys
import networkx as nx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation import IModelSupervisor


class NodeSelector:
    @staticmethod
    def select_nodes_margin(target_runner:IModelSupervisor):
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
        other = np.random.choice(other, 20, replace=False).tolist()

        return high + low + other
    
    @staticmethod
    def select_nodes_margin_degree(target_runner:IModelSupervisor):
        '''
        selecting nodes as reported in nettack paper:
        (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
        (ii) the 10 nodes with lowest margin (but still correctly classified) and
        (iii) 10 nodes with highest degree
        (iv) 10 nodes with lowest degree
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
        degree_values = NodeSelector._calculate_degrees(target_runner.adj,other)
        highest_degree,lowest_degree = NodeSelector._select_node_margin(degree_values)

        return high,low, highest_degree,lowest_degree
    
    @staticmethod
    def select_node_margin_centrality(target_runner:IModelSupervisor):
        '''
        selecting nodes as reported in nettack paper:
        (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
        (ii) the 10 nodes with lowest margin (but still correctly classified) and
        (iii) 10 nodes with highest centrality
        (iv) 10 nodes with lowest centrality
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
        central_values = NodeSelector._calculate_centrality(target_runner.adj,other)
        highest_central,lowest_central = NodeSelector._select_node_margin(central_values)

        return high,low, highest_central,lowest_central
    
    
    
    @staticmethod
    def _select_node_margin(tuples_values,top_n = 10):
        tuples_values.sort(key=lambda x: x[1])
        highest = [node for node,_ in tuples_values[-top_n:]]
        lowest = [node for node,_ in tuples_values[:top_n]]
        return highest,lowest
    
    @staticmethod
    def _calculate_degrees (adj_matrix,node_list):
        # For undirected graphs
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()  # Sum of rows (or columns, as it's symmetric)
        degrees_tuple = [(node,degrees[node]) for node in node_list]
        return degrees_tuple
    
    @staticmethod
    def _calculate_centrality (adj_matrix,node_list):
       row_indices,col_indices = adj_matrix.nonzero()
       edgelists = list(zip(row_indices,col_indices))
       G = nx.Graph(edgelists)
       centrality_node = nx.degree_centrality(G)
       centrality_values = [(node,centrality_node[node]) for node in node_list]
       return centrality_values
import os.path as osp

import torch
import torch_geometric.transforms as T
import sys
import os
import statistics as stats
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GreatX.greatx.attack.targeted import SGAttack
from GreatX.greatx.datasets import GraphDataset
from GreatX.greatx.nn.models import SGC,GCN
from GreatX.greatx.training import Trainer
from GreatX.greatx.utils import mark, split_nodes
from adversarial_attack.utils import select_nodes_margin,select_nodes_margin_v1
from data_loader.DataLoader import DataLoader
from utility.ConfigHandler import ConfigHandler
from evaluation.ModelSupervisor import ModelSupervisor





dataset = 'pubmed'
root = 'data/dataset/'
dataset = GraphDataset(root=root, name=dataset,
                       transform=T.LargestConnectedComponents())

data = dataset[0]
print(data)
splits = split_nodes(data.y, random_state=15)

num_features = data.x.size(-1)
num_classes = data.y.max().item() + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


surrogate_model = Trainer(SGC(num_features, num_classes), device=device, lr=0.1,weight_decay=1e-5)
surrogate_model.fit(data, mask=(splits.train_nodes, splits.val_nodes))
gin_config = ConfigHandler.load_model_config(model_type="GIN")


attacker = SGAttack(data, device=device)

row = []
for budget in range(1,6):
    budget_result = []
    for i in range(5):
        data_our = DataLoader.load("pubmed")
        supervisor_surrogate = ModelSupervisor(data_our,device=device,**gin_config)
        supervisor_surrogate.update_split(splits.train_nodes,splits.val_nodes,splits.test_nodes)
        supervisor_surrogate.train_model()
        target_nodes_list = select_nodes_margin_v1(supervisor_surrogate)
        miss = 0
        for target_node in target_nodes_list:

            attacker.setup_surrogate(surrogate_model.model)
            attacker.reset()
            attacker.set_max_perturbations(budget)

            attacker.attack(target_node,num_budgets=budget)

            supervisor_surrogate = ModelSupervisor(data_our,device=device,**gin_config)
            supervisor_surrogate.update_structure_with_edgelist(attacker.data().edge_index.cpu().numpy())
            supervisor_surrogate.update_split(splits.train_nodes,splits.val_nodes,splits.test_nodes)
            supervisor_surrogate.train_model()
            
            
            output = supervisor_surrogate.get_model_output()[target_node]
            if torch.argmax(output).item() != data.y[target_node]:
                miss += 1

        budget_result.append(miss/40)
    
    row.append("{:.3f},{:.3f}".format(stats.mean(budget_result),stats.stdev(budget_result)))

df_result = pd.DataFrame([row],columns=[1,2,3,4,5])
print(df_result)
df_result.to_csv('pubmed_gin.csv')
    
    

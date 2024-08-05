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
from adversarial_attack.utils import select_nodes_margin


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


attacker = SGAttack(data, device=device)

row = []
for budget in range(1,5):
    budget_result = []
    for i in range(5):
        node_select_model = Trainer(GCN(num_features, num_classes), device=device)
        node_select_model.fit(data, mask=(splits.train_nodes, splits.val_nodes))
        target_nodes_list = select_nodes_margin(node_select_model,data,splits.test_nodes,data.y)
        miss = 0
        for target_node in target_nodes_list:

            attacker.setup_surrogate(surrogate_model.model)
            attacker.reset()
            attacker.set_max_perturbations(budget)

            attacker.attack(target_node,num_budgets=budget)

            #SGA
            trainer_after = Trainer(GCN(num_features, num_classes), device=device)
            trainer_after.fit(attacker.data(), mask=(splits.train_nodes, splits.val_nodes),verbose = False)
            trainer_after.cache_clear()
            output = trainer_after.predict(attacker.data(), mask=target_node)
            if torch.argmax(output).item() != data.y[target_node]:
                miss += 1

        budget_result.append(miss/40)
    
    row.append("{:.3f},{:.3f}".format(stats.mean(budget_result),stats.stdev(budget_result)))

df_result = pd.DataFrame([row],columns=[1,2,3,4,5])
print(df_result)
df_result.to_csv('pubmed_gcn.csv')
    
    

import os.path as osp

import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
import os 
import sys
import numpy as np
import statistics as stats
import pandas as pd
import timeit


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GreatX.greatx.attack.targeted import SGAttack,Nettack,RandomAttack
from GreatX.greatx.nn.models import SGC,GCN
from GreatX.greatx.training import Trainer
from GreatX.greatx.utils import BunchDict, mark
from adversarial_attack.utils import select_nodes_margin


root = 'data/dataset/'
dataset = PygNodePropPredDataset(root=root, name='ogbn-arxiv',
                                 transform=T.ToUndirected())
data = dataset[0]
splits = dataset.get_idx_split()
splits = BunchDict(train_nodes=splits['train'], val_nodes=splits['valid'],
                   test_nodes=splits['test'])

num_features = data.x.size(-1)
num_classes = data.y.max().item() + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


surrogate_model = Trainer(SGC(num_features, num_classes), device=device, lr=0.1,weight_decay=1e-5)
surrogate_model.fit(data, mask=(splits.train_nodes, splits.val_nodes))

node_select_model = Trainer(GCN(num_features, num_classes), device=device)
node_select_model.fit(data, mask=(splits.train_nodes, splits.val_nodes))


target_nodes_list = select_nodes_margin(node_select_model,data,splits.test_nodes,data.y)
surrogate_model.cache_clear()
attacker_sga = SGAttack(data, device=device)
attacker_rnd = RandomAttack(data,device=device)
rowlist = []
for budget in range(1,6):

    for i in range(5):
        node_select_model = Trainer(GCN(num_features, num_classes), device=device)
        target_nodes_list = select_nodes_margin(node_select_model,data,splits.test_nodes,data.y)
        start = timeit.default_timer()
        miss = 0
        for target_node in target_nodes_list:

            attacker_sga.setup_surrogate(surrogate_model.model)
            attacker_sga.reset()
            attacker_sga.set_max_perturbations(budget)

            attacker_sga.attack(target_node,num_budgets=budget)

            #SGA
            trainer_after = Trainer(GCN(num_features, num_classes), device=device)
            trainer_after.fit(attacker_sga.data(), mask=(splits.train_nodes, splits.val_nodes),verbose = False)
            trainer_after.cache_clear()
            output = trainer_after.predict(attacker_sga.data(), mask=target_node)
            if torch.argmax(output).item() != data.y[target_node]:
                miss[0] += 1

            
    







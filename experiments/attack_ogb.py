import os.path as osp

import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
import os 
import sys
import numpy as np
import statistics as stats
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GreatX.greatx.attack.targeted import SGAttack,Nettack,RandomAttack
from GreatX.greatx.nn.models import SGC,GCN
from GreatX.greatx.training import Trainer
from GreatX.greatx.training.callbacks import ModelCheckpoint
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
    row = []
    budget_result = {'sga':[],'rnd':[]}

    for i in range(5):
        miss = [0,0]
        node_select_model = Trainer(GCN(num_features, num_classes), device=device)
        target_nodes_list = select_nodes_margin(node_select_model,data,splits.test_nodes,data.y)
        
        for target_node in target_nodes_list:

            attacker_sga.setup_surrogate(surrogate_model.model)
            attacker_sga.reset()
            attacker_rnd.reset()
            attacker_sga.attack(target_node,num_budgets=budget)
            attacker_rnd.attack(target_node,num_budgets=budget)

            #SGA
            trainer_after = Trainer(GCN(num_features, num_classes), device=device)
            trainer_after.fit(attacker_sga.data(), mask=(splits.train_nodes, splits.val_nodes),verbose = False)
            trainer_after.cache_clear()
            output = trainer_after.predict(attacker_sga.data(), mask=target_node)
            if torch.argmax(output).item() != data.y[target_node]:
                miss[0] += 1

            #RND
            trainer_after = Trainer(GCN(num_features, num_classes), device=device)
            trainer_after.fit(attacker_rnd.data(), mask=(splits.train_nodes, splits.val_nodes),verbose = False)
            trainer_after.cache_clear()
            output = trainer_after.predict(attacker_rnd.data(), mask=target_node)
            if torch.argmax(output).item() != data.y[target_node]:
                miss[1] += 1
        miss = np.array(miss)  
        miss = miss/40
    
        budget_result['sga'].append(miss[0])
        budget_result['rnd'].append(miss[1])
    row.append("{:.3f},{:.3f}".format(stats.mean(budget_result['sga']), stats.stdev(budget_result['sga'])))
    row.append("{:.3f},{:.3f}".format(stats.mean(budget_result['rnd']), stats.stdev(budget_result['rnd'])))
    
    rowlist.append(row)
df_result = pd.DataFrame(rowlist,columns=['sga','rnd'])
df_result.index = [1,2,3,4,5,6]
print(df_result)
df_result.to_csv("arxiv_sga_rnd.csv")
    
    







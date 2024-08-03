import os.path as osp

import torch
import torch_geometric.transforms as T
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GreatX.greatx.attack.targeted import SGAttack
from GreatX.greatx.datasets import GraphDataset
from GreatX.greatx.nn.models import SGC
from GreatX.greatx.training import Trainer
from GreatX.greatx.training.callbacks import ModelCheckpoint
from GreatX.greatx.utils import mark, split_nodes

dataset = 'pubmed'
root = 'data/dataset/'
dataset = GraphDataset(root=root, name=dataset,
                       transform=T.LargestConnectedComponents())

data = dataset[0]
print(data)
# splits = split_nodes(data.y, random_state=15)

# num_features = data.x.size(-1)
# num_classes = data.y.max().item() + 1

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# target = 1  # target node to attack
# target_label = data.y[target].item()

# trainer_before = Trainer(SGC(num_features, num_classes), device=device, lr=0.1,
#                          weight_decay=1e-5)
# ckp = ModelCheckpoint('model_before.pth', monitor='val_acc')
# trainer_before.fit(data, mask=(splits.train_nodes, splits.val_nodes),
#                    callbacks=[ckp])
# output = trainer_before.predict(data, mask=target)
# print("Before attack:")
# print(mark(output, target_label))

# attacker = SGAttack(data, device=device)
# attacker.setup_surrogate(trainer_before.model)
# attacker.reset()
# attacker.attack(target)

# trainer_after = Trainer(SGC(num_features, num_classes), device=device)
# ckp = ModelCheckpoint('model_after.pth', monitor='val_acc')
# trainer_after.fit(attacker.data(), mask=(splits.train_nodes, splits.val_nodes),
#                   callbacks=[ckp])
# output = trainer_after.predict(attacker.data(), mask=target)
# print("After poisoning attack:")
# print(torch.argmax(output).item())
# print(mark(output, target_label))

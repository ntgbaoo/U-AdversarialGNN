import torch
print(torch.cuda.is_available())

import pandas as pd
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from tqdm import tqdm
import warnings

# from OrbitAttack import OrbitAttack

warnings.filterwarnings("ignore")
# from orbit_table_generator import OrbitTableGenerator
import random
from deeprobust.graph.data import Dataset, Dpr2Pyg
# from model.GIN import GIN
# from model.GSAGE import GraphSAGE
from deeprobust.graph.targeted_attack import Nettack
# from ogb.nodeproppred import PygNodePropPredDataset


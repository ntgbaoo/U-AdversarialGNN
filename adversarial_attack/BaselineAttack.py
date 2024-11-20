from torch_geometric.data import Data
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utility.util import set_random

class DegreeAttack:
    def __init__(self, data:Data , device='cpu',seed = 720):
        set_random(seed)
        self.ori_data = data

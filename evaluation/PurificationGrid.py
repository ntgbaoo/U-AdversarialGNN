from copy import deepcopy 
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.Grid import Grid

from static import *

class PurificationGrid(Grid):
    def __init__(self,config_dict:dict) -> None:
        super.__init__(config_dict)
        self.config_dict = config_dict
    
    def assert_config_dict(self):
        pass
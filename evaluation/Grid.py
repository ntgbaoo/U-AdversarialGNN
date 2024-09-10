from copy import deepcopy 
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from static import *

class Grid:
    def __init__(self,config_dict:dict) -> None:
        self.config_dict = config_dict
    
    def assert_config_dict(self):
        pass